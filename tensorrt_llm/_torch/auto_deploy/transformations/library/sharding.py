"""Transformations to support graph sharding.

Our sharding algorithm for tensor parallelism (TP) is based on the following steps:

    1. Initialize/construct unsharded model. Ideally, this should be done on device="meta" to avoid
       unnecessary memory allocation. In some cases, this is necessary if the model is too large to
       fit on a single device.
    2. Shard the graph IR of the model:
        a. Identify linear nodes that correspond to TP tuples.
        b. Reduce/Shard shape of weights in the corresponding linear nodes accordingly (either in
           row or column dimension). Add all_reduce nodes where necessary (--> only needed for
           fusing results in final linear node of the TP tuple).
        c. Add a checkpoint loading hook to the sharded linear nodes so that only the correct shard
           of the weight from the checkpoint gets loaded.
    3. Load the checkpoint and allocate the tensor. Loading the correct shard from the checkpoint
       happens automatically via the checkpoint loading hook added in step 2c.
"""

import operator
from collections import defaultdict
from functools import partial
from typing import Callable, DefaultDict, Dict, List, Set

import torch
import torch.nn as nn
from torch.fx import GraphModule, Node

from ...distributed.distributed_classes import DistributedTensor, PGrid
from ...utils.logger import ad_logger
from ...utils.node_utils import (
    bfs,
    successors,
    predecessors,
    extract_param_names_from_lin_node,
    identify_regions_between_residuals,
    is_aggregation_op,
    is_linear_op,
    is_contraction_op,
    is_attention_op,
    is_op,
    num_users_of_weight_node,
)
from ...utils.quantization_utils import QuantizationImpl
from .._graph import canonicalize_graph


def _load_hook(
    state_dict,
    prefix,
    *args,
    f_split: Callable[[torch.Tensor, int], torch.Tensor],
    param_key: str,
    param_shape: torch.Size,
):
    # TODO: we need to support loading either a sharded or unsharded checkpoint.
    # Otherwise, basic workflows like
    # model.load_state_dict(model.state_dict()) will fail.
    # This is quite a hacky solution. A better solution would be to store extra_state in
    # the state_dict to identify whether the state_dict is sharded or not.
    key = prefix + param_key
    ad_logger.debug(f"Sharder LOAD hook is called for '{key}'")
    if key not in state_dict:
        return
    p_to_load = state_dict[key]
    p_to_load = p_to_load if param_shape == p_to_load.shape else f_split(p_to_load)
    state_dict[key] = p_to_load


def _load_hook_remove(
    state_dict: Dict,
    prefix: str,
    *args,
    param_key: str,
):
    key = prefix + param_key
    ad_logger.debug(f"Sharder LOAD hook is called for '{key}'")
    state_dict.pop(key, None)


def _insert_sharded_matmul(
    gm: GraphModule, node: Node, dim: int, rank: int, world_size: int, add_dist: bool = False
):
    """Replaces the matmul node with a new matmul node that accepts sharded weights.

    The state_dict is also updated to contain the sharded weights.
    """
    assert dim in [0, 1], "Only dim 0 and 1 are supported for sharding"
    assert add_dist or dim == 0, "For dim=1 sharding, dist_op is required."

    quantization_impl = QuantizationImpl.create(node)

    def split_tensor(
        t: torch.Tensor, d: int = dim, r: int = rank, ws: int = world_size
    ) -> torch.Tensor:
        return torch.tensor_split(t, ws, dim=d)[r]

    num_users = num_users_of_weight_node(node)
    if num_users > 1 or num_users == 0:
        ad_logger.warning(
            f"Weight node {node} has {num_users} users. This is not supported for sharding. Skipping."
        )
        return
    # get weight and bias key
    weight_key, bias_key = extract_param_names_from_lin_node(node)

    modname = weight_key.rpartition(".")[0]
    submod = gm.get_submodule(modname)

    def set_new_param(submod: nn.Module, param_key: str, remove: bool = False) -> torch.Size:
        # split or remove it
        param_new = (
            None
            if remove
            else nn.Parameter(
                split_tensor(gm.get_parameter(param_key)).detach().clone(),
                requires_grad=quantization_impl is None,
            )
        )

        # update the parameter
        param_name = param_key.rpartition(".")[-1]
        setattr(submod, param_name, param_new)
        return torch.Size() if param_new is None else param_new.shape

    # update weight
    weight_new_shape = set_new_param(submod, weight_key)
    gm._register_load_state_dict_pre_hook(
        partial(
            _load_hook, f_split=split_tensor, param_key=weight_key, param_shape=weight_new_shape
        )
    )

    if bias_key is not None and dim == 0:
        # update bias for dim 0 --> we can handle it like the weight
        bias_new_shape = set_new_param(submod, bias_key)
        gm._register_load_state_dict_pre_hook(
            partial(
                _load_hook, f_split=split_tensor, param_key=bias_key, param_shape=bias_new_shape
            )
        )
    elif bias_key is not None and rank != world_size - 1:
        # update the bias for dim 1 --> in this case only the last rank gets the bias to avoid
        # double counting it. For all other we will delete the bias.
        args = list(node.args)
        node_bias = args[2]
        args[2] = None
        node.args = tuple(args)
        gm.graph.erase_node(node_bias)
        set_new_param(submod, bias_key, remove=True)
        gm._register_load_state_dict_pre_hook(partial(_load_hook_remove, param_key=bias_key))

    if quantization_impl:
        scales = {}
        for scale_name in quantization_impl.scale_names():
            scales[scale_name] = submod.get_buffer(scale_name)
        scales["weight_shape"] = weight_new_shape
        sharded_scales = quantization_impl.shard_scales(dim, rank, world_size, **scales)
        for k, v in sharded_scales.items():
            submod.register_buffer(k, v)

        gm._register_load_state_dict_pre_hook(
            partial(
                quantization_impl.shard_load_hook,
                weight_name=weight_key,
                weight_shape=weight_new_shape,
                dim=dim,
                rank=rank,
                world_size=world_size,
            )
        )

    # no comm node needed for single device
    if not add_dist:
        return

    # figure out the right dist op
    dist_lookup = {
        0: (torch.ops.dist.all_gather, -1),
        1: (torch.ops.dist.all_reduce,),
    }
    fn_dist, *dist_args = dist_lookup[dim]

    # add reduction node
    with gm.graph.inserting_after(node):
        dist_node = gm.graph.call_function(fn_dist, args=(node, *dist_args))
        node.replace_all_uses_with(dist_node)
        dist_node.replace_input_with(dist_node, node)


def _simple_shard(
    gm: GraphModule, nodes_linear: Dict[Node, List[Node]], rank: int, world_size: int
):
    # for every linear node:
    # --> row_split (dim 0 of weight) + all_gather (dim -1 of output)
    for node_group in nodes_linear.values():
        for n in node_group:
            _insert_sharded_matmul(gm, n, 0, rank, world_size, add_dist=True)


def column_row_shard(
    gm: GraphModule, rank: int, world_size: int, simple_shard_only: bool = False
) -> GraphModule:
    """A transformation to apply sharding to the model following tensor parallelism.

    The transformation is based on the following steps:

    1. Identify boundary nodes between residual nodes to identify shardable regions.
    2. Identify the GEMM nodes that can be sharded
    3. Trace through the subgraph using DFS/BFS between each pair of boundary nodes
    4. Account for each node in the trace to ensure the op is correct even after sharding. This is
       necessary to ensure that the sharding is correct and we need to be able to account for
       **all** nodes in the subgraph. The subgraph here is defined as the region between the first
       linear node to the last linear node of an identified sharding region.
    # 5. Shard the GEMM nodes or skip accordingly.
    """
    ad_logger.debug("Before sharding graph: " + str(gm))

    if world_size < 2:
        ad_logger.info("Skipping sharding for single device")
        return gm

    assert isinstance(gm, GraphModule), "Expecting GraphModule"

    # find boundary nodes of regions we want to shard
    boundary_nodes = identify_regions_between_residuals(gm)

    # TODO: continue updating these lists
    # pointwise ops that don't affect the sharder
    pointwise_ops = {
        torch.ops.aten.gelu,
        torch.ops.aten.leaky_relu,
        torch.ops.aten.mul,
        torch.ops.aten.relu,
        torch.ops.aten.sigmoid,
        torch.ops.aten.silu,
        torch.ops.aten.tanh,
        torch.ops.aten.contiguous,
    }

    # acceptable attention nodes between sharded GEMMs
    shardable_attention_nodes = {
        torch.ops.attention.scaled_dot_product_attention,
        torch.ops.attention.grouped_sdpa,
        torch.ops.attention.bsnd_grouped_sdpa,
    }

    # This is a heuristic. Basically, we assume those are okay to shard if we also encounter an
    # attention node because we know that those ops must be compatible with the attention op. Now
    # since the attention op is shardable, we will assume those are as well if used in conjunction
    # with the attention op.
    shardable_nodes_with_attention = {
        torch.ops.aten.view,
        torch.ops.aten.reshape,
        torch.ops.rope.flashinfer,
        operator.getitem,
    }

    # let's look at linear nodes we can identify between pairs of boundary nodes
    # There is three potential cases we can handle:
    # 1. No linear nodes:
    #       --> just continue
    # 2. Two groups of linear nodes and we can account for all to the view nodes:
    #       --> row_split (dim 0) 1st group + check for supported nodes +
    #           col_split (dim 1) 2nd group + all_reduce output of 2nd group
    # 3. Linear nodes that are not in two groups or we cannot account for all nodes:
    #       --> row_split (dim 0 of weight) + all_gather (dim -1 of output) output
    num_shards = 0
    for n_start, n_end in zip(boundary_nodes[:-1], boundary_nodes[1:]):
        # we iterate through all nodes between the two boundary nodes and store linear nodes
        # sorted by their input activation node. We also store remaining nodes.
        nodes_linear: DefaultDict[Node, List[Node]] = defaultdict(list)
        attention_nodes: Set[Node] = set()
        attention_related_nodes: Set[Node] = set()
        unaccounted_nodes: Set[Node] = set()
        current_node = n_start
        while current_node != n_end:
            if is_linear_op(current_node, include_quantization=True):
                nodes_linear[current_node.args[0]].append(current_node)
            elif is_op(current_node, shardable_attention_nodes):
                attention_nodes.add(current_node)
            elif is_op(current_node, shardable_nodes_with_attention):
                attention_related_nodes.add(current_node)
            elif not is_op(current_node, pointwise_ops):
                unaccounted_nodes.add(current_node)
            current_node = current_node.next
            assert current_node, "Could not identify next node"

        all_nodes_between_start_end = [n for n in gm.graph.nodes if n_start <= n < n_end]

        # nothing to shard
        if len(nodes_linear) == 0:
            continue

        # simple shard when we have != 2 groups of linear nodes
        if len(nodes_linear) != 2:
            ad_logger.debug(f"Linear groups: {nodes_linear}")
            _simple_shard(gm, nodes_linear, rank, world_size)
            continue

        # let's look at the unnacounted nodes. They are okay as long as they fall before the
        # first linear node or after the last linear node, i.e., outside the sharded region
        lin_nodes_flat: Set[Node] = {n for group in nodes_linear.values() for n in group}
        lin_nodes_passed: Set[Node] = set()
        current_node = n_start
        while current_node != n_end:
            # check if this is another linear node
            if current_node in lin_nodes_flat:
                lin_nodes_passed.add(current_node)

            # check if we are OUTSIDE sharded region
            if len(lin_nodes_passed) == 0 or lin_nodes_passed == lin_nodes_flat:
                # remove node from unaccounted nodes since we are outside and it doesn't matter
                unaccounted_nodes.discard(current_node)
                attention_related_nodes.discard(current_node)
                attention_nodes.discard(current_node)

            current_node = current_node.next

        # let's post-process the attention-related nodes
        # we can disregard them if we also see attention nodes and we assume they are compatible
        if len(attention_nodes) > 0:
            attention_related_nodes.clear()

        # check if any unaccounted nodes are left. If so, do a simply shard
        if unaccounted_nodes or attention_related_nodes:
            ad_logger.debug(f"Unaccounted nodes: {unaccounted_nodes}")
            _simple_shard(gm, nodes_linear, rank, world_size)
            continue

        # If we can account for all sharded nodes, we can do a two-way shard
        # --> row_split (dim 0) + col_split (dim 1) + all_reduce
        for i, group in enumerate(nodes_linear.values()):
            for n in group:
                _insert_sharded_matmul(gm, n, i, rank, world_size, add_dist=i > 0)

    # canonicalize and return
    gm = canonicalize_graph(gm)
    ad_logger.debug("After sharding: " + str(gm))
    return gm


def get_node_dict(gm: GraphModule) -> Dict[str, Node]:
    node_dict = {}
    for n in gm.graph.nodes:
        node_dict[n.name] = n
    return node_dict

def get_nth_predecessor(n: Node, nth: int) -> Node:
    for i, p in enumerate(n.args):
        if p is not None and isinstance(p, Node):
            if i == nth:
                return p
            else:
                return get_nth_predecessor(p, nth - 1)
            
def get_nth_successor(n: Node, nth: int) -> Node:
    for i, p in enumerate(n.users):
        if p is not None and isinstance(p, Node):
            if i == nth:
                return p
            else:
                return get_nth_successor(p, nth - 1)

def column_row_shard_2(gm: GraphModule, rank: int, world_size: int, config) -> GraphModule:
    g = get_node_dict(gm)
    for n in gm.graph.nodes:
        if "distributed" not in n.meta:
            n.meta["distributed"] = {}
        
        if "val" in n.meta:
            # tensors are initially, by default, replicated
            input_is_column_sharded = False
        else:
            # these are states, parameters, constants, but also, "flashinfer" falls here
            input_is_column_sharded = True
        
        n.meta["distributed"]["is_column_sharded"] = input_is_column_sharded
        
        
        # find the input distribution
        
        all_inputs_are_column_sharded = set([s.meta["distributed"]["is_column_sharded"] 
                            for s in n.args 
                            if s is not None and 
                            isinstance(s,Node) and 
                            'weight' not in s.name and
                            # 'val' in s.meta and
                            "distributed" in s.meta])
        if len(all_inputs_are_column_sharded) > 1:
            # We have conflicting input distributions: some inputs are sharded, some are not.
            # We have three options:
            # 1. We check if indeed this operation could potentially be sharded. If not,
            #    we made mistake in the previous sharding step and raise an error.
            # 2. We can shard the input that is not sharded
            # 3. We all-gather the input that is not sharded
            
            # 1. Check if this is legal
            if is_aggregation_op(n) and is_aggregation_op(n)[1] == 2:
                raise ValueError(f"Operation {n} has some of its inputs sharded, which is not allowed.")
            
            sinks = find_all_boundary_nodes(
                n,
                lambda x: is_aggregation_op(x),
                attr_next="users",
            )
            if sinks:
                all_sink_aggregation_dims = set([is_aggregation_op(s)[1] for s in sinks])
                # dims are [batch, sequence, embedding]
                if 2 in all_sink_aggregation_dims:
                    # dim == 2 means that the sink aggregation operation
                    # performs aggregation across the embedding dimension,
                    # therefore, X cannot be shared (otherwise, that would 
                    # imply distributed aggegation)
                    raise ValueError(f"Operation {n} has some of its inputs sharded, which is not allowed.")
                
            # # 2. Shard the input that is not sharded
            # for s in n.args:
            #     if s is not None and isinstance(s,Node) and "distributed" in s.meta:
            #         if not s.meta["distributed"]["is_column_sharded"]:
            #             with gm.graph.inserting_before(s):
            #                 tensor_slice = gm.graph.call_function(
            #                     torch.ops.aten.slice.Tensor, args=(tensor_node, 0, start_idx, end_idx, 1)
            #                 )
            #             # Update BMM node to use the sliced tensor
            #             bmm_node.update_arg(arg_idx, tensor_slice)
            # 3. All-gather the input that is not sharded
            
        if all_inputs_are_column_sharded:
            input_is_column_sharded = all_inputs_are_column_sharded.pop()
        n.meta["distributed"]["is_column_sharded"] = input_is_column_sharded
        
        # only linear ops can change distribution. All other ops preserve the distribution    
        if is_linear_op(n):            
            # find the required output distribution
            sinks = find_all_boundary_nodes(
                n,
                lambda x: is_aggregation_op(x),
                attr_next="users",
            )
            weigh = n.args[1]
            can_output_be_column_sharded = True
            if sinks:
                # check if all sinks are aggregation operations
                all_sink_aggregation_dims = set([is_aggregation_op(s)[1] for s in sinks])
                # dims are [batch, sequence, embedding]
                if 2 in all_sink_aggregation_dims:
                    # dim == 2 means that the sink aggregation operation
                    # performs aggregation across the embedding dimension,
                    # therefore, X cannot be shared (otherwise, that would 
                    # imply distributed aggegation)
                    can_output_be_column_sharded = False
            
            
            output_is_column_sharded = can_output_be_column_sharded and not input_is_column_sharded
            _insert_sharded_matmul(gm, 
                                   n, 
                                   dim = 1 if input_is_column_sharded else 0, 
                                   rank = rank, 
                                   world_size = world_size, 
                                   add_dist = not output_is_column_sharded)
            if "distributed" not in n.meta:
                n.meta["distributed"] = {}
            n.meta["distributed"]["is_column_sharded"] = output_is_column_sharded
            stat = (n, weigh.name, input_is_column_sharded, can_output_be_column_sharded, output_is_column_sharded)
            print(f"stat: {stat}")
            if not can_output_be_column_sharded and not input_is_column_sharded:
                print(f"\nWarning, simple shard detected! {stat}")
            
        # but attention nodes, if their inputs are NOT sharded, 
        # can do a column-split to allow distributed attention computation
        if is_attention_op(n) and not input_is_column_sharded:
            # find the required output distribution
            sinks = find_all_boundary_nodes(
                n,
                lambda x: is_aggregation_op(x),
                attr_next="users",
            )
            can_output_be_column_sharded = True
            if sinks:
                # check if all sinks are aggregation operations
                all_sink_aggregation_dims = set([is_aggregation_op(s)[1] for s in sinks])
                # dims are [batch, sequence, embedding]
                if 2 in all_sink_aggregation_dims:
                    # dim == 2 means that the sink aggregation operation
                    # performs aggregation across the embedding dimension,
                    # therefore, X cannot be shared (otherwise, that would 
                    # imply distributed aggegation)
                    can_output_be_column_sharded = False
            
            if can_output_be_column_sharded:
                inputs = [s for s in n.args if s is not None and isinstance(s, Node)]
                assert len(inputs) >= 3, "Attention node should have at least 3 inputs"
                q_node, k_node, v_node = inputs[:3]
                # get the num_heads and head_dims (potentially, latent_dim > head_dim for q and k)
                if is_op(n, torch.ops.attention.bsnd_grouped_sdpa):
                    head_dim_no = 2
                else:
                    head_dim_no = 3                
                num_heads = set([s.meta["val"].shape[head_dim_no] for s in [q_node, k_node, v_node]])
                assert len(num_heads) == 1, "All inputs to attention node should have the same number of heads"
                num_heads = num_heads.pop()
                qk_head_dim =  set([s.meta["val"].shape[-1] for s in [q_node, k_node]])
                assert len(qk_head_dim) == 1, "All inputs to attention node should have the same head dimension"
                qk_head_dim = qk_head_dim.pop()
                v_head_dim = v_node.meta["val"].shape[-1]
                
                assert (num_heads > world_size) and num_heads % world_size == 0, "Number of heads must be divisible by world size"
                heads_per_rank = num_heads // world_size
                for i, src in enumerate([q_node, k_node, v_node]):
                    # column-split the input
                    distribute_tensor(gm, 
                                      consumer_node = n, 
                                      tensor_node = src, 
                                      split_dim = head_dim_no, 
                                      arg_idx = i, 
                                      start_idx = heads_per_rank * rank, 
                                      end_idx = heads_per_rank * (rank + 1))
                n.meta["distributed"]["is_column_sharded"] = True
                
                
    return gm


def find_all_boundary_nodes(
    node: Node,
    target: Callable,
    attr_next: str = "users",
) -> Node:
    queue = [node]
    visited = set()
    boundary_nodes = []
    
    visited.add(node)
    queue = list(n for n in getattr(node, attr_next) if n is not None 
                    and isinstance(n, Node))
    while queue:
        cur_node = queue.pop(0)
        if target(cur_node):
            # don't continue pass the boundary condition
            boundary_nodes.append(cur_node)
        else:
            for next_node in getattr(cur_node, attr_next):
                if next_node is None or not isinstance(next_node, Node):
                    continue
                if next_node not in visited:
                    visited.add(next_node)
                    queue.append(next_node)
    return boundary_nodes


def distribute_3d(gm: GraphModule, rank: int, world_size: int, config) -> GraphModule:
    ad_logger.debug("Before sharding graph: " + str(gm))

    if world_size < 2:
        ad_logger.info("Skipping sharding for single device")
        return gm

    assert isinstance(gm, GraphModule), "Expecting GraphModule"

    batch_size = config.batch_size
    seq_len = config.seq_len
    embd = config.hidden_size
    num_heads = config.num_attention_heads
    head_dim = config.hidden_size // config.num_attention_heads
    if "num_key_value_heads" in config:
        num_kv_heads = config.num_key_value_heads
    else:
        num_kv_heads = num_heads
    inter_dim = embd
    mlp_dim = embd
    if "intermediate_size" in config:
        inter_dim = config.intermediate_size
    if "intermediate_size_mlp" in config:
        mlp_dim = config.intermediate_size_mlp
    vocab_size = config.vocab_size
    kv_dim = embd * num_kv_heads // num_heads
    if "q_lora_rank" in config:
        q_lora_rank = config.q_lora_rank
    else:
        q_lora_rank = -1
    if "kv_lora_rank" in config:
        kv_lora_rank = config.kv_lora_rank
    else:
        kv_lora_rank = -1

    modes_extents = {
        "b": batch_size,
        "s": seq_len,
        "e": embd,
        "f": embd,
        "h": num_heads,
        "d": head_dim,
        "v": vocab_size,
        "m": mlp_dim,
        "i": inter_dim,
        "n": num_kv_heads,
        "k": kv_dim,
        "P": world_size,
        "q": q_lora_rank,
        "l": kv_lora_rank,
    }

    def shape_to_einsum(fake_tensor_shape: torch.Size) -> str:
        # b: batch size
        # e: embedding size
        # h: num heads
        # d: head dim
        # v: vocab size
        # m: mlp dim
        # n: num kv heads
        # P: world size
        einsum_str = ""
        e_used = False
        for dim in fake_tensor_shape:
            if dim == embd:
                if not e_used:
                    einsum_str += "e"
                    e_used = True
                else:
                    einsum_str += "f"
            elif dim == num_heads:
                einsum_str += "h"
            elif dim == head_dim:
                einsum_str += "d"
            elif dim == vocab_size:
                einsum_str += "v"
            elif dim == mlp_dim:
                einsum_str += "m"
            elif dim == num_kv_heads:
                einsum_str += "n"
            elif dim == world_size:
                einsum_str += "P"
            elif dim == seq_len:
                einsum_str += "s"
            elif dim == batch_size:
                einsum_str += "b"
            elif dim == kv_dim:
                einsum_str += "k"
            elif dim == q_lora_rank:
                einsum_str += "q"
            elif dim == kv_lora_rank:
                einsum_str += "l"
            else:
                einsum_str += "u"
                ad_logger.debug(f"Unknown dimension: {dim}")

        return einsum_str

    import copy

    seq_len_copy = copy.deepcopy(seq_len)
    a = 1
    # first pass over nodes - assign dist_tensor to each node
    for n in gm.graph.nodes:
        # infer data shapes
        # if tensor_meta is available, use it to infer the shape
        if "tensor_meta" in n.meta:
            tensor_meta = n.meta["tensor_meta"]
        else:
            # get it from parent or child node
            if len(n.args) > 0:
                parent_node = n.args[0]
                if isinstance(parent_node, tuple):
                    parent_node = parent_node[0]
                if "tensor_meta" not in parent_node.meta:
                    tensor_meta = None
                else:
                    tensor_meta = parent_node.meta["tensor_meta"]
            elif len(n.users) > 0:
                child_node = list(n.users)[0]
                tensor_meta = child_node.meta["tensor_meta"]
            else:
                raise ValueError(f"Node {n} has no args or users")
            
                # figure out the tensor shape as an einsum string
        if tensor_meta is not None:
            einsum_str = shape_to_einsum(tensor_meta.shape)
        else:
            einsum_str = "u"  # u for unknown

        dist_tensor = DistributedTensor(modes_extents, einsum_str)
        n.meta["dist_tensor"] = dist_tensor

        if is_contraction_op(n):
            # find the input distribution
            sources = find_all_boundary_nodes(
                n,
                # lambda x: is_aggregation_op(x),
                lambda x: is_contraction_op(x),
                attr_next="args",
            )
            if sources:
                if len(sources) > 1:
                    a = 1
                input_p_grid = sources.meta["p_grid"]
                prev_rank_order = input_p_grid.rank_order
            else:
                prev_rank_order = (2, 1, 0)
            
            # find the required output distribution
            sinks = find_all_boundary_nodes(
                n,
                lambda x: is_aggregation_op(x),
                attr_next="users",
            )
            dim = None
            if sinks:
                if len(sinks) > 1:
                    a = 1
                op, dim = is_aggregation_op(sinks[0])
            if dim is not None:
                pass

            # Infer the contraction einsum string
            A = n.args[0].meta["dist_tensor"]
            B = n.args[1].meta["dist_tensor"]
            C = n.meta["dist_tensor"]
            
            contraction_einsum_str = f"{A.einsum_str},{B.einsum_str}->{C.einsum_str}"
            a = 1
            # assert A.N == B.N, "Input tensors must have the same embedding dimension"
            # assert A.K == B.K, "Input tensors must have the same embedding dimension"
            # assert A.M == C.M, "Output tensor must have the same embedding dimension"
            # assert A.K == C.K, "Output tensor must have the same embedding dimension"
            # We assume that the linear node performs contraction:
            # Y = X @ W^T
            # defined by the einsum string:
            # bse, ef -> bsf
            # the mode extent f differs depending on the layer:
            # for MLP, that is the intermediate size (mlp_dim)
            # for standard attention, Q and Out projections that is the embedding size (embd)
            # for the key and value projections, that is the embedding size (embd)
            # for the output, that is the embedding size (embd)
            # for the input, that is the embedding size (embd)
            # for the key, that is the embedding size (embd)
            # for the value, that is the embedding size (embd)
            # for the query, that is the embedding size (embd)

            n.meta["p_grid"] = PGrid(
                M_global=batch_size * seq_len,
                N_global=embd,
                K_global=embd,
                world_size=world_size,
                rank_order=(prev_rank_order[2], prev_rank_order[1], prev_rank_order[0]),
                nonparallelizable_dim=dim,
            )

            



    linear_layers = [n for n in gm.graph.nodes if is_linear_op(n)]
    aggregation_nodes = [n for n in gm.graph.nodes if is_aggregation_op(n)]

    # second pass over nodes - find all linear nodes and assign distributed computation grid PGrid to each node
    for n in gm.graph.nodes:
        if is_linear_op(n, include_quantization=True):
            prev = bfs(
                n,
                lambda x: is_aggregation_op(x),
                attr_next="args",
                skip_root=True,
                allow_empty=True,
            )
            if prev is not None:
                input_p_grid = prev.meta["p_grid"]
                prev_rank_order = input_p_grid.rank_order
            else:
                prev_rank_order = (2, 1, 0)
            successor = bfs(
                n,
                lambda x: is_aggregation_op(x),
                attr_next="users",
                skip_root=True,
                allow_empty=True,
            )
            dim = None
            if successor is not None:
                op, dim = is_aggregation_op(successor)
            if dim is not None:
                pass
            n.meta["p_grid"] = PGrid(
                M_global=batch_size * seq_len,
                N_global=embd,
                K_global=embd,
                world_size=world_size,
                rank_order=(prev_rank_order[2], prev_rank_order[1], prev_rank_order[0]),
            )

    # TODO: continue updating these lists
    # pointwise ops that don't affect the sharder
    pointwise_ops = {
        torch.ops.aten.gelu,
        torch.ops.aten.leaky_relu,
        torch.ops.aten.mul,
        torch.ops.aten.relu,
        torch.ops.aten.sigmoid,
        torch.ops.aten.silu,
        torch.ops.aten.tanh,
        torch.ops.aten.contiguous,
    }

    # acceptable attention nodes between sharded GEMMs
    shardable_attention_nodes = {
        torch.ops.attention.scaled_dot_product_attention,
        torch.ops.attention.grouped_sdpa,
        torch.ops.attention.bsnd_grouped_sdpa,
    }

    # This is a heuristic. Basically, we assume those are okay to shard if we also encounter an
    # attention node because we know that those ops must be compatible with the attention op. Now
    # since the attention op is shardable, we will assume those are as well if used in conjunction
    # with the attention op.
    shardable_nodes_with_attention = {
        torch.ops.aten.view,
        torch.ops.aten.reshape,
        torch.ops.rope.flashinfer,
        operator.getitem,
    }

    # let's look at linear nodes we can identify between pairs of boundary nodes
    # There is three potential cases we can handle:
    # 1. No linear nodes:
    #       --> just continue
    # 2. Two groups of linear nodes and we can account for all to the view nodes:
    #       --> row_split (dim 0) 1st group + check for supported nodes +
    #           col_split (dim 1) 2nd group + all_reduce output of 2nd group
    # 3. Linear nodes that are not in two groups or we cannot account for all nodes:
    #       --> row_split (dim 0 of weight) + all_gather (dim -1 of output) output
    for n_start, n_end in zip(boundary_nodes[:-1], boundary_nodes[1:]):
        # we iterate through all nodes between the two boundary nodes and store linear nodes
        # sorted by their input activation node. We also store remaining nodes.
        nodes_linear: DefaultDict[Node, List[Node]] = defaultdict(list)
        attention_nodes: Set[Node] = set()
        attention_related_nodes: Set[Node] = set()
        unaccounted_nodes: Set[Node] = set()
        current_node = n_start
        while current_node != n_end:
            if is_linear_op(current_node, include_quantization=True):
                nodes_linear[current_node.args[0]].append(current_node)
            elif is_op(current_node, shardable_attention_nodes):
                attention_nodes.add(current_node)
            elif is_op(current_node, shardable_nodes_with_attention):
                attention_related_nodes.add(current_node)
            elif not is_op(current_node, pointwise_ops):
                unaccounted_nodes.add(current_node)
            current_node = current_node.next
            assert current_node, "Could not identify next node"

        all_nodes_between_start_end = [n for n in gm.graph.nodes if n_start <= n < n_end]

        # nothing to shard
        if len(nodes_linear) == 0:
            continue

        num_shards += 1

        if simple_shard_only:
            ad_logger.debug(f"Forcing Simple Shard: Linear groups: {nodes_linear}")
            _simple_shard(gm, nodes_linear, rank, world_size)
            continue

        # simple shard when we have != 2 groups of linear nodes
        if len(nodes_linear) != 2:
            ad_logger.debug(f"Linear groups: {nodes_linear}")
            _simple_shard(gm, nodes_linear, rank, world_size)
            continue

        # let's look at the unnacounted nodes. They are okay as long as they fall before the
        # first linear node or after the last linear node, i.e., outside the sharded region
        lin_nodes_flat: Set[Node] = {n for group in nodes_linear.values() for n in group}
        lin_nodes_passed: Set[Node] = set()
        current_node = n_start
        while current_node != n_end:
            # check if this is another linear node
            if current_node in lin_nodes_flat:
                lin_nodes_passed.add(current_node)

            # check if we are OUTSIDE sharded region
            if len(lin_nodes_passed) == 0 or lin_nodes_passed == lin_nodes_flat:
                # remove node from unaccounted nodes since we are outside and it doesn't matter
                unaccounted_nodes.discard(current_node)
                attention_related_nodes.discard(current_node)
                attention_nodes.discard(current_node)

            current_node = current_node.next

        # let's post-process the attention-related nodes
        # we can disregard them if we also see attention nodes and we assume they are compatible
        if len(attention_nodes) > 0:
            attention_related_nodes.clear()

        # check if any unaccounted nodes are left. If so, do a simply shard
        if unaccounted_nodes or attention_related_nodes:
            ad_logger.debug(f"Unaccounted nodes: {unaccounted_nodes}")
            _simple_shard(gm, nodes_linear, rank, world_size)
            continue

        # If we can account for all sharded nodes, we can do a two-way shard
        # --> row_split (dim 0) + col_split (dim 1) + all_reduce
        for i, group in enumerate(nodes_linear.values()):
            for n in group:
                _insert_sharded_matmul(gm, n, i, rank, world_size, add_dist=i > 0)

    # canonicalize and return
    if num_shards:
        gm = canonicalize_graph(gm)
    ad_logger.debug("After sharding: " + str(gm))
    ad_logger.info(f"Found {num_shards} TP shards")
    return gm


def dp_bmm_shard(gm: GraphModule, rank: int, world_size: int) -> GraphModule:
    """A transformation to apply sharding to batched matrix multiplications in the graph.

    We'll shard the BMM nodes by slicing the batch dimension of input tensors into world_size number of slices.
    After sharding each BMM node, we'll insert an all_gather node to gather the results across the different devices.
    This transformation handles any combination of tensor types for both inputs to the BMM operation.

    We'll also assume that the inputs to BMM are broadcasted across the devices already.
    """
    ad_logger.debug("Before sharding graph: " + str(gm))

    if world_size < 2:
        ad_logger.info("Skipping sharding for single device")
        return gm

    assert isinstance(gm, GraphModule), "Expecting GraphModule"

    num_bmm_shards = 0

    def handle_tensor(
        bmm_node: Node, tensor_node: Node, arg_idx: int, start_idx: int, end_idx: int
    ):
        """Unified helper function to shard either a parameter tensor or a dynamic tensor.

        Args:
            bmm_node: The BMM node that is being processed
            tensor_node: The input tensor node to shard
            arg_idx: The argument index of the tensor in the BMM node
            start_idx: Start index for sharding
            end_idx: End index for sharding
        """

        # Define slice function for the sharding
        def slice_tensor(t: torch.Tensor) -> torch.Tensor:
            return t[start_idx:end_idx]

        if tensor_node.op == "get_attr":
            # Handle parameter tensor
            weight_key = tensor_node.target
            modname, _, param_name = weight_key.rpartition(".")
            param = gm.get_parameter(weight_key)

            # Update the parameter with its shard
            param_new = nn.Parameter(slice_tensor(param).detach().clone(), requires_grad=True)
            gm.get_submodule(modname).register_parameter(param_name, param_new)

            # Register load state dict hook
            gm._register_load_state_dict_pre_hook(
                partial(
                    _load_hook,
                    f_split=slice_tensor,
                    param_key=weight_key,
                    param_shape=param_new.shape,
                )
            )
        else:
            # Handle dynamic tensor
            with gm.graph.inserting_before(bmm_node):
                tensor_slice = gm.graph.call_function(
                    torch.ops.aten.slice.Tensor, args=(tensor_node, 0, start_idx, end_idx, 1)
                )
            # Update BMM node to use the sliced tensor
            bmm_node.update_arg(arg_idx, tensor_slice)

    for node in gm.graph.nodes:
        if not is_op(node, {torch.ops.aten.bmm}):
            continue

        ad_logger.debug(f"Found BMM node: {node}")

        # Get the input tensors
        lhs_tensor = node.args[0]
        rhs_tensor = node.args[1]

        # Check batch sizes from meta information
        lhs_batch_size = lhs_tensor.meta["val"].shape[0]
        rhs_batch_size = rhs_tensor.meta["val"].shape[0]

        assert lhs_batch_size == rhs_batch_size, "Batch sizes of both tensors must match"
        bmm_batch_size = lhs_batch_size

        # Calculate balanced distribution
        base_size = bmm_batch_size // world_size
        remainder = bmm_batch_size % world_size

        # NOTE: our torch.ops.dist.all_gather doesn't support uneven splits at the moment.
        if remainder:
            ad_logger.warning(
                f"BMM batch size {bmm_batch_size} is not divisible by world size {world_size}. "
                f"This will result in uneven distribution of work across devices. Skipping."
            )
            continue

        # Calculate start and end indices for this rank
        if rank < remainder:
            start_idx = rank * (base_size + 1)
            end_idx = start_idx + base_size + 1
        else:
            start_idx = remainder + rank * base_size
            end_idx = start_idx + base_size

        ad_logger.debug(
            f"Sharding BMM for rank {rank}: batch_size={bmm_batch_size}, start_idx={start_idx}, end_idx={end_idx}"
        )

        # Handle both tensors
        handle_tensor(node, lhs_tensor, 0, start_idx, end_idx)
        handle_tensor(node, rhs_tensor, 1, start_idx, end_idx)

        # Add all_gather node after BMM to collect results
        with gm.graph.inserting_after(node):
            gather_node = gm.graph.call_function(
                torch.ops.dist.all_gather,
                args=(node, 0),  # Gather along batch dimension (0)
            )
            node.replace_all_uses_with(gather_node)
            gather_node.replace_input_with(gather_node, node)

        num_bmm_shards += 1

    # Canonicalize and return
    if num_bmm_shards:
        gm = canonicalize_graph(gm)
    ad_logger.debug("After sharding BMM: " + str(gm))
    ad_logger.info(f"Found {num_bmm_shards} BMM shards")
    return gm





def distribute_tensor(gm: GraphModule,
        consumer_node: Node, 
        tensor_node: Node, 
        split_dim: int,
        arg_idx: int, 
        start_idx: int, end_idx: int
    ):
        """Unified helper function to shard either a parameter tensor or a dynamic tensor.

        Args:
            consumer_node: The node that is being processed
            tensor_node: The input tensor node to shard
            arg_idx: The argument index of the tensor in the consumer_node node
            start_idx: Start index for sharding
            end_idx: End index for sharding
        """

        # Define slice function for the sharding
        def slice_tensor(t: torch.Tensor) -> torch.Tensor:
            """
            Args:
                t: torch.Tensor: tensor to slice
                split_dim: int: dimension across which the slice is performed
                start_idx: int: start index of the slice
                end_idx: int: end index of the slice
            Returns:
                torch.Tensor: sliced tensor
                
            Example:
                t = torch.rand(8,16,32,64)
                slice_tensor(t, 1, 0, 4) # returns a tensor of shape (8,4,32,64)
                slice_tensor(t, 2, 4, 8) # returns a tensor of shape (8,16,4,64)
                slice_tensor(t, 3, 16, 32) # returns a tensor of shape (8,16,32,16)
            """
            # Create a list of slice objects for all dimensions
            slices = [slice(None)] * t.dim()
            # Set the specific dimension to slice from start_idx to end_idx
            slices[split_dim] = slice(start_idx, end_idx)
            # Apply the slicing and return the result
            return t[tuple(slices)]


        if tensor_node.op == "get_attr":
            # Handle parameter tensor
            weight_key = tensor_node.target
            modname, _, param_name = weight_key.rpartition(".")
            param = gm.get_parameter(weight_key)

            # Update the parameter with its shard
            param_new = nn.Parameter(slice_tensor(param).detach().clone(), requires_grad=True)
            gm.get_submodule(modname).register_parameter(param_name, param_new)

            # Register load state dict hook
            gm._register_load_state_dict_pre_hook(
                partial(
                    _load_hook,
                    f_split=slice_tensor,
                    param_key=weight_key,
                    param_shape=param_new.shape,
                )
            )
        else:
            # Handle dynamic tensor
            with gm.graph.inserting_before(consumer_node):
                tensor_slice = gm.graph.call_function(
                    torch.ops.aten.slice.Tensor, args=(tensor_node, split_dim, start_idx, end_idx, 1)
                )
            # Update BMM node to use the sliced tensor
            consumer_node.update_arg(arg_idx, tensor_slice)
