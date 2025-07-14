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

import math
import operator
from collections import defaultdict
from functools import partial
from typing import Callable, DefaultDict, Dict, List, Set

import torch
import torch.nn as nn
from torch.fx import GraphModule, Node

from ...utils.logger import ad_logger
from ...utils.node_utils import (
    is_attention_op,
    is_aggregation_op,
    extract_param_names_from_lin_node,
    identify_regions_between_residuals,
    is_linear_op,
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
    gm: GraphModule,
    node: Node,
    dim: int,
    rank: int,
    world_size: int,
    add_dist: bool = False,
    min_local_shape: int = 1,
):
    """Replaces the matmul node with a new matmul node that accepts sharded weights.

    The state_dict is also updated to contain the sharded weights.
    """
    assert dim in [0, 1], "Only dim 0 and 1 are supported for sharding"
    assert add_dist or dim == 0, "For dim=1 sharding, dist_op is required."

    quantization_impl = QuantizationImpl.create(node)

    def split_tensor(
        t: torch.Tensor,
        d: int = dim,
        r: int = rank,
        ws: int = world_size,
        min_d_shape: int = min_local_shape,
    ) -> torch.Tensor:
        # The local tensor shape has to be divisible by min_d_shape
        max_split_size = t.shape[d] // min_d_shape
        if ws > max_split_size:
            num_groups = math.ceil(ws / max_split_size)
            ad_logger.debug(
                f"World size {ws} is greater than the max split size {max_split_size}. "
                + f"Splitting tensor to {num_groups} chunks"
            )
            return torch.tensor_split(t, max_split_size, dim=d)[r // num_groups]
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
        0: (torch.ops.auto_deploy.torch_dist_all_gather, -1),
        1: (torch.ops.auto_deploy.torch_dist_all_reduce,),
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
    gm: GraphModule,
    rank: int,
    world_size: int,
    simple_shard_only: bool = False,
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

    min_local_shape is the minimum size of the local tensor shard, to prevent TP parallelism
    splitting, e.g., the individual heads into smaller shards.
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
        torch.ops.auto_deploy.torch_attention_sdpa,
        torch.ops.auto_deploy.torch_attention_grouped_sdpa,
        torch.ops.auto_deploy.torch_attention_bsnd_grouped_sdpa,
    }

    # This is a heuristic. Basically, we assume those are okay to shard if we also encounter an
    # attention node because we know that those ops must be compatible with the attention op. Now
    # since the attention op is shardable, we will assume those are as well if used in conjunction
    # with the attention op.
    shardable_nodes_with_attention = {
        torch.ops.aten.view,
        torch.ops.aten.reshape,
        torch.ops.auto_deploy.flashinfer_rope,
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

        # nothing to shard
        if len(nodes_linear) == 0:
            continue

        num_shards += 1

        if simple_shard_only:
            print(f"Forcing Simple Shard: Linear groups: {nodes_linear}")
            _simple_shard(gm, nodes_linear, rank, world_size)
            continue

        # simple shard when we have != 2 groups of linear nodes
        if len(nodes_linear) != 2:
            print(f"Linear groups: {nodes_linear}")
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
            print(f"Unaccounted nodes: {unaccounted_nodes}")
            _simple_shard(gm, nodes_linear, rank, world_size)
            continue

        # If we can account for all sharded nodes, we can do a two-way shard
        # --> row_split (dim 0) + col_split (dim 1) + all_reduce

        # check if we are sharding the attention block
        if attention_nodes:
            if len(attention_nodes) > 1:
                # Column-row shard boundary region detection is probably wrong - there should be
                # only one attention operation. Fall back to simple shard.
                print(f"More than one attention node: {unaccounted_nodes}")
                _simple_shard(gm, nodes_linear, rank, world_size)
                continue
            # Extract head dimension. We cannot shard below the head_dim size.
            # Assume that head_dim is the last (innermost) dimension of the tensor
            min_local_shape = attention_nodes.pop().meta["val"].shape[-1]
        else:
            min_local_shape = 1
        for i, group in enumerate(nodes_linear.values()):
            for n in group:
                print(f"column-row shard for node: {n.name}")
                _insert_sharded_matmul(
                    gm, n, i, rank, world_size, add_dist=i > 0, min_local_shape=min_local_shape
                )

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

        # NOTE: our torch.ops.auto_deploy.torch_dist_all_gather doesn't support uneven splits at the moment.
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
                torch.ops.auto_deploy.torch_dist_all_gather,
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


def get_node_dict(gm: GraphModule) -> Dict[str, Node]:
    node_dict = {}
    for n in gm.graph.nodes:
        node_dict[n.name] = n
    return node_dict


def column_row_shard_2(gm: GraphModule, rank: int, world_size: int) -> GraphModule:
    g = get_node_dict(gm)
    for n in gm.graph.nodes:
        if "zeros_" in n.name:
            print(f"\n\nn: {n.name}\nargs: {n.args}\n meta: {n.meta}")
            print(f"n.meta['val']: {n.meta['val']}")
        if "distributed" not in n.meta:
            n.meta["distributed"] = {}
        
        if 'val' in n.meta:
            # dynamic tensors are initially, by default, replicated
            input_is_column_sharded = False
        else:
            if len(n.args) == 1 and isinstance(n.args[0], Node) and 'val' in n.args[0].meta:
                # if 'val' in n.args[0].meta:
                    n.meta['val'] = n.args[0].meta['val']
                    input_is_column_sharded = False
                # else:
                #     input_is_column_sharded = True
            else:
                # these are states, parameters, 
                # constants (also tensor constants like sin/cos for rope),
                # "flashinfer" falls here
                input_is_column_sharded = None
        
        n.meta["distributed"]["is_column_sharded"] = input_is_column_sharded
        
        
        # find the input distribution
        shardable_inputs = [
            s
            for s in n.args
            if s is not None
            and isinstance(s, Node)
            and not s.op == "get_attr"
            and (
                ('val' in s.meta
                    and (isinstance(s.meta['val'], tuple)
                    or isinstance(s.meta['val'], list)
                    or (
                        hasattr(s.meta['val'], 'shape')
                        and
                        len(s.meta['val'].shape) >= 2)
                    )
                )
                or s.meta['distributed']['is_column_sharded']
            )
        ]
        
        all_inputs_are_column_sharded = set([
            s.meta["distributed"]["is_column_sharded"]
            for s in shardable_inputs
        ])        

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
            
            
            if is_attention_op(n):
                # First three arguments are Q, K, V, and there is an optional fourth argument, the attention mask.
                # Check if Q, K, V were sharded
                if all([s.meta["distributed"]["is_column_sharded"] for s in shardable_inputs[:3]]):
                    # check if this is the mask that was not sharded.
                    #  It's static and may never had a chance to pass through the sharding
                    # linear layers' logic.
                    assert len(shardable_inputs) == 4, "Expecting Q, K, V, and mask inputs for attention"                    
                    assert not shardable_inputs[3].meta["distributed"]["is_column_sharded"], "Expecting mask to be replicated"
                    shardable_inputs[3].meta["distributed"]["is_column_sharded"] = True
                    all_inputs_are_column_sharded = set([True])
                
                else:
                    # Now we are in the situation, where at least one of Q, K, V is not sharded.
                    # Check whether none of Q, K, V are sharded and only mask was sharded.
                    if all([not s.meta["distributed"]["is_column_sharded"] for s in shardable_inputs[:3]]):
                        assert len(shardable_inputs) == 4, "Expecting Q, K, V, and mask inputs for attention"                    
                        assert shardable_inputs[3].meta["distributed"]["is_column_sharded"], "Expecting mask to be sharded"
                        # then, we don't shard it at all.
                        all_inputs_are_column_sharded = set([False])
                    else:
                        # Some of Q, K, V are sharded, and some are not.
                        q_node, k_node, v_node = shardable_inputs[:3]
                        # Probably, Q and K are NOT sharded because of the normalization,
                        # while V is sharded.
                        assert q_node.meta["distributed"]["is_column_sharded"] == False, "Expecting Q to be replicated"
                        assert k_node.meta["distributed"]["is_column_sharded"] == False, "Expecting K to be replicated"
                        assert v_node.meta["distributed"]["is_column_sharded"] == True, "Expecting V to be sharded"
                        
                        distribute_attention_node(n, gm, rank, world_size)                                    
                        all_inputs_are_column_sharded = set([True])
            else:               
                # Check whether this is a RoPe-type situation, that is:
                # 1. we have some QK-related inputs that are already reshaped into heads and 
                #    they are column-sharded 
                # 2. we have cos/sin inputs that are "static" (don't have any aggregation op predecessors)
                #    and they are not sharded.
                # In that case, don't do anything and mark it as sharded.
                # This means that dynamic inputs Q/K are sharded, static inputs are replicated,
                # and the rope cos/sin transformations are applied to sharded Q/K states.
                sharded_shardable = [s for s in shardable_inputs if s.meta["distributed"]["is_column_sharded"]]
                static_not_sharded = [s for s in shardable_inputs if not s.meta["distributed"]["is_column_sharded"]]
                
                # check whether all tensors in sharded_shardable are 4-dimensional
                # and all tensors in static_not_sharded are 3-dimensional
                # if so, we can proceed
                if not (all([len(s.meta["val"].shape) == 4 for s in sharded_shardable]) and \
                        all([len(s.meta["val"].shape) == 3 for s in static_not_sharded])):
                    sharded_shape = sharded_shardable[0].meta["val"].shape if sharded_shardable else "unknown"
                    static_shape = static_not_sharded[0].meta["val"].shape if static_not_sharded else "unknown"
                    print(f"\nnode {n.name}, args: {n.args}, shardable_inputs: {[(s, s.args, s.meta["distributed"]["is_column_sharded"], s.meta['val']) for s in shardable_inputs]}")
                    all_inputs_are_column_sharded = set([False])
                    # raise ValueError(f"Sharded and static inputs have different shapes: {sharded_shape} and {static_shape}")

                # check whether the shapes of all static_not_sharded are the same
                if not all([s.meta["val"].shape == static_not_sharded[0].meta["val"].shape for s in static_not_sharded]):
                    sharded_shape = sharded_shardable[0].meta["val"].shape if sharded_shardable else "unknown"
                    static_shape = static_not_sharded[0].meta["val"].shape if static_not_sharded else "unknown"
                    raise ValueError(f"Sharded and static inputs have different shapes: {sharded_shape} and {static_shape}")
                
                # sharded_shardable shapes may differ in the 3rd dimension (number of heads), since the number of Q heads
                # may be different than KV heads. 
                
                # take the shape of the static_not_sharded, add a dummy dimension to the 3rd position
                static_not_sharded_dummy_shape = static_not_sharded[0].meta["val"].shape[:2] + (1,) + static_not_sharded[0].meta["val"].shape[2:]
                
                # now check whether dimensions 0, 1, and 3 (batch, sequence, head_dim) are the same for all 
                # sharded_shardable and static_not_sharded_dummy_shape
                if not all([s.meta["val"].shape[i] == static_not_sharded_dummy_shape[i] for s in sharded_shardable for i in [0, 1, 3]]):
                    sharded_shape = sharded_shardable[0].meta["val"].shape if sharded_shardable else "unknown"
                    static_shape = static_not_sharded[0].meta["val"].shape if static_not_sharded else "unknown"
                    print(f"\nnode {n.name}, args: {n.args}, shardable_inputs: {[(s, s.args, s.meta["distributed"]["is_column_sharded"], s.meta['val']) for s in shardable_inputs]}")
                    all_inputs_are_column_sharded = set([False])
                    # raise ValueError(f"Sharded and static inputs have different shapes: {sharded_shape} and {static_shape}")
                              
                sharded_shape = sharded_shardable[0].meta["val"].shape
                static_shape = static_not_sharded[0].meta["val"].shape
                  
                head_dim_nos = [dim for dim, shape in enumerate(sharded_shape) if shape not in static_shape]
                if len(head_dim_nos) != 1:
                    raise ValueError(f"Sharded and static inputs have different shapes: {sharded_shape} and {static_shape}")
                
                # good to go. Keep not_sharded not sharded (replicated), and mark this node as sharded
                all_inputs_are_column_sharded = set([True])
                

            
            
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
            can_output_be_column_sharded = False
            if sinks:
                # check if all sinks are aggregation operations
                all_sink_aggregation_dims = set([is_aggregation_op(s)[1] for s in sinks])
                # dims are [batch, sequence, embedding]
                if 2 not in all_sink_aggregation_dims:
                    # dim == 2 means that the sink aggregation operation
                    # performs aggregation across the embedding dimension,
                    # therefore, X cannot be shared (otherwise, that would 
                    # imply distributed aggegation)
                    can_output_be_column_sharded = True
            
            output_is_column_sharded = can_output_be_column_sharded and not input_is_column_sharded
            _insert_sharded_matmul(gm,
                                   n,
                                   dim=1 if input_is_column_sharded else 0,
                                   rank=rank,
                                   world_size=world_size,
                                   add_dist=not output_is_column_sharded)
            if "distributed" not in n.meta:
                n.meta["distributed"] = {}
            n.meta["distributed"]["is_column_sharded"] = output_is_column_sharded
            stat = (n, weigh.name, input_is_column_sharded, can_output_be_column_sharded, output_is_column_sharded)
            # print(f"stat: {stat}")
            if not can_output_be_column_sharded and not input_is_column_sharded:
                pass
                print(f"SIMPLE shard detected! {stat}")
            else:
                print(f"COL-ROW shard detected! {stat}")
            
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
                if is_op(n, torch.ops.auto_deploy.torch_attention_bsnd_grouped_sdpa):
                    head_dim_no = 2
                else:
                    head_dim_no = 3
                num_heads = q_node.meta["val"].shape[head_dim_no]
                num_kv_heads = set([s.meta["val"].shape[head_dim_no] for s in [k_node, v_node]])
                assert len(num_kv_heads) == 1, "K and V inputs to attention node should have the same number of heads"
                num_kv_heads = num_kv_heads.pop()
                assert num_heads >= num_kv_heads, "Number of heads must be greater than or equal to number of KV heads"
                
                assert (num_heads > world_size) and num_heads % world_size == 0, "Number of heads must be divisible by world size"
                heads_per_rank = num_heads // world_size
                
                print(f"Distributing ATTENTION right before execution: {n.name}")
                distribute_tensor(gm,
                                      consumer_node=n,
                                      tensor_node=q_node,
                                      split_dim=head_dim_no,
                                      arg_idx=0,
                                      start_idx=heads_per_rank * rank,
                                      end_idx=heads_per_rank * (rank + 1))

                
                # NOTE: this is a floating point division!
                # For models with small num_kv_heads, like Qwen, where num_kv_heads = 8,
                # TP may be larger. This means that, e.g., if TP = 16, every two ranks will
                # share the same single KV head.
                kv_heads_per_rank = num_kv_heads / world_size
                for i, src in enumerate([k_node, v_node]):
                    # column-split the input
                    distribute_tensor(gm,
                                      consumer_node=n,
                                      tensor_node=src,
                                      split_dim=head_dim_no,
                                      arg_idx=i+1,
                                      # NOTE: rounding down to int is intentional!
                                      start_idx=int(kv_heads_per_rank * rank),
                                      end_idx=int(kv_heads_per_rank * (rank + 1)))
                n.meta["distributed"]["is_column_sharded"] = True
                
        if is_attention_op(n) and not n.meta["distributed"]["is_column_sharded"]:
            raise ValueError(f"Attention node {n} is not sharded")
        
    # exit(0)
    return gm



def find_all_boundary_nodes(
    node: Node,
    target: Callable,
    attr_next: str = "users",
) -> List[Node]:
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


def distribute_attention_node(n: Node, gm: GraphModule, rank: int, world_size: int):
    print(f"Distributing PARTIALLY distributed ATTENTION node: {n.name}")
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
        if is_op(n, torch.ops.auto_deploy.torch_attention_bsnd_grouped_sdpa):
            head_dim_no = 2
        else:
            head_dim_no = 3
        num_heads = q_node.meta["val"].shape[head_dim_no]
        num_kv_heads = set([s.meta["val"].shape[head_dim_no] for s in [k_node, v_node]])
        assert len(num_kv_heads) == 1, "K and V inputs to attention node should have the same number of heads"
        num_kv_heads = num_kv_heads.pop()
        assert num_heads >= num_kv_heads, "Number of heads must be greater than or equal to number of KV heads"
        
        assert (num_heads > world_size) and num_heads % world_size == 0, "Number of heads must be divisible by world size"
        heads_per_rank = num_heads // world_size
        if not q_node.meta["distributed"]["is_column_sharded"]:
            distribute_tensor(gm,
                                    consumer_node=n,
                                    tensor_node=q_node,
                                    split_dim=head_dim_no,
                                    arg_idx=0,
                                    start_idx=heads_per_rank * rank,
                                    end_idx=heads_per_rank * (rank + 1))
        else:
            print(f"Warning: {q_node} is already sharded")
        
        # NOTE: this is a floating point division!
        # For models with small num_kv_heads, like Qwen, where num_kv_heads = 8,
        # TP may be larger. This means that, e.g., if TP = 16, every two ranks will
        # share the same single KV head.
        kv_heads_per_rank = num_kv_heads / world_size
        for i, src in enumerate([k_node, v_node]):
            if not src.meta["distributed"]["is_column_sharded"]:
                # column-split the input
                distribute_tensor(gm,
                                    consumer_node=n,
                                    tensor_node=src,
                                    split_dim=head_dim_no,
                                    arg_idx=i+1,
                                    # NOTE: rounding down to int is intentional!
                                    start_idx=int(kv_heads_per_rank * rank),
                                    end_idx=int(kv_heads_per_rank * (rank + 1)))
            else:
                print(f"Warning: {src} is already sharded")
        n.meta["distributed"]["is_column_sharded"] = True
        
