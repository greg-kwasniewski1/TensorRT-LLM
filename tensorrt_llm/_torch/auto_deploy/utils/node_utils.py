"""Common utils for torch fx graph transformation."""

import operator
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Tuple, Union

import torch
from torch._ops import OpOverload, OpOverloadPacket
from torch.fx import Graph, GraphModule, Node

from ..custom_ops.quant import QUANT_OPS
from .logger import ad_logger

try:
    # import modelopt to get quantize_op
    from modelopt.torch.quantization import tensor_quant  # noqa: F401

    if hasattr(torch.ops, "tensorrt"):
        modelopt_quantize_op = torch.ops.tensorrt.quantize_op
        modelopt_dynamic_block_quantize_op = torch.ops.tensorrt.dynamic_block_quantize_op
    else:
        modelopt_quantize_op = None
        modelopt_dynamic_block_quantize_op = None
except ImportError:
    modelopt_quantize_op = None
    modelopt_dynamic_block_quantize_op = None

OperatorLike = Union[OpOverloadPacket, OpOverload, Callable]


@dataclass
class modelopt_quant_params:
    input_node: torch.fx.node.Node = None
    amax: torch.fx.node.Node = None
    num_bits: int = 0
    exp_bits: int = 0
    is_unsigned: bool = False
    narrow_range: bool = False
    is_dynamic_block_quant: bool = False
    block_size: int = 0
    scale_num_bits: int = 0
    scale_exponent_bits: int = 0

    def is_fp8_e4m3(self):
        return self.num_bits == 8 and self.exp_bits == 4

    def is_fp4_e2m1(self):
        return self.num_bits == 4 and self.exp_bits == 2

    def get_quant_format_str(self):
        return (
            None
            if self.num_bits == 0 and self.exp_bits == 0
            else f"num_bits: {self.num_bits}, exp_bits: {self.exp_bits}"
        )

    @staticmethod
    def get_quant_params_from_quantize_node(input_node):
        params = None
        if is_op(input_node, modelopt_quantize_op):
            params = modelopt_quant_params(
                *input_node.args,
                is_dynamic_block_quant=False,
                block_size=0,
                scale_num_bits=0,
                scale_exponent_bits=0,
            )
        elif is_op(input_node, modelopt_dynamic_block_quantize_op):
            params = modelopt_quant_params(
                input_node=input_node.args[0],
                block_size=input_node.args[1],
                amax=input_node.args[2],
                num_bits=input_node.args[3],
                exp_bits=input_node.args[4],
                is_dynamic_block_quant=True,
                scale_num_bits=input_node.args[5],
                scale_exponent_bits=input_node.args[6],
            )

        def get_amax_from_detach(detach_node):
            if (
                not is_op(detach_node, torch.ops.aten.detach)
                or len(detach_node.all_input_nodes) != 1
            ):
                return detach_node
            return detach_node.all_input_nodes[0]

        if params:
            params.amax = get_amax_from_detach(params.amax)
        return params


def get_quantization_params_from_linear_node(linear_op: torch.fx.node.Node):
    """Return quantization parameters of the linear node."""
    input_params, weight_params, output_params = None, None, None

    if modelopt_quantize_op is not None and is_linear_op(linear_op):
        input_node, weight_node = linear_op.all_input_nodes[:2]
        # check if activation, weight, and output are quantized
        input_params = modelopt_quant_params.get_quant_params_from_quantize_node(input_node)
        weight_params = modelopt_quant_params.get_quant_params_from_quantize_node(weight_node)
        output_params = modelopt_quant_params.get_quant_params_from_quantize_node(
            list(linear_op.users.keys())[0]
        )

    return input_params, weight_params, output_params


def is_match(node: Node, names_to_skip: List[str]):
    if names_to_skip is None:
        return False
    for n in names_to_skip:
        module_stack = node.meta.get("nn_module_stack", None)
        if module_stack is None:
            return False
        module_stack = list(module_stack.keys())
        if n in module_stack[-1]:
            return True
    return False


def extract_weight_node(mm_node: Node) -> int:
    """Extracts the weight node from the given matmul node."""

    def find_get_attr_node(node: Node) -> Node:
        """Recursively traverse inputs of allowed nodes to find a node with 'get_attr' op."""
        # If node is a get_attr node return node
        # List of nodes allowed in between a get_attr node and the matmul node
        allowed_ops = {torch.ops.aten.to.dtype}

        if node.op == "get_attr":
            return node

        # If node is not in the list of allowable ops then return None
        if node.target not in allowed_ops:
            return None

        for input_node in node.all_input_nodes:
            result = find_get_attr_node(input_node)
            if result:
                return result
        return None

    weight_node = mm_node.args[1]
    # for modelopt quantized graph, there will be a quantize_op
    _, weight_params, _ = get_quantization_params_from_linear_node(mm_node)
    weight_node = weight_params.input_node if weight_params else weight_node

    return find_get_attr_node(weight_node)


def num_users_of_weight_node(mm_node: Node) -> int:
    """Returns the number of users of the weight node of the given matmul node."""
    weight_node = extract_weight_node(mm_node)
    return len(weight_node.users) if weight_node is not None else 0


def extract_param_names_from_lin_node(mm_node: Node) -> Tuple[str, Optional[str]]:
    """Extracts the name of the parameter associated with the given matmul node.

    Args:
        mm_node: Matmul node in the graph.
    """
    assert is_linear_op(mm_node, include_quantization=True), (
        f"Expecting linear node, Found: {mm_node}"
    )
    weight_node = extract_weight_node(mm_node)

    assert weight_node, "Cannot identify weight parameter of linear node."

    # Map arg to named parameter
    weight_name = weight_node.target

    # check for bias
    bias_node = mm_node.args[2] if len(mm_node.args) > 2 else None
    assert bias_node is None or bias_node.op == "get_attr"
    bias_name = bias_node.target if bias_node is not None else None

    return weight_name, bias_name


def get_op_overload_packet(node: Union[OpOverloadPacket, OpOverload]) -> OpOverloadPacket:
    """Get the overload packet from the op overload."""
    if isinstance(node, OpOverloadPacket):
        return node
    elif isinstance(node, OpOverload):
        return node.overloadpacket
    else:
        raise ValueError(f"Expected OpOverloadPacket or OpOverload, got {type(node)}")


def is_op(node: Node, ops: Union[OperatorLike, Iterable[OperatorLike]]) -> bool:
    """Check if the node is a call to one of the ops."""
    if not isinstance(node, Node):
        return False

    if node.op != "call_function":
        return False

    # check if it's a single op that's provided by checking if it's iterable
    if isinstance(ops, OpOverloadPacket) or not isinstance(ops, Iterable):
        ops = [ops]

    # now iterate through the operator list and see if there is a match
    is_match = True
    for op in ops:
        if node.target == op:
            break
        if isinstance(op, OpOverloadPacket):
            if any(node.target == getattr(op, overload) for overload in op):
                break
    else:
        is_match = False

    return is_match


def is_linear_op(node: Node, include_quantization: bool = False) -> bool:
    """Check if the node is a linear op.

    Using this function is preferred over `is_op` for linear ops to ensure all variants are covered.
    """
    lin_ops = {
        torch.ops.aten.linear,
        torch.ops.linear.simple,
    }

    if include_quantization:
        lin_ops.update(QUANT_OPS)
    return is_op(node, lin_ops)


def is_nonlinear_reduction_op(node: Node, include_quantization: bool = False) -> bool:
    """Check if the node is a nonlinear reduction op.

    Using this function is preferred over `is_op` for linear ops to ensure all variants are covered.
    """
    lin_ops = {
        torch.ops.aten.mean,
        torch.ops.aten.sum,
        torch.ops.aten.max,
        torch.ops.aten.min,
        torch.ops.aten.amax,
        torch.ops.aten.amin,
        torch.ops.aten.norm,
        torch.ops.aten.std,
        torch.ops.attention.scaled_dot_product_attention,
        torch.ops.attention.grouped_sdpa,
        torch.ops.attention.bsnd_grouped_sdpa,
    }

    if include_quantization:
        lin_ops.update(QUANT_OPS)
    return is_op(node, lin_ops)


def is_dist_op(node: Node) -> bool:
    """Check if the node is a distributed op."""
    dist_ops = {torch.ops.dist.all_gather, torch.ops.dist.all_reduce, torch.distributed.P2POp}
    return is_op(node, dist_ops)


def get_all_input_output_nodes(graph: Graph) -> Tuple[List[Node], List[Node]]:
    input_nodes: List[Node] = graph.find_nodes(op="placeholder")
    output_nodes: List[Node] = graph.find_nodes(op="output")
    return (input_nodes, output_nodes)


def get_user_if_pattern_match(node, ops, numusers, user_idx: int = 0):
    """Get a user from a node if the node matches a given op set and num of users."""
    if node is None:
        return None
    assert len(node.users) > user_idx
    return (
        list(node.users.keys())[user_idx]
        if node and len(list(node.users.keys())) == numusers and is_op(node, ops)
        else None
    )


def identify_regions_between_residuals(gm: GraphModule) -> List[Node]:
    """Identify regions of the graph that we can investigate further for patterning matching.

    Right now, we split the regions according to the following structure:
        1. Input node
        2. Embedding node
        3. Residual nodes from the embedding node onwards (no other nodes in-between0)
        4. Output node

    The list will contain the boundary nodes between the regions.
    """
    assert gm.graph.nodes, "Graph is empty"

    # get first input node and last output node
    input_id_node = None
    output_node = None
    for node in gm.graph.nodes:
        if input_id_node is None and node.op == "placeholder":
            input_id_node = node
        output_node = node
    assert input_id_node, "Could not find input node"
    assert output_node.op == "output", "Could not find output node"

    # start list of boundary nodes
    boundary_nodes = [input_id_node]

    # find embedding node which we assume to be the first node in a sequence of residual nodes
    for n_user in input_id_node.users:
        if is_op(n_user, torch.ops.aten.embedding):
            break
    else:
        # we could not identify any boundary regions via embedding nodes
        boundary_nodes.append(output_node)
        return boundary_nodes

    # add embedding node to boundary nodes
    boundary_nodes.append(n_user)

    # find residual nodes from here on
    # NOTE: for now, we assume that the residual nodes do not go through point-wise operations like
    # activations. We are just looking for a "straight" path to the output.
    for node in gm.graph.nodes:
        if is_op(node, torch.ops.aten.add) and any(n == node for n in boundary_nodes[-1].users):
            boundary_nodes.append(node)

    # sanity check: we expect at most two users for any residual node
    res_nodes_more_users = [n for n in boundary_nodes[2:] if len(n.users) > 2]
    if res_nodes_more_users:
        ad_logger.debug(f"Unexpected # of users for residuals: {res_nodes_more_users}")

    # add output node to boundary nodes
    boundary_nodes.append(output_node)

    return boundary_nodes


def bfs(
    node: Node,
    target: Callable,
    attr_next: str = "users",
    boundary: Optional[Node] = None,
    skip_root=False,
    allow_empty=False,
) -> Node:
    queue = [node]
    visited = set()
    if skip_root:
        visited.add(node)
        queue = list(n for n in getattr(node, attr_next) if n is not None)
    while queue:
        cur_node = queue.pop(0)
        if boundary is not None and cur_node == boundary:
            continue  # Skip the boundary node.
        if target(cur_node):
            return cur_node
        for next_node in getattr(cur_node, attr_next):
            if next_node is None or not isinstance(next_node, Node):
                continue
            if boundary is not None and next_node == boundary:
                continue  # Do not expand past the boundary.
            if next_node not in visited:
                visited.add(next_node)
                queue.append(next_node)
    if allow_empty:
        return None
    raise RuntimeError(f"Could not find node with target condition {target}.")


def bfs_(
    node: Node, target: Callable, attr_next: str = "users", boundary: Optional[Node] = None
) -> Node:
    queue = [node]
    visited = set()
    while queue:
        cur_node = queue.pop(0)
        if boundary is not None and cur_node == boundary:
            continue  # Skip the boundary node.
        if target(cur_node):
            return cur_node
        for next_node in getattr(cur_node, attr_next):
            if boundary is not None and next_node == boundary:
                continue  # Do not expand past the boundary.
            if next_node not in visited:
                visited.add(next_node)
                queue.append(next_node)
    raise RuntimeError(f"Could not find node with target condition {target}.")


def extract_output_tuple(node: Node, count: int = 2):
    """
    Extract up to `count` outputs from a tuple-producing node.
    Returns a list of length `count`, with None if an output isn't found.
    """
    results = []
    for idx in range(count):
        user_node = next(
            (
                u
                for u in node.users
                if u.op == "call_function" and u.target == operator.getitem and u.args[1] == idx
            ),
            None,
        )
        results.append(user_node)
    return results


def extract_op_args(node: Node, *arg_names):
    """
    Given a call_function node for torch custom op,
    returns a tuple of values for each name in arg_names, trying in order:
    1. node.kwargs[name]
    2. node.args[position_in_schema]
    3. the schema default
    """
    if node.op != "call_function":
        raise ValueError(f"extract_op_args only supports call_function nodes, got {node.op}")

    op = node.target
    if hasattr(op, "_schemas"):
        schema = next(iter(op._schemas.values()))
    elif hasattr(op, "_schema"):
        schema = op._schema
    else:
        raise RuntimeError(f"No schema found on op {op}")
    args_meta = schema.arguments

    # name→index in signature, and name→default_value
    pos = {a.name: i for i, a in enumerate(args_meta)}
    defs = {a.name: a.default_value for a in args_meta if a.has_default_value}

    args = list(node.args)
    kwargs = node.kwargs or {}

    def _get(name):
        if name in kwargs:
            return kwargs[name]
        i = pos.get(name)
        if i is not None and i < len(args):
            return args[i]
        if name in defs:
            return defs[name]
        raise RuntimeError(f"Could not find a value for '{name}' on op {op}")

    return [_get(n) for n in arg_names]


def create_symint_mapping(gm: GraphModule):
    """Create a mapping from SymInt expressions to their corresponding nodes."""
    symint_to_node = {}

    for node in gm.graph.nodes:
        if node.op == "placeholder":
            # Check if the node itself is a SymInt
            if isinstance(node.meta.get("val"), torch.SymInt):
                symint_to_node[str(node.meta["val"])] = node

            # Check tensor shapes for SymInt dimensions
            elif hasattr(node.meta.get("val"), "shape"):
                fake_tensor = node.meta["val"]
                for dim_idx, dim_size in enumerate(fake_tensor.shape):
                    if isinstance(dim_size, torch.SymInt):
                        symint_to_node[str(dim_size)] = dim_size

    return symint_to_node


def classify_operation_type(node: Node) -> Tuple[str, Optional[int]]:
    """
    Classify a PyTorch FX node into operation categories for sharding purposes.

    Returns:
        Tuple[str, Optional[int]]: (operation_type, aggregation_dimension)
        - operation_type: 'pointwise', 'linear', or 'nonlinear_reduction'
        - aggregation_dimension: For nonlinear operations, the dimension that's being aggregated over
    """

    if not isinstance(node, Node) or node.op != "call_function":
        return ("unknown", None)

    # Category A: Pointwise Operations
    pointwise_ops = {
        # Element-wise arithmetic
        torch.ops.aten.add,
        torch.ops.aten.add_,
        torch.ops.aten.sub,
        torch.ops.aten.sub_,
        torch.ops.aten.mul,
        torch.ops.aten.mul_,
        torch.ops.aten.div,
        torch.ops.aten.div_,
        torch.ops.aten.pow,
        torch.ops.aten.pow_,
        # Element-wise functions
        torch.ops.aten.relu,
        torch.ops.aten.relu_,
        torch.ops.aten.gelu,
        torch.ops.aten.gelu_approximate,
        torch.ops.aten.silu,
        torch.ops.aten.silu_,
        torch.ops.aten.tanh,
        torch.ops.aten.tanh_,
        torch.ops.aten.sigmoid,
        torch.ops.aten.sigmoid_,
        torch.ops.aten.exp,
        torch.ops.aten.exp_,
        torch.ops.aten.sin,
        torch.ops.aten.cos,
        torch.ops.aten.abs,
        torch.ops.aten.sqrt,
        torch.ops.aten.log,
        torch.ops.aten.log_,
        torch.ops.aten.rsqrt,
        torch.ops.aten.neg,
        # Element-wise comparisons
        torch.ops.aten.eq,
        torch.ops.aten.ne,
        torch.ops.aten.lt,
        torch.ops.aten.le,
        torch.ops.aten.gt,
        torch.ops.aten.ge,
        # Element-wise logical
        torch.ops.aten.logical_and,
        torch.ops.aten.logical_or,
        torch.ops.aten.logical_not,
        torch.ops.aten.logical_xor,
        # Shape manipulations (no computation)
        torch.ops.aten.view,
        torch.ops.aten.reshape,
        torch.ops.aten.transpose,
        torch.ops.aten.permute,
        torch.ops.aten.squeeze,
        torch.ops.aten.unsqueeze,
        torch.ops.aten.contiguous,
        torch.ops.aten.flatten,
        # Element-wise conversions
        torch.ops.aten.to,
        torch.ops.aten.type_as,
        torch.ops.aten.clone,
        torch.ops.aten.detach,
        # Activation functions
        torch.ops.aten.leaky_relu,
        torch.ops.aten.elu,
        torch.ops.aten.hardtanh,
        torch.ops.aten.hardswish,
    }

    # Category B: Linear Operations (Matrix/Tensor Contractions)
    linear_ops = {
        torch.ops.aten.linear,
        torch.ops.linear.simple,
        torch.ops.aten.matmul,
        torch.ops.aten.mm,
        torch.ops.aten.bmm,
        torch.ops.aten.addmm,
        torch.ops.aten.baddbmm,
        torch.ops.aten.addmv,
        torch.ops.aten.mv,
        torch.ops.aten.dot,
        torch.ops.aten.conv1d,
        torch.ops.aten.conv2d,
        torch.ops.aten.conv3d,
        torch.ops.aten.conv_transpose1d,
        torch.ops.aten.conv_transpose2d,
        torch.ops.aten.embedding,
        torch.ops.aten.embedding_bag,
    }

    # Category C: Nonlinear Reduction Operations
    # We need to analyze both the operation and its dimension parameter
    nonlinear_reduction_ops = {
        # Reduction operations - need to check 'dim' parameter
        torch.ops.aten.mean,
        torch.ops.aten.sum,
        torch.ops.aten.max,
        torch.ops.aten.min,
        torch.ops.aten.amax,
        torch.ops.aten.amin,
        torch.ops.aten.std,
        torch.ops.aten.var,
        torch.ops.aten.norm,
        torch.ops.aten.linalg_norm,
        torch.ops.aten.prod,
        torch.ops.aten.any,
        torch.ops.aten.all,
        # Normalization operations - aggregate over specific dimensions
        torch.ops.aten.layer_norm,
        torch.ops.aten.group_norm,
        torch.ops.aten.batch_norm,
        torch.ops.aten.instance_norm,
        torch.ops.aten.rms_norm,  # if available
        # Softmax and related - aggregate over specific dimensions
        torch.ops.aten.softmax,
        torch.ops.aten.log_softmax,
        torch.ops.aten.gumbel_softmax,
    }

    # Attention operations - special case (aggregate over sequence dimension)
    attention_ops = {
        torch.ops.attention.scaled_dot_product_attention,
        torch.ops.attention.grouped_sdpa,
        torch.ops.attention.bsnd_grouped_sdpa,
    }

    # Check operation type
    if node.target in pointwise_ops:
        return ("pointwise", None)

    elif node.target in linear_ops:
        return ("linear", None)

    elif node.target in nonlinear_reduction_ops:
        # Extract the aggregation dimension
        agg_dim = _extract_aggregation_dimension(node)
        return ("nonlinear_reduction", agg_dim)

    elif node.target in attention_ops:
        # Attention operations aggregate over sequence dimension
        # For standard attention layouts: [batch, num_heads, seq_len, head_dim]
        # The aggregation happens over seq_len (dimension -2 or 2)
        return ("nonlinear_reduction", -2)  # sequence dimension

    else:
        return ("unknown", None)


def is_aggregation_op(node: Node) -> bool:
    """
    Classify a PyTorch FX node into operation categories for sharding purposes.

    Mode dimension convention:
    We always assume that tensor X is of shape [batch, sequence, embedding].
    Therfore, 0 = batch, 1 = sequence, 2 = embedding.

    Returns:
        Tuple[str, Optional[int]]: (operation_type, aggregation_dimension)
        - operation_type: 'pointwise', 'linear', or 'nonlinear_reduction'
        - aggregation_dimension: For nonlinear operations, the dimension that's being aggregated over
    """

    if not isinstance(node, Node) or node.op != "call_function":
        return False

    # Category B: Linear Operations (Matrix/Tensor Contractions)
    linear_ops = {
        torch.ops.aten.linear,
        torch.ops.linear.simple,
        torch.ops.aten.matmul,
        torch.ops.aten.mm,
        torch.ops.aten.bmm,
        torch.ops.aten.addmm,
        torch.ops.aten.baddbmm,
        torch.ops.aten.addmv,
        torch.ops.aten.mv,
        torch.ops.aten.dot,
        torch.ops.aten.conv1d,
        torch.ops.aten.conv2d,
        torch.ops.aten.conv3d,
        torch.ops.aten.conv_transpose1d,
        torch.ops.aten.conv_transpose2d,
        torch.ops.aten.embedding,
        torch.ops.aten.embedding_bag,
    }

    # Category C: Nonlinear Reduction Operations
    # We need to analyze both the operation and its dimension parameter
    nonlinear_reduction_ops = {
        # Reduction operations - need to check 'dim' parameter
        torch.ops.aten.mean,
        torch.ops.aten.sum,
        torch.ops.aten.max,
        torch.ops.aten.min,
        torch.ops.aten.amax,
        torch.ops.aten.amin,
        torch.ops.aten.std,
        torch.ops.aten.var,
        torch.ops.aten.norm,
        torch.ops.aten.linalg_norm,
        torch.ops.aten.prod,
        torch.ops.aten.any,
        torch.ops.aten.all,
        # Normalization operations - aggregate over specific dimensions
        torch.ops.aten.layer_norm,
        torch.ops.aten.group_norm,
        torch.ops.aten.batch_norm,
        torch.ops.aten.instance_norm,
        torch.ops.aten.rms_norm,  # if available
        # Softmax and related - aggregate over specific dimensions
        torch.ops.aten.softmax,
        torch.ops.aten.log_softmax,
        torch.nn.functional.gumbel_softmax,
    }

    # Attention operations - special case (aggregate over sequence dimension)
    attention_ops = {
        torch.ops.attention.scaled_dot_product_attention,
        torch.ops.attention.grouped_sdpa,
        torch.ops.attention.bsnd_grouped_sdpa,
    }

    if is_op(node, linear_ops):
        return ("linear", None)

    elif is_op(node, nonlinear_reduction_ops):
        # Extract the aggregation dimension
        agg_dim = _extract_aggregation_dimension(node)
        if isinstance(agg_dim, Iterable):
            agg_dim = agg_dim[0]
        return ("nonlinear_reduction", agg_dim)

    elif is_op(node, attention_ops):
        # Attention operations aggregate over sequence dimension
        # For standard attention layouts: [batch, num_heads, seq_len]
        # The aggregation happens over seq_len (dimension -1)
        return ("nonlinear_reduction", -1)  # sequence dimension

    else:
        return False


def _extract_aggregation_dimension(node: Node) -> Optional[int]:
    """Extract the dimension being aggregated over for reduction operations."""

    # Common parameter names for reduction dimension
    dim_param_names = ["dim", "axis", "dims", "axes"]

    # Check args first (positional parameters)
    if is_op(node, {torch.ops.aten.softmax, torch.ops.aten.log_softmax}):
        # For softmax: softmax(input, dim, dtype=None)
        if len(node.args) >= 2:
            return node.args[1]

    elif is_op(
        node,
        {
            torch.ops.aten.mean,
            torch.ops.aten.sum,
            torch.ops.aten.max,
            torch.ops.aten.min,
            torch.ops.aten.std,
            torch.ops.aten.var,
        },
    ):
        # For reductions: mean(input, dim=None, keepdim=False, *, dtype=None)
        if len(node.args) >= 2:
            return node.args[1]

    elif is_op(node, torch.ops.aten.layer_norm):
        # LayerNorm normalizes over the last N dimensions
        # layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-05)
        if len(node.args) >= 2:
            normalized_shape = node.args[1]
            if isinstance(normalized_shape, (list, tuple)):
                # Aggregates over the last len(normalized_shape) dimensions
                return list(range(-len(normalized_shape), 0))
            elif isinstance(normalized_shape, int):
                # Aggregates over the last dimension
                return -1

    # Check kwargs
    for param_name in dim_param_names:
        if param_name in node.kwargs:
            return node.kwargs[param_name]

    # Default heuristics based on operation type
    if node.target in {torch.ops.aten.softmax, torch.ops.aten.log_softmax}:
        return -1  # Usually applied to last dimension

    elif node.target == torch.ops.aten.layer_norm:
        return -1  # Usually normalizes embedding dimension

    elif node.target == torch.ops.aten.batch_norm:
        return 1  # Usually normalizes channel dimension

    return None


def analyze_sharding_constraints(node: Node) -> dict[str, any]:
    """
    Analyze sharding constraints for a given operation.

    Returns a dictionary with sharding recommendations.
    """
    op_type, agg_dim = classify_operation_type(node)

    constraints = {
        "operation_type": op_type,
        "aggregation_dimension": agg_dim,
        "sharding_recommendation": None,
        "communication_pattern": None,
        "parallelizable_dimensions": None,
    }

    if op_type == "pointwise":
        constraints.update(
            {
                "sharding_recommendation": "Can be sharded along any dimension",
                "communication_pattern": "No communication required",
                "parallelizable_dimensions": "all",
            }
        )

    elif op_type == "linear":
        constraints.update(
            {
                "sharding_recommendation": "Shard to minimize communication volume",
                "communication_pattern": "All-reduce on output or weight gathering",
                "parallelizable_dimensions": _analyze_linear_parallelization(node),
            }
        )

    elif op_type == "nonlinear_reduction":
        forbidden_dims = (
            agg_dim if isinstance(agg_dim, list) else [agg_dim] if agg_dim is not None else []
        )
        constraints.update(
            {
                "sharding_recommendation": f"Do NOT shard along dimensions {forbidden_dims}",
                "communication_pattern": "All-reduce before operation",
                "parallelizable_dimensions": f"all except {forbidden_dims}",
                "forbidden_sharding_dimensions": forbidden_dims,
            }
        )

    return constraints


def _analyze_linear_parallelization(node: Node) -> dict[str, str]:
    """Analyze how linear operations can be parallelized."""
    if node.target in {torch.ops.aten.linear, torch.ops.linear.simple}:
        return {
            "input_batch_dim": "parallelizable",
            "input_feature_dim": "requires all-gather or weight replication",
            "weight_input_dim": "requires all-gather or input replication",
            "weight_output_dim": "parallelizable (column parallel)",
            "output_batch_dim": "inherits from input",
            "output_feature_dim": "requires all-reduce if weight is column-parallel",
        }

    elif node.target in {torch.ops.aten.matmul, torch.ops.aten.mm, torch.ops.aten.bmm}:
        return {
            "matrix_a_batch": "parallelizable",
            "matrix_a_rows": "parallelizable",
            "matrix_a_cols": "requires synchronization with matrix_b_rows",
            "matrix_b_rows": "requires synchronization with matrix_a_cols",
            "matrix_b_cols": "parallelizable",
            "output": "requires all-reduce if inner dimension is sharded",
        }

    return {}


# Example usage function
def get_sharding_strategy(node: Node) -> str:
    """Get a high-level sharding strategy recommendation."""
    op_type, agg_dim = classify_operation_type(node)

    if op_type == "pointwise":
        return "REPLICATE_COMPUTATION - can distribute along any dimension"

    elif op_type == "linear":
        return "TENSOR_PARALLEL - use row/column parallelism with communication"

    elif op_type == "nonlinear_reduction":
        if agg_dim == -1:  # embedding/feature dimension
            return "NO_SHARD_EMBEDDING - do not distribute embedding dimension"
        elif agg_dim == -2:  # sequence dimension
            return "NO_SHARD_SEQUENCE - do not distribute sequence dimension"
        else:
            return f"NO_SHARD_DIM_{agg_dim} - do not distribute dimension {agg_dim}"

    return "ANALYZE_MANUALLY - unknown operation type"


def analyze_attention_sharding(node: Node) -> dict[str, str]:
    """Specific analysis for attention operations."""
    if node.target in {
        torch.ops.attention.scaled_dot_product_attention,
        torch.ops.attention.grouped_sdpa,
        torch.ops.attention.bsnd_grouped_sdpa,
    }:
        return {
            "batch_dimension": "parallelizable",
            "num_heads_dimension": "parallelizable",
            "sequence_dimension": "DO NOT SHARD - breaks attention semantics",
            "head_dimension": "parallelizable with all-reduce",
            "recommendation": "Use sequence-parallel only with specialized attention kernels",
        }
    return {}
