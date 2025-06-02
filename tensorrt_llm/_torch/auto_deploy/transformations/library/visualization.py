"""Transformation to the graph to render nicely in model_explorer."""

import json
from typing import Tuple

import model_explorer
import torch
from model_explorer.graph_builder import GraphNode, KeyValue, MetadataItem
from model_explorer.pytorch_exported_program_adater_impl import PytorchExportedProgramAdapterImpl
from torch import fx
from torch.fx import GraphModule
from torch.fx.passes.graph_drawer import FxGraphDrawer
import torch.nn as nn

from ..export import torch_export


def print_tensor(self, tensor: torch.Tensor, size_limit: int = 16):
    shape = tensor.shape
    total_size = 1
    for dim in shape:
        total_size *= dim

    if size_limit < 0 or size_limit >= total_size:
        return json.dumps(tensor.to(torch.float32).cpu().detach().clone().numpy().tolist())

    return json.dumps(
        (tensor.to(torch.float32).cpu().detach().clone().numpy().flatten())[:size_limit].tolist()
    )


def _get_shape(val):
    return json.dumps(
        list(
            map(
                lambda x: int(x) if str(x).isdigit() else str(x),
                val.shape,
            )
        )
    )


def add_outputs_metadata(self, fx_node: torch.fx.node.Node, node: GraphNode):
    out_vals = fx_node.meta.get("val")
    if out_vals is None:
        return

    if isinstance(out_vals, (tuple, list)):
        for idx, val in enumerate(out_vals):
            metadata = MetadataItem(id=str(idx), attrs=[])
            if val is None:
                continue
            dtype = str(val.dtype)
            shape = _get_shape(val)
            metadata.attrs.append(KeyValue(key="tensor_shape", value=dtype + shape))
            node.outputsMetadata.append(metadata)
    elif isinstance(out_vals, torch.Tensor):
        dtype = str(out_vals.dtype)
        shape = _get_shape(out_vals)
        metadata = MetadataItem(id="0", attrs=[KeyValue(key="tensor_shape", value=dtype + shape)])
        node.outputsMetadata.append(metadata)
    elif isinstance(out_vals, bool):
        metadata = MetadataItem(id="0", attrs=[KeyValue(key="tensor_shape", value="bool[1]")])
        node.outputsMetadata.append(metadata)
    else:
        raise ValueError(f"Unsupported output type: {type(out_vals)}")


PytorchExportedProgramAdapterImpl.print_tensor = print_tensor
PytorchExportedProgramAdapterImpl.add_outputs_metadata = add_outputs_metadata

# TODO(yudong): make custom_ops configurable
CUSTOM_OPS = (
    torch.ops.dist.all_reduce.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.attention.fused_mha_with_cache.default,
    torch.ops.linear.fused_linear_all_reduce.default,
    torch.ops.linear.simple.default,
    torch.ops.aten.split_with_sizes.default,
)


# TODO(yudong): make viz as non-block call.
def visualize_namespace(gm: fx.GraphModule, args: Tuple[torch.Tensor, ...], dynamic_shapes):
    ep = torch_export(gm, args=args, dynamic_shapes=dynamic_shapes)
    graph = ep.graph
    # Ensure the ops land up in the right module for better viz
    for n in graph.nodes:
        if n.target in CUSTOM_OPS:
            n.meta["nn_module_stack"] = n.args[0].meta["nn_module_stack"]

    model_explorer.visualize_pytorch("model-viz", ep)



def visualize_model(model: nn.Module, filename: str = "model.svg"):
    gm = torch.fx.symbolic_trace(model)
    visualize_graph(gm, model.dag, filename)


def visualize_graph(gm: GraphModule, filename: str = "graph.svg"):
    # Use FxGraphDrawer to visualize the graph
    drawer = FxGraphDrawer(gm, "my_module")
    dot_graph = drawer.get_dot_graph()
    
    # # Fix node labels by escaping special characters
    # for node in dot_graph.get_nodes():
    #     if node.get_label():
    #         # Get current label
    #         label = node.get_label()
    #         # Remove quotes at beginning and end if they exist
    #         if label.startswith('"') and label.endswith('"'):
    #             label = label[1:-1]
            
    #         # clean up the label
    #         # label is a string of the form:
    #         # '{key1=value1|key2=value2|...}'
    #         # we want to create the actual dictionary. Strip curly brackets from the string, split key-value pairs by |, then create a dictionary from the key-value pairs.
    #         label = label.strip('{}')
    #         label = label.split('|')
    #         label = [item.split('=') for item in label]
    #         label = dict(label)

    #         # start cleaning up.
    #         # Remove key "op_code"
    #         label.pop("op_code")
    #         # remove key "name"
    #         label.pop("name")
    #         # rename key "target" to "input"
    #         label["input"] = label.pop("target")
    #         # if key "args" exists, extract the einsum string of the contraction.
    #         if "args" in label:
    #             einsum_str = label["args"][1:-2]
    #             input_tensors = einsum_str.split("->")[0].split(",")
    #             # get contraction modes
    #             output_tensor = einsum_str.split("->")[1]
    #             contr = dag.nodes[output_tensor]
    #             modes_M = ''.join(contr.compute_grid.non_contracted_modes_A)
    #             modes_N = ''.join(contr.compute_grid.non_contracted_modes_B)
    #             modes_K = ''.join(contr.compute_grid.contracted_modes)
    #             label["contraction"] = einsum_str.replace("->", " = ")
    #             label["flattened"] = f"({modes_M})({modes_K}), ({modes_K})({modes_N}) = ({modes_M})({modes_N})"

    #             # get gloabl and local sizes M_global, N_global, K_global, M_local, N_local, K_local
    #             label["global_sizes"] = f"[{contr.compute_grid.M_global},{contr.compute_grid.K_global}] "\
    #                 f"x [{contr.compute_grid.K_global},{contr.compute_grid.N_global}] = "\
    #                 f"[{contr.compute_grid.M_global},{contr.compute_grid.N_global}]"
    #             label["rank_grid (P_m, P_k, P_n)"] = f"({contr.compute_grid.P_m},{contr.compute_grid.P_k},{contr.compute_grid.P_n})"
    #             label["local_sizes"] = f"[{contr.compute_grid.M_local},{contr.compute_grid.K_local}] "\
    #                 f"x [{contr.compute_grid.K_local},{contr.compute_grid.N_local}] = "\
    #                 f"[{contr.compute_grid.M_local},{contr.compute_grid.N_local}]"

    #             # label["flattened"] = f"{modes_M}{modes_K},{modes_K}{modes_N}={modes_M}{modes_N}"
                
    #             # label["input"] = einsum_str.split("->")[0]
    #             # label["contraction"] = ''.join(contraction_modes)
    #             # label["result"] = output_tensor
    #             label.pop("args")
    #             label.pop("input")

            
    #         # create a label string according to the dot format (keys separated by |, no whitespace)
    #         label_str = "{" + "|".join([f"{k}: {v}" for k, v in label.items()]) + "}"
    #         # Set the fixed label
    #         node.set_label(f'"{label_str}"')

    # Remove disconnected nodes
    # edges = dot_graph.get_edges()
    
    # # Collect all node names that appear in edges
    # connected_node_names = set()
    # for edge in edges:
    #     src = edge.get_source()
    #     dst = edge.get_destination()
    #     # Remove quotes if present
    #     if src.startswith('"') and src.endswith('"'):
    #         src = src[1:-1]
    #     if dst.startswith('"') and dst.endswith('"'):
    #         dst = dst[1:-1]
    #     connected_node_names.add(src)
    #     connected_node_names.add(dst)
    
    # # Find and remove disconnected nodes
    # nodes_to_remove = []
    # for node in dot_graph.get_nodes():
    #     node_name = node.get_name()
    #     # Remove quotes if present
    #     if node_name.startswith('"') and node_name.endswith('"'):
    #         node_name = node_name[1:-1]
    #     if node_name not in connected_node_names:
    #         nodes_to_remove.append(node)
    
    # for node in nodes_to_remove:
    #     dot_graph.del_node(node)
    
    # Save the modified graph
    dot_graph.write_svg(filename)
    a = 1