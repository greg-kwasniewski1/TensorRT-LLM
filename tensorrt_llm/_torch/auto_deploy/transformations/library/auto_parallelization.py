import torch
from ml_kernels.distributed_layers import Distributed3DLinear, DistributedRMSNorm
import torch.fx as fx
from transformations.export import torch_export_to_gm
from utils.node_utils import is_linear_op
from transformations.graph import lift_to_meta
from ml_kernels.elementary_seq_kernels import RMSNorm
import gc
from utils.debug_utils import DEBUG, log, INFO, CRITICAL, visualize_model
seed_offset = 0
torch.manual_seed(1337 + seed_offset)



# Registry to track all distributed linear layers by their original module path
_distributed_linear_registry = {}
# Registry to track data dependencies between layers
_layer_dependency_graph = {}


def distribute_linear_layer(layer, device, pretrained, root_layer: bool = False, seq_len: int = 128, batch_size: int = 1, layer_name=None, input_p_grid=None):
    """
    Convert a standard linear layer to a Distributed3DLinear layer.
    
    Args:
        layer: The nn.Linear layer to convert
        device: Device to place the new layer on
        pretrained: Whether to load weights from the original layer
        layer_depth: Depth of this layer in the model (affects grid dimensions)
        seq_len: Sequence length for sharding calculations
        batch_size: Batch size for sharding calculations
        layer_name: Optional name of the layer for debugging
        input_p_grid: Optional input PGrid for the layer
    Returns:
        A Distributed3DLinear layer configured to replace the input layer
    """
    if isinstance(layer, torch.nn.Linear):
        distr_linear = Distributed3DLinear(embd_dim=layer.in_features, 
                                         hidden_dim=layer.out_features,
                                         bias=layer.bias is not None, 
                                         root_layer=root_layer,
                                         seq_len=seq_len,
                                         batch_size=batch_size,
                                         input_p_grid=input_p_grid)
        distr_linear = distr_linear.to(device)
        if layer.bias is not None:
            distr_linear.bias.data = layer.bias.data
            
        if pretrained:
            distr_linear.weight.data = layer.weight.data
            if layer.bias is not None:
                distr_linear.bias.data = layer.bias.data
                
        # Register the new distributed linear layer with its name
        if layer_name:
            _distributed_linear_registry[layer_name] = distr_linear
        
        return distr_linear
    else:
        log(f"\n\nWARNING! Layer {layer_name} is not a Linear layer. Skipping.\n\n", log_level=CRITICAL)
        return layer
    
def trace_model_dependencies(model, example_input=None):
    """
    Trace a model to build a dependency graph between layers.
    
    Args:
        model: The model to trace
        example_input: An example input to use for tracing. If None, a dummy input is created.
        
    Returns:
        A dictionary mapping layer names to lists of their preceding layers.
    """
    # Create a dummy input if none provided
    if example_input is None:
        # Try to determine input shape from model architecture
        input_shape = (1, 512, 2048)  # Default (batch, seq_len, dim)
        device = next(model.parameters()).device
        example_input = torch.zeros(input_shape, device=device)
    
    # try:
        # Use FX to symbolically trace the model
    with lift_to_meta(model):
        dummy_input = torch.zeros_like(example_input, device="meta")
        traced_model = fx.symbolic_trace(model)
    # except Exception as e:
    #     log(f"Warning: Failed to symbolically trace model: {e}")
    #     log("Falling back to export-based graph creation")
        
    #     # Fall back to torch_export_to_gm which is more robust for complex models
    #     with lift_to_meta(model) as state_dict:
    #         dummy_input = torch.zeros_like(example_input, device="meta")
    #         traced_model = torch_export_to_gm(model, (dummy_input,))
    
    # Build dependencies graph
    dependencies = {}
    
    # Maps from nodes to the linearmodules they represent
    node_to_linear = {}
    
    # First pass: identify all linear layers
    for node in traced_model.graph.nodes:
        if node.op == "call_module":
            target_module = traced_model.get_submodule(node.target)
            if isinstance(target_module, torch.nn.Linear):
                node_to_linear[node] = node.target
    
    # Second pass: trace dependencies
    for node in traced_model.graph.nodes:
        if node in node_to_linear:
            current_name = node_to_linear[node]
            dependencies[current_name] = []
            
            # Recursively trace back through input nodes to find LinearLayers
            def find_linear_inputs(n, visited=None):
                if visited is None:
                    visited = set()
                
                if n in visited:
                    return []
                visited.add(n)
                
                if n in node_to_linear:
                    return [node_to_linear[n]]
                
                linear_deps = []
                for input_node in n.all_input_nodes:
                    linear_deps.extend(find_linear_inputs(input_node, visited))
                
                return linear_deps
            
            # Get all linear modules that feed into this one
            for input_node in node.all_input_nodes:
                dependencies[current_name].extend(find_linear_inputs(input_node))
    
    return dependencies

def get_previous_linear_layers(layer, layer_name=None, model=None):
    """
    Find all previous Distributed3DLinear layers in the model's dataflow.
    
    This function returns all predecessor layers based on the data dependencies
    in the model's dataflow graph.
    
    Args:
        layer: The current layer
        layer_name: Name of the current layer, if known
        model: The model containing the layers (needed for first run)
        
    Returns:
        List of preceding Distributed3DLinear layers in the dataflow
    """
    global _layer_dependency_graph
    
    # If this is the first time running on this model, trace the model
    if not _layer_dependency_graph and model is not None:
        # Trace the model to build dependency graph
        _layer_dependency_graph = trace_model_dependencies(model)
        log(f"Traced model dependencies: found {len(_layer_dependency_graph)} linear layers")
    
    # Find the name of the layer if not provided
    if layer_name is None:
        for name, registered_layer in _distributed_linear_registry.items():
            if registered_layer is layer:
                layer_name = name
                break
    
    # If we can't find the layer or have no dependency graph, fall back to simple approach
    if layer_name is None or not _layer_dependency_graph:
        # Simple fallback - return empty list
        return []
    
    # Look up dependencies
    if layer_name in _layer_dependency_graph:
        # Get predecessor layer names
        predecessor_names = _layer_dependency_graph[layer_name]
        
        # Get the actual Distributed3DLinear instances
        predecessors = []
        for pred_name in predecessor_names:
            if pred_name in _distributed_linear_registry:
                predecessors.append(_distributed_linear_registry[pred_name])
        
        return predecessors
    
    return []

def get_previous_linear_layer(layer, layer_name=None, model=None):
    """
    Find the previous Distributed3DLinear layer in the model.
    
    This simplified version maintains backward compatibility with the original function
    by returning only the first previous layer if multiple exist.
    
    Args:
        layer: Current layer (torch.nn.Linear or Distributed3DLinear)
        layer_name: Name of the current layer, if known
        model: The model containing the layers (needed for first run)
        
    Returns:
        The first previous Distributed3DLinear layer in the dataflow, or None if none exist
    """
    predecessors = get_previous_linear_layers(layer, layer_name, model)
    
    # Return the first predecessor, if any
    if predecessors:
        return predecessors[0]
    return None

def parallelize_model(model, 
                      device, 
                      input_distributer: Distributed3DLinear,
                      pretrained: bool = False,
                      layers_to_skip: list = [],
                      seq_len: int = 128,
                      batch_size: int = 1):
    """
    Convert a standard model with nn.Linear layers to a distributed model using Distributed3DLinear.
    
    This function handles the process of finding all linear layers in the model and replacing them with
    distributed versions, while ensuring connections between consecutive layers are properly configured.
    
    Args:
        model: The model to parallelize
        device: Device to place the new layers on
        pretrained: Whether to load weights from the original layers
        layers_to_skip: List of layer type names to skip
        seq_len: Sequence length for sharding calculations
        batch_size: Batch size for sharding calculations
        dtype: Data type for the model
        
    Returns:
        The parallelized model
    """

    
    visualize_model(model, "model_pre.svg")

    global _distributed_linear_registry, _layer_dependency_graph
    
    # Reset the registries
    _distributed_linear_registry = {}
    _layer_dependency_graph = {}

    # First trace the model to build the dependency graph
    # try:
    log("Tracing model to analyze layer dependencies...")
    # Create a dummy input based on model structure
    # This is a simplified approach - we might need more complex logic for different models
    example_shape = (1, seq_len, 2048)  # Assuming a transformer model
    dummy_input = torch.zeros(example_shape, device=device)
    _layer_dependency_graph = trace_model_dependencies(model, dummy_input)
    # except Exception as e:
    #     log(f"Warning: Failed to trace model: {e}")
    #     log("Will rely on parallelization-time dependency tracking")
    
    layer_depth = 0
    # Collection of all linear layers first to avoid modifying during iteration
    linear_layers = [(name, module) for name, module in model.named_modules() 
                    if isinstance(module, torch.nn.Linear)]
    
    log(f"Found {len(linear_layers)} linear layers to parallelize", log_level=DEBUG)
    
    # Replace linear layers with distributed linear layers
    for name, module in linear_layers:
        # Get parent module name by getting everything before the last dot
        parent_name = '.'.join(name.split('.')[:-1])
        parent_module = dict(model.named_modules())[parent_name]
      
        if any(lr_to_skip in str(type(parent_module).__name__) for lr_to_skip in layers_to_skip):
            log(f"Skipping layer {name} (part of {type(parent_module).__name__})")
            continue
            
        # Create the new distributed linear module
        log(f"Converting layer {name} to Distributed3DLinear (depth {layer_depth}). First, trying to find the predecessors distributed linear layers.")        
        # Find all previous layers based on dataflow dependencies
        prev_layers = get_previous_linear_layers(module, name, model)
        input_p_grid = None
        if prev_layers:
            # In the case of multiple predecessors, we need a strategy
            # For now, use the first predecessor's grid as our input grid
            # A more sophisticated approach would be to ensure all predecessors use compatible grids
            prev_d3d_layer = prev_layers[0]
            
            # Use the output PGrid of the previous layer as the input PGrid for this layer
            input_p_grid = prev_d3d_layer.grid
            
            # Log this connection for debugging
            prev_names = []
            for i, prev_layer in enumerate(prev_layers):
                for reg_name, reg_layer in _distributed_linear_registry.items():
                    if reg_layer is prev_layer:
                        prev_names.append(reg_name)
                        break
                # Only use the first predecessor for now
                if i == 0:
                    log(f"Setting input grid for {name} ({module}) to {prev_layer.grid}")
                    # new_module.input_p_grid = prev_layer.grid
            
            log(f"Connected layer {name} to previous layers: {prev_names}")
            
            # TODO: Handle multiple predecessors better - we need to ensure they all use compatible grids
            if len(prev_layers) > 1:
                log(f"Warning: Layer {name} has multiple predecessors. Using the first one's grid.")
        else:
            log(f"No previous layer found for {name}")

        new_module = distribute_linear_layer(
            module, device, pretrained, root_layer= len(prev_layers) == 0, 
            seq_len=seq_len, batch_size=batch_size, layer_name=name, input_p_grid=input_p_grid)
        
        # Get the parent module and set the new module
        name_parts = name.split(".")
        parent = model
        for attr in name_parts[:-1]:
            parent = getattr(parent, attr)
        setattr(parent, name_parts[-1], new_module)

        model.last_layer = new_module
        
    log(f"Successfully parallelized {len(_distributed_linear_registry)} linear layers", log_level=INFO)
    

    # parallelize th RMSNorm layers
    norm_layers = [(name, module) for name, module in model.named_modules() if isinstance(module, RMSNorm)]
    for name, norm_layer in norm_layers:
        log(f"Converting RMSNorm layer {name} to DistributedRMSNorm")
        # find the previous linear layer
        prev_layer = get_previous_linear_layer(norm_layer, model=model)
        previous_p_grid = None
        if prev_layer:
            previous_p_grid = prev_layer.grid
        else:
            previous_p_grid = input_distributer.grid

        # create a new RMSNorm layer
        distr_norm_layer = DistributedRMSNorm(p_grid=previous_p_grid, 
                                              weight=norm_layer.weight, 
                                              norm_eps=norm_layer.norm_eps)

        # get the parent module and set the new module
        name_parts = name.split(".")
        parent = model
        for attr in name_parts[:-1]:
            parent = getattr(parent, attr)
        setattr(parent, name_parts[-1], distr_norm_layer)

    
    # parallelize attention layers
    # possible names for number of heads attributes:
    head_param_names = ["n_heads", "num_heads", "n_head", "num_head", "nHeads", "numHeads", "n_kv_heads", "num_kv_heads"]
    attn_layers = [(name, module) for name, module in model.named_modules() if 
                   "attention" in str(type(module)).lower() and 
                   any(hasattr(module, param_name) for param_name in head_param_names)]

    for name, attn_layer in attn_layers:
        log(f"Parallelizing attention layer {name}")
        for head_param_name in head_param_names:
            if hasattr(attn_layer, head_param_name):
                log(f"{head_param_name} in this layer: {getattr(attn_layer, head_param_name)}")
                # divide the number of heads by the number of ranks and set new value
                setattr(attn_layer, head_param_name, getattr(attn_layer, head_param_name) // input_distributer.grid.P)                    
        
    log(f"Successfully parallelized {len(attn_layers)} attention layers", log_level=INFO)
    visualize_model(model, "model_post.svg")
    exit()
                
    return model
        


def initialize_weight_matrices(layer: torch.nn.Linear, grid: tuple, fixed_init: bool = True):
    """Initialize weights and bias with same values on all ranks."""
    P_k, P_n = grid
    K_global, N_global = layer.weight.shape
    K_local, N_local =  K_global // P_k, N_global // P_n
    if fixed_init:
        # # split X_global accordingly to the distribute_input_tensor function, that is [M//P_m, K//P_k]
        # tmp = layer.weight.view(P_k, K_local, P_n, N_local).detach()
        # # initialize each tile with its rank number
        # for i in range(P_k):
        #     for j in range(P_n):
        #         tmp[i, :, j, :] = i * P_n + j + 1
          # split X_global accordingly to the distribute_input_tensor function, that is [M//P_m, K//P_k]
        # initialize each tile with its rank number
        for i in range(K_global):
            for j in range(N_global):
                layer.weight[i, j] = ((i % 3) + (j % 2))/3  # (i % 3) * N_global + (j % 5) 
        
        # log(f"init global weight: \n{layer.weight.detach().cpu().numpy()}\n")


def initialize_all_linear_layers(model, grid: tuple, fixed_init: bool = True):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            initialize_weight_matrices(module, grid, fixed_init)
    