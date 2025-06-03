import copy
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import math
import os
import datetime
from dataclasses import dataclass

def cdiv(a, b):
    return (a + b - 1) // b

# from utils.debug_utils import CRITICAL, VALUES, log, INFO, DEBUG
# from ml_kernels.elementary_seq_kernels import precompute_freqs_cis, apply_rotary_emb

# STRATEGY = "megatron"
STRATEGY = "COSMA"


@dataclass
class DistributedTensor:
    p: int
    P: int
    einsum_str: str

    
    grid_extents: dict[str, int]
    modes_extents: dict[str, int]
    
    # modes_extens_local = modes_extents / grid_extents
    modes_extents_local: dict[str, int]
    extent_order: dict[str, int]
    extent_order_sorted = list[str]

    def __init__(self,
                 modes_extents: dict[str, int],
                 einsum_str: str,
                 extent_order_sorted: list[str] = None,
                 grid_extents: dict[str, int] = None,
                 world_size: int = None):       
        rank = dist.get_rank()
        if world_size is None:
            world_size = dist.get_world_size()
        self.P = world_size
        self.p = rank
        self.modes_extents = modes_extents
        self.einsum_str = einsum_str

        if not extent_order_sorted:
            self.extent_order_sorted = list(sorted(modes_extents.keys()))
        else:
            self.extent_order_sorted = extent_order_sorted
        
        if not grid_extents:
            self.grid_extents = {k: 1 for k in self.extent_order_sorted}
        else:
            self.grid_extents = grid_extents

        self.extent_order = {}
        i = 0
        for k in self.extent_order_sorted:
            if k in einsum_str:
                self.extent_order[k] = i
                i += 1        
        

        self.update_local_extents()
        self.calculate_partial_prod()
        self.p_coords = self.rank_to_coords()


    def reinit(self):
        self.update_local_extents()
        self.calculate_partial_prod()
        self.p_coords = self.rank_to_coords()
        
    
    def calculate_partial_prod(self):
        grid_extents_sorted = [self.grid_extents[mode] for mode in self.extent_order_sorted]
        
        self.partial_prod = {}
        prev = 1
        for i in range(len(self.extent_order_sorted)-1, -1, -1):
            self.partial_prod[self.extent_order_sorted[i]] = prev            
            prev *= grid_extents_sorted[i]

    
    def init_flattening(self):
        """Initialize the flattening of the input tensors."""
        self.non_contracted_modes_A = [mode for mode in self.tensor_A if mode in self.tensor_C]
        if self.tensor_B is not None:
            self.non_contracted_modes_B = [mode for mode in self.tensor_B if mode in self.tensor_C and mode not in self.tensor_A]
        else:
            self.non_contracted_modes_B = []
        self.contracted_modes = [mode for mode in self.tensor_A if mode not in self.tensor_C]

        self.M_global = int(np.prod([self.modes_extents[mode] for mode in self.non_contracted_modes_A]))
        self.N_global = int(np.prod([self.modes_extents[mode] for mode in self.non_contracted_modes_B]))
        self.K_global = int(np.prod([self.modes_extents[mode] for mode in self.contracted_modes]))    
        

    
    def update_local_extents(self):
        # update local extents of non_contracted_modes_A, non_contracted_modes_B, contracted_modes
        self.modes_extents_local = {}
        for mode in self.extent_order_sorted:
            self.modes_extents_local[mode] = cdiv(self.modes_extents[mode], self.grid_extents[mode])
    
    def matches(self, other, modes: str):
        """
        Custom equality operator to compare PGrid objects.                
        """
        previous_modes = other.tensor_C
        for i, mode in enumerate(modes):
            if mode not in other.modes_extents_local:
                # get matching mode in previous_modes and replace it
                prev_mode = previous_modes[i]
                other.grid_extents[mode] = other.modes_extents_local[prev_mode]
                other.modes_extents_local[mode] = other.modes_extents_local[prev_mode]
                other.modes_extents[mode] = other.modes_extents[prev_mode]
                other.extent_order[mode] = other.extent_order[prev_mode]
                # find index of prev_mode in extent_order_sorted
                index = other.extent_order_sorted.index(prev_mode)
                other.extent_order_sorted[index] = mode
                # add mode to partial_prod
                other.partial_prod[mode] = other.partial_prod[prev_mode]
                # remove prev_mode from partial_prod
                other.partial_prod.pop(prev_mode)

        return all(self.modes_extents_local[mode] == other.modes_extents_local[mode] for mode in modes) \
        and all(self.extent_order[mode] == other.extent_order[mode] for mode in modes)
                

    def rank_to_coords(self, p: int = None) -> tuple[int, int, int]:
        """Convert a rank to its coordinates in the process grid."""
        if p is None:
            p = self.p
        p_remaining = p
        p_coords = {}
        for m in self.extent_order_sorted:
            p_coords[m] = p_remaining // self.partial_prod[m]
            p_remaining = p_remaining % self.partial_prod[m]
        return p_coords
            


    def coords_to_rank(self, p_coords:dict[str, int] = None) -> int:
        """Convert coordinates to rank."""
        if p_coords is None:
            p_coords = self.p_coords
        p = 0
        for i, mode in enumerate(self.extent_order_sorted):
            p += p_coords[mode] * self.partial_prod[mode]
        return p

    
    def coords_to_global_slice(self, p_coords: dict[str, int] = None) -> dict[str, slice]:
        """Convert coordinates to slice."""
        if p_coords is None:
            p_coords = self.p_coords
        
        slices = {}
        for mode in self.extent_order_sorted:
            slices[mode] = (p_coords[mode] * self.modes_extents_local[mode], 
                            (p_coords[mode] + 1) * self.modes_extents_local[mode])
        return slices

    
    def element_to_coords(self, x: dict[str, int]) -> int:
        """Convert global element address, given as a vector x of element indices, 
        to its owners rank given by the process grid."""
        owner = {}
        for mode in self.extent_order_sorted:
            owner[mode] = x[mode] // self.modes_extents_local[mode]
        return owner
    
    
    def distribute_input_tensor(self, modes: str, X_global: torch.Tensor = None) -> torch.Tensor:
        """
        Extract local input tensor slice given by the modes string.
        
        Args:
            X_global: Input tensor, defined by the modes string
            modes: String of modes to extract
            
        Returns:
            X_local: Local tensor of shape self.modes_extents_local[mode] for each mode in modes
        """
        local_slice = self.coords_to_global_slice()
        slicer = [slice(None)] * len(modes)  # Full slices for all axes       
        # slice is a dictionary of slices for each mode
        # iterate over modes present in X_global
        for i, mode in enumerate(modes):
            mode_i_min, mode_i_max = local_slice[mode]
            # get the slice of the global tensor of the i-th mode
            slicer[i] = slice(mode_i_min, mode_i_max)
        tmp = X_global[tuple(slicer)]
        return X_global[tuple(slicer)]



        

@dataclass
class PGrid:
    P_k: int
    P_m: int
    P_n: int

    rank: int
    P: int
    M_global: int
    K_global: int
    N_global: int

    M_local: int
    K_local: int
    N_local: int


    rank_order: tuple[int, int, int]

    def __init__(self, 
                 M_global: int, K_global: int, N_global: int, 
                 world_size: int = None,
                 rank: int = None,
                 rank_order: tuple[int, int, int] = None,
                 grid: tuple[int, int, int] = None):
        self.M_global = M_global
        self.K_global = K_global
        self.N_global = N_global
        if world_size is None:
            world_size = dist.get_world_size()
        self.P = world_size
        if rank is None:
            rank = dist.get_rank()
        self.rank = rank
        
        if grid is None:
            self.grid = self.find_optimal_grid(M_global, K_global, N_global, P)
        else:
            self.grid = grid
        self.P_k = self.grid[0]
        self.P_m = self.grid[1]
        self.P_n = self.grid[2]
        self.K_local = K_global // self.grid[0]
        self.M_local = M_global // self.grid[1]
        self.N_local = N_global // self.grid[2]
        if not rank_order:
            self.rank_order = (0, 1, 2)
        else:
            self.rank_order = rank_order
        self.pk, self.pm, self.pn = self.rank_to_coords()
    
    def matches(self, other):
        """
        Custom equality operator to compare PGrid objects.
        Two PGrids are considered equal if they have the same dimensions and rank order.
        """
        if not isinstance(other, PGrid):
            return False
        
        # Compare the core grid properties
        grid_match = (self.P_k == other.P_n and 
                     self.P_m == other.P_m and 
                     self.P_n == other.P_k)
        
        # # Compare the global dimensions 
        # dims_match = (self.M_global == other.M_global and
        #              self.K_global == other.K_global and
        #              self.N_global == other.N_global)
        
        # Compare rank ordering (this is critical for reshuffling decisions)
        order_match = self.rank_order == (other.rank_order[2], other.rank_order[1], other.rank_order[0])
        
        return grid_match and order_match
    

    def calc_commvol(self, P_k: int, P_m: int, P_n: int, K: int, M: int, N: int):
        """Calculate the communication volume for a given grid."""
        # Calculate the communication volume for the given grid
        m_local = M // P_m
        n_local = N // P_n
        k_local = K // P_k
        comm_vol = m_local*n_local + m_local*k_local + n_local*k_local
        return comm_vol
    
    def find_optimal_grid(self, M: int, K: int, N: int, P: int):
        V = M * K * N/P
        a = V**(1/3)
        grid = [int(K/a), int(M/a), int(N/a)]

        # different approach. Exhaustively try all possible grids,
        # for each grid, calculate the communication volume, and choose the one with the least communication volume
        min_comm_vol = float('inf')
        best_grid = None
        for P_k in range(1, P+1):
            for P_m in range(1, P//P_k+1):
                P_n = P//P_k//P_m
                comm_vol = self.calc_commvol(P_k, P_m, P_n, K, M, N)

                if comm_vol == min_comm_vol:
                    # break ties.
                    if N > K:
                        # for larger N, choose the grid with the smaller P_k
                        if P_k < best_grid[0]:
                            best_grid = (P_k, P_m, P_n)
                    else:
                        # for larger K, choose the grid with the smaller P_n
                        if P_n < best_grid[2]:
                            best_grid = (P_k, P_m, P_n)

                if comm_vol < min_comm_vol:
                    min_comm_vol = comm_vol
                    best_grid = (P_k, P_m, P_n)                    

        if STRATEGY == "megatron":
            if N > K:
                best_grid = (1, 1, P)
            else:
                best_grid = (P, 1, 1)

        # best_grid = (2, 2, 2)
        log(f"\n\nbest_grid: {best_grid}")
        return best_grid
                

    def rank_to_coords(self) -> tuple[int, int, int]:
        """Convert a rank to its coordinates in the process grid."""

        if self.rank_order == (0, 1, 2):
            pn = self.rank % self.P_n
            pm = (self.rank // self.P_n) % self.P_m
            pk = self.rank // (self.P_m * self.P_n)
        elif self.rank_order == (2, 1, 0):
            pk = self.rank % self.P_k
            pm = (self.rank // self.P_k) % self.P_m
            pn = self.rank // (self.P_m * self.P_k)
        elif self.rank_order == (1, 0, 2):
            pk = self.rank % self.P_n
            pm = (self.rank // self.P_n) % self.P_k
            pn = self.rank // (self.P_k * self.P_n)
        elif self.rank_order == (1, 2, 0):
            pk = self.rank % self.P_k
            pn = (self.rank // self.P_m) % self.P_n
            pm = self.rank // (self.P_n * self.P_m)
        elif self.rank_order == (2, 0, 1):
            pk = self.rank % self.P_m
            pm = (self.rank // self.P_m) % self.P_k
            pn = self.rank // (self.P_k * self.P_m)
        else:
            raise ValueError(f"Invalid rank order: {self.rank_order}")
        return (pk, pm, pn)

    def coords_to_rank(self, grid_coords: tuple[int, int, int] = None, input: bool = False) -> int:
        """Convert coordinates to rank."""
        if grid_coords is None:
            grid_coords = (self.pk, self.pm, self.pn)
        pk, pm, pn = grid_coords
        if input:
            if self.rank_order == (0, 1, 2):
                return pn * self.P_m * self.P_n + pm * self.P_n + pk
            elif self.rank_order == (2, 1, 0):
                return pk * self.P_m * self.P_k + pm * self.P_k + pn
            elif self.rank_order == (1, 0, 2):
                return pm * self.P_k * self.P_m + pk * self.P_k + pn
            else:
                # print(f"\n\nInvalid rank order: {rank_order}. Type: {type(rank_order)}")
                # print(f"rank_order == (0, 1, 2): {rank_order == (0, 1, 2)}\n\n")
                raise ValueError(f"Invalid rank order: {self.rank_order}")
        else:
            if self.rank_order == (0, 1, 2):
                return pk * self.P_m * self.P_n + pm * self.P_n + pn
            elif self.rank_order == (2, 1, 0):
                return pn * self.P_m * self.P_k + pm * self.P_k + pk
            elif self.rank_order == (1, 0, 2):
                return pm * self.P_k * self.P_m + pk * self.P_k + pn
            else:
                # print(f"\n\nInvalid rank order: {rank_order}. Type: {type(rank_order)}")
                # print(f"rank_order == (0, 1, 2): {rank_order == (0, 1, 2)}\n\n")
                raise ValueError(f"Invalid rank order: {self.rank_order}")

    
    def coords_to_slice(self, grid_coords: tuple[int, int, int] = None, input: bool = False) -> tuple[int, int, int, int]:
        """Convert coordinates to slice."""
        if grid_coords is None:
            grid_coords = (self.pk, self.pm, self.pn)
        
        pk, pm, pn = grid_coords
        if input:
            m_min = pm * self.M_local
            m_max = (pm + 1) * self.M_local
            n_min = pk * self.K_local
            n_max = (pk + 1) * self.K_local
        else:
            m_min = pm * self.M_local
            m_max = (pm + 1) * self.M_local
            n_min = pn * self.N_local
            n_max = (pn + 1) * self.N_local
        return m_min, m_max, n_min, n_max

    
    def slice_to_coords(self, m_min: int, m_max: int, n_min: int, n_max: int, input: bool = False) -> tuple[int, int, int]:
        """Convert slice to coordinates."""
        if input:
            pm_min = m_min // self.M_local
            pm_max = cdiv(m_max, self.M_local)
            pn_min = n_min // self.N_local
            pn_max = cdiv(n_max, self.N_local)
        else:
            pm_min = m_min // self.M_local
            pm_max = cdiv(m_max, self.M_local)
            pn_min = n_min // self.K_local
            pn_max = cdiv(n_max, self.K_local)
        return (pm_min, pm_max, pn_min, pn_max)


    def slice_to_coord(self, m_global: int, n_global: int, input: bool = False) -> tuple[int, int, int]:
        """Convert slice to coordinates."""
        pm = m_global // self.M_local
        if input:
            pn = n_global // self.N_local
        else:
            pn = n_global // self.K_local
        return pm, pn
        
        


# class DistributedRMSNorm(torch.nn.Module):
#     def __init__(self,
#                     p_grid: PGrid = None,
#                     weight: torch.Tensor = None,
#                     norm_eps: float = 1e-5,
#                     root_layer: bool = False,
#                     device: str = SurrogateConfig.device,
#                     norm_dtype: torch.dtype = SurrogateConfig.norm_dtype,
#                     in_dtype: torch.dtype = torch.float16):
#         super().__init__()
#         self.device = device
#         self.p_grid = p_grid
#         self.norm_dtype = norm_dtype
#         self.in_dtype = in_dtype
#         # counter that will increase by the amount of data communicated over the network each time torch.dist is called
#         self.comm_vol = 0
#         # Initialize process groups for communication
#         self._init_process_groups()
#         # Initialize local weight slice
#         self.init_local_weight_slice(weight, norm_eps)

            

#     def init_local_weight_slice(self, weight, norm_eps):
#         # Initialize global weight matrix on all ranks
#         self.weight = weight
#         self.norm_eps = norm_eps
            
#         # Slice the global weight matrix according to process grid coordinates
#         # Reshape weight to [P_k, K//P_k, P_n, N//P_n]
#         w_reshaped = self.weight.view(self.p_grid.P_n, self.p_grid.N_local)
#         # Get local slice based on process grid coordinates
#         self.weight_local = w_reshaped[self.p_grid.pn]
        
#         # Remove the global weight parameter since we'll only use weight_local
#         del self.weight


        
#     def _init_process_groups(self):
#         """Initialize process groups for different communication patterns."""
#         self.reduction_groups = []
        
#         # All processes create all groups for all combinations of pm and pn
#         for pk in range(self.p_grid.P_k):
#             for pm in range(self.p_grid.P_m):
#                 reduction_ranks = [self.p_grid.coords_to_rank(grid_coords=(pk, pm, pn)) for pn in range(self.p_grid.P_n)]
                
#                 # !!!!!!!!!!!!!!!!!!!!! IMPORTANT !!!!!!!!!!!!!!!!!!!!!
#                 # DEBUG ONLY
#                 group = reduction_ranks
#                 # correct version
#                 # group = dist.new_group(reduction_ranks, timeout=datetime.timedelta(seconds=30))
#                 # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                
#                 # Store the group if this process is part of it
#                 if self.p_grid.rank in reduction_ranks:
#                     self.reduction_groups.append((group, (pk, pm)))

#     def forward(self, x_local):
#         x_local = x_local.to(dtype=self.norm_dtype)
#         x2_mean_local = x_local.pow(2).mean(-1, keepdim=True) + self.norm_eps
#         # perform allreduce on x2_mean_local

#         # dist.barrier()
#         # log(f"\n\nRMSnorm reduction groups: {self.reduction_groups}\ngrid: {self.p_grid}", log_level=DEBUG)
#         # exit()
#         for group, (pk, pm) in self.reduction_groups:
#             if self.p_grid.pm == pm and self.p_grid.pk == pk:
                
#                 # !!!!!!!!!!!!!!!!!!!!! IMPORTANT !!!!!!!!!!!!!!!!!!!!!
#                 # DEBUG ONLY
#                 # group_ranks = dist.get_process_group_ranks(group)    
#                 group_ranks = copy.deepcopy(group)            
#                 group = dist.new_group(group, timeout=datetime.timedelta(seconds=30))
                
#                 # correct version - do nothing, use the already created group
#                 # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#                 if dist.get_world_size(group) > 1:
#                     # log(f"\n\nx2_mean_local.shape: {x2_mean_local.shape}, type: {x2_mean_local.dtype}", log_level=DEBUG)
#                     # exit()
#                     dist.all_reduce(x2_mean_local, op=dist.ReduceOp.SUM, group=group)
#                     x2_mean_local /= dist.get_world_size(group)
#                     # log(f"Rank {self.grid.rank}: all_reduce comm volume:  {self.comm_vol} += {torch.prod(torch.tensor(y_local.shape))} * 2 * ({dist.get_world_size(group)} - 1)/{dist.get_world_size(group)}",
#                         #  log_level=INFO, ranks=[0])
#                     dist.barrier()
#                     log(f"Rank {self.p_grid.rank}, allreduce within group: {group_ranks}, " +  \
#                         f"reduce buffer size: {torch.prod(torch.tensor(x2_mean_local.shape))}", log_level=DEBUG)
                        
#                     log(f"result:\n{x2_mean_local.detach().cpu().numpy()}",
#                         log_level=VALUES, ranks=[0])
#                     dist.barrier()
            
#                     self.comm_vol += torch.prod(torch.tensor(x2_mean_local.shape)) * 2 * (dist.get_world_size(group) - 1)/dist.get_world_size(group)
                                
#         x_local = x_local * torch.rsqrt(x2_mean_local)
#         out = x_local.to(dtype=self.in_dtype)

        
#         return out * self.weight_local 


    # def _norm(self, x):
    #     return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.norm_eps)
    
    # def forward(self, x):
    #     out = self._norm(x.float()).type_as(x)
    #     return out * self.weight # (2, 8, DIM) Values stays the same. We make the tensor grad_fn.

class Distributed3DLinear(nn.Module):
    """Distributed 3D linear layer implementing matrix multiplication across a 3D grid of GPUs."""
    def __init__(self,
                  input_p_grid: PGrid = None,
                  output_p_grid: PGrid = None,
                  bias: bool = False,
                  batch_size: int = 1,
                  seq_len: int = 1,
                  embd_dim: int = 1,
                  hidden_dim: int = 1,
                  root_layer: bool = False,
                  device: str = "cuda"):
        super().__init__()
        self.device = device
        self.input_p_grid = input_p_grid

        if output_p_grid is None:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            if input_p_grid is None:
                rank_order = (0, 1, 2) if root_layer else (2, 1, 0)
            else:
                rank_order = (0, 1, 2) if input_p_grid.rank_order == (2, 1, 0) else (2, 1, 0)
            output_p_grid = PGrid(rank=rank, P=world_size, 
                                  M_global=batch_size*seq_len, 
                                  K_global=embd_dim, 
                                  N_global=hidden_dim, rank_order=rank_order)
        self.grid = output_p_grid

        

        # counter that will increase by the amount of data communicated over the network each time torch.dist is called
        self.comm_vol = 0
        # Initialize process groups for communication
        self._init_process_groups()
        # Initialize local weight slice
        # log(f"Initializing local weight slice. Grid: {self.grid}, global sizes: [M, N, K] = [{self.grid.M_global}, {self.grid.N_global}, {self.grid.K_global}]", log_level=DEBUG)
        self.init_local_weight_slice(bias)

        
            

    def init_local_weight_slice(self, bias: bool = False, dtype: torch.dtype = torch.float16):
        # Initialize global weight matrix on all ranks
        self.weight = nn.Parameter(torch.empty(self.grid.N_global, self.grid.K_global, device=self.device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.grid.N_global, device=self.device, dtype=dtype))
        else:
            self.register_parameter('bias', None)
            
        
        # Initialize weights with same values on all ranks
        self.initialize_weight_matrices()
        # Slice the global weight matrix according to process grid coordinates
        # Reshape weight to [P_k, K//P_k, P_n, N//P_n]
        w_reshaped = self.weight.view(self.grid.P_n, self.grid.N_local, self.grid.P_k, self.grid.K_local)
        # Get local slice based on process grid coordinates
        self.weight_local = w_reshaped[self.grid.pn, :, self.grid.pk, :]
        
        # Remove the global weight parameter since we'll only use weight_local
        del self.weight

    
        
    def _init_process_groups(self):
        """Initialize process groups for different communication patterns."""
        self.reduction_groups = []
        
        # All processes create all groups for all combinations of pm and pn
        for pm in range(self.grid.P_m):
            for pn in range(self.grid.P_n):
                reduction_ranks = [self.grid.coords_to_rank(grid_coords=(pk, pm, pn)) for pk in range(self.grid.P_k)]
                
                # !!!!!!!!!!!!!!!!!!!!! IMPORTANT !!!!!!!!!!!!!!!!!!!!!
                # DEBUG ONLY
                group = reduction_ranks
                # correct version
                # group = dist.new_group(reduction_ranks, timeout=datetime.timedelta(seconds=30))
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                
                # Store the group if this process is part of it
                if self.grid.rank in reduction_ranks:
                    self.reduction_groups.append((group, (pm, pn)))
        
        

    def initialize_weight_matrices(self, fixed_init: bool = True, dtype: torch.dtype = torch.float16):
        """Initialize weights and bias with same values on all ranks."""
        if fixed_init:
            # split X_global accordingly to the distribute_input_tensor function, that is [M//P_m, K//P_k]
            tmp = self.weight.detach().to(dtype=dtype)
            # # initialize each tile with its rank number
            # for i in range(self.grid.P_k):
            #     for j in range(self.grid.P_n):
            #         tmp[i, :, j, :] = i * self.grid.P_n + j + 1
            # # reshape X_global back to [batch, seq, embd]
            # self.weight = torch.nn.Parameter(tmp.view(self.grid.K_global, self.grid.N_global))
            for i in range(self.grid.N_global):
                for j in range(self.grid.K_global):
                    tmp[i, j] = ((i % 3) + (j % 2))/3 # (i % 3) * self.grid.K_global + (j % 5) 
            self.weight = torch.nn.Parameter(tmp)
            # log(f"init weight. Weight shape: {self.weight.shape}, N_global: {self.grid.N_global}, K_global: {self.grid.K_global}\n{self.weight.detach().cpu().numpy()}\n")
        else:
            # Initialize global weight matrix
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                fan_in = self.grid.K_global
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x_local: torch.Tensor, start_pos = None, mask = None, freqs_cis = None) -> torch.Tensor:
        """
        Forward pass of distributed 3D linear layer.
        
        Args:
            x_local: Pre-distributed input tensor of shape [M//P_m, K//P_k]
            
        Returns:
            y_local: Local output tensor of shape [M//P_m, N//P_n]
        """
        log(f"Distributed3DLinear forward. x_local type: {type(x_local)}", log_level=DEBUG)
        log(f"x_local dtype: {x_local.dtype}", log_level=DEBUG)
        if isinstance(x_local, torch.fx.proxy.Proxy) or x_local.device.type == "meta":
            # Create a meta output tensor with the correct shape
            # if x_local.ndim == 3:
            # batch, seq_len, _ = self.config. self.grid.M_global, self.grid.N_global
            # M, N = self.grid.M_local, self.grid.N_local
            # M, N = 1, 1
            return torch.matmul(x_local, self.weight_local.T)    
            # else:
            #     m_local = x_local.shape[0]
            #     return torch.empty((m_local, self.grid.N_local), device="meta", dtype=x_local.dtype)

        # Get local rank for device synchronization
        local_rank = int(os.environ.get("LOCAL_RANK", self.grid.rank))

        # if X is 3D, flatten it to 2D
        if x_local.ndim == 3:
            x_local = x_local.view(x_local.shape[0] * x_local.shape[1], x_local.shape[2])
        
        # check if x_local has any nans
        log(f"\n\n\nType of x_local: {type(x_local)}", log_level=DEBUG)
        log(f"device of x_local: {x_local.device}\n\n", log_level=DEBUG)
        if torch.isnan(x_local).any() or torch.isinf(x_local).any():
            log(f"Rank {self.grid.rank}, distributed linear layer beginning with nans", log_level=CRITICAL)
            exit()
        # if self.grid.rank == 0:
        #     print(f"BEFORE RESHUFFLE: x_local.shape: {x_local.shape}, self.weight_local.shape: {self.weight_local.shape}, \nself.input_p_grid: \n{self.input_p_grid}, \nself.output_p_grid: \n{self.grid}")

        # Reshuffle if needed
        x_local = self.reshuffle(x_local)
        # if self.grid.rank == 0:
        #     print(f"AFTER RESHUFFLE: x_local.shape: {x_local.shape}, self.weight_local.shape: {self.weight_local.shape}, \nself.input_p_grid: \n{self.input_p_grid}, \nself.output_p_grid: \n{self.grid}")
        # print(f"Rank {self.grid.rank}: inside forward pass, x_local:\n{x_local.detach().cpu().numpy()}\nweight_local:\n{self.weight_local.detach().cpu().numpy()}")
        y_local = torch.matmul(x_local, self.weight_local.T)       
        log(f"Rank {self.grid.rank}: Forward pass after matmul. x_local:\n{x_local.detach().cpu().numpy()}" +\
            f"\nweight_local:\n{self.weight_local.detach().cpu().numpy()}\ny_local:\n{y_local.detach().cpu().numpy()}\n", 
            ranks=[0], log_level=VALUES)
        # Perform allreduce across ranks with same [pm, pn] coordinates
        for group, (pm, pn) in self.reduction_groups:
            if self.grid.pm == pm and self.grid.pn == pn:
                
                # !!!!!!!!!!!!!!!!!!!!! IMPORTANT !!!!!!!!!!!!!!!!!!!!!
                # DEBUG ONLY
                # group_ranks = dist.get_process_group_ranks(group)    
                group_ranks = copy.deepcopy(group)            
                group = dist.new_group(group, timeout=datetime.timedelta(seconds=30))
                
                # correct version - do nothing, use the already created group
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                if dist.get_world_size(group) > 1:
                    dist.all_reduce(y_local, op=dist.ReduceOp.SUM, group=group)
                    
                    log(f"Rank {self.grid.rank}: all_reduce comm volume:  {self.comm_vol} += {torch.prod(torch.tensor(y_local.shape))} * 2 * ({dist.get_world_size(group)} - 1)/{dist.get_world_size(group)}",
                         log_level=DEBUG, ranks=[0])
                    dist.barrier()
                    log(f"Rank {self.grid.rank}, allreduce within group: {group_ranks}, result:\n{y_local.detach().cpu().numpy()}",
                        log_level=VALUES, ranks=[0])
                    dist.barrier()
            
                    self.comm_vol += torch.prod(torch.tensor(y_local.shape)) * 2 * (dist.get_world_size(group) - 1)/dist.get_world_size(group)
                                
        
        # Add bias if present
        if self.bias is not None:
            # Get local slice of bias
            bias_local = self.bias[self.grid.pn * self.grid.N_local:(self.grid.pn + 1) * self.grid.N_local]
            y_local = y_local + bias_local
          
        # check if y_local has any nans
        if torch.isnan(y_local).any() or torch.isinf(y_local).any():
            log(f"Rank {self.grid.rank}, distributed linear layer ending with nans", log_level=CRITICAL)
            exit()
        return y_local
        
        
    def reshuffle(self, local_tensor_pre: torch.Tensor) -> torch.Tensor:
        """
        Reshuffle a distributed tensor according to specified pre and post distributions.
        
        Careful! We are matching required input with the received input.
        The received input is the output of the previous layer.
        That means that the required input is:
        - column-distributed over P_m
        - row-distributed over P_k
        - replicated over P_n
        BUT the output of the previous layer is:
        - row-distributed over P_m
        - column-distributed over P_n
        - replicated over P_k
        So we need to switch between P_k in the input and P_n in the output.

        Args:
            local_tensor_pre: Local tensor to reshuffle
            input_p_grid: PGrid object specifying the pre distribution
            
        Returns:
            local_tensor_post: Reshuffled local tensor
        """  
        # If the input tensor is on meta device, return a meta tensor of the correct shape
        if local_tensor_pre.device.type == "meta":
            print("\nEntering reshuffle in the meta-analysis mode. Returning a meta tensor of the correct shape.")
            return torch.empty((self.grid.M_local, self.grid.K_local), device="meta", dtype=local_tensor_pre.dtype)

        # Get local rank for device synchronization
        local_rank = int(os.environ.get("LOCAL_RANK", self.grid.rank))
        
        if self.grid.matches(self.input_p_grid):
            log(f"reshuffle grid matches input grid! Nice........\n")
            return local_tensor_pre

        log(f"reshuffle grid does not match input grid. current grid: {self.grid}, input grid: {self.input_p_grid}\n")
        assert max(self.grid.M_local, self.input_p_grid.M_local) % min(self.grid.M_local, self.input_p_grid.M_local) == 0, f"M_local must be divisible by input_p_grid.M_local, M_local: {self.grid.M_local}, input_p_grid.M_local: {self.input_p_grid.M_local}"
        assert max(self.grid.K_local, self.input_p_grid.N_local) % min(self.grid.K_local, self.input_p_grid.N_local) == 0, f"N_local must be divisible by input_p_grid.N_local, N_local: {self.grid.N_local}, input_p_grid.N_local: {self.input_p_grid.N_local}"

        # new version. Let's start from scratch
        chunk_size_m = math.gcd(self.grid.M_local, self.input_p_grid.M_local)
        chunk_size_n = math.gcd(self.grid.K_local, self.input_p_grid.N_local)

        num_chunks_m = self.grid.M_global // chunk_size_m
        num_chunks_n = self.grid.K_global // chunk_size_n

        # print(f"Rank {self.grid.rank}: chunk_size: [{chunk_size_m}, {chunk_size_n}], num_chunks: [{num_chunks_m}, {num_chunks_n}]")

        # since grids are divisible, for each dimension (m,n), either we have one-to-many or many-to-one mapping
        # m_many_to_one = self.grid.M_local > self.input_p_grid.M_local
        # n_many_to_one = self.grid.N_local > self.input_p_grid.K_local

        m_num_chunks_to_receive = cdiv(self.grid.M_local, chunk_size_m)
        n_num_chunks_to_receive = cdiv(self.grid.K_local, chunk_size_n)

        m_num_chunks_to_send = cdiv(self.input_p_grid.M_local, chunk_size_m)
        n_num_chunks_to_send = cdiv(self.input_p_grid.N_local, chunk_size_n)


        local_tensor_send = rearrange(local_tensor_pre, '(m_b b_m) (n_b b_n) -> m_b n_b b_m b_n',
                    m_b = m_num_chunks_to_send,
                    n_b = n_num_chunks_to_send,
                    b_m = chunk_size_m,
                    b_n = chunk_size_n
        )
        
        dist.barrier()
        # make it contiguous
        local_tensor_send = local_tensor_send.contiguous()

        local_tensor_rcv = torch.empty(m_num_chunks_to_receive, 
                                n_num_chunks_to_receive, 
                                chunk_size_m, 
                                chunk_size_n,
                                device=self.device,
                                dtype=local_tensor_pre.dtype)
        
     
        ops = []


        for chunk_m in range(num_chunks_m):
            m_chunk_ind = chunk_m % m_num_chunks_to_receive
            m_global = chunk_m * chunk_size_m
            for chunk_n in range(num_chunks_n):
                n_chunk_ind = chunk_n % n_num_chunks_to_receive
                n_global = chunk_n * chunk_size_n
                # figure out owners (srcs) and recipients (dsts) for this chunk
                # owners are the ranks that have the data for this chunk
                # recipients are the ranks that need the data for this chunk

                pm_src, pn_src = self.input_p_grid.slice_to_coord(m_global=m_global, n_global=n_global, input=True)
                pm_dst, pk_dst = self.grid.slice_to_coord(m_global=m_global, n_global=n_global, input=False)

                # all ranks in range(self.grid.P_n) have to receive the same chunk (input is P_n-replicated)
                for pn_dst in range(self.grid.P_n):
                    src_rank = self.input_p_grid.coords_to_rank(
                        grid_coords=((m_chunk_ind + n_chunk_ind + pn_dst) % self.input_p_grid.P_k, pm_src, pn_src), 
                        input=False)
                    dst_rank = self.grid.coords_to_rank(grid_coords=(pk_dst, pm_dst, pn_dst))

                    if src_rank == self.grid.rank:
                        # log(f"\n\nRank {self.grid.rank}: sending to {dst_rank}: chunk_ind: [{chunk_m}, {chunk_n}], " +\
                        #         f"src_rank: {src_rank}, dst_rank: {dst_rank}, " +\
                        #         f"src_coords: ({m_chunk_ind + n_chunk_ind}, {pm_src}, {pn_src}), " +\
                        #         f"dst_coords: ({pk_dst}, {pm_dst}, {pn_dst}), " +\
                        #         f"grid: [{self.grid.P_k}, {self.grid.P_m}, {self.grid.P_n}], " +\
                        #         f"input grid: [{self.input_p_grid.P_k}, {self.input_p_grid.P_m}, {self.input_p_grid.P_n}]" +\
                        #         f"sending data:\n{local_tensor_send[m_chunk_ind % m_num_chunks_to_send, n_chunk_ind % n_num_chunks_to_send].detach().cpu().numpy()}\n",
                        #         log_level=VALUES
                        #         )                  
                        send_op = dist.P2POp(dist.isend, 
                            local_tensor_send[chunk_m % m_num_chunks_to_send, 
                                              chunk_n % n_num_chunks_to_send],
                             dst_rank)
                        ops.append(send_op)

                        if dst_rank != self.grid.rank:
                            log(f"Rank {self.grid.rank}: Current vol: {self.comm_vol}, reshuffle send volume = {chunk_size_m} * {chunk_size_n}", log_level=DEBUG)
                            self.comm_vol += chunk_size_m * chunk_size_n
                    if dst_rank == self.grid.rank:
                        tmp = local_tensor_rcv[m_chunk_ind, n_chunk_ind]
                        if not tmp.is_contiguous():
                            print(f"Rank {self.grid.rank}: local_tensor_rcv[m_chunk_ind, n_chunk_ind] is not contiguous! " +\
                                f"shape: {tmp.shape}, dtype: {tmp.dtype}, device: {tmp.device}")
                            exit()
                        recv_op = dist.P2POp(dist.irecv, local_tensor_rcv[m_chunk_ind, n_chunk_ind], src_rank)            
                        ops.append(recv_op)

        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:  
            req.wait()

        # log(f"Rank {self.grid.rank}: received data:\n{local_tensor_rcv.detach().cpu().numpy()}\n",
        #     log_level=VALUES, ranks=[2])

        local_tensor_post = rearrange(local_tensor_rcv, 'm_b n_b b_m b_n -> (m_b b_m) (n_b b_n)',
                    m_b = m_num_chunks_to_receive,
                    n_b = n_num_chunks_to_receive,
                    b_m = chunk_size_m,
                    b_n = chunk_size_n
        )
        # print(f"Rank {self.grid.rank}: local_tensor_post.shape: {local_tensor_post.shape}, DONE!")
        # print(f"local_tensor_post:\n{local_tensor_post.detach().cpu().numpy()}")
        # exit()

        # make sure that local_tensor_post is contiguous
        local_tensor_post = local_tensor_post.contiguous()

        return local_tensor_post


           
    def output_reshuffle(self, local_tensor_pre: torch.Tensor, output_p_grid: PGrid) -> torch.Tensor:
        """
        Reshuffle a distributed tensor according to specified post distribution.
        
        Args:
            local_tensor_pre: Local tensor to reshuffle
            output_p_grid: PGrid object specifying the post distribution
            
        Returns:
            local_tensor_post: Reshuffled local tensor
        """  
        # If the input tensor is on meta device, return a meta tensor of the correct shape
        if local_tensor_pre.device.type == "meta":
            print("\n\n\nEntering reshuffle in the meta-analysis mode. Returning a meta tensor of the correct shape.\n\n\n")
            return torch.empty((self.grid.M_local, self.grid.K_local), device="meta", dtype=local_tensor_pre.dtype)

        # Get local rank for device synchronization
        local_rank = int(os.environ.get("LOCAL_RANK", self.grid.rank))
        
        if self.grid == output_p_grid:
            log(f"reshuffle grid matches output grid! Nice........\n")
            return local_tensor_pre

        log(f"Output reshuffler. grid does not match output grid. current grid: {self.grid}, output grid: {output_p_grid}\n")
        assert max(self.grid.M_local, output_p_grid.M_local) % min(self.grid.M_local, output_p_grid.M_local) == 0, f"M_local must be divisible by output_p_grid.M_local, M_local: {self.grid.M_local}, output_p_grid.M_local: {output_p_grid.M_local}"
        assert max(self.grid.N_local, self.input_p_grid.N_local) % min(self.grid.N_local, self.input_p_grid.N_local) == 0, f"N_local must be divisible by input_p_grid.N_local, N_local: {self.grid.N_local}, input_p_grid.N_local: {self.input_p_grid.N_local}"

        # new version. Let's start from scratch
        chunk_size_m = math.gcd(self.grid.M_local, output_p_grid.M_local)
        chunk_size_n = math.gcd(self.grid.N_local, output_p_grid.N_local)

        num_chunks_m = self.grid.M_global // chunk_size_m
        num_chunks_n = self.grid.N_global // chunk_size_n

        # print(f"Rank {self.grid.rank}: chunk_size: [{chunk_size_m}, {chunk_size_n}], num_chunks: [{num_chunks_m}, {num_chunks_n}]")

        # since grids are divisible, for each dimension (m,n), either we have one-to-many or many-to-one mapping
        # m_many_to_one = self.grid.M_local > self.input_p_grid.M_local
        # n_many_to_one = self.grid.N_local > self.input_p_grid.K_local

        m_num_chunks_to_send = cdiv(self.grid.M_local, chunk_size_m)
        n_num_chunks_to_send = cdiv(self.grid.N_local, chunk_size_n)

        m_num_chunks_to_receive = cdiv(output_p_grid.M_local, chunk_size_m)
        n_num_chunks_to_receive = cdiv(output_p_grid.N_local, chunk_size_n)


        local_tensor_send = rearrange(local_tensor_pre, '(m_b b_m) (n_b b_n) -> m_b n_b b_m b_n',
                    m_b = m_num_chunks_to_send,
                    n_b = n_num_chunks_to_send,
                    b_m = chunk_size_m,
                    b_n = chunk_size_n
        )
        
        dist.barrier()
        # make it contiguous
        local_tensor_send = local_tensor_send.contiguous()

        local_tensor_rcv = torch.empty(m_num_chunks_to_receive, 
                                n_num_chunks_to_receive, 
                                chunk_size_m, 
                                chunk_size_n,
                                device=self.device,
                                dtype=local_tensor_pre.dtype)
        
     
        ops = []


        for chunk_m in range(num_chunks_m):
            m_chunk_ind = chunk_m % m_num_chunks_to_receive
            m_global = chunk_m * chunk_size_m
            for chunk_n in range(num_chunks_n):
                n_chunk_ind = chunk_n % n_num_chunks_to_receive
                n_global = chunk_n * chunk_size_n
                # figure out owners (srcs) and recipients (dsts) for this chunk
                # owners are the ranks that have the data for this chunk
                # recipients are the ranks that need the data for this chunk

                pm_src, pn_src = self.grid.slice_to_coord(m_global=m_global, n_global=n_global, input=True)
                pm_dst, pn_dst = output_p_grid.slice_to_coord(m_global=m_global, n_global=n_global, input=True)

                # all ranks in range(self.grid.P_n) have to receive the same chunk (input is P_n-replicated)
                for pk_dst in range(output_p_grid.P_k):
                    src_rank = self.grid.coords_to_rank(
                        grid_coords=((m_chunk_ind + n_chunk_ind + pn_dst) % self.grid.P_k, pm_src, pn_src), 
                        input=False)
                    dst_rank = output_p_grid.coords_to_rank(grid_coords=(pk_dst, pm_dst, pn_dst))

                    if src_rank == self.grid.rank:
                        log(f"Output reshuffle: Rank {self.grid.rank}: sending to {dst_rank}: chunk_ind: [{chunk_m}, {chunk_n}], " +\
                                f"src_rank: {src_rank}, dst_rank: {dst_rank}, " +\
                                f"src_coords: ({m_chunk_ind + n_chunk_ind}, {pm_src}, {pn_src}), " +\
                                f"dst_coords: ({pk_dst}, {pm_dst}, {pn_dst}), " +\
                                f"grid: [{self.grid.P_k}, {self.grid.P_m}, {self.grid.P_n}], " +\
                                f"input grid: [{self.input_p_grid.P_k}, {self.input_p_grid.P_m}, {self.input_p_grid.P_n}]" +\
                                f"sending data:\n{local_tensor_send[m_chunk_ind % m_num_chunks_to_send, n_chunk_ind % n_num_chunks_to_send].detach().cpu().numpy()}\n",
                                log_level=VALUES,
                                ranks= []
                                )#list(range(8))
                        log(f"Output reshuffle SEND. Rank {self.grid.rank}: sending to {dst_rank}: chunk_ind: [{chunk_m}, {chunk_n}], " +\
                                f"src_rank: {src_rank}, dst_rank: {dst_rank}, " +\
                                f"src_coords: ({m_chunk_ind + n_chunk_ind}, {pm_src}, {pn_src}), " +\
                                f"dst_coords: ({pk_dst}, {pm_dst}, {pn_dst}), " +\
                                f"grid: [{self.grid.P_k}, {self.grid.P_m}, {self.grid.P_n}], " +\
                                f"input grid: [{self.input_p_grid.P_k}, {self.input_p_grid.P_m}, {self.input_p_grid.P_n}]",
                                log_level=DEBUG,
                                ranks= []# list(range(8))
                                )             
                        send_op = dist.P2POp(dist.isend, 
                            local_tensor_send[chunk_m % m_num_chunks_to_send, 
                                              chunk_n % n_num_chunks_to_send],
                             dst_rank)
                        ops.append(send_op)

                        if dst_rank != self.grid.rank:
                            if self.grid.rank == 0:
                                print(f"Rank {self.grid.rank}: Current vol: {self.comm_vol}, reshuffle send volume = {chunk_size_m} * {chunk_size_n}")
                            self.comm_vol += chunk_size_m * chunk_size_n
                    if dst_rank == self.grid.rank:
                        tmp = local_tensor_rcv[m_chunk_ind, n_chunk_ind]
                        if not tmp.is_contiguous():
                            print(f"Rank {self.grid.rank}: local_tensor_rcv[m_chunk_ind, n_chunk_ind] is not contiguous! " +\
                                f"shape: {tmp.shape}, dtype: {tmp.dtype}, device: {tmp.device}")
                            exit()

                        log(f"Output reshuffle RECEIVE. Rank {self.grid.rank}: receiving from {src_rank}: chunk_ind: [{chunk_m}, {chunk_n}], " +\
                                f"src_rank: {src_rank}, dst_rank: {dst_rank}, " +\
                                f"src_coords: ({m_chunk_ind + n_chunk_ind}, {pm_src}, {pn_src}), " +\
                                f"dst_coords: ({pk_dst}, {pm_dst}, {pn_dst}), " +\
                                f"grid: [{self.grid.P_k}, {self.grid.P_m}, {self.grid.P_n}], " +\
                                f"input grid: [{self.input_p_grid.P_k}, {self.input_p_grid.P_m}, {self.input_p_grid.P_n}]",
                                log_level=DEBUG,
                                ranks= []#list(range(8))
                                )         
                        recv_op = dist.P2POp(dist.irecv, local_tensor_rcv[m_chunk_ind, n_chunk_ind], src_rank)            
                        ops.append(recv_op)

        dist.barrier()
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:  
            req.wait()

        # log(f"Rank {self.grid.rank}: received data:\n{local_tensor_rcv.detach().cpu().numpy()}\n",
        #     log_level=VALUES, ranks=[2])
    
        log(f"\n\nSuccessfully finished output reshuffle\n")
        dist.barrier()

        local_tensor_post = rearrange(local_tensor_rcv, 'm_b n_b b_m b_n -> (m_b b_m) (n_b b_n)',
                    m_b = m_num_chunks_to_receive,
                    n_b = n_num_chunks_to_receive,
                    b_m = chunk_size_m,
                    b_n = chunk_size_n
        )
        # print(f"Rank {self.grid.rank}: local_tensor_post.shape: {local_tensor_post.shape}, DONE!")
        # print(f"local_tensor_post:\n{local_tensor_post.detach().cpu().numpy()}")
        # exit()

        # make sure that local_tensor_post is contiguous
        local_tensor_post = local_tensor_post.contiguous()

        return local_tensor_post




    def distribute_input_tensor(self, X_global: torch.Tensor = None, 
                                fixed_init: bool = True, 
                                batch: int = 1, 
                                seq: int = 1, 
                                embd: int = 1, 
                                dtype: torch.dtype = torch.float16) -> torch.Tensor:
        """
        Distribute input tensor across a 3D process grid.
        
        Args:
            X_global: Input tensor of shape [batch, seq, embd] on rank 0, None on other ranks
            grid: Tuple of (P_k, P_m, P_n) process grid dimensions
            
        Returns:
            X_local: Local tensor of shape [M//P_m, K//P_k] for each rank
        """
        P_m, P_k, P_n = self.grid.P_m, self.grid.P_k, self.grid.P_n
        pm, pn, pk = self.grid.pm, self.grid.pn, self.grid.pk
        if X_global is None:
            if fixed_init:
                X_global = torch.zeros(batch, seq, embd, device=self.device, dtype=dtype)
                # X_global = X_global.view(batch, P_m, seq//P_m, P_n, embd//P_n).detach()
                # # initialize each tile with its rank number
                for b in range(batch):
                    for i in range(seq):
                        for j in range(embd):
                            X_global[b, i, j] = ((i % 3) + (j % 5) + b) / 8
                        if i % 7 == 0:
                            X_global[b, i, :] = 0
                # # reshape back to [batch, seq, embd]
                # X_global = X_global.view(batch, seq, embd)
            else:
                X_global = torch.randn(batch, seq, embd, device=self.device)
        else:
            batch, seq, embd = X_global.shape

        # reshape X_global to [batch*seq, embd]
        X_global = X_global.reshape(batch*seq, embd)
        M = batch * seq
        N = embd
        # carve out local tensor from global tensor
            # slice rows across P_m columns across P_k. Replicate the slices across P_n
        log(f"Rank {self.grid.rank}, about to reshape X_global from [M, K] = [{M}, {N}] to [P_m, M//P_m, P_k, K//P_k] = [{P_m}, {M//P_m}, {P_k}, {N//P_k}]")
        X_global = rearrange(X_global,
                             '(b_m m_b) (b_n n_b) -> b_m b_n m_b n_b',
                             b_m = P_m,
                             m_b = M//P_m,
                             b_n = P_n,
                             n_b = N // P_n)
        # take pm, pk patch from X_global
        X_local = X_global[pm, pn]
        log(f"Rank {self.grid.rank}, distribute input tensor. Shapes: X_global: {X_global.shape}, " +\
            f"X_local: {X_local.shape}, [M, K] = [{M}, {N}], [P_m, P_k, P_n] = [{P_m}, {P_k}, {P_n}], "+\
            f"[pm, pk, pn] = [{pm}, {pk}, {pn}]. [batch, seq, N] = [{batch}, {seq}, {N}], [({batch}, {seq//P_m}, {N//P_n})]" +\
            f"\nX_global:\n{X_global.detach().cpu().numpy()}\n" +\
            f"X_local:\n{X_local.detach().cpu().numpy()}", ranks = [0,2], log_level=VALUES)
        # exit()
        
        return X_local.reshape(batch, seq//P_m, N//P_n), X_global





# GQA With Cache
class DistributedAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(self.config.dim,
                            self.config.n_heads * self.config.head_dim,
                            bias=False, device=self.config.device, dtype=self.config.in_dtype)
        self.k_proj = nn.Linear(self.config.dim,
                            self.config.n_kv_heads * self.config.head_dim,
                            bias=False, device=self.config.device, dtype=self.config.in_dtype)
        self.v_proj = nn.Linear(self.config.dim,
                            self.config.n_kv_heads * self.config.head_dim,
                            bias=False, device=self.config.device, dtype=self.config.in_dtype)
        self.o_proj = nn.Linear(self.config.n_heads * self.config.head_dim,
                            self.config.dim,
                            bias=False, device=self.config.device, dtype=self.config.in_dtype)
        
        self.total_params = self.q_proj.weight.numel() + self.k_proj.weight.numel() + self.v_proj.weight.numel() + self.o_proj.weight.numel()

        # IMPORTANT ANALYSIS OF DISTRIBUTED ATTENTION
        # THE FOLLOWING COMMENTS DETERMINE THE WHOLE PARALLEL GRID FOR ATTENTION
        # AND THE INDUCED REDISTRIBUTION.
        #
        # OBSERVATION:
        # For linear layers (matmuls), batch dimension is stacked with sequence length.
        # Therfore, matmul M x K x N is (seq_len * bsz) x dim x hidden_dim.
        # We make no distinction in parallelizing batch and sequence.
        # That means that the activation tensor X is viewed as
        # X.size() = (seq_len * bsz, dim)
        #
        # HOWEVER:
        # For attention, we do make a distinction. We parallelize across batch and heads.
        # Therefore, we can view the activation tensor X as
        # X.size() = (seq_len, bsz * n_heads * head_dim)
        #
        # If we want to avoid RingAttention (parallelizing across sequence length),
        # we can "easily" parallelize across batch and heads (bsz * n_heads).
        #
        #
        # Now, onto creating the parallel grid to reshuffle the activation tensor X.
        # from the output of the linear layers (q_proj, k_proj, v_proj), to prepare it
        # for the attention operation.
        #
        # how to calculate parallel grid for attentiion:
        # grid = [P_k, P_m, P_n]
        # where:
        # P_k: replication (which ranks own the same slice of X)
        # P_m: sequence parallelization.
        # P_n: batch and head parallelization.
        # since we are doing batch and head parallelization, we need to have P_n <= n_heads.
        # for sequence  parallelization (ringattention), we would have P_m > 1.

        # This means that if we have more heads and batches than GPUs, we are good to go, just a 1D grid.
        # and simple head parallelism.
        # If we have more GPUs than heads, we need to parallelize also across sequence length,
        # effectively requiring RingAttention, which is NOT IMPLEMENTED YET.
        P = torch.distributed.get_world_size()
        p = dist.get_rank()
        assert(self.config.n_heads * self.config.cur_bsz >= P)
        # require that P divides n_heads * cur_bsz
        assert(self.config.n_heads * self.config.cur_bsz % P == 0)

        self.q_p_grid = PGrid(
          rank=p,
          P=P,
          M_global= self.config.cur_seqlen, 
          K_global=self.config.dim,
          N_global= self.config.cur_bsz * self.config.n_heads * self.config.head_dim,
          grid=(1, 1, P),
          rank_order=(0, 1, 2),
        )
        
        self.kv_p_grid = PGrid(
          rank=p,
          P=P,
          M_global= self.config.cur_seqlen, 
          K_global=self.config.dim,
          N_global= self.config.cur_bsz * self.config.n_kv_heads * self.config.head_dim,
          # P_k = 1: we don't replicate Q, K, Vs for attention
          # P_m = 1: no RingAttention
          # P_n: throw all GPUs at it.
          grid=(1, 1, P),
          rank_order=(0, 1, 2),
        )

        self.o_p_grid = PGrid(
          rank=p,
          P=P,
          M_global= self.config.cur_seqlen, 
          K_global=self.config.dim,
          N_global= self.config.cur_bsz * self.config.n_kv_heads * self.config.head_dim,
          # P_k = 1: we don't replicate Q, K, Vs for attention
          # P_m = 1: no RingAttention
          # P_n: throw all GPUs at it.
          grid=(1, 1, P),
          rank_order=(0, 1, 2),
        )


    def forward(self, x, start_pos, mask, freqs_cis):
        """
        Non-caching version: ignore 'start_pos' and compute standard MHA over 'x'.
        """        
        bsz, seqlen = self.config.cur_bsz, self.config.cur_seqlen

        # check if x contains nans
        if not isinstance(x, torch.fx.proxy.Proxy):
          assert not torch.isnan(x).any(), "x contains nans"
        # Project to query, key, value     
        #
        queries = self.q_proj(x)  # (bsz, seqlen, n_heads * head_dim)
        keys    = self.k_proj(x)  # (bsz, seqlen, n_kv_heads * head_dim)
        values  = self.v_proj(x)  # (bsz, seqlen, n_kv_heads * head_dim)

        if isinstance(self.q_proj, Distributed3DLinear):  
          queries = self.q_proj.output_reshuffle(queries, self.q_p_grid)
          keys = self.k_proj.output_reshuffle(keys, self.kv_p_grid)
          values = self.v_proj.output_reshuffle(values, self.kv_p_grid)
                # # Reshape
          queries = queries.view(bsz, seqlen, self.config.n_heads // self.q_p_grid.P_n, self.config.head_dim)
          keys    = keys.view(bsz, seqlen, self.config.n_kv_heads // self.kv_p_grid.P_n, self.config.head_dim)
          values  = values.view(bsz, seqlen, self.config.n_kv_heads // self.kv_p_grid.P_n, self.config.head_dim)
        else:
          # # Reshape
          queries = queries.view(bsz, seqlen, self.config.n_heads, self.config.head_dim)
          keys    = keys.view(bsz, seqlen, self.config.n_kv_heads, self.config.head_dim)
          values  = values.view(bsz, seqlen, self.config.n_kv_heads, self.config.head_dim)

        # # Rotary embeddings
        if not isinstance(x, torch.fx.proxy.Proxy) and not isinstance(freqs_cis, torch.fx.proxy.Proxy):
          # try:
          queries, keys = apply_rotary_emb(queries, keys, freqs_cis=freqs_cis)
          assert not torch.isnan(queries).any(), "queries contains nans"
          assert not torch.isnan(keys).any(), "keys contains nans"
          # except Exception as e:
          #   print(f"\n\nError applying rotary embeddings: {e}\n\n")            

        # # GQA "repeat" if needed, to match n_heads
        keys = torch.repeat_interleave(keys, dim=2, repeats=self.config.n_kv_head_rep)
        values = torch.repeat_interleave(values, dim=2, repeats=self.config.n_kv_head_rep)
        # # Now keys, values each have shape (bsz, seqlen, n_heads, head_dim)

        # # Prepare for scaled_dot_product_attention
        queries = queries.transpose(1, 2)  # (bsz, n_heads, seqlen, head_dim)
        keys    = keys.transpose(1, 2)     # (bsz, n_heads, seqlen, head_dim)
        values  = values.transpose(1, 2)   # (bsz, n_heads, seqlen, head_dim)


        if not isinstance(queries, torch.fx.proxy.Proxy):
          assert not torch.isnan(queries).any(), "queries contains nans"
          assert not torch.isnan(keys).any(), "keys contains nans"
          assert not torch.isnan(values).any(), "values contains nans"
          if mask is not None:
            assert not torch.isnan(mask).any(), "mask contains nans"

        out = F.scaled_dot_product_attention(
            queries,  # (bsz, n_heads, L, head_dim)
            keys,     # (bsz, n_heads, L, head_dim)
            values,   # (bsz, n_heads, L, head_dim)
            attn_mask=mask,  # shape must broadcast to (bsz, n_heads, L, L)
        )  # => (bsz, n_heads, seqlen, head_dim)

        if not isinstance(out, torch.fx.proxy.Proxy):
          assert not torch.isnan(queries).any(), "queries contains nans"
          assert not torch.isnan(keys).any(), "keys contains nans"
          assert not torch.isnan(values).any(), "values contains nans"
          if mask is not None:
            assert not torch.isnan(mask).any(), "mask contains nans"
          if torch.isnan(out).any():
            # print datatypes
            log(f"\n\n\n!!!queries.dtype: {queries.dtype}, keys.dtype: {keys.dtype}, values.dtype: {values.dtype}, mask.dtype: {mask.dtype}, out.dtype: {out.dtype}", log_level=DEBUG)
          assert not torch.isnan(out).any(), "out contains nans"
        # Merge heads back
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)  # (bsz, seqlen, dim)

        # print datatypes
        # print(f"x dtype: {x.dtype}, out dtype: {out.dtype}, mask dtype: {mask.dtype}, queries dtype: {queries.dtype}, keys dtype: {keys.dtype}, values dtype: {values.dtype}")
        # exit()
        if isinstance(self.o_proj, Distributed3DLinear):
          # the input_p_grid for o_proj comes from the attention module, which is self.q_p_grid
          self.o_proj.input_p_grid = self.q_p_grid

        return self.o_proj(out)  # final linear projection