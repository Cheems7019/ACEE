"""
Conditional Diffusion Models with Causal Structure

This module implements a per-group conditional diffusion approach where:
- Each group has its own conditional diffusion model
- Each group's model conditions on parent group values (not on other groups in the network)
- Sampling follows topological order of the causal DAG
- Supports both unconditional sampling and conditional imputation

Key differences from standard ddpm.py:
- Standard: Single network with masked connections encoding causal structure
- Conditional: Separate network per group, explicitly conditioning on parent data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Set
import math


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class ConditionalMLPDiffusion(nn.Module):
    """
    MLP-based conditional diffusion model for a single group.
    
    Takes as input:
    - Noisy samples from this group (x_group^t)
    - Original data from parent groups (x_parents)
    - Time step (t)
    
    Outputs:
    - Predicted noise for this group
    """
    
    def __init__(
        self,
        d_group: int,          # dimension of this group
        d_parents: int,        # total dimension of all parent groups
        hidden_dims: List[int],  # e.g., [512, 256, 256, 128]
        dim_t: int = 128,      # time embedding dimension
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.d_group = d_group
        self.d_parents = d_parents
        self.dim_t = dim_t
        
        # Time embedding MLP
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t),
        )
        
        # Input: [noisy group data, parent data, time embedding]
        d_in = d_group + d_parents + dim_t
        
        # Main MLP backbone
        layers = []
        prev_dim = d_in
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.SiLU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            ])
            prev_dim = h_dim
        
        # Output layer: predict noise for this group
        layers.append(nn.Linear(prev_dim, d_group))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x_group: torch.Tensor, x_parents: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_group: [batch_size, d_group] - noisy samples from this group
            x_parents: [batch_size, d_parents] - clean values from parent groups
            timesteps: [batch_size] - time steps
        
        Returns:
            [batch_size, d_group] - predicted noise for this group
        """
        # Compute time embedding
        te = timestep_embedding(timesteps, self.dim_t)  # [B, dim_t]
        te = self.time_embed(te)                        # [B, dim_t]
        
        # Concatenate inputs: [noisy_group, parents, time]
        if self.d_parents > 0:
            x_input = torch.cat([x_group, x_parents, te], dim=-1)
        else:
            # No parents (root group)
            x_input = torch.cat([x_group, te], dim=-1)
        
        # Predict noise
        return self.network(x_input)


class ConditionalDDPM(nn.Module):
    """
    Conditional DDPM for a single group that conditions on parent groups.
    """
    
    def __init__(
        self,
        network: ConditionalMLPDiffusion,
        n_steps: int = 1000,
        min_beta: float = 1e-4,
        max_beta: float = 0.02,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.network = network.to(self.device)
        
        # Compute diffusion schedule, then register as buffers for proper state management
        betas = torch.linspace(min_beta, max_beta, n_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        
        # Move to device
        self.to(self.device)
    
    def forward(self, x0_group: torch.Tensor, t: torch.Tensor, eta: Optional[torch.Tensor] = None):
        """
        Forward diffusion process: add noise to clean group samples.
        
        Args:
            x0_group: [batch_size, d_group] - clean samples from this group
            t: [batch_size] - time steps
            eta: [batch_size, d_group] - noise (optional, will be sampled if None)
        
        Returns:
            [batch_size, d_group] - noisy samples at time t
        """
        a_bar = self.alpha_bars[t]
        n = len(a_bar)
        
        if eta is None:
            eta = torch.randn_like(x0_group).to(self.device)
        
        x0_noisy = (
            a_bar.sqrt().reshape(n, 1) * x0_group + 
            (1 - a_bar).sqrt().reshape(n, 1) * eta
        )
        return x0_noisy
    
    def backward(self, x_group: torch.Tensor, x_parents: torch.Tensor, t: torch.Tensor):
        """
        Backward process: predict noise given noisy samples and parent values.
        
        Args:
            x_group: [batch_size, d_group] - noisy samples from this group
            x_parents: [batch_size, d_parents] - clean parent values
            t: [batch_size] - time steps
        
        Returns:
            [batch_size, d_group] - predicted noise
        """
        return self.network(x_group, x_parents, t)


def topological_sort(groups: torch.Tensor, A: torch.Tensor) -> List[int]:
    """
    Perform topological sort on groups based on adjacency matrix A.
    
    Args:
        groups: [d_in] - group assignment for each variable
        A: [K, K] - adjacency matrix where A[i,j] != 0 means group i influences group j
    
    Returns:
        List of group indices in topological order
    
    Raises:
        ValueError: If the graph contains cycles
    """
    K = A.shape[0]
    
    # Build adjacency list (parent -> children)
    adj_list = {i: [] for i in range(K)}
    in_degree = {i: 0 for i in range(K)}
    
    for i in range(K):
        for j in range(K):
            if i != j and A[i, j].item() != 0:
                adj_list[i].append(j)
                in_degree[j] += 1
    
    # Kahn's algorithm
    queue = [i for i in range(K) if in_degree[i] == 0]
    topo_order = []
    
    while queue:
        # Sort queue for deterministic ordering
        queue.sort()
        node = queue.pop(0)
        topo_order.append(node)
        
        for neighbor in adj_list[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    # Check for cycles
    if len(topo_order) != K:
        raise ValueError(
            f"Graph contains cycles! Only {len(topo_order)} of {K} groups could be sorted. "
            "Conditional diffusion requires a DAG structure."
        )
    
    return topo_order


def get_group_info(groups: torch.Tensor) -> Tuple[int, Dict[int, List[int]], Dict[int, int]]:
    """
    Extract group information from group assignment tensor.
    
    Args:
        groups: [d_in] - group assignment for each variable
    
    Returns:
        - K: number of groups
        - group_to_vars: dict mapping group_id -> list of variable indices
        - var_to_group: dict mapping variable_id -> group_id
    """
    K = int(groups.max().item()) + 1
    
    group_to_vars = {i: [] for i in range(K)}
    var_to_group = {}
    
    for var_idx, group_id in enumerate(groups.tolist()):
        group_to_vars[group_id].append(var_idx)
        var_to_group[var_idx] = group_id
    
    return K, group_to_vars, var_to_group


def get_parent_groups(group_id: int, A: torch.Tensor) -> List[int]:
    """
    Get parent groups for a given group.
    
    Args:
        group_id: target group
        A: [K, K] - adjacency matrix where A[i,j] != 0 means group i influences group j
    
    Returns:
        List of parent group indices
    """
    parents = []
    for i in range(A.shape[0]):
        if i != group_id and A[i, group_id].item() != 0:
            parents.append(i)
    return parents


def find_ancestor_groups(
    target_group: int,
    A: torch.Tensor,
    observed_vars: set,
    group_to_vars: Dict[int, List[int]]
) -> set:
    """
    Find all ancestor groups needed to sample target_group.
    
    Uses BFS to traverse backwards through the DAG, stopping at:
    - Fully observed groups (ALL variables in the group are observed)
    - Root nodes (no parents)
    
    Args:
        target_group: group to sample
        A: [K, K] - adjacency matrix
        observed_vars: set of variable IDs that are observed
        group_to_vars: dict mapping group_id -> list of variable indices
    
    Returns:
        Set of ancestor group IDs that need to be sampled
    
    Example:
        Target: G2 (Jnk)
        Parents: G0 (PKC), G5 (PKA)
        - G0: ALL vars observed → stop
        - G5: NOT all vars observed → add G5 to ancestors
          → Recursively check G5's parents
            → G5's parent: G0 (all vars observed) → stop
        Result: ancestors = {G5}
    """
    ancestors = set()
    to_visit = [target_group]
    visited = set()
    
    while to_visit:
        current = to_visit.pop(0)
        
        # Skip if already visited
        if current in visited:
            continue
        visited.add(current)
        
        # Find parent groups (A[i, current] != 0 means i is parent of current)
        parent_groups = [
            i
            for i in range(A.shape[0])
            if A[i, current].item() != 0 and i != current
        ]
        
        for parent in parent_groups:
            # Check if ALL variables in this parent group are observed
            parent_vars = group_to_vars[parent]
            all_observed = all(v in observed_vars for v in parent_vars)
            
            if not all_observed:
                # This parent group has unobserved variables, needs to be sampled
                ancestors.add(parent)
                # Recursively check this parent's parents
                to_visit.append(parent)
            # If all parent vars are observed, we stop (no need to go further)
    
    return ancestors


def compute_required_ancestors(
    sample_vars: List[int],
    observed_vars: set,
    groups: torch.Tensor,
    A: torch.Tensor,
) -> List[int]:
    """
    Compute all variables that need to be sampled to generate sample_vars.
    
    Algorithm: For each variable in sample_vars, recursively find all ancestors
    (parents, grandparents, etc.) that are not in observed_vars.
    
    This implements the recursive parent-checking rule:
    - To sample X, need all parents of X
    - To sample parent P, need all parents of P
    - Continue until reaching observed variables or root nodes
    
    Args:
        sample_vars: list of variable indices user wants to sample
        observed_vars: set of variable indices that are observed
        groups: [d_in] - group assignment
        A: [K, K] - adjacency matrix
    
    Returns:
        List of variable indices in topological order (ready to sample in sequence)
    
    Example (Sachs):
        sample_vars = [5, 6, 7]  # Jnk, P38, PIP2
        observed_vars = {0, 1}   # PKC, Plcg
        
        Check X5 (Jnk):
          - Parents: X0 (observed), X2 (PKA, not observed)
          - Need to sample X2
          - X2's parents: X0 (observed)
          - Add X2 to required
        
        Check X6 (P38):
          - Parents: X0 (observed), X2 (already added)
          - No new variables needed
        
        Check X7 (PIP2):
          - Parents: X1 (observed), X3 (PIP3, not observed)
          - Need to sample X3
          - X3's parents: X1 (observed)
          - Add X3 to required
        
        Result: [2, 3, 5, 6, 7] (in topological order)
    """
    K, group_to_vars, var_to_group = get_group_info(groups)
    
    # Map to groups
    sample_groups = {var_to_group[v] for v in sample_vars}
    
    # Find all groups needed (requested + their ancestors)
    required_groups = set()
    
    for target_group in sample_groups:
        # Recursively find all ancestors until we hit fully observed groups
        ancestors = find_ancestor_groups(target_group, A, observed_vars, group_to_vars)
        required_groups.update(ancestors)
        required_groups.add(target_group)
    
    # Convert back to variables in topological order
    required_vars = []
    topo_order = topological_sort(groups, A)
    
    for group_id in topo_order:
        if group_id in required_groups:
            # Add all variables in this group that aren't observed
            for var_idx in group_to_vars[group_id]:
                if var_idx not in observed_vars:
                    required_vars.append(var_idx)
    
    return required_vars


def train_conditional_ddpm_group(
    group_id: int,
    train_data: torch.Tensor,
    groups: torch.Tensor,
    A: torch.Tensor,
    n_epochs: int,
    lr: float = 5e-5,
    hidden_dims: List[int] = [512, 256, 256, 128],
    dim_t: int = 128,
    n_steps: int = 1000,
    device: str = 'cuda',
    verbose: bool = True,
) -> ConditionalDDPM:
    """
    Train conditional diffusion model for a single group.
    
    Args:
        group_id: which group to train
        train_data: [n_samples, d_in] - training data for all variables
        groups: [d_in] - group assignment
        A: [K, K] - adjacency matrix
        n_epochs: number of training epochs
        lr: learning rate
        hidden_dims: hidden layer dimensions
        dim_t: time embedding dimension
        n_steps: number of diffusion steps
        device: device to use
        verbose: whether to print progress
    
    Returns:
        Trained ConditionalDDPM model
    """
    K, group_to_vars, _ = get_group_info(groups)
    parent_groups = get_parent_groups(group_id, A)
    
    # Get variable indices for this group and parents
    group_vars = group_to_vars[group_id]
    parent_vars = []
    for p in parent_groups:
        parent_vars.extend(group_to_vars[p])
    
    d_group = len(group_vars)
    d_parents = len(parent_vars)
    
    if verbose:
        print(f"\nTraining group {group_id}:")
        print(f"  Variables: {group_vars} (dim={d_group})")
        print(f"  Parent groups: {parent_groups}")
        print(f"  Parent variables: {parent_vars} (dim={d_parents})")
    
    # Create network and DDPM
    network = ConditionalMLPDiffusion(
        d_group=d_group,
        d_parents=d_parents,
        hidden_dims=hidden_dims,
        dim_t=dim_t,
    )
    
    ddpm = ConditionalDDPM(
        network=network,
        n_steps=n_steps,
        device=device,
    )
    
    # Optimize only the network parameters (buffers like betas/alphas are not trainable)
    optimizer = torch.optim.Adam(ddpm.network.parameters(), lr=lr)
    mse = nn.MSELoss()
    
    # Extract data
    x0_full = train_data.to(device)
    x0_group = x0_full[:, group_vars]  # [n_samples, d_group]
    if d_parents > 0:
        x0_parents = x0_full[:, parent_vars]  # [n_samples, d_parents]
    else:
        x0_parents = torch.zeros(len(x0_full), 0).to(device)
    
    best_loss = float("inf")
    best_state = None
    
    # Training loop
    iterator = tqdm(range(n_epochs), desc=f"Training group {group_id}") if verbose else range(n_epochs)
    
    for epoch in iterator:
        n = len(x0_group)
        
        # Sample noise and timesteps
        eta = torch.randn_like(x0_group).to(device)
        t = torch.randint(0, n_steps, (n,)).to(device)
        
        # Forward process: add noise to group
        noisy_group = ddpm(x0_group, t, eta)
        
        # Backward: predict noise given noisy group and parent values
        eta_pred = ddpm.backward(noisy_group, x0_parents, t)
        
        # Compute loss
        loss = mse(eta_pred, eta)
        
        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track best loss (training loss)
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.detach().cpu().clone() for k, v in ddpm.state_dict().items()}
        
        if verbose and (epoch % 100 == 0 or epoch == n_epochs - 1):
            print(f"  Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.6f}, Best: {best_loss:.6f}")
    
    if best_state is not None:
        ddpm.load_state_dict(best_state)
    return ddpm


def train_all_conditional_ddpms(
    train_data: torch.Tensor,
    groups: torch.Tensor,
    A: torch.Tensor,
    n_epochs: int = 3000,
    lr: float = 5e-5,
    hidden_dims: List[int] = [512, 256, 256, 128],
    dim_t: int = 128,
    n_steps: int = 1000,
    device: str = 'cuda',
    verbose: bool = True,
) -> Dict[int, ConditionalDDPM]:
    """
    Train conditional diffusion models for all groups.
    
    Args:
        train_data: [n_samples, d_in] - training data
        groups: [d_in] - group assignment
        A: [K, K] - adjacency matrix
        n_epochs: number of training epochs per group
        lr: learning rate
        hidden_dims: hidden layer dimensions
        dim_t: time embedding dimension
        n_steps: number of diffusion steps
        device: device to use
        verbose: whether to print progress
    
    Returns:
        Dictionary mapping group_id -> trained ConditionalDDPM
    """
    # Check for cycles
    try:
        topo_order = topological_sort(groups, A)
    except ValueError as e:
        raise e
    
    K = A.shape[0]
    models = {}
    
    if verbose:
        print(f"Training {K} conditional diffusion models in topological order: {topo_order}")
    
    # Train in topological order for consistency (though training is independent per group)
    for group_id in topo_order:
        models[group_id] = train_conditional_ddpm_group(
            group_id=group_id,
            train_data=train_data,
            groups=groups,
            A=A,
            n_epochs=n_epochs,
            lr=lr,
            hidden_dims=hidden_dims,
            dim_t=dim_t,
            n_steps=n_steps,
            device=device,
            verbose=verbose,
        )
    
    return models


def generate_conditional_samples(
    models: Dict[int, ConditionalDDPM],
    groups: torch.Tensor,
    A: torch.Tensor,
    do_vars: Optional[Dict[int, torch.Tensor]] = None,
    sample_vars: Optional[List[int]] = None,
    n_samples: int = 1,
    verbose: bool = False,
    auto_intermediate: bool = True,
) -> torch.Tensor:
    """
    Generate samples following causal order with automatic batching.
    
    NEW API: Automatically replicates conditioning values and returns full vectors in normalized space.
    
    Args:
        models: dict mapping group_id -> trained ConditionalDDPM
        groups: [d_in] - group assignment
        A: [K, K] - adjacency matrix
        do_vars: dict mapping var_idx -> value (variables to fix/condition on).
                 Values should be [M] or [M, 1] tensors/arrays (in NORMALIZED space),
                 or scalars (treated as M=1)
                 where M is the number of conditioning cases
        sample_vars: list of variable indices to sample (if None, sample all variables not in do_vars)
        n_samples: number of MC repetitions per conditioning case (default: 1)
        verbose: whether to print progress
        auto_intermediate: if True, automatically detect and sample intermediate variables
                          needed to generate sample_vars (default: True)
    
    Returns:
        If sample_vars is specified:
            [n_samples, M, len(sample_vars)] - ONLY sampled variables in NORMALIZED space
            Variables returned in the order specified by sample_vars
        
        If sample_vars is None:
            [n_samples, M, d_in] - ALL variables in original order (0, 1, ..., d_in-1)
            - Variables in do_vars: filled with specified values (replicated across n_samples)
            - Other variables: sampled values (different across n_samples)
        
        NOTE: User must apply inverse_transform themselves to get back to original space.
    
    Example with auto_intermediate:
        # User sets some variables and requests targets
        do_vars = {0: PKC_norm, 1: Plcg_norm}  # Each has shape [200], NORMALIZED
        sample_vars = [5, 6, 7]  # Jnk, P38, PIP2
        n_samples = 50  # Generate 50 MC samples per conditioning case
        
        # Function automatically detects and adds intermediate nodes [2, 3]
        # Internally samples: [2, 3, 5, 6, 7] (PKA, PIP3, Jnk, P38, PIP2)
        # Returns: [50, 200, 3] - ONLY the 3 requested variables [5, 6, 7] in that order
        # User already has X0, X1 from do_vars if needed for inverse_transform
    """
    K, group_to_vars, var_to_group = get_group_info(groups)
    d_in = len(groups)
    
    # Get topological order
    topo_order = topological_sort(groups, A)
    
    # Keep the user's requested order stable
    requested_vars = None
    if sample_vars is not None:
        requested_vars = list(sample_vars)  # copy
    
    # Determine M (number of conditioning cases) from do_vars
    M = 1
    if do_vars is not None and len(do_vars) > 0:
        first_var_idx = list(do_vars.keys())[0]
        first_value = do_vars[first_var_idx]
        if isinstance(first_value, np.ndarray):
            first_value_arr = np.asarray(first_value)
            M = 1 if first_value_arr.ndim == 0 else len(first_value_arr)
        else:
            first_value_tensor = torch.tensor(first_value) if not isinstance(first_value, torch.Tensor) else first_value
            M = 1 if first_value_tensor.ndim == 0 else first_value_tensor.shape[0]
    
    # Total number of samples to generate internally
    total_samples = n_samples * M
    
    # Use model's device for consistency
    # This ensures output is on the same device as where sampling happens
    model_device = next(iter(models.values())).device
    
    # Initialize output
    output = torch.zeros(total_samples, d_in).to(model_device)
    
    # Track which variables are already filled
    filled_vars = set()
    
    # Fill in do_vars (replicated n_samples times)
    if do_vars is not None:
        for var_idx, value in do_vars.items():
            # Convert to tensor
            if isinstance(value, np.ndarray):
                value = torch.tensor(value, dtype=torch.float32)
            elif not isinstance(value, torch.Tensor):
                value = torch.tensor(value, dtype=torch.float32)
            
            value = value.to(model_device)
            
            # Replicate for n_samples: each of M conditioning cases gets n_samples copies
            # Shape: [M] -> [n_samples, M] -> [n_samples * M]
            # Use view(-1) instead of squeeze() to avoid edge case when M==1
            v = value.view(-1)  # Always [M]
            value_repeated = v.unsqueeze(0).repeat(n_samples, 1).reshape(-1)  # [n_samples * M]
            output[:, var_idx] = value_repeated
            filled_vars.add(var_idx)
    
    # Determine which variables to sample (use separate list for internal logic)
    if sample_vars is None:
        # Sample all variables not in do_vars
        sampling_vars = [i for i in range(d_in) if i not in filled_vars]
    else:
        sampling_vars = list(sample_vars)
        # Auto-detect intermediate variables if requested
        if auto_intermediate and len(sampling_vars) > 0:
            sampling_vars = compute_required_ancestors(
                sample_vars=sampling_vars,
                observed_vars=filled_vars,  # Note: internal name still uses observed_vars
                groups=groups,
                A=A,
            )
            if verbose:
                added_vars = set(sampling_vars) - set(requested_vars)
                if added_vars:
                    print(f"Auto-detected intermediate variables: {sorted(added_vars)}")
                print(f"Requested vars: {requested_vars}")
                print(f"Internal sampling vars (topo): {sampling_vars}")
        else:
            if verbose:
                print(f"Requested vars: {requested_vars}")
                print(f"Internal sampling vars: {sampling_vars}")
    
    # Group sampling_vars by group
    groups_to_sample = set()
    for var_idx in sampling_vars:
        groups_to_sample.add(var_to_group[var_idx])
    
    # Sample groups in topological order
    if verbose:
        print(f"Sampling groups in order: {topo_order}")
        print(f"Groups to sample: {groups_to_sample}")
    
    for group_id in topo_order:
        group_vars = group_to_vars[group_id]
        
        # Check if we need to sample this group
        needs_sampling = any(var in sampling_vars for var in group_vars)
        
        if not needs_sampling:
            # All vars in this group are either observed or not requested
            if verbose:
                print(f"  Group {group_id}: Skipping (all vars observed or not requested)")
            continue
        
        # Check if some vars in this group are observed
        observed_in_group = [var for var in group_vars if var in filled_vars]
        unobserved_in_group = [var for var in group_vars if var not in filled_vars]
        
        # Safety check: ensure all parent group variables are filled before conditioning
        parent_groups = get_parent_groups(group_id, A)
        for parent_group_id in parent_groups:
            parent_group_vars = group_to_vars[parent_group_id]
            unfilled_parent_vars = [v for v in parent_group_vars if v not in filled_vars]
            if unfilled_parent_vars:
                raise RuntimeError(
                    f"Attempting to sample group {group_id} but parent group {parent_group_id} "
                    f"has unfilled variables {unfilled_parent_vars}. This likely indicates "
                    f"auto_intermediate=False with incomplete sample_vars, or a bug in dependency resolution."
                )
        
        if observed_in_group and unobserved_in_group:
            # Need to impute within the group
            if verbose:
                print(f"  Group {group_id}: Imputing {unobserved_in_group} (observed: {observed_in_group})")
            # Use imputation for this group
            output[:, group_vars] = generate_group_imputation(
                models[group_id],
                output[:, group_vars],
                torch.tensor([1 if var in filled_vars else 0 for var in group_vars], dtype=torch.float32).to(model_device),
                get_parent_values(group_id, output, groups, A, group_to_vars),
                device=model_device,
                verbose=verbose,
            )
        else:
            # Sample entire group from scratch
            if verbose:
                print(f"  Group {group_id}: Sampling entire group {group_vars}")
            
            parent_values = get_parent_values(group_id, output, groups, A, group_to_vars)
            sampled_group = generate_group_samples(
                models[group_id],
                total_samples,
                parent_values,
                device=model_device,
                verbose=verbose,
            )
            output[:, group_vars] = sampled_group
        
        # Mark these variables as filled
        filled_vars.update(group_vars)
    
    # Reshape and extract requested variables
    if requested_vars is None:
        # Return all variables in original order
        output_reshaped = output.view(n_samples, M, d_in)
    else:
        # Return ONLY what the user requested, in the user-provided order
        out_vars = requested_vars
        output_selected = output[:, out_vars]  # Shape: [n_samples * M, len(out_vars)]
        output_reshaped = output_selected.view(n_samples, M, len(out_vars))
    
    return output_reshaped


def get_parent_values(
    group_id: int,
    current_data: torch.Tensor,
    groups: torch.Tensor,
    A: torch.Tensor,
    group_to_vars: Dict[int, List[int]],
) -> torch.Tensor:
    """
    Extract parent group values from current data.
    
    Args:
        group_id: target group
        current_data: [n_samples, d_in] - current state of all variables
        groups: [d_in] - group assignment
        A: [K, K] - adjacency matrix
        group_to_vars: dict mapping group_id -> list of variable indices
    
    Returns:
        [n_samples, d_parents] - parent values
    """
    parent_groups = get_parent_groups(group_id, A)
    
    if not parent_groups:
        # No parents, return empty tensor
        return torch.zeros(len(current_data), 0).to(current_data.device)
    
    parent_vars = []
    for p in parent_groups:
        parent_vars.extend(group_to_vars[p])
    
    return current_data[:, parent_vars]


def generate_group_samples(
    ddpm: ConditionalDDPM,
    n_samples: int,
    parent_values: torch.Tensor,
    device: str = 'cuda',
    verbose: bool = False,
) -> torch.Tensor:
    """
    Generate samples for a single group conditioned on parent values.
    
    Args:
        ddpm: trained ConditionalDDPM for this group
        n_samples: number of samples
        parent_values: [n_samples, d_parents] - parent group values
        device: device to use (ignored, uses ddpm.device for consistency)
        verbose: whether to print progress
    
    Returns:
        [n_samples, d_group] - generated samples
    """
    d_group = ddpm.network.d_group
    
    # Use model's device to avoid mismatches
    device = ddpm.device
    
    with torch.no_grad():
        # Start from random noise
        x = torch.randn(n_samples, d_group).to(device)
        parent_values = parent_values.to(device)
        
        # Reverse diffusion
        iterator = reversed(range(ddpm.n_steps))
        if verbose:
            iterator = tqdm(list(iterator), desc="Sampling group")
        
        for t in iterator:
            time_tensor = torch.ones(n_samples).long().to(device) * t
            
            # Predict noise
            eta_theta = ddpm.backward(x, parent_values, time_tensor)
            
            alpha_t = ddpm.alphas[t]
            alpha_t_bar = ddpm.alpha_bars[t]
            
            # Denoise
            x = (1 / alpha_t.sqrt()) * (
                x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta
            )
            
            # Add noise (if not final step)
            if t > 0:
                z = torch.randn(n_samples, d_group).to(device)
                # Use correct DDPM posterior variance: β̃_t = β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
                beta_t = ddpm.betas[t]
                alpha_bar_prev = ddpm.alpha_bars[t - 1]
                beta_t_tilde = beta_t * (1 - alpha_bar_prev) / (1 - alpha_t_bar)
                sigma_t = torch.sqrt(beta_t_tilde)
                x = x + sigma_t * z
    
    return x


def generate_group_imputation(
    ddpm: ConditionalDDPM,
    group_data: torch.Tensor,
    mask: torch.Tensor,
    parent_values: torch.Tensor,
    resampling_steps: int = 10,
    device: str = 'cuda',
    verbose: bool = False,
) -> torch.Tensor:
    """
    Impute missing values within a single group using conditional diffusion.
    
    Args:
        ddpm: trained ConditionalDDPM for this group
        group_data: [n_samples, d_group] - data for this group (observed values filled in)
        mask: [d_group] - binary mask (1 for observed, 0 for missing)
        parent_values: [n_samples, d_parents] - parent group values
        resampling_steps: number of resampling steps
        device: device to use (ignored, uses ddpm.device for consistency)
        verbose: whether to print progress
    
    Returns:
        [n_samples, d_group] - imputed group data
    """
    # Use model's device to avoid mismatches
    device = ddpm.device
    
    n_samples = group_data.shape[0]
    x0 = group_data.clone().to(device)
    m = mask.clone().to(device)
    parent_values = parent_values.to(device)
    
    with torch.no_grad():
        # Start from random noise
        x = torch.randn_like(x0).to(device)
        
        # Reverse diffusion with resampling
        iterator = reversed(range(ddpm.n_steps))
        if verbose:
            iterator = tqdm(list(iterator), desc="Imputing group")
        
        for t in iterator:
            time_tensor = torch.ones(n_samples).long().to(device) * t
            
            for u in range(resampling_steps):
                # Get known part from original data (add appropriate noise)
                x_known = ddpm(x0, time_tensor)
                
                # Denoise unknown part
                alpha_t = ddpm.alphas[t]
                alpha_t_bar = ddpm.alpha_bars[t]
                beta_t = ddpm.betas[t]
                
                # Use correct DDPM posterior variance: β̃_t = β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
                if t > 0:
                    alpha_bar_prev = ddpm.alpha_bars[t - 1]
                    beta_t_tilde = beta_t * (1 - alpha_bar_prev) / (1 - alpha_t_bar)
                    sigma_t = torch.sqrt(beta_t_tilde)
                else:
                    sigma_t = 0
                
                eta_theta = ddpm.backward(x, parent_values, time_tensor)
                z = torch.randn_like(x0).to(device) if t > 0 else 0
                x_unknown = (1 / alpha_t.sqrt()) * (
                    x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta
                ) + sigma_t * z
                
                # Combine known and unknown
                x = m * x_known + (1 - m) * x_unknown
                
                # Resampling step: forward noising uses forward variance beta_t (not posterior)
                if u < resampling_steps - 1 and t > 0:
                    x = (1 - beta_t).sqrt() * x + beta_t.sqrt() * torch.randn_like(x0).to(device)
    
    return x
