"""
Clean conditional DDPM implementation.

This module provides a single conditional diffusion model that learns p(Y | X)
without any group/DAG logic. It is intended as a lightweight baseline.
"""

from typing import List, Optional, Tuple

import math
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.
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
    MLP-based conditional diffusion model for a single target variable vector.

    Inputs:
    - x_target_t: noisy target at time t
    - x_cond: conditioning covariates (optional)
    - t: timestep
    """

    def __init__(
        self,
        d_target: int,
        d_cond: int,
        hidden_dims: List[int],
        dim_t: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_target = d_target
        self.d_cond = d_cond
        self.dim_t = dim_t

        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.ReLU(),
            nn.Linear(dim_t, dim_t),
        )

        d_in = d_target + d_cond + dim_t
        layers = []
        prev_dim = d_in
        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                ]
            )
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, d_target))
        self.network = nn.Sequential(*layers)

    def forward(self, x_target_t: torch.Tensor, x_cond: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        te = timestep_embedding(timesteps, self.dim_t)
        te = self.time_embed(te)
        if self.d_cond > 0:
            x_input = torch.cat([x_target_t, x_cond, te], dim=-1)
        else:
            x_input = torch.cat([x_target_t, te], dim=-1)
        return self.network(x_input)


class ConditionalDDPM(nn.Module):
    """
    Conditional DDPM for a single target vector given covariates.
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
        self.device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        self.network = network.to(self.device)

        betas = torch.linspace(min_beta, max_beta, n_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.to(self.device)

    def forward(self, x0: torch.Tensor, t: torch.Tensor, eta: Optional[torch.Tensor] = None) -> torch.Tensor:
        a_bar = self.alpha_bars[t]
        n = len(a_bar)
        if eta is None:
            eta = torch.randn_like(x0).to(self.device)
        x_t = a_bar.sqrt().reshape(n, 1) * x0 + (1 - a_bar).sqrt().reshape(n, 1) * eta
        return x_t

    def backward(self, x_t: torch.Tensor, x_cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.network(x_t, x_cond, t)


def _to_tensor(x, device: str) -> torch.Tensor:
    if x is None:
        return torch.zeros(0, device=device)
    if isinstance(x, np.ndarray):
        return torch.tensor(x, dtype=torch.float32, device=device)
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.tensor(x, dtype=torch.float32, device=device)


def train_conditional_ddpm(
    train_target: torch.Tensor,
    train_cond: Optional[torch.Tensor],
    n_epochs: int,
    lr: float = 5e-5,
    hidden_dims: List[int] = [256, 256, 128],
    dim_t: int = 128,
    n_steps: int = 1000,
    device: str = "cuda",
    batch_size: int = 0,
    verbose: bool = True,
    use_best_loss: bool = True,
) -> ConditionalDDPM:
    """
    Train a conditional DDPM model for p(target | cond).
    
    Args:
        train_target: Training target data [n, d_target]
        train_cond: Conditioning data [n, d_cond] (optional)
        n_epochs: Number of training epochs
        lr: Learning rate
        hidden_dims: Hidden layer dimensions
        dim_t: Time embedding dimension
        n_steps: Number of diffusion steps
        device: Device to train on
        batch_size: Mini-batch size (0 or None for full-batch)
        verbose: Print training progress
        use_best_loss: Restore the best training-loss checkpoint before returning
    
    Returns:
        Trained ConditionalDDPM model
    """
    train_target = _to_tensor(train_target, device)
    train_cond = _to_tensor(train_cond, device) if train_cond is not None else torch.zeros(len(train_target), 0, device=device)

    d_target = train_target.shape[1]
    d_cond = train_cond.shape[1]
    n = len(train_target)

    network = ConditionalMLPDiffusion(
        d_target=d_target,
        d_cond=d_cond,
        hidden_dims=hidden_dims,
        dim_t=dim_t,
    )
    ddpm = ConditionalDDPM(network=network, n_steps=n_steps, device=device)

    optimizer = torch.optim.Adam(ddpm.network.parameters(), lr=lr)
    mse = nn.MSELoss()

    # Determine if using mini-batching
    use_minibatch = batch_size is not None and batch_size > 0 and batch_size < n
    
    iterator = tqdm(range(n_epochs), desc="Training conditional DDPM") if verbose else range(n_epochs)
    best_loss = float("inf")
    best_state = None

    for epoch in iterator:
        if use_minibatch:
            # Mini-batch training
            epoch_loss = 0.0
            n_batches = 0
            
            # Create random permutation for shuffling
            indices = torch.randperm(n, device=device)
            
            for i in range(0, n, batch_size):
                batch_indices = indices[i:i + batch_size]
                batch_target = train_target[batch_indices]
                batch_cond = train_cond[batch_indices]
                batch_n = len(batch_indices)
                
                eta = torch.randn_like(batch_target).to(device)
                t = torch.randint(0, n_steps, (batch_n,), device=device)
                noisy_target = ddpm(batch_target, t, eta)
                eta_pred = ddpm.backward(noisy_target, batch_cond, t)

                loss = mse(eta_pred, eta)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * batch_n
                n_batches += 1
            
            epoch_loss = epoch_loss / n
            
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                if use_best_loss:
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in ddpm.network.state_dict().items()
                    }
            if verbose and (epoch % 100 == 0 or epoch == n_epochs - 1):
                print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {epoch_loss:.6f}, Best: {best_loss:.6f}, Batches: {n_batches}")
        else:
            # Full-batch training (original behavior)
            eta = torch.randn_like(train_target).to(device)
            t = torch.randint(0, n_steps, (n,), device=device)
            noisy_target = ddpm(train_target, t, eta)
            eta_pred = ddpm.backward(noisy_target, train_cond, t)

            loss = mse(eta_pred, eta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                if use_best_loss:
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in ddpm.network.state_dict().items()
                    }
            if verbose and (epoch % 100 == 0 or epoch == n_epochs - 1):
                print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item():.6f}, Best: {best_loss:.6f}")

    if use_best_loss and best_state is not None:
        ddpm.network.load_state_dict(best_state)

    return ddpm


def generate_conditional_samples(
    ddpm: ConditionalDDPM,
    cond: Optional[torch.Tensor],
    n_samples: int = 1,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Generate samples from p(target | cond).

    Args:
        ddpm: trained ConditionalDDPM
        cond: [M, d_cond] conditioning covariates (can be None)
        n_samples: number of MC samples per conditioning case

    Returns:
        [n_samples, M, d_target] tensor
    """
    device = ddpm.device
    cond = _to_tensor(cond, device) if cond is not None else torch.zeros(1, 0, device=device)
    if cond.ndim == 1:
        cond = cond.unsqueeze(0)
    m = cond.shape[0]

    cond_rep = cond.unsqueeze(0).repeat(n_samples, 1, 1).reshape(-1, cond.shape[1])
    total_samples = n_samples * m
    d_target = ddpm.network.d_target

    with torch.no_grad():
        x = torch.randn(total_samples, d_target, device=device)
        iterator = reversed(range(ddpm.n_steps))
        if verbose:
            iterator = tqdm(list(iterator), desc="Sampling conditional DDPM")

        for t in iterator:
            time_tensor = torch.ones(total_samples, device=device, dtype=torch.long) * t
            eta_theta = ddpm.backward(x, cond_rep, time_tensor)

            alpha_t = ddpm.alphas[t]
            alpha_t_bar = ddpm.alpha_bars[t]
            x = (1 / alpha_t.sqrt()) * (
                x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta
            )

            if t > 0:
                z = torch.randn_like(x)
                beta_t = ddpm.betas[t]
                alpha_bar_prev = ddpm.alpha_bars[t - 1]
                beta_t_tilde = beta_t * (1 - alpha_bar_prev) / (1 - alpha_t_bar)
                sigma_t = torch.sqrt(beta_t_tilde)
                x = x + sigma_t * z

    return x.view(n_samples, m, d_target)


def prepare_conditional_data(
    X: np.ndarray,
    D: np.ndarray,
    Y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Utility to build conditional training arrays for Y | (X, D).

    Returns:
        target: Y with shape [n, d_y]
        cond: concatenated [X, D] with shape [n, d_x + d_d]
    """
    if D.ndim == 1:
        D = D.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    cond = np.concatenate([X, D], axis=1)
    return Y, cond
