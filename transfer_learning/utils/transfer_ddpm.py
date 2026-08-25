"""
Transfer learning conditional DDPM.

Modes:
- Shared encoder + domain-specific heads (orig/aux)
- Flat conditional MLP on [x_t, x_cond, t_embed] (no domain heads)
"""

from typing import List, Optional, Tuple, Union

import math
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


DomainType = Union[str, int]


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


def _normalize_domain(domain: DomainType) -> str:
    if isinstance(domain, str):
        key = domain.strip().lower()
        if key in ("orig", "original"):
            return "orig"
        if key in ("aux", "auxiliary"):
            return "aux"
    if isinstance(domain, (int, np.integer, bool)):
        return "orig" if int(domain) == 0 else "aux"
    raise ValueError(f"Unknown domain: {domain}. Use 'orig' or 'aux'.")


class SharedMLPEncoder(nn.Module):
    """
    Shared encoder for covariates x_cond (only X, not D) -> phi.
    When hidden_dims is empty, applies: X -> Linear -> ReLU -> phi
    """

    def __init__(
        self,
        d_cond: int,
        hidden_dims: List[int],
        dim_phi: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_cond = d_cond
        self.dim_phi = dim_phi
        if d_cond <= 0 or dim_phi <= 0:
            self.network = None
        else:
            layers = []
            prev_dim = d_cond
            for h_dim in hidden_dims:
                layers.extend(
                    [
                        nn.Linear(prev_dim, h_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                    ]
                )
                prev_dim = h_dim
            # Final layer to phi with ReLU activation
            layers.append(nn.Linear(prev_dim, dim_phi))
            layers.append(nn.ReLU())
            self.network = nn.Sequential(*layers)

    def forward(self, x_cond: torch.Tensor) -> torch.Tensor:
        if self.network is None:
            return torch.zeros((x_cond.shape[0], self.dim_phi), device=x_cond.device)
        return self.network(x_cond)


class DomainMLPHead(nn.Module):
    """
    Domain-specific head that predicts noise given [x_t, phi, t_embed].
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        hidden_dims: List[int],
        dropout: float = 0.0,
    ):
        super().__init__()
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
        layers.append(nn.Linear(prev_dim, d_out))
        self.network = nn.Sequential(*layers)

    def forward(self, x_input: torch.Tensor) -> torch.Tensor:
        return self.network(x_input)


class ConditionalMLPDiffusion(nn.Module):
    """
    Flat conditional diffusion network that consumes [x_t, x_cond, t_embed].

    x_cond is expected to be [X, D] (all covariates and treatment).
    """

    def __init__(
        self,
        d_target: int,
        d_cond: int,
        hidden_dims: List[int],
        dim_phi: Optional[int] = None,
        dim_t: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_target = d_target
        self.d_cond = d_cond
        self.dim_phi = dim_phi
        self.dim_t = dim_t
        self.uses_domain = False

        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.ReLU(),
            nn.Linear(dim_t, dim_t),
        )

        layers = []
        prev_dim = d_target + d_cond + dim_t
        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                ]
            )
            prev_dim = h_dim

        if dim_phi is not None and dim_phi > 0:
            layers.append(nn.Linear(prev_dim, dim_phi))
            layers.append(nn.ReLU())
            prev_dim = dim_phi

        self.final_linear = nn.Linear(prev_dim, d_target)
        layers.append(self.final_linear)
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        x_target_t: torch.Tensor,
        x_cond: torch.Tensor,
        timesteps: torch.Tensor,
        domain: Optional[DomainType] = None,
    ) -> torch.Tensor:
        te = timestep_embedding(timesteps, self.dim_t)
        te = self.time_embed(te)
        x_input = torch.cat([x_target_t, x_cond, te], dim=-1)
        return self.network(x_input)


class TransferConditionalMLPDiffusion(nn.Module):
    """
    Conditional diffusion network with a shared covariate encoder and
    domain-specific heads.
    
    x_cond is expected to be [X, D] where:
    - X: covariates (all columns except last)
    - D: treatment (last column)
    
    Shared encoder uses only X -> phi
    Domain heads use [x_t, phi, D, t_embed]
    """

    def __init__(
        self,
        d_target: int,
        d_cond: int,
        dim_phi: int,
        shared_hidden_dims: List[int],
        head_hidden_dims_orig: List[int],
        head_hidden_dims_aux: Optional[List[int]] = None,
        dim_t: int = 128,
        dropout: float = 0.0,
        head_orig: Optional[nn.Module] = None,
        head_aux: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.d_target = d_target
        self.d_cond = d_cond
        self.dim_phi = dim_phi
        self.dim_t = dim_t
        self.uses_domain = True

        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.ReLU(),
            nn.Linear(dim_t, dim_t),
        )

        # Shared encoder takes only X (d_cond - 1 dimensions)
        self.shared_encoder = SharedMLPEncoder(
            d_cond=d_cond - 1,
            hidden_dims=shared_hidden_dims,
            dim_phi=dim_phi,
            dropout=dropout,
        )

        # Domain heads take [x_t, phi, D, t_embed]
        head_in_dim = d_target + dim_phi + 1 + dim_t
        if head_hidden_dims_aux is None:
            head_hidden_dims_aux = list(head_hidden_dims_orig)

        self.head_orig = head_orig or DomainMLPHead(
            d_in=head_in_dim,
            d_out=d_target,
            hidden_dims=head_hidden_dims_orig,
            dropout=dropout,
        )
        self.head_aux = head_aux or DomainMLPHead(
            d_in=head_in_dim,
            d_out=d_target,
            hidden_dims=head_hidden_dims_aux,
            dropout=dropout,
        )

    def encode_cond(self, x_cond: torch.Tensor) -> torch.Tensor:
        # Extract only X (all columns except last)
        X = x_cond[:, :-1]
        return self.shared_encoder(X)

    def forward(
        self,
        x_target_t: torch.Tensor,
        x_cond: torch.Tensor,
        timesteps: torch.Tensor,
        domain: DomainType = "orig",
    ) -> torch.Tensor:
        # Split x_cond into X and D
        X = x_cond[:, :-1]  # All columns except last
        D = x_cond[:, -1:]  # Last column (treatment)
        
        te = timestep_embedding(timesteps, self.dim_t)
        te = self.time_embed(te)
        phi = self.shared_encoder(X)
        
        # Concatenate [x_t, phi, D, t_embed] for domain heads
        x_input = torch.cat([x_target_t, phi, D, te], dim=-1)

        domain = _normalize_domain(domain)
        if domain == "orig":
            return self.head_orig(x_input)
        return self.head_aux(x_input)


class TransferConditionalDDPM(nn.Module):
    """
    Conditional DDPM with shared covariate encoder and domain-specific heads.
    """

    def __init__(
        self,
        network: nn.Module,
        n_steps: int = 1000,
        min_beta: float = 1e-4,
        max_beta: float = 0.02,
        device: Optional[str] = None,
        uses_domain: bool = True,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        self.network = network.to(self.device)
        self.uses_domain = uses_domain

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

    def backward(
        self,
        x_t: torch.Tensor,
        x_cond: torch.Tensor,
        t: torch.Tensor,
        domain: DomainType = "orig",
    ) -> torch.Tensor:
        if self.uses_domain:
            return self.network(x_t, x_cond, t, domain=domain)
        return self.network(x_t, x_cond, t)


def _to_tensor(x, device: str) -> torch.Tensor:
    if x is None:
        return torch.zeros(0, device=device)
    if isinstance(x, np.ndarray):
        return torch.tensor(x, dtype=torch.float32, device=device)
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.tensor(x, dtype=torch.float32, device=device)


def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1:
        return x.unsqueeze(1)
    return x


def _train_domain_epoch(
    ddpm: TransferConditionalDDPM,
    optimizer: torch.optim.Optimizer,
    mse: nn.Module,
    target: torch.Tensor,
    cond: torch.Tensor,
    n_steps: int,
    batch_size: int,
    domain: str,
    device: str,
    weight: float,
) -> float:
    n = len(target)
    use_minibatch = batch_size is not None and batch_size > 0 and batch_size < n

    if use_minibatch:
        indices = torch.randperm(n, device=device)
        epoch_loss = 0.0
        total_seen = 0
        for i in range(0, n, batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_target = target[batch_indices]
            batch_cond = cond[batch_indices]
            batch_n = len(batch_indices)

            eta = torch.randn_like(batch_target).to(device)
            t = torch.randint(0, n_steps, (batch_n,), device=device)
            noisy_target = ddpm(batch_target, t, eta)
            eta_pred = ddpm.backward(noisy_target, batch_cond, t, domain=domain)

            raw_loss = mse(eta_pred, eta)
            loss = raw_loss * weight
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += raw_loss.item() * batch_n
            total_seen += batch_n
        return epoch_loss / total_seen

    eta = torch.randn_like(target).to(device)
    t = torch.randint(0, n_steps, (n,), device=device)
    noisy_target = ddpm(target, t, eta)
    eta_pred = ddpm.backward(noisy_target, cond, t, domain=domain)

    raw_loss = mse(eta_pred, eta)
    loss = raw_loss * weight
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return raw_loss.item()


def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad = requires_grad


def train_transfer_ddpm(
    orig_target: torch.Tensor,
    orig_cond: Optional[torch.Tensor],
    aux_target: Optional[torch.Tensor] = None,
    aux_cond: Optional[torch.Tensor] = None,
    n_epochs: int = 1000,
    lr: float = 5e-5,
    shared_hidden_dims: List[int] = [256, 256],
    dim_phi: int = 128,
    head_hidden_dims_orig: List[int] = [256, 256, 128],
    head_hidden_dims_aux: Optional[List[int]] = None,
    dim_t: int = 128,
    n_steps: int = 1000,
    device: str = "cuda",
    batch_size: int = 0,
    aux_batch_size: Optional[int] = None,
    pretrain_aux_epochs: int = 0,
    pretrain_aux_lr: Optional[float] = None,
    pretrain_aux_batch_size: Optional[int] = None,
    finetune_orig_epochs: int = 0,
    finetune_orig_lr: Optional[float] = None,
    finetune_orig_batch_size: Optional[int] = None,
    orig_weight: float = 1.0,
    aux_weight: float = 1.0,
    dropout: float = 0.0,
    verbose: bool = True,
    use_best_loss: bool = True,
    head_orig: Optional[nn.Module] = None,
    head_aux: Optional[nn.Module] = None,
    use_domain_heads: bool = True,
) -> TransferConditionalDDPM:
    """
    Train a transfer-learning conditional DDPM with shared covariate encoder.

    Args:
        orig_target: original target data [n, d_target]
        orig_cond: original conditioning data [n, d_cond] (optional)
        aux_target: auxiliary target data [m, d_target] (optional)
        aux_cond: auxiliary conditioning data [m, d_cond] (optional)
        n_epochs: number of joint-training epochs
        lr: learning rate for joint training
        shared_hidden_dims: shared encoder hidden dimensions
        dim_phi: shared representation size
        head_hidden_dims_orig: original head hidden dimensions
        head_hidden_dims_aux: auxiliary head hidden dimensions (defaults to orig head dims)
        dim_t: time embedding dimension
        n_steps: number of diffusion steps
        device: device to train on
        batch_size: batch size for original data (0 or None for full-batch)
        aux_batch_size: batch size for auxiliary data (defaults to batch_size)
        pretrain_aux_epochs: auxiliary-only pretraining epochs
        pretrain_aux_lr: learning rate for pretraining (defaults to lr)
        pretrain_aux_batch_size: batch size for pretraining (defaults to aux_batch_size)
        finetune_orig_epochs: original-only finetuning epochs (head only)
        finetune_orig_lr: learning rate for finetuning (defaults to lr)
        finetune_orig_batch_size: batch size for finetuning (defaults to batch_size)
        orig_weight: loss weight for original data
        aux_weight: loss weight for auxiliary data
        dropout: dropout probability for encoder and heads
        verbose: print training progress
        use_best_loss: restore best joint-training checkpoint before returning
        head_orig: optional custom head module for original data
        head_aux: optional custom head module for auxiliary data
    """
    if orig_target is None:
        raise ValueError("orig_target is required for training.")

    orig_target = _ensure_2d(_to_tensor(orig_target, device))
    orig_cond = _to_tensor(orig_cond, device) if orig_cond is not None else torch.zeros(len(orig_target), 0, device=device)
    orig_cond = _ensure_2d(orig_cond)

    if len(orig_cond) != len(orig_target):
        raise ValueError("orig_cond and orig_target must have the same number of rows.")

    aux_available = aux_target is not None
    if aux_available:
        aux_target = _ensure_2d(_to_tensor(aux_target, device))
        aux_cond = _to_tensor(aux_cond, device) if aux_cond is not None else torch.zeros(len(aux_target), 0, device=device)
        aux_cond = _ensure_2d(aux_cond)

        if len(aux_cond) != len(aux_target):
            raise ValueError("aux_cond and aux_target must have the same number of rows.")
        if aux_target.shape[1] != orig_target.shape[1]:
            raise ValueError("aux_target must match orig_target feature dimension.")
        if aux_cond.shape[1] != orig_cond.shape[1]:
            raise ValueError("aux_cond must match orig_cond feature dimension.")

    d_target = orig_target.shape[1]
    d_cond = orig_cond.shape[1]

    if use_domain_heads:
        network = TransferConditionalMLPDiffusion(
            d_target=d_target,
            d_cond=d_cond,
            dim_phi=dim_phi,
            shared_hidden_dims=shared_hidden_dims,
            head_hidden_dims_orig=head_hidden_dims_orig,
            head_hidden_dims_aux=head_hidden_dims_aux,
            dim_t=dim_t,
            dropout=dropout,
            head_orig=head_orig,
            head_aux=head_aux,
        )
        ddpm = TransferConditionalDDPM(
            network=network,
            n_steps=n_steps,
            device=device,
            uses_domain=True,
        )
    else:
        flat_hidden_dims = list(shared_hidden_dims) if shared_hidden_dims else list(head_hidden_dims_orig)
        network = ConditionalMLPDiffusion(
            d_target=d_target,
            d_cond=d_cond,
            hidden_dims=flat_hidden_dims,
            dim_phi=dim_phi,
            dim_t=dim_t,
            dropout=dropout,
        )
        ddpm = TransferConditionalDDPM(
            network=network,
            n_steps=n_steps,
            device=device,
            uses_domain=False,
        )

    optimizer = torch.optim.Adam(ddpm.network.parameters(), lr=lr)
    mse = nn.MSELoss()

    aux_batch_size = batch_size if aux_batch_size is None else aux_batch_size
    pretrain_aux_batch_size = aux_batch_size if pretrain_aux_batch_size is None else pretrain_aux_batch_size

    if aux_available and pretrain_aux_epochs > 0:
        pretrain_lr = lr if pretrain_aux_lr is None else pretrain_aux_lr
        pretrain_optimizer = optimizer if pretrain_lr == lr else torch.optim.Adam(ddpm.network.parameters(), lr=pretrain_lr)
        iterator = tqdm(range(pretrain_aux_epochs), desc="Pretrain aux DDPM") if verbose else range(pretrain_aux_epochs)
        for epoch in iterator:
            aux_loss = _train_domain_epoch(
                ddpm=ddpm,
                optimizer=pretrain_optimizer,
                mse=mse,
                target=aux_target,
                cond=aux_cond,
                n_steps=n_steps,
                batch_size=pretrain_aux_batch_size,
                domain="aux",
                device=device,
                weight=1.0,
            )
            if verbose and (epoch % 100 == 0 or epoch == pretrain_aux_epochs - 1):
                print(f"Pretrain Epoch {epoch + 1}/{pretrain_aux_epochs}, Aux Loss: {aux_loss:.6f}")

    best_loss = float("inf")
    best_state = None

    iterator = tqdm(range(n_epochs), desc="Training transfer DDPM") if verbose else range(n_epochs)
    for epoch in iterator:
        orig_loss = _train_domain_epoch(
            ddpm=ddpm,
            optimizer=optimizer,
            mse=mse,
            target=orig_target,
            cond=orig_cond,
            n_steps=n_steps,
            batch_size=batch_size,
            domain="orig",
            device=device,
            weight=orig_weight,
        )

        aux_loss = None
        if aux_available and aux_weight > 0:
            aux_loss = _train_domain_epoch(
                ddpm=ddpm,
                optimizer=optimizer,
                mse=mse,
                target=aux_target,
                cond=aux_cond,
                n_steps=n_steps,
                batch_size=aux_batch_size,
                domain="aux",
                device=device,
                weight=aux_weight,
            )

        if aux_loss is None:
            combined_loss = orig_weight * orig_loss
        else:
            combined_loss = orig_weight * orig_loss + aux_weight * aux_loss

        if use_best_loss and combined_loss < best_loss:
            best_loss = combined_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in ddpm.network.state_dict().items()
            }

        if verbose and (epoch % 100 == 0 or epoch == n_epochs - 1):
            if aux_loss is None:
                print(f"Epoch {epoch + 1}/{n_epochs}, Orig Loss: {orig_loss:.6f}, Best: {best_loss:.6f}")
            else:
                print(
                    f"Epoch {epoch + 1}/{n_epochs}, Orig Loss: {orig_loss:.6f}, "
                    f"Aux Loss: {aux_loss:.6f}, Best: {best_loss:.6f}"
                )

    if use_best_loss and best_state is not None:
        ddpm.network.load_state_dict(best_state)

    if finetune_orig_epochs > 0:
        _set_requires_grad(ddpm.network, False)
        if use_domain_heads:
            _set_requires_grad(ddpm.network.head_orig, True)
            finetune_params = ddpm.network.head_orig.parameters()
        else:
            _set_requires_grad(ddpm.network.final_linear, True)
            finetune_params = ddpm.network.final_linear.parameters()

        finetune_lr = lr if finetune_orig_lr is None else finetune_orig_lr
        finetune_batch_size = batch_size if finetune_orig_batch_size is None else finetune_orig_batch_size
        finetune_optimizer = torch.optim.Adam(finetune_params, lr=finetune_lr)

        best_finetune_loss = float("inf")
        best_finetune_state = None
        finetune_desc = "Finetune orig head" if use_domain_heads else "Finetune final linear"
        iterator = tqdm(range(finetune_orig_epochs), desc=finetune_desc) if verbose else range(finetune_orig_epochs)
        for epoch in iterator:
            orig_loss = _train_domain_epoch(
                ddpm=ddpm,
                optimizer=finetune_optimizer,
                mse=mse,
                target=orig_target,
                cond=orig_cond,
                n_steps=n_steps,
                batch_size=finetune_batch_size,
                domain="orig",
                device=device,
                weight=orig_weight,
            )

            if use_best_loss and orig_loss < best_finetune_loss:
                best_finetune_loss = orig_loss
                best_finetune_state = {
                    k: v.detach().cpu().clone()
                    for k, v in ddpm.network.state_dict().items()
                }
            if verbose and (epoch % 100 == 0 or epoch == finetune_orig_epochs - 1):
                print(
                    f"Finetune Epoch {epoch + 1}/{finetune_orig_epochs}, Orig Loss: {orig_loss:.6f}"
                )

        if use_best_loss and best_finetune_state is not None:
            ddpm.network.load_state_dict(best_finetune_state)

        _set_requires_grad(ddpm.network, True)

    return ddpm


def generate_transfer_samples(
    ddpm: TransferConditionalDDPM,
    cond: Optional[torch.Tensor],
    n_samples: int = 1,
    domain: DomainType = "orig",
    verbose: bool = False,
) -> torch.Tensor:
    """
    Generate samples from p(target | cond) for a given domain.
    """
    device = ddpm.device
    cond = _to_tensor(cond, device) if cond is not None else torch.zeros(1, 0, device=device)
    cond = _ensure_2d(cond)
    m = cond.shape[0]

    cond_rep = cond.unsqueeze(0).repeat(n_samples, 1, 1).reshape(-1, cond.shape[1])
    total_samples = n_samples * m
    d_target = ddpm.network.d_target

    with torch.no_grad():
        x = torch.randn(total_samples, d_target, device=device)
        iterator = reversed(range(ddpm.n_steps))
        if verbose:
            iterator = tqdm(list(iterator), desc="Sampling transfer DDPM")

        for t in iterator:
            time_tensor = torch.ones(total_samples, device=device, dtype=torch.long) * t
            eta_theta = ddpm.backward(x, cond_rep, time_tensor, domain=domain)

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
