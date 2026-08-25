# confounder_vae_xdy.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

TensorLike = Union[np.ndarray, torch.Tensor]


# -------------------------
# Building blocks (CEVAE-ish MLPs)
# -------------------------
class FullyConnected(nn.Sequential):
    """Fully connected multi-layer network with ELU activations."""
    def __init__(self, sizes, final_activation: Optional[nn.Module] = None):
        layers = []
        for in_size, out_size in zip(sizes, sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ELU())
        if layers:
            layers.pop(-1)  # drop last activation
        if final_activation is not None:
            layers.append(final_activation)
        super().__init__(*layers)


class DiagNormalHead(nn.Module):
    """Outputs (loc, scale) for diagonal Normal."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.out_dim = out_dim
        self.fc = nn.Linear(in_dim, 2 * out_dim)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        loc_scale = self.fc(h)
        loc = loc_scale[..., : self.out_dim].clamp(min=-1e2, max=1e2)
        raw = loc_scale[..., self.out_dim :]
        scale = torch.nn.functional.softplus(raw).add(1e-3).clamp(max=1e2)
        return loc, scale


class BernoulliHead(nn.Module):
    """Outputs logits for Bernoulli."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h).clamp(min=-10, max=10)


# -------------------------
# Standardize continuous columns only (leave binary alone)
# -------------------------
class MixedStandardizer(nn.Module):
    """
    Standardizes selected continuous columns, leaving others (binary) unchanged.
    Stores training mean/std for continuous columns.
    """
    def __init__(self, dim: int, continuous_idx: List[int], data: torch.Tensor):
        super().__init__()
        self.dim = int(dim)
        self.continuous_idx = sorted(list(set(int(i) for i in continuous_idx)))

        mask = torch.zeros(self.dim, dtype=torch.bool)
        if self.continuous_idx:
            mask[self.continuous_idx] = True
        self.register_buffer("cont_mask", mask)

        with torch.no_grad():
            if self.continuous_idx:
                cont = data[:, self.continuous_idx]
                loc = cont.mean(0)
                scale = cont.std(0, unbiased=False)
                scale[~(scale > 0)] = 1.0
                self.register_buffer("loc", loc)
                self.register_buffer("inv_scale", scale.reciprocal())
            else:
                self.register_buffer("loc", torch.zeros(0))
                self.register_buffer("inv_scale", torch.ones(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.continuous_idx:
            return x
        out = x.clone()
        out[:, self.continuous_idx] = (out[:, self.continuous_idx] - self.loc) * self.inv_scale
        return out


# -------------------------
# Core VAE-like model
# -------------------------
class ConfounderVAE_XDY(nn.Module):
    """
    Encoder: q(z | x,d,y) = Normal(mu(x,d,y), diag(sigma^2(x,d,y)))
    Decoder: p(x | z)      (mixed: Normal for continuous dims + Bernoulli for binary dims)
             p(d | z)      (binary or continuous)
             p(y | d,z)    (binary or continuous)
    Prior:   p(z) = N(0,I)
    """

    def __init__(
        self,
        x_dim: int,
        latent_dim: int,
        x_binary_idx: Optional[List[int]] = None,
        d_type: str = "binary",      # "binary" or "continuous"
        y_type: str = "binary",      # "binary" or "continuous"
        hidden_dim: int = 200,
        num_layers: int = 3,
    ):
        super().__init__()
        self.x_dim = int(x_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)

        if x_binary_idx is None:
            x_binary_idx = []
        self.x_binary_idx = sorted(list(set(int(i) for i in x_binary_idx)))
        for i in self.x_binary_idx:
            if i < 0 or i >= self.x_dim:
                raise ValueError(f"x_binary_idx out of range: {i} for x_dim={self.x_dim}")

        self.x_cont_idx = [i for i in range(self.x_dim) if i not in set(self.x_binary_idx)]
        self.x_bin_dim = len(self.x_binary_idx)
        self.x_cont_dim = len(self.x_cont_idx)

        if d_type not in ("binary", "continuous"):
            raise ValueError("d_type must be 'binary' or 'continuous'")
        if y_type not in ("binary", "continuous"):
            raise ValueError("y_type must be 'binary' or 'continuous'")
        self.d_type = d_type
        self.y_type = y_type

        # ----- Encoder q(z|x,d,y) -----
        enc_in = self.x_dim + 1 + 1
        self.enc_trunk = FullyConnected([enc_in] + [self.hidden_dim] * self.num_layers, final_activation=nn.ELU())
        self.enc_head = DiagNormalHead(self.hidden_dim, self.latent_dim)

        # ----- Decoder p(x|z) -----
        self.decx_trunk = FullyConnected([self.latent_dim] + [self.hidden_dim] * self.num_layers, final_activation=nn.ELU())
        self.decx_cont = DiagNormalHead(self.hidden_dim, self.x_cont_dim) if self.x_cont_dim > 0 else None
        self.decx_bin = BernoulliHead(self.hidden_dim, self.x_bin_dim) if self.x_bin_dim > 0 else None

        # ----- Decoder p(d|z) -----
        self.decd_trunk = FullyConnected([self.latent_dim] + [self.hidden_dim] * self.num_layers, final_activation=nn.ELU())
        self.decd_head = DiagNormalHead(self.hidden_dim, 1) if self.d_type == "continuous" else BernoulliHead(self.hidden_dim, 1)

        # ----- Decoder p(y|d,z) -----
        self.decy_trunk = FullyConnected([self.latent_dim + 1] + [self.hidden_dim] * self.num_layers, final_activation=nn.ELU())
        self.decy_head = DiagNormalHead(self.hidden_dim, 1) if self.y_type == "continuous" else BernoulliHead(self.hidden_dim, 1)

    @staticmethod
    def _reparameterize(mu: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * scale

    @staticmethod
    def _kl_standard_normal(mu: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        var = scale.pow(2)
        return 0.5 * (var + mu.pow(2) - 1.0 - var.log()).sum(dim=-1)

    def encode(self, x: torch.Tensor, d: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.enc_trunk(torch.cat([x, d, y], dim=-1))
        return self.enc_head(h)

    def _log_p_x_given_z(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        h = self.decx_trunk(z)
        logp = torch.zeros(x.size(0), device=x.device)

        if self.x_cont_dim > 0:
            loc, scale = self.decx_cont(h)
            x_cont = x[:, self.x_cont_idx]
            logp = logp + torch.distributions.Normal(loc, scale).log_prob(x_cont).sum(dim=-1)

        if self.x_bin_dim > 0:
            logits = self.decx_bin(h)
            x_bin = x[:, self.x_binary_idx]
            logp = logp + torch.distributions.Bernoulli(logits=logits).log_prob(x_bin).sum(dim=-1)

        return logp

    def _log_p_d_given_z(self, d: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        h = self.decd_trunk(z)
        if self.d_type == "continuous":
            loc, scale = self.decd_head(h)
            return torch.distributions.Normal(loc, scale).log_prob(d).sum(dim=-1)
        else:
            logits = self.decd_head(h)
            return torch.distributions.Bernoulli(logits=logits).log_prob(d).sum(dim=-1)

    def _log_p_y_given_dz(self, y: torch.Tensor, d: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        h = self.decy_trunk(torch.cat([z, d], dim=-1))
        if self.y_type == "continuous":
            loc, scale = self.decy_head(h)
            return torch.distributions.Normal(loc, scale).log_prob(y).sum(dim=-1)
        else:
            logits = self.decy_head(h)
            return torch.distributions.Bernoulli(logits=logits).log_prob(y).sum(dim=-1)

    def loss(self, x: torch.Tensor, d: torch.Tensor, y: torch.Tensor, beta: float = 1.0, mc_samples: int = 1) -> torch.Tensor:
        mc = int(mc_samples)
        if mc < 1:
            raise ValueError("mc_samples must be >= 1")
        mu, scale = self.encode(x, d, y)
        kl = self._kl_standard_normal(mu, scale)  # [N]

        recon = 0.0
        for _ in range(mc):
            z = self._reparameterize(mu, scale)
            recon = recon + (
                self._log_p_x_given_z(x, z)
                + self._log_p_d_given_z(d, z)
                + self._log_p_y_given_dz(y, d, z)
            )
        recon = recon / float(mc)

        elbo = recon - beta * kl
        return (-elbo).mean()


class ConfounderVAE_X(nn.Module):
    """
    Encoder: q(z | x) = Normal(mu(x), diag(sigma^2(x)))
    Decoder: p(x | z) (mixed: Normal for continuous dims + Bernoulli for binary dims)
    Prior:   p(z) = N(0, I)
    """

    def __init__(
        self,
        x_dim: int,
        latent_dim: int,
        x_binary_idx: Optional[List[int]] = None,
        hidden_dim: int = 200,
        num_layers: int = 3,
    ):
        super().__init__()
        self.x_dim = int(x_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)

        if x_binary_idx is None:
            x_binary_idx = []
        self.x_binary_idx = sorted(list(set(int(i) for i in x_binary_idx)))
        for i in self.x_binary_idx:
            if i < 0 or i >= self.x_dim:
                raise ValueError(f"x_binary_idx out of range: {i} for x_dim={self.x_dim}")

        self.x_cont_idx = [i for i in range(self.x_dim) if i not in set(self.x_binary_idx)]
        self.x_bin_dim = len(self.x_binary_idx)
        self.x_cont_dim = len(self.x_cont_idx)

        # ----- Encoder q(z|x) -----
        enc_in = self.x_dim
        self.enc_trunk = FullyConnected([enc_in] + [self.hidden_dim] * self.num_layers, final_activation=nn.ELU())
        self.enc_head = DiagNormalHead(self.hidden_dim, self.latent_dim)

        # ----- Decoder p(x|z) -----
        self.decx_trunk = FullyConnected([self.latent_dim] + [self.hidden_dim] * self.num_layers, final_activation=nn.ELU())
        self.decx_cont = DiagNormalHead(self.hidden_dim, self.x_cont_dim) if self.x_cont_dim > 0 else None
        self.decx_bin = BernoulliHead(self.hidden_dim, self.x_bin_dim) if self.x_bin_dim > 0 else None

    @staticmethod
    def _reparameterize(mu: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * scale

    @staticmethod
    def _kl_standard_normal(mu: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        var = scale.pow(2)
        return 0.5 * (var + mu.pow(2) - 1.0 - var.log()).sum(dim=-1)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.enc_trunk(x)
        return self.enc_head(h)

    def _log_p_x_given_z(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        h = self.decx_trunk(z)
        logp = torch.zeros(x.size(0), device=x.device)

        if self.x_cont_dim > 0:
            loc, scale = self.decx_cont(h)
            x_cont = x[:, self.x_cont_idx]
            logp = logp + torch.distributions.Normal(loc, scale).log_prob(x_cont).sum(dim=-1)

        if self.x_bin_dim > 0:
            logits = self.decx_bin(h)
            x_bin = x[:, self.x_binary_idx]
            logp = logp + torch.distributions.Bernoulli(logits=logits).log_prob(x_bin).sum(dim=-1)

        return logp

    def loss(self, x: torch.Tensor, beta: float = 1.0, mc_samples: int = 1) -> torch.Tensor:
        mc = int(mc_samples)
        if mc < 1:
            raise ValueError("mc_samples must be >= 1")
        mu, scale = self.encode(x)
        kl = self._kl_standard_normal(mu, scale)  # [N]

        recon = 0.0
        for _ in range(mc):
            z = self._reparameterize(mu, scale)
            recon = recon + self._log_p_x_given_z(x, z)
        recon = recon / float(mc)

        elbo = recon - beta * kl
        return (-elbo).mean()


# -------------------------
# User-facing wrapper
# -------------------------
@dataclass
class FitConfig:
    epochs: int = 100
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    beta: float = 1.0
    mc_samples: int = 1
    log_every: int = 50


class ConfounderExtractor:
    """
    Fit and extract z (posterior mean) from (X,D,Y).

    Example:
      ex = ConfounderExtractor(x_dim=p, latent_dim=10, x_binary_idx=[0,3], d_type="binary", y_type="continuous")
      ex.fit(X_train, D_train, Y_train)
      Z = ex.transform(X_train, D_train, Y_train)   # [N,10]
    """

    def __init__(
        self,
        x_dim: int,
        latent_dim: int,
        x_binary_idx: Optional[List[int]] = None,
        d_type: str = "binary",
        y_type: str = "binary",
        hidden_dim: int = 200,
        num_layers: int = 3,
        device: Optional[str] = None,
    ):
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.x_dim = int(x_dim)
        self.latent_dim = int(latent_dim)
        self.x_binary_idx = x_binary_idx or []
        self.d_type = d_type
        self.y_type = y_type

        self.model = ConfounderVAE_XDY(
            x_dim=self.x_dim,
            latent_dim=self.latent_dim,
            x_binary_idx=self.x_binary_idx,
            d_type=self.d_type,
            y_type=self.y_type,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        ).to(self.device)

        self.x_std: Optional[MixedStandardizer] = None
        self.d_std: Optional[MixedStandardizer] = None
        self.y_std: Optional[MixedStandardizer] = None

    @staticmethod
    def _to_tensor(a: TensorLike, device: torch.device) -> torch.Tensor:
        if isinstance(a, np.ndarray):
            a = torch.from_numpy(a)
        if not torch.is_tensor(a):
            raise TypeError("Input must be numpy array or torch tensor")
        return a.to(device=device, dtype=torch.float32)

    @staticmethod
    def _ensure_colvec(v: torch.Tensor) -> torch.Tensor:
        if v.dim() == 1:
            return v.unsqueeze(-1)
        if v.dim() == 2 and v.size(-1) == 1:
            return v
        raise ValueError(f"Expected shape [N] or [N,1], got {tuple(v.shape)}")

    def fit(self, X: TensorLike, D: TensorLike, Y: TensorLike, cfg: FitConfig = FitConfig()) -> List[float]:
        X = self._to_tensor(X, self.device)
        D = self._ensure_colvec(self._to_tensor(D, self.device))
        Y = self._ensure_colvec(self._to_tensor(Y, self.device))

        if X.dim() != 2 or X.size(-1) != self.x_dim:
            raise ValueError(f"Expected X shape [N,{self.x_dim}], got {tuple(X.shape)}")
        if D.size(0) != X.size(0) or Y.size(0) != X.size(0):
            raise ValueError("X, D, Y must have same first dimension N")

        # Standardizers: X continuous cols only; D/Y if continuous
        x_cont_idx = [i for i in range(self.x_dim) if i not in set(self.x_binary_idx)]
        self.x_std = MixedStandardizer(self.x_dim, continuous_idx=x_cont_idx, data=X).to(self.device)

        self.d_std = MixedStandardizer(1, continuous_idx=([0] if self.d_type == "continuous" else []), data=D).to(self.device)
        self.y_std = MixedStandardizer(1, continuous_idx=([0] if self.y_type == "continuous" else []), data=Y).to(self.device)

        loader = DataLoader(TensorDataset(X, D, Y), batch_size=cfg.batch_size, shuffle=True)
        opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        self.model.train(True)
        losses: List[float] = []
        for epoch in range(cfg.epochs):
            for step, (xb, db, yb) in enumerate(loader):
                xb = self.x_std(xb)
                db = self.d_std(db)
                yb = self.y_std(yb)

                loss = self.model.loss(xb, db, yb, beta=cfg.beta, mc_samples=cfg.mc_samples)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                val = float(loss.detach().cpu().item())
                losses.append(val)
                if cfg.log_every and (len(losses) % cfg.log_every == 0):
                    print(f"[epoch {epoch+1:03d}] step {step:04d} loss={val:.6g}")

        return losses

    @torch.no_grad()
    def encode(self, X: TensorLike, D: TensorLike, Y: TensorLike) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.x_std is None or self.d_std is None or self.y_std is None:
            raise RuntimeError("You must call fit() before encode()/transform().")

        X = self._to_tensor(X, self.device)
        D = self._ensure_colvec(self._to_tensor(D, self.device))
        Y = self._ensure_colvec(self._to_tensor(Y, self.device))

        if X.dim() != 2 or X.size(-1) != self.x_dim:
            raise ValueError(f"Expected X shape [N,{self.x_dim}], got {tuple(X.shape)}")
        if D.size(0) != X.size(0) or Y.size(0) != X.size(0):
            raise ValueError("X, D, Y must have same first dimension N")

        self.model.train(False)
        xw = self.x_std(X)
        dw = self.d_std(D)
        yw = self.y_std(Y)
        return self.model.encode(xw, dw, yw)

    @torch.no_grad()
    def transform(self, X: TensorLike, D: TensorLike, Y: TensorLike) -> np.ndarray:
        """Returns posterior mean of z (shape [N, latent_dim])."""
        mu, _ = self.encode(X, D, Y)
        return mu.detach().cpu().numpy()


class ConfounderExtractorX:
    """
    Fit and extract z (posterior mean) from X only.
    """

    def __init__(
        self,
        x_dim: int,
        latent_dim: int,
        x_binary_idx: Optional[List[int]] = None,
        hidden_dim: int = 200,
        num_layers: int = 3,
        device: Optional[str] = None,
    ):
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.x_dim = int(x_dim)
        self.latent_dim = int(latent_dim)
        self.x_binary_idx = x_binary_idx or []

        self.model = ConfounderVAE_X(
            x_dim=self.x_dim,
            latent_dim=self.latent_dim,
            x_binary_idx=self.x_binary_idx,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        ).to(self.device)

        self.x_std: Optional[MixedStandardizer] = None

    @staticmethod
    def _to_tensor(a: TensorLike, device: torch.device) -> torch.Tensor:
        if isinstance(a, np.ndarray):
            a = torch.from_numpy(a)
        if not torch.is_tensor(a):
            raise TypeError("Input must be numpy array or torch tensor")
        return a.to(device=device, dtype=torch.float32)

    def fit(self, X: TensorLike, cfg: FitConfig = FitConfig()) -> List[float]:
        X = self._to_tensor(X, self.device)

        if X.dim() != 2 or X.size(-1) != self.x_dim:
            raise ValueError(f"Expected X shape [N,{self.x_dim}], got {tuple(X.shape)}")

        x_cont_idx = [i for i in range(self.x_dim) if i not in set(self.x_binary_idx)]
        self.x_std = MixedStandardizer(self.x_dim, continuous_idx=x_cont_idx, data=X).to(self.device)

        loader = DataLoader(TensorDataset(X), batch_size=cfg.batch_size, shuffle=True)
        opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        self.model.train(True)
        losses: List[float] = []
        for epoch in range(cfg.epochs):
            for step, (xb,) in enumerate(loader):
                xb = self.x_std(xb)
                loss = self.model.loss(xb, beta=cfg.beta, mc_samples=cfg.mc_samples)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                val = float(loss.detach().cpu().item())
                losses.append(val)
                if cfg.log_every and (len(losses) % cfg.log_every == 0):
                    print(f"[epoch {epoch+1:03d}] step {step:04d} loss={val:.6g}")

        return losses

    @torch.no_grad()
    def encode(self, X: TensorLike) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.x_std is None:
            raise RuntimeError("You must call fit() before encode()/transform().")

        X = self._to_tensor(X, self.device)
        if X.dim() != 2 or X.size(-1) != self.x_dim:
            raise ValueError(f"Expected X shape [N,{self.x_dim}], got {tuple(X.shape)}")

        self.model.train(False)
        xw = self.x_std(X)
        return self.model.encode(xw)

    @torch.no_grad()
    def transform(self, X: TensorLike) -> np.ndarray:
        """Returns posterior mean of z (shape [N, latent_dim])."""
        mu, _ = self.encode(X)
        return mu.detach().cpu().numpy()
