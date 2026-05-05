import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy
import numpy as np
import pandas as pd
from .base import InverseModel


class EpsilonSampler(nn.Module):
    """Concatenates input with i.i.d. Gaussian noise draws along a new sample axis."""

    def __init__(self, n_dim):
        super().__init__()
        self.n_dim = n_dim

    def forward(self, x, n_samples=10):
        # x: (..., input_dim)  ->  output: (..., n_samples, input_dim + n_dim)
        eps = torch.randn(*x.shape[:-1], n_samples, self.n_dim, device=x.device)
        return torch.cat(
            [x.unsqueeze(-2).expand(*[-1] * (x.ndim - 1), n_samples, -1), eps],
            dim=-1,
        )


# ---------------------------------------------------------------------------
# Scoring-rule loss functions (utility library — not all are used in training
# but are provided for experimentation with alternative objectives).
# ---------------------------------------------------------------------------

def crps_loss(yps, y):
    """Univariate CRPS via rank-based formula.

    Parameters
    ----------
    yps : Tensor, shape (batch, n_samples)
    y   : Tensor, shape (batch,)

    Returns
    -------
    Tensor, shape (batch,) — per-sample CRPS (lower is better).
    """
    # y must be broadcast-compatible with yps along the sample dim.
    y = y.unsqueeze(-1)  # (batch, 1)
    ml = yps.shape[-1]
    mrank = torch.argsort(torch.argsort(yps, dim=-1), dim=-1)
    return (
        (2 / (ml * (ml - 1)))
        * (yps - y)
        * ((ml - 1) * (y < yps).float() - mrank)
    ).sum(dim=-1)


def crps_loss_weighted(yps, w, y):
    """Weighted univariate CRPS.

    Parameters
    ----------
    yps : Tensor, shape (batch, n_samples)
    w   : Tensor, shape (batch, n_samples)  — non-negative sample weights
    y   : Tensor, shape (batch,)

    Returns
    -------
    Tensor, shape (batch,).
    """
    # y must be broadcast-compatible with yps along the sample dim.
    y = y.unsqueeze(-1)  # (batch, 1)
    ml = yps.shape[-1]
    sort_ix = torch.argsort(yps, dim=-1)
    sort_ix_reverse = torch.argsort(sort_ix, dim=-1)
    s = torch.take_along_dim(
        torch.cumsum(torch.take_along_dim(w, sort_ix, dim=-1), dim=-1),
        sort_ix_reverse,
        dim=-1,
    )
    W = w.sum(dim=-1, keepdim=True)
    return (
        (2 / (ml * (ml - 1)))
        * (w * (yps - y) * ((ml - 1) * (y < yps).float() - s + (W - ml + w + 1) / 2))
    ).sum(dim=-1)


def crps_loss_mv(yps, y):
    """Multivariate energy score (unbiased estimator of the proper scoring rule).

    The energy score is defined as
        ES(F, y) = E_F[||X - y||] - (1/2) E_F[||X - X'||]
    where X, X' are independent draws from F.

    Implementation note: the second expectation uses the unbiased estimator
        (1 / (m*(m-1))) * sum_{i != j} ||s_i - s_j||
    which avoids the zero-diagonal self-pairs. This is computed as
    sum_i(mean_j ||s_i - s_j||) / (m-1), algebraically equivalent to
    1/(m*(m-1)) * sum_{i,j} ||s_i - s_j||.

    Parameters
    ----------
    yps : Tensor, shape (batch, n_samples, output_dim)
    y   : Tensor, shape (batch, output_dim)

    Returns
    -------
    Tensor, shape (batch,).
    """
    # Term 1: E[||X - y||]  — mean over samples
    term1 = (yps - y.unsqueeze(-2)).norm(dim=-1).mean(dim=-1)  # (batch,)

    # Term 2: (1/2) * E[||X - X'||]  — unbiased estimator excluding self-pairs.
    # pairwise norms: (batch, n_samples, n_samples)
    # mean(dim=-1): (batch, n_samples)  mean over j for each i
    # sum(dim=-1):  (batch,)            sum over i
    # divide by (n-1) -> 1/(n*(n-1)) * sum_{i,j}  (the unbiased inter-sample mean).
    n = yps.shape[-2]
    pairwise = (yps.unsqueeze(-2) - yps.unsqueeze(-3)).norm(dim=-1)  # (batch, n, n)
    term2 = pairwise.mean(dim=-1).sum(dim=-1) / (n - 1)              # (batch,)

    return term1 - 0.5 * term2


def crps_loss_mv_weighted(yps, w, y):
    """Weighted multivariate energy score.

    Parameters
    ----------
    yps : Tensor, shape (batch, n_samples, output_dim)
    w   : Tensor, shape (batch, n_samples)  — non-negative sample weights
    y   : Tensor, shape (batch, output_dim)

    Returns
    -------
    Tensor, shape (batch,).
    """
    t1 = ((yps - y.unsqueeze(-2)).norm(dim=-1) * w).mean(dim=-1)
    n = yps.shape[-2]
    t2 = (
        (yps.unsqueeze(-2) - yps.unsqueeze(-3)).norm(dim=-1)
        * (w.unsqueeze(-1) * w.unsqueeze(-2))
    ).mean(dim=-1).sum(dim=-1) / (n - 1)
    return t1 - 0.5 * t2


def nll_gpu(pred_samples, y):
    """Adaptive-KDE negative log-likelihood from ensemble samples (1-D per call).

    Uses a two-pass adaptive bandwidth KDE (Abramson's method with Silverman's
    global pilot bandwidth).

    Parameters
    ----------
    pred_samples : Tensor or array, shape (batch, n_samples)
    y            : Tensor or array, shape (batch,)

    Returns
    -------
    Tensor, shape (batch,)  — per-sample NLL (lower is better).
    """
    if not torch.is_tensor(pred_samples):
        pred_samples = torch.tensor(pred_samples, dtype=torch.float32)
    if not torch.is_tensor(y):
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    n = pred_samples.shape[1]
    t_sqcov = pred_samples.std(dim=1, keepdim=True).clamp(min=1e-8)  # (batch, 1)
    t_mean  = pred_samples.mean(dim=1, keepdim=True)                  # (batch, 1)
    t_std_X = (pred_samples - t_mean) / t_sqcov                       # (batch, n)

    glob_bw = np.power(n * 0.75, -0.2)  # Silverman's rule pilot bandwidth

    # --- Pass 1: pilot KDE density at each sample point ---
    t_invbw = torch.ones(1, 1, n, device=pred_samples.device) / glob_bw
    t_norm  = t_invbw / (t_sqcov.unsqueeze(2) * np.sqrt(2 * np.pi)) / n
    kde_vals = (
        torch.exp(
            -0.5 * t_invbw ** 2
            * (t_std_X.unsqueeze(1) - t_std_X.unsqueeze(2)) ** 2
        )
        * t_norm
    ).sum(dim=2)  # (batch, n)

    # Geometric mean of pilot densities (used for adaptive scaling).
    t_g = torch.exp(
        torch.log(kde_vals.clamp(min=1e-30)).sum(dim=1) / n
    ).unsqueeze(1)  # (batch, 1)

    t_inv_loc_bw = (kde_vals / t_g.clamp(min=1e-30)).sqrt()  # (batch, n)

    # --- Pass 2: adaptive KDE evaluated at query points y ---
    t_p_ = (y - t_mean) / t_sqcov                         # (batch, 1)
    # t_inv_loc_bw reshaped to (batch, 1, n) so it broadcasts against t_std_X
    # of shape (batch, n) treated as (batch, 1, n) for the kernel evaluation.
    t_invbw = t_inv_loc_bw.unsqueeze(1) / glob_bw         # (batch, 1, n)
    t_norm  = t_invbw / (t_sqcov.unsqueeze(2) * np.sqrt(2 * np.pi)) / n
    likelihoods = (
        torch.exp(
            -0.5 * t_invbw ** 2
            * (t_std_X.unsqueeze(1) - t_p_.unsqueeze(2)) ** 2
        )
        * t_norm
    ).sum(dim=2).squeeze(1)  # (batch,)

    return -torch.log(likelihoods.clamp(min=1e-30))


# ---------------------------------------------------------------------------
# CRPS neural network architecture
# ---------------------------------------------------------------------------

# Default architecture constants — named so changes propagate consistently.
_CRPS_HIDDEN1_DIM = 64   # output dim of the first linear layer
_CRPS_EPS_DIM = 4        # number of noise dimensions injected by EpsilonSampler


class CRPSModule(nn.Module):
    """Stochastic neural network for CRPS-trained probabilistic regression.

    Architecture: Linear -> EpsilonSampler (noise injection) -> MLP -> LeakyReLU.
    The network maps a single measurement y to an ensemble of parameter samples,
    enabling training via the multivariate energy score.

    Parameters
    ----------
    input_dim  : int  — dimension of the measurement vector y
    output_dim : int  — number of physical parameters to recover
    eps_dim    : int  — number of Gaussian noise channels injected (default: 4)
    hidden1    : int  — output size of the first Linear before noise injection (default: 64)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        eps_dim: int = _CRPS_EPS_DIM,
        hidden1: int = _CRPS_HIDDEN1_DIM,
    ):
        super().__init__()
        self.eps_dim = eps_dim
        self.hidden1 = hidden1
        # After EpsilonSampler the feature size is hidden1 + eps_dim.
        post_eps_dim = hidden1 + eps_dim
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, hidden1),
            EpsilonSampler(eps_dim),
            nn.Linear(post_eps_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        ])

    def forward(self, x, n_samples=10):
        for layer in self.layers:
            if isinstance(layer, EpsilonSampler):
                x = layer(x, n_samples=n_samples)
            else:
                x = layer(x)
        return nn.functional.leaky_relu(x)


# ---------------------------------------------------------------------------
# CRPSModel — InverseModel subclass
# ---------------------------------------------------------------------------

class CRPSModel(InverseModel):
    """Inverse model trained with the multivariate energy score (CRPS generalisation).

    Maps measurements y to a posterior ensemble over physical parameters q.
    Call ``sample()`` to obtain full posterior samples; ``predict()`` returns the
    ensemble mean as a point estimate.
    """

    def __init__(self, Q, QoI_names, eps_dim: int = _CRPS_EPS_DIM, **kwargs):
        super().__init__(Q, QoI_names, "CRPS", **kwargs)
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_dim      = len(QoI_names)
        output_dim     = Q.num_params()
        self.model     = CRPSModule(input_dim, output_dim, eps_dim=eps_dim).double().to(self.device)
        self.history   = []

    def train_and_validate(
        self,
        y_tr, q_tr,
        y_vl, q_vl,
        epochs=50,
        batch_size=64,
        n_samples=10,
        lr=0.001,
        patience=20,
    ):
        y_tr_t = torch.tensor(y_tr, dtype=torch.float64, device=self.device)
        q_tr_t = torch.tensor(q_tr, dtype=torch.float64, device=self.device)
        y_vl_t = torch.tensor(y_vl, dtype=torch.float64, device=self.device)
        q_vl_t = torch.tensor(q_vl, dtype=torch.float64, device=self.device)

        train_loader = DataLoader(
            TensorDataset(y_tr_t, q_tr_t), batch_size=batch_size, shuffle=True
        )
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-9)
        scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=10)

        best_val_loss = np.inf
        best_state    = None
        no_improve    = 0

        for epoch in range(epochs + 1):
            self.model.train()
            total_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self.model(inputs, n_samples=n_samples)
                loss = crps_loss_mv(outputs, labels).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()

            self.model.eval()
            with torch.no_grad():
                val_loss = crps_loss_mv(
                    self.model(y_vl_t, n_samples=n_samples), q_vl_t
                ).mean().item()

            train_loss_avg = total_loss / len(train_loader)
            self.history.append({"epoch": epoch, "train_loss": train_loss_avg, "val_loss": val_loss})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = deepcopy(self.model.state_dict())
                no_improve    = 0
            else:
                no_improve += 1

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch}: "
                    f"Train Loss = {train_loss_avg:.6f}, "
                    f"Val Loss = {val_loss:.6f}"
                )

            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

    def model_predict(self, y_scaled, n_samples=10):
        """Return ensemble-mean parameter predictions for scaled measurements."""
        self.model.eval()
        y_t = torch.tensor(y_scaled, dtype=torch.float64, device=self.device)
        with torch.no_grad():
            outputs = self.model(y_t, n_samples=n_samples)  # (batch, n_samples, output_dim)
        return outputs.mean(dim=1).cpu().numpy()

    def sample(self, y, n_samples=500):
        """Draw n_samples posterior samples from p(q | y) for a single observation.

        Parameters
        ----------
        y : array-like or DataFrame, shape (1, n_qoi)
            A single measurement vector. Only the first row is used.
        n_samples : int
            Number of posterior samples to draw.

        Returns
        -------
        DataFrame of shape (n_samples, n_params).
        """
        if not isinstance(y, pd.DataFrame):
            y = pd.DataFrame(y, columns=self.QoI_names)
        y_scaled = self.y_scaler.transform(y)
        self.model.eval()
        y_t = torch.tensor(y_scaled, dtype=torch.float64, device=self.device)
        with torch.no_grad():
            outputs = self.model(y_t, n_samples=n_samples)  # (batch, n_samples, output_dim)
        # Take the first (and typically only) batch item.
        samples_scaled = outputs[0].cpu().numpy()            # (n_samples, output_dim)
        samples_orig   = self.q_scaler.inverse_transform(samples_scaled)
        return pd.DataFrame(samples_orig, columns=self.Q.param_names())
