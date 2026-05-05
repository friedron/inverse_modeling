import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy
import numpy as np
import pandas as pd
from .base import InverseModel


# ---------------------------------------------------------------------------
# MDN helper functions
# ---------------------------------------------------------------------------

def _mdn_transform_output(x, min_std=0.0):
    """Apply activation functions to raw network outputs for one parameter dimension.

    Parameters
    ----------
    x      : Tensor, shape (batch, K, 3)
             Raw network slice for a single output dimension.
             Channels: [mu (raw), sigma (raw), logit (raw)].
    min_std : float
             Minimum standard deviation added after softplus (default: 0).

    Returns
    -------
    Tensor, shape (batch, K, 3) with channels [mu, sigma > 0, logit].

    Notes
    -----
    The logit channel (x[:, :, 2]) is passed through unchanged. The clamp to
    [-15, 15] that appeared in a previous version was removed: log_softmax in
    _mdn_log_likelihood is numerically stable for any finite input, and the
    clamp needlessly restricts gradient flow during early training when one
    component may legitimately dominate.
    """
    return torch.cat([
        x[:, :, 0:1],                                                        # mu — unconstrained
        nn.functional.softplus(x[:, :, 1:2]) + min_std,                     # sigma > 0
        x[:, :, 2:3],                                                        # logit — unconstrained
    ], dim=2)


def _mdn_log_likelihood(p, y, min_log_proba=-np.inf):
    """Per-sample log-likelihood under a Gaussian mixture model.

    Parameters
    ----------
    p              : Tensor, shape (batch, K, 3) — transformed MDN output [mu, sigma, logit].
    y              : Tensor, shape (batch,) — target scalar values.
    min_log_proba  : float — floor applied to log-likelihoods (useful for gradient stability).

    Returns
    -------
    Tensor, shape (batch,) — log p(y | x) for each sample.
    """
    dist = torch.distributions.Normal(p[:, :, 0], p[:, :, 1])
    log_likelihood_terms = dist.log_prob(y.reshape(-1, 1))         # (batch, K)
    mixture_logcoefs     = nn.functional.log_softmax(p[:, :, 2], dim=1)
    log_likelihoods      = torch.logsumexp(log_likelihood_terms + mixture_logcoefs, dim=1)
    return torch.clamp(log_likelihoods, min=min_log_proba)


def _mdn_expected_value(p):
    """Mixture-weighted mean of the Gaussian components.

    Parameters
    ----------
    p : Tensor, shape (batch, K, 3) — transformed MDN output [mu, sigma, logit].

    Returns
    -------
    Tensor, shape (batch,).
    """
    mixture_coefs = nn.functional.softmax(p[:, :, 2], dim=1)
    return (mixture_coefs * p[:, :, 0]).sum(dim=1)


def _mdn_sample(p, n=100):
    """Draw n samples per batch item from the Gaussian mixture.

    Parameters
    ----------
    p : Tensor, shape (batch, K, 3) — transformed MDN output [mu, sigma, logit].
    n : int — number of samples per batch item.

    Returns
    -------
    Tensor, shape (batch, n).
    """
    k      = torch.distributions.Categorical(logits=p[:, :, 2]).sample((n,)).T  # (batch, n)
    params = torch.gather(p, 1, k.unsqueeze(-1).expand(-1, -1, 2))              # (batch, n, 2)
    return torch.distributions.Normal(params[:, :, 0], params[:, :, 1]).sample()


# ---------------------------------------------------------------------------
# MDN network module
# ---------------------------------------------------------------------------

class MDNModule(nn.Module):
    """Mixture Density Network producing K Gaussian components per output dimension.

    Architecture note — tensor layout
    ----------------------------------
    The final linear layer produces ``num_components * 3 * output_dim`` values.
    These are reshaped to ``(batch, num_components, 3 * output_dim)`` so that
    ``raw[:, :, i*3:(i+1)*3]`` extracts a ``(batch, K, 3)`` slice holding
    [mu_raw, sigma_raw, logit_raw] for output dimension i across all K components.
    This layout (consecutive triplets per output dim within the component axis)
    matches the slicing convention used in MDNModel and the helper functions above.

    Parameters
    ----------
    input_dim      : int
    output_dim     : int
    num_components : int — number of Gaussian mixture components K (default: 5)
    """

    def __init__(self, input_dim: int, output_dim: int, num_components: int = 5):
        super().__init__()
        self.output_dim     = output_dim
        self.num_components = num_components
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_components * 3 * output_dim),
        )

    def forward(self, x):
        # Shape: (batch, num_components, 3 * output_dim)
        # Slicing raw[:, :, i*3:(i+1)*3] gives (batch, K, 3) for output dim i.
        return self.network(x).view(-1, self.num_components, 3 * self.output_dim)


# ---------------------------------------------------------------------------
# MDNModel — InverseModel subclass
# ---------------------------------------------------------------------------

class MDNModel(InverseModel):
    """Inverse model based on a Mixture Density Network (Bishop, 1994).

    Learns the full conditional distribution p(q | y) as a Gaussian mixture,
    trained by maximum log-likelihood. Call ``sample()`` for posterior samples;
    ``predict()`` returns the mixture-weighted mean as a point estimate.
    """

    def __init__(self, Q, QoI_names, num_components: int = 5, **kwargs):
        super().__init__(Q, QoI_names, "MDN", **kwargs)
        self.device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim      = len(QoI_names)
        self.output_dim     = Q.num_params()
        self.num_components = num_components
        self.model   = MDNModule(
            self.input_dim, self.output_dim, num_components
        ).double().to(self.device)
        self.history = []

    def train_and_validate(
        self,
        y_tr, q_tr,
        y_vl, q_vl,
        epochs=100,
        batch_size=64,
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
                raw_params = self.model(inputs)
                loss = sum(
                    -_mdn_log_likelihood(
                        _mdn_transform_output(raw_params[:, :, i * 3:(i + 1) * 3]),
                        labels[:, i],
                        min_log_proba=-20,
                    ).mean()
                    for i in range(self.output_dim)
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()

            self.model.eval()
            with torch.no_grad():
                raw_vl    = self.model(y_vl_t)
                val_loss  = sum(
                    -_mdn_log_likelihood(
                        _mdn_transform_output(raw_vl[:, :, i * 3:(i + 1) * 3]),
                        q_vl_t[:, i],
                        min_log_proba=-20,
                    ).mean()
                    for i in range(self.output_dim)
                ).item()

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

    def model_predict(self, y_scaled):
        """Return mixture-mean parameter predictions for scaled measurements."""
        self.model.eval()
        y_t = torch.tensor(y_scaled, dtype=torch.float64, device=self.device)
        with torch.no_grad():
            raw_params = self.model(y_t)
            means = [
                _mdn_expected_value(
                    _mdn_transform_output(raw_params[:, :, i * 3:(i + 1) * 3])
                ).cpu().numpy()
                for i in range(self.output_dim)
            ]
        return np.stack(means, axis=1)

    def sample(self, y, n_samples=500):
        """Draw n_samples from the learned Gaussian mixture p(q | y).

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
            raw_params = self.model(y_t)   # (1, num_components, 3*output_dim)

        samples = [
            _mdn_sample(
                _mdn_transform_output(raw_params[:, :, i * 3:(i + 1) * 3]),
                n=n_samples,
            )[0].cpu().numpy()
            for i in range(self.output_dim)
        ]
        samples_orig = self.q_scaler.inverse_transform(np.stack(samples, axis=1))
        return pd.DataFrame(samples_orig, columns=self.Q.param_names())
