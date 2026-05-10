import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def _mdn_log_likelihood(p, y, min_log_proba=-np.inf):
    """Per-sample log-likelihood under a Gaussian mixture model.

    p : (batch, K, 3) — [mu, sigma, logit] per component
    y : (batch,) — target scalar
    """
    dist = torch.distributions.Normal(p[:, :, 0], p[:, :, 1])
    log_terms = dist.log_prob(y.reshape(-1, 1))
    log_weights = nn.functional.log_softmax(p[:, :, 2], dim=1)
    return torch.clamp(torch.logsumexp(log_terms + log_weights, dim=1), min=min_log_proba)


def _mdn_expected_value(p):
    """Mixture-weighted mean of Gaussian components. p : (batch, K, 3)"""
    weights = nn.functional.softmax(p[:, :, 2], dim=1)
    return (weights * p[:, :, 0]).sum(dim=1)


def _mdn_sample(p, n=100):
    """Draw n samples per batch item from the Gaussian mixture. p : (batch, K, 3)"""
    k = torch.distributions.Categorical(logits=p[:, :, 2]).sample((n,)).T  # (batch, n)
    params = torch.gather(p, 1, k.unsqueeze(-1).expand(-1, -1, 2))         # (batch, n, 2)
    return torch.distributions.Normal(params[:, :, 0], params[:, :, 1]).sample()


def _transform_output(x, min_std=0.0):
    """Apply activations to raw network output slice for one parameter dimension.

    x : (batch, K, 3) raw — channels: [mu, sigma_raw, logit]
    Returns (batch, K, 3) with sigma > 0.
    """
    return torch.cat([
        x[:, :, 0:1],
        nn.functional.softplus(x[:, :, 1:2]) + min_std,
        x[:, :, 2:3],
    ], dim=2)


class MDNNetwork(nn.Module):
    """Mixture Density Network: K Gaussian components per output dimension."""

    def __init__(self, input_dim, output_dim, num_components=5):
        super().__init__()
        self.output_dim = output_dim
        self.num_components = num_components
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, num_components * 3 * output_dim),
        )

    def forward(self, x):
        # Output shape: (batch, num_components, 3 * output_dim)
        # Slice raw[:, :, i*3:(i+1)*3] to get (batch, K, 3) for output dim i.
        return self.network(x).view(-1, self.num_components, 3 * self.output_dim)


class MDNModel:
    """Inverse model based on a Mixture Density Network (Bishop, 1994).

    Learns p(q | y) as a Gaussian mixture via maximum log-likelihood.
    Call sample() for posterior samples; predict() returns the mixture mean.
    """

    def __init__(self, Q, QoI_names, num_components=5):
        self.Q = Q
        self.QoI_names = QoI_names
        self.num_components = num_components
        self.output_dim = Q.num_params()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = MDNNetwork(len(QoI_names), self.output_dim, num_components).double().to(self.device)
        self.y_scaler = MinMaxScaler((0.05, 0.95))
        self.q_scaler = MinMaxScaler((0.05, 0.95))
        self.history = []

    def train(self, y_train, q_train, epochs=100, batch_size=64, lr=1e-3, patience=20):
        y_sc = self.y_scaler.fit_transform(np.array(y_train))
        q_sc = self.q_scaler.fit_transform(np.array(q_train))
        y_tr, y_vl, q_tr, q_vl = train_test_split(y_sc, q_sc, test_size=0.2, random_state=42)

        loader = DataLoader(
            TensorDataset(
                torch.tensor(y_tr, dtype=torch.float64, device=self.device),
                torch.tensor(q_tr, dtype=torch.float64, device=self.device),
            ),
            batch_size=batch_size, shuffle=True,
        )
        y_vl_t = torch.tensor(y_vl, dtype=torch.float64, device=self.device)
        q_vl_t = torch.tensor(q_vl, dtype=torch.float64, device=self.device)

        optimizer = optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-9)
        scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=10)

        best_val, best_state, no_improve = np.inf, None, 0
        self.history = []
        print("----- Training started for 'MDN' inverse model -----")

        for epoch in range(epochs + 1):
            self.net.train()
            total = 0.0
            for y_b, q_b in loader:
                optimizer.zero_grad()
                raw = self.net(y_b)
                loss = sum(
                    -_mdn_log_likelihood(
                        _transform_output(raw[:, :, i * 3:(i + 1) * 3]),
                        q_b[:, i], min_log_proba=-20,
                    ).mean()
                    for i in range(self.output_dim)
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=10)
                optimizer.step()
                total += loss.item()
            scheduler.step()

            self.net.eval()
            with torch.no_grad():
                raw_vl = self.net(y_vl_t)
                val = sum(
                    -_mdn_log_likelihood(
                        _transform_output(raw_vl[:, :, i * 3:(i + 1) * 3]),
                        q_vl_t[:, i], min_log_proba=-20,
                    ).mean()
                    for i in range(self.output_dim)
                ).item()

            avg = total / len(loader)
            self.history.append({"epoch": epoch, "train_loss": avg, "val_loss": val})
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {avg:.6f}, Val Loss = {val:.6f}")

            if val < best_val:
                best_val, best_state, no_improve = val, deepcopy(self.net.state_dict()), 0
            else:
                no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if best_state is not None:
            self.net.load_state_dict(best_state)
        print("----- Training ended for 'MDN' inverse model -----")

    def predict(self, y):
        """Return mixture-mean parameter estimates for one or more measurements."""
        if not isinstance(y, pd.DataFrame):
            y = pd.DataFrame(y, columns=self.QoI_names)
        y_sc = self.y_scaler.transform(y)
        self.net.eval()
        with torch.no_grad():
            raw = self.net(torch.tensor(y_sc, dtype=torch.float64, device=self.device))
            means = [
                _mdn_expected_value(_transform_output(raw[:, :, i * 3:(i + 1) * 3])).cpu().numpy()
                for i in range(self.output_dim)
            ]
        q_sc = np.stack(means, axis=1)
        return pd.DataFrame(self.q_scaler.inverse_transform(q_sc), columns=self.Q.param_names())

    def sample(self, y, n_samples=500):
        """Draw n_samples from the learned mixture p(q | y) for a single measurement."""
        if not isinstance(y, pd.DataFrame):
            y = pd.DataFrame(y, columns=self.QoI_names)
        y_sc = self.y_scaler.transform(y)
        self.net.eval()
        with torch.no_grad():
            raw = self.net(torch.tensor(y_sc, dtype=torch.float64, device=self.device))
        samples = [
            _mdn_sample(_transform_output(raw[:, :, i * 3:(i + 1) * 3]), n=n_samples)[0].cpu().numpy()
            for i in range(self.output_dim)
        ]
        samples_sc = np.stack(samples, axis=1)
        return pd.DataFrame(self.q_scaler.inverse_transform(samples_sc), columns=self.Q.param_names())
