import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class EpsilonSampler(nn.Module):
    """Concatenates input with i.i.d. Gaussian noise along a new sample axis."""

    def __init__(self, n_dim):
        super().__init__()
        self.n_dim = n_dim

    def forward(self, x, n_samples=10):
        # x: (batch, input_dim) -> (batch, n_samples, input_dim + n_dim)
        eps = torch.randn(*x.shape[:-1], n_samples, self.n_dim, device=x.device)
        return torch.cat(
            [x.unsqueeze(-2).expand(*[-1] * (x.ndim - 1), n_samples, -1), eps],
            dim=-1,
        )


def energy_score(yps, y):
    """Multivariate energy score (proper scoring rule for ensemble predictions).

    yps : (batch, n_samples, output_dim)
    y   : (batch, output_dim)
    """
    term1 = (yps - y.unsqueeze(-2)).norm(dim=-1).mean(dim=-1)
    n = yps.shape[-2]
    pairwise = (yps.unsqueeze(-2) - yps.unsqueeze(-3)).norm(dim=-1)
    term2 = pairwise.mean(dim=-1).sum(dim=-1) / (n - 1)
    return term1 - 0.5 * term2


class CRPSNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, eps_dim=4, hidden1=64):
        super().__init__()
        self.pre = nn.Linear(input_dim, hidden1)
        self.eps_sampler = EpsilonSampler(eps_dim)
        self.post = nn.Sequential(
            nn.Linear(hidden1 + eps_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x, n_samples=10):
        x = self.pre(x)
        x = self.eps_sampler(x, n_samples=n_samples)
        return self.post(x)


class CRPSModel:
    """Inverse model trained with the multivariate energy score.

    Maps measurements y to a posterior ensemble over physical parameters q.
    Call sample() for full posterior samples; predict() returns the ensemble mean.
    """

    def __init__(self, Q, QoI_names, eps_dim=4):
        self.Q = Q
        self.QoI_names = QoI_names
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = CRPSNetwork(len(QoI_names), Q.num_params(), eps_dim).double().to(self.device)
        self.y_scaler = MinMaxScaler((0.05, 0.95))
        self.q_scaler = MinMaxScaler((0.05, 0.95))
        self.history = []

    def train(self, y_train, q_train, epochs=50, batch_size=64, n_samples=10, lr=1e-3, patience=20):
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
        print("----- Training started for 'CRPS' inverse model -----")

        for epoch in range(epochs + 1):
            self.net.train()
            total = 0.0
            for y_b, q_b in loader:
                optimizer.zero_grad()
                loss = energy_score(self.net(y_b, n_samples=n_samples), q_b).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=10)
                optimizer.step()
                total += loss.item()
            scheduler.step()

            self.net.eval()
            with torch.no_grad():
                val = energy_score(self.net(y_vl_t, n_samples=n_samples), q_vl_t).mean().item()

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
        print("----- Training ended for 'CRPS' inverse model -----")

    def predict(self, y, n_samples=10):
        """Return ensemble-mean parameter estimates for one or more measurements."""
        if not isinstance(y, pd.DataFrame):
            y = pd.DataFrame(y, columns=self.QoI_names)
        y_sc = self.y_scaler.transform(y)
        self.net.eval()
        with torch.no_grad():
            out = self.net(torch.tensor(y_sc, dtype=torch.float64, device=self.device), n_samples=n_samples)
        q_sc = out.mean(dim=1).cpu().numpy()
        return pd.DataFrame(self.q_scaler.inverse_transform(q_sc), columns=self.Q.param_names())

    def sample(self, y, n_samples=500):
        """Draw n_samples posterior samples for a single measurement vector."""
        if not isinstance(y, pd.DataFrame):
            y = pd.DataFrame(y, columns=self.QoI_names)
        y_sc = self.y_scaler.transform(y)
        self.net.eval()
        with torch.no_grad():
            out = self.net(torch.tensor(y_sc, dtype=torch.float64, device=self.device), n_samples=n_samples)
        samples_sc = out[0].cpu().numpy()
        return pd.DataFrame(self.q_scaler.inverse_transform(samples_sc), columns=self.Q.param_names())
