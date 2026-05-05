import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer


def identity_func(x):
    if isinstance(x, np.ndarray):
        return x
    return x.to_numpy()


def identity_inverse_func(x):
    return x


class InverseModel:
    """Base class for Inverse Models (mapping measurements y to parameters q)."""

    def __init__(self, Q, QoI_names, method, **kwargs):
        self.Q = Q                   # VariableSet for parameters
        self.QoI_names = QoI_names   # Names for measurements
        self.method = method
        self.init_config = kwargs
        self.model = None            # To be initialized by subclasses
        self.q_scaler = None
        self.y_scaler = None

    @staticmethod
    def create(Q, QoI_names, method, **kwargs):
        """Factory: instantiate the appropriate InverseModel subclass by name."""
        from .crps import CRPSModel
        from .mdn import MDNModel
        from .jax_fem import JaxFemGradientModel

        if method == "CRPS":
            return CRPSModel(Q, QoI_names, **kwargs)
        elif method == "MDN":
            return MDNModel(Q, QoI_names, **kwargs)
        elif method == "JAX-FEM":
            return JaxFemGradientModel(Q, QoI_names, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

    def get_q_scaler(self, q, scale_method='minmax'):
        """Fit and store a scaler for the parameter space q."""
        if scale_method == 'identity':
            self.q_scaler = FunctionTransformer(
                func=identity_func, inverse_func=identity_inverse_func
            )
        elif scale_method == 'minmax':
            self.q_scaler = MinMaxScaler((0.05, 0.95))
            self.q_scaler.fit(q)
        return self.q_scaler

    def get_y_scaler(self, y, scale_method='minmax'):
        """Fit and store a scaler for the measurement space y."""
        if scale_method == 'identity':
            self.y_scaler = FunctionTransformer(
                func=identity_func, inverse_func=identity_inverse_func
            )
        elif scale_method == 'minmax':
            self.y_scaler = MinMaxScaler((0.05, 0.95))
            self.y_scaler.fit(y)
        return self.y_scaler

    def train(self, y_train, q_train, y_scaler=None, q_scaler=None, k_fold=None, **params):
        """
        Fit the inverse model on paired (y, q) data.

        Parameters
        ----------
        y_train : array-like, shape (n, n_qoi)
            Measurement inputs for the inverse model.
        q_train : array-like, shape (n, n_params)
            Parameter targets for the inverse model.
        y_scaler : fitted scaler, optional
            Pre-fitted scaler for y; if None, a MinMax scaler is fitted on y_train.
        q_scaler : fitted scaler, optional
            Pre-fitted scaler for q; if None, a MinMax scaler is fitted on q_train.
        k_fold : int or None
            Number of K-Fold cross-validation splits. If None, an 80/20 hold-out
            split is used instead.
        **params
            Forwarded to train_and_validate (e.g. epochs, batch_size, lr).
        """
        if y_scaler is not None:
            self.y_scaler = y_scaler
        else:
            self.get_y_scaler(y_train)
        if q_scaler is not None:
            self.q_scaler = q_scaler
        else:
            self.get_q_scaler(q_train)

        y_scaled = self.y_scaler.transform(y_train)
        q_scaled = self.q_scaler.transform(q_train)

        # Reset epoch history so a fresh train() call starts with a clean slate.
        # Subclasses that expose self.history (CRPS, MDN) benefit from this;
        # others simply have the attribute set harmlessly.
        self.history = []

        print(f"----- Training started for '{self.method}' inverse model -----")
        if k_fold is not None:
            kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)
            for train_idx, val_idx in kf.split(y_scaled):
                y_tr, y_vl = y_scaled[train_idx], y_scaled[val_idx]
                q_tr, q_vl = q_scaled[train_idx], q_scaled[val_idx]
                self.train_and_validate(y_tr, q_tr, y_vl, q_vl, **params)
        else:
            y_tr, y_vl, q_tr, q_vl = train_test_split(
                y_scaled, q_scaled, test_size=0.2, random_state=42
            )
            self.train_and_validate(y_tr, q_tr, y_vl, q_vl, **params)
        print(f"----- Training ended for '{self.method}' inverse model -----")

    def train_and_validate(self, y_tr, q_tr, y_vl, q_vl, **params):
        """Run one training + validation pass. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement train_and_validate")

    def predict(self, y, **params):
        """
        Map measurements y to parameter estimates q̂.

        Parameters
        ----------
        y : array-like or DataFrame, shape (n, n_qoi)
        **params : forwarded to model_predict.

        Returns
        -------
        DataFrame of shape (n, n_params) with columns from Q.param_names().
        """
        if not isinstance(y, pd.DataFrame):
            y = pd.DataFrame(y, columns=self.QoI_names)

        y_scaled = self.y_scaler.transform(y)
        q_scaled = self.model_predict(y_scaled, **params)
        q_orig = self.q_scaler.inverse_transform(q_scaled)
        return pd.DataFrame(q_orig, columns=self.Q.param_names())

    def model_predict(self, y_scaled, **params):
        """Return scaled parameter predictions for scaled inputs. Implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement model_predict")
