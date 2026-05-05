import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax_fem.solver import ad_wrapper
from .base import InverseModel


class JaxFemGradientModel(InverseModel):
    """Gradient-based inverse model that differentiates through a JAX-FEM solve.

    Parameters
    ----------
    Q : VariableSet
        Parameter space (bounds, names).
    QoI_names : list[str]
        Names of the measurement quantities (length must match qoi_fn output).
    problem_class : type
        JAX-FEM Problem subclass to instantiate.
    mesh : Mesh
        JAX-FEM mesh object.
    params_fn : callable  (params: array) -> list[jnp.ndarray]
        Maps a 1-D parameter array to the list of arrays expected by fwd_pred.
        Example for uniform E/nu on HEX8:
            lambda p: [jnp.full((n_cells, 8), p[0]), jnp.full((n_cells, 8), p[1])]
    qoi_fn : callable  (u: jnp.ndarray, mesh: Mesh) -> jnp.ndarray
        Extracts the measurement vector from the full displacement field.
        Must be JAX-differentiable.
    problem_args : dict, optional
        Keyword arguments forwarded to problem_class constructor.
    solver_options : dict, optional
        Forwarded to ad_wrapper for the forward linear solve.
    adjoint_solver_options : dict, optional
        Forwarded to ad_wrapper for the adjoint linear solve.  Defaults to
        ``{'umfpack_solver': {}}`` because JAX's BiCGStab can diverge on the
        non-symmetric transposed system produced by row-elimination Dirichlet BCs.
    """

    def __init__(
        self,
        Q,
        QoI_names,
        problem_class,
        mesh,
        params_fn,
        qoi_fn,
        problem_args=None,
        solver_options=None,
        adjoint_solver_options=None,
    ):
        super().__init__(Q, QoI_names, "JAX-FEM")
        self.mesh      = mesh
        self.params_fn = params_fn
        self.qoi_fn    = qoi_fn
        self.problem   = problem_class(mesh, **(problem_args or {}))
        # Default the adjoint solver to umfpack (direct) to avoid BiCGStab
        # divergence on the transposed non-symmetric system that arises from
        # row-elimination Dirichlet BCs.
        adj_opts = adjoint_solver_options if adjoint_solver_options is not None else {"umfpack_solver": {}}
        self.fwd_pred  = ad_wrapper(
            self.problem,
            solver_options=solver_options or {},
            adjoint_solver_options=adj_opts,
        )

    def simulate(self, params):
        """Forward pass: run FEM and return the QoI vector."""
        u = self.fwd_pred(self.params_fn(params))[0]
        return self.qoi_fn(u, self.mesh)

    def get_loss_and_grad(self, params, target_y):
        """MSE loss and its gradient w.r.t. params via adjoint AD.

        Parameters
        ----------
        params   : array-like, shape (n_params,)
        target_y : array-like, shape (n_qoi,)

        Returns
        -------
        (loss, grad) — scalar loss value and gradient array of the same shape as params.
        """
        def loss_fn(p):
            return jnp.mean(jnp.square(self.simulate(p) - target_y))

        return jax.value_and_grad(loss_fn)(params)

    def predict(self, q):
        """Forward prediction for a batch of parameter sets: q -> y.

        Parameters
        ----------
        q : DataFrame or array-like, shape (n, n_params)

        Returns
        -------
        DataFrame of shape (n, n_qoi) with columns from QoI_names.
        """
        q_vals = q.values if hasattr(q, "values") else np.atleast_2d(q)
        results = [np.array(self.simulate(sample)) for sample in q_vals]
        return pd.DataFrame(results, columns=self.QoI_names)

    def train(self, *args, **kwargs):
        """No-op: physics-based model requires no data-driven training.

        The base-class signature ``train(y_train, q_train, ...)`` is accepted
        so that generic code that calls model.train() does not raise TypeError.
        Use InverseCalibrator.identify() for gradient-based parameter recovery.
        """
        print("JaxFemGradientModel is physics-based and requires no training.")
        print("Use InverseCalibrator.identify() for gradient-based calibration.")
