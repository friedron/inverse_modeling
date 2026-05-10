import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax_fem.solver import ad_wrapper
from jax.scipy.stats import multivariate_normal


class JaxFemGradientModel:
    """Gradient-based inverse model that differentiates through a JAX-FEM solve.

    Parameters
    ----------
    Q : VariableSet
        Parameter space (bounds, names).
    QoI_names : list[str]
        Names of the measurement quantities (must match qoi_fn output length).
    problem_class : type
        JAX-FEM Problem subclass to instantiate.
    mesh : Mesh
        JAX-FEM mesh object.
    params_fn : callable  (params: array) -> list[jnp.ndarray]
        Maps a 1-D parameter array to the list of arrays expected by fwd_pred.
    qoi_fn : callable  (u: jnp.ndarray, mesh: Mesh) -> jnp.ndarray
        Extracts the measurement vector from the displacement field.
        Must be JAX-differentiable.
    problem_args : dict, optional
        Keyword arguments forwarded to problem_class constructor.
    solver_options : dict, optional
        Forwarded to ad_wrapper for the forward linear solve.
    adjoint_solver_options : dict, optional
        Forwarded to ad_wrapper for the adjoint linear solve. Defaults to
        {'umfpack_solver': {}} to avoid BiCGStab divergence on the transposed
        non-symmetric system from row-elimination Dirichlet BCs.
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
        self.Q = Q
        self.QoI_names = QoI_names
        self.mesh = mesh
        self.params_fn = params_fn
        self.qoi_fn = qoi_fn
        self.problem = problem_class(mesh, **(problem_args or {}))
        adj_opts = adjoint_solver_options if adjoint_solver_options is not None else {"umfpack_solver": {}}
        self.fwd_pred = ad_wrapper(
            self.problem,
            solver_options=solver_options or {},
            adjoint_solver_options=adj_opts,
        )

    def simulate(self, params):
        """Run FEM forward pass and return the QoI vector."""
        u = self.fwd_pred(self.params_fn(params))[0]
        return self.qoi_fn(u, self.mesh)

    def get_loss_and_grad(self, params, target_y):
        """MSE loss and its gradient w.r.t. params via adjoint AD."""
        def loss_fn(p):
            return jnp.mean(jnp.square(self.simulate(p) - target_y))
        return jax.value_and_grad(loss_fn)(params)

    def get_likelihood_and_grad(self, params, target_y, sigmas):
        """Gaussian log-likelihood and its gradient w.r.t. params via adjoint AD."""
        def get_likelihood(p):
            mean = jnp.zeros_like(target_y)
            cov = jnp.diag(sigmas ** 2)
            return multivariate_normal.pdf(p, mean=mean, cov=cov)
        return jax.value_and_grad(get_likelihood)(params)

    def predict(self, q):
        """Forward prediction for a batch of parameter sets: q -> y.

        q : DataFrame or array-like, shape (n, n_params)
        Returns DataFrame of shape (n, n_qoi).
        """
        q_vals = q.values if hasattr(q, "values") else np.atleast_2d(q)
        results = [np.array(self.simulate(sample)) for sample in q_vals]
        return pd.DataFrame(results, columns=self.QoI_names)
