import numpy as np


def gradient_based_model_calibration(
    model,
    target_y,
    initial_params=None,
    learning_rate=0.1,
    steps=100,
    verbose=True,
    return_history=False,
):
    """Calibrate a model via projected gradient descent in normalised parameter space.

    Parameters are normalised to [0, 1] before optimisation so that a single
    learning rate is effective across parameters with different physical scales.
    Gradients are transformed via the chain rule before each update, and the
    normalised iterates are projected back onto [0, 1] after every step.

    Parameters
    ----------
    model : InverseModel
        Must implement ``get_loss_and_grad(params, target_y)`` and expose
        ``model.Q`` with ``get_bounds()`` and ``param_names()`` methods.
    target_y : array-like
        Observed measurement vector to match.
    initial_params : array-like or None
        Starting parameter values in physical units.  Defaults to the midpoint
        of each parameter's bounds.
    learning_rate : float
        Step size applied to normalised gradients (default: 0.1).
    steps : int
        Number of gradient descent iterations (default: 100).
    verbose : bool
        Print a progress table every ``steps // 10`` iterations and at the
        final step (default: True).
    return_history : bool
        If True, return a tuple ``(params, history)`` where ``history`` is a
        list of dicts with keys ``'step'``, ``'loss'``, and ``'params'``
        (physical-unit parameter values at that step).

    Returns
    -------
    numpy.ndarray, shape (n_params,)
        Best-estimate parameter vector in physical units.
        If ``return_history=True``, returns ``(params, history)`` instead.
    """
    bounds      = np.array(model.Q.get_bounds())
    low         = bounds[:, 0]
    high        = bounds[:, 1]
    param_range = high - low

    if initial_params is None:
        params = (low + high) / 2.0
    else:
        params = np.array(initial_params, dtype=float)

    # Normalise initial params to [0, 1].
    params_norm = (params - low) / param_range

    if verbose:
        print(f"\n--- Starting Normalised Gradient Calibration ({model.method}) ---")
        param_names = model.Q.param_names()
        headers     = ["Step", "Loss"] + param_names
        print(" | ".join(f"{h:<10}" for h in headers))
        print("-" * (13 * len(headers)))

    log_interval = max(1, steps // 10)
    history = []

    for i in range(steps):
        # Denormalise for model evaluation.
        p_real = params_norm * param_range + low
        loss, grads = model.get_loss_and_grad(p_real, target_y)

        if return_history:
            history.append({"step": i, "loss": float(loss), "params": np.array(p_real)})

        # Chain rule: dL/dp_norm = dL/dp_real * (dp_real/dp_norm) = grads * param_range
        grads_norm = np.array(grads) * param_range

        # SGD update on normalised params.
        params_norm = params_norm - learning_rate * grads_norm

        # Project back onto [0, 1].
        params_norm = np.clip(params_norm, 0.0, 1.0)

        if verbose and (i % log_interval == 0 or i == steps - 1):
            row = [i, float(loss)] + p_real.tolist()
            print(
                f"{row[0]:<10} | {row[1]:<10.4e} | "
                + " | ".join(f"{p:<10.4f}" for p in row[2:])
            )

    result = params_norm * param_range + low
    if return_history:
        return result, history
    return result