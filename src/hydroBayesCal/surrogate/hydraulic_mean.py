"""Physics-informed mean functions of the Gaussian process emulators.

The emulators of :mod:`hydroBayesCal.surrogate.gpe_gpytorch` model every output column as a Gaussian process with a zero (single output) or constant (multitask) mean on the normalized outputs. Far from the training runs, the prediction then returns to the average of the training runs. A *trend* mean, m(θ) = a + Σ_k b_k f_k(θ_k), lets the Gaussian process learn only the departure of the model response from that trend, which pays off with few training runs and toward the bounds of the prior.

``gp_mean="linear"`` uses f_k(θ_k) = θ_k for every calibration parameter (universal kriging with a linear trend). ``gp_mean="hydraulic"`` takes the shape of f_k from the friction law of a roughness parameter, following the normal-flow (Gauckler-Manning-Strickler) scaling of a wide channel at a fixed unit discharge:

* water depth (and water-surface elevation) h ∝ n^(3/5) ∝ k_St^(-3/5) ∝ C^(-2/3),
* depth-averaged velocity U = q / h ∝ n^(-3/5) ∝ k_St^(3/5) ∝ C^(2/3),
* for an equivalent sand roughness k_s (Nikuradse, Haaland, log wall, Colebrook-White), U ∝ ln(12 h / k_s), so f is ln(k_s) for both.

A parameter without a known friction law (GAIA, turbulence, a fortran constant) and a target that is neither a depth nor a velocity get the linear trend. The coefficients a and b_k are trained with the kernel hyperparameters, so their signs are free: a zone whose roughness raises the velocity at a point in another zone gets a negative coefficient with the same curvature.
"""
import gpytorch
import numpy as np
import torch

from hydroBayesCal.utils.config_logging import logger, logger_warn

#: Values of ``sampling['gp_mean']``.
GP_MEANS = ("constant", "linear", "hydraulic")

#: Exponent of the trend feature θ^p for the value 0: the feature is ln(θ).
LOG_FEATURE = 0.0

#: Exponent p of the water depth in the friction coefficient of each law (h ∝ c^p), keyed by the four-letter law names of a TELEMAC friction table. The velocity takes -p. ``LOG_FEATURE`` marks laws with an equivalent sand roughness. ``None`` (no friction) falls back to the linear trend.
DEPTH_EXPONENTS = {
    "MANN": 0.6,
    "STRI": -0.6,
    "CHEZ": -2.0 / 3.0,
    "NIKU": LOG_FEATURE,
    "HAAL": LOG_FEATURE,
    "LOGW": LOG_FEATURE,
    "COWH": LOG_FEATURE,
    "NOFR": None,
}

#: TELEMAC ``LAW OF BOTTOM FRICTION`` codes and their table names.
TELEMAC_LAW_CODES = {0: "NOFR", 1: "HAAL", 2: "CHEZ", 3: "STRI", 4: "MANN", 5: "NIKU", 6: "LOGW", 7: "COWH"}

#: Calibration targets that scale like the water depth, with underscores read as blanks (TELEMAC, Delft3D). A water-surface elevation is the depth plus the bed level, a constant the coefficient a absorbs.
DEPTH_LIKE = ("WATER DEPTH", "FREE SURFACE", "WATER LEVEL", "WATER SURFACE")
#: Calibration targets that scale like the depth-averaged velocity (TELEMAC, Delft3D, OpenFOAM names). Any name that starts with VELOCITY counts as well.
VELOCITY_LIKE = ("SCALAR VELOCITY", "U MAG", "U X", "U Y", "U Z")

#: The trend has 1 + n_parameters coefficients. With fewer than this many training runs per coefficient it nearly interpolates the runs and leaves the Gaussian process without residual to learn from, which makes the emulator overconfident; the emulator then keeps the constant mean.
MIN_RUNS_PER_COEFFICIENT = 2

#: Ridge term of the least-squares start of the trend coefficients, on standardized features and outputs.
RIDGE = 1e-3


def check_gp_mean(gp_mean):
    """Refuse an unknown ``sampling['gp_mean']`` before any simulation runs. Returns the value in lower case."""
    value = str(gp_mean).strip().lower()
    if value not in GP_MEANS:
        raise ValueError(f"Unknown sampling['gp_mean'] = '{gp_mean}'. Use one of {GP_MEANS}.")
    return value


def target_kind(quantity):
    """``"depth"``, ``"velocity"`` or ``None`` for a calibration target name (case-insensitive)."""
    name = " ".join(str(quantity).replace("_", " ").upper().split())
    if name in DEPTH_LIKE:
        return "depth"
    if name in VELOCITY_LIKE or name.startswith("VELOCITY"):
        return "velocity"
    return None


def trend_exponent(law, quantity):
    """Exponent p of the trend feature θ^p for a parameter with this friction law and a calibration target.

    Returns 1.0 (linear) for an unknown law or target, :data:`LOG_FEATURE` for a sand roughness.
    """
    if law is None:
        return 1.0
    depth_exponent = DEPTH_EXPONENTS.get(str(law).strip().upper()[:4])
    kind = target_kind(quantity)
    if depth_exponent is None or kind is None:
        return 1.0
    if depth_exponent == LOG_FEATURE:
        return LOG_FEATURE
    return depth_exponent if kind == "depth" else -depth_exponent


def mean_exponents(gp_mean, parameters, quantities, nloc, parameter_ranges=None, friction_laws=None):
    """Exponents of the trend features for every output column of a calibration.

    Parameters
    ----------
    gp_mean : {"constant", "linear", "hydraulic"}
        ``sampling['gp_mean']``.
    parameters : list of str
        Calibration parameter names.
    quantities : list of str
        Calibration target names, in the order of the output columns of one calibration point.
    nloc : int
        Number of calibration points; the outputs interleave the targets per point.
    parameter_ranges : list of [float, float], optional
        Prior bounds. A power or logarithm of a parameter whose range reaches zero or below is undefined, and that parameter falls back to the linear trend.
    friction_laws : dict, optional
        ``{parameter: law}`` with a law name of :data:`DEPTH_EXPONENTS` for every parameter that is a friction coefficient. Only used by ``"hydraulic"``.

    Returns
    -------
    numpy.ndarray or None
        Shape ``[nloc * n_quantities, n_parameters]``; ``None`` for ``"constant"``.
    """
    gp_mean = check_gp_mean(gp_mean)
    if gp_mean == "constant":
        return None
    parameters, quantities = list(parameters), list(quantities)
    per_target = np.ones((len(quantities), len(parameters)))
    if gp_mean == "hydraulic":
        laws = dict(friction_laws or {})
        for j, parameter in enumerate(parameters):
            for q, quantity in enumerate(quantities):
                per_target[q, j] = trend_exponent(laws.get(parameter), quantity)
        ranges_reach_zero = [
            parameters[j] for j in range(len(parameters))
            if parameter_ranges is not None and np.any(per_target[:, j] != 1.0) and min(parameter_ranges[j]) <= 0.0]
        if ranges_reach_zero:
            for j, parameter in enumerate(parameters):
                if parameter in ranges_reach_zero:
                    per_target[:, j] = 1.0
            logger_warn.warning(
                f"The prior range of {ranges_reach_zero} reaches zero or below, where the power or logarithm of the "
                f"friction law is undefined. These parameters get a linear trend in the hydraulic mean.")
        shaped = {parameters[j]: per_target[:, j].tolist() for j in range(len(parameters))
                  if np.any(per_target[:, j] != 1.0)}
        if shaped:
            logger.info(f"Hydraulic GP mean, trend exponents per calibration target {quantities} "
                        f"(0 = logarithm; parameters not listed have a linear trend): {shaped}")
        else:
            logger_warn.warning(
                f"gp_mean='hydraulic': none of the calibration parameters {parameters} is a friction coefficient of a "
                f"known law, or none of the targets {quantities} is a depth or a velocity. The emulator uses the "
                f"linear trend of gp_mean='linear'.")
    return np.tile(per_target, (int(nloc), 1))


def _as_column_exponents(mean_exponents, n_columns, n_params):
    """Broadcast ``None``, ``[n_params]`` or ``[n_columns, n_params]`` exponents to one row per output column."""
    if mean_exponents is None:
        return None
    exponents = np.atleast_2d(np.asarray(mean_exponents, dtype=float))
    if exponents.shape[0] == 1:
        exponents = np.repeat(exponents, n_columns, axis=0)
    if exponents.shape != (n_columns, n_params):
        raise ValueError(f"mean_exponents has the shape {exponents.shape}; expected ({n_columns}, {n_params}) "
                         f"(one row per output column, one column per calibration parameter) or ({n_params},).")
    return exponents


class HydraulicMean(gpytorch.means.Mean):
    """Trend mean m(θ) = a + Σ_k b_k (f_k(θ_k) - c_k) / s_k of one output of a Gaussian process.

    f_k is θ_k^p_k, or ln(θ_k) for p_k = 0 (:data:`LOG_FEATURE`). The features are standardized with their mean c_k and spread s_k over the training runs, so that a and b_k live on the scale of the normalized outputs. The emulator may see the parameters scaled to [0, 1] (``tp_normalization``); ``x_min`` and ``x_range`` map them back, since the friction law acts on the physical value.

    Parameters
    ----------
    exponents : array_like
        p_k per calibration parameter, shape ``[n_params]``.
    train_x : torch.Tensor
        Training inputs as the emulator sees them, shape ``[n_runs, n_params]``.
    train_y : torch.Tensor, optional
        Normalized training outputs of this column, shape ``[n_runs]``. When given, a and b_k start from their ridge least-squares estimate instead of zero: the drivers train for 150 Adam steps with a learning rate that halves every 5 steps, which moves a coefficient by less than one from where it starts.
    x_min, x_range : array_like, optional
        Input scaling of the emulator; physical θ = x_min + x * x_range.
    """

    def __init__(self, exponents, train_x, train_y=None, x_min=None, x_range=None):
        super().__init__()
        train_x = torch.as_tensor(train_x)
        dtype = train_x.dtype
        exponents = torch.as_tensor(np.asarray(exponents, dtype=float), dtype=dtype).reshape(-1)
        n_params = exponents.numel()
        if train_x.shape[-1] != n_params:
            raise ValueError(f"HydraulicMean has {n_params} exponents for {train_x.shape[-1]} parameters.")
        offset = torch.zeros(n_params, dtype=dtype) if x_min is None else torch.as_tensor(np.asarray(x_min, dtype=float), dtype=dtype)
        scale = torch.ones(n_params, dtype=dtype) if x_range is None else torch.as_tensor(np.asarray(x_range, dtype=float), dtype=dtype)
        self.register_buffer("exponents", exponents)
        self.register_buffer("x_offset", offset)
        self.register_buffer("x_scale", scale)
        self.register_buffer("f_center", torch.zeros(n_params, dtype=dtype))
        self.register_buffer("f_scale", torch.ones(n_params, dtype=dtype))
        with torch.no_grad():
            features = self._raw_features(train_x)
            spread = features.std(dim=0) if features.shape[0] > 1 else torch.ones(n_params, dtype=dtype)
            self.f_center.copy_(features.mean(dim=0))
            self.f_scale.copy_(torch.where(torch.isfinite(spread) & (spread > 0), spread, torch.ones_like(spread)))
        constant, weights = torch.zeros(1, dtype=dtype), torch.zeros(n_params, dtype=dtype)
        if train_y is not None:
            constant, weights = self._least_squares(train_x, torch.as_tensor(train_y, dtype=dtype).reshape(-1))
        self.register_parameter("constant", torch.nn.Parameter(constant))
        self.register_parameter("weights", torch.nn.Parameter(weights))

    def _raw_features(self, x):
        theta = self.x_offset + x * self.x_scale
        shaped = self.exponents != 1.0
        # A power or logarithm needs a positive value; mean_exponents keeps such features to positive prior ranges, the clamp only guards against round-off at a bound.
        positive = torch.where(shaped, theta.clamp_min(torch.finfo(theta.dtype).tiny), theta)
        powered = positive ** self.exponents
        return torch.where(self.exponents == LOG_FEATURE, torch.log(positive), powered)

    def features(self, x):
        """Standardized trend features, shape ``[..., n_points, n_params]``."""
        return (self._raw_features(x) - self.f_center) / self.f_scale

    def _least_squares(self, train_x, train_y):
        with torch.no_grad():
            design = torch.cat([torch.ones(train_x.shape[0], 1, dtype=train_y.dtype), self.features(train_x)], dim=1)
            gram = design.T @ design + RIDGE * train_x.shape[0] * torch.eye(design.shape[1], dtype=train_y.dtype)
            gram[0, 0] -= RIDGE * train_x.shape[0]  # the intercept is not shrunk
            coefficients = torch.linalg.solve(gram, design.T @ train_y)
        if not torch.all(torch.isfinite(coefficients)):
            return torch.zeros(1, dtype=train_y.dtype), torch.zeros(design.shape[1] - 1, dtype=train_y.dtype)
        return coefficients[:1].clone(), coefficients[1:].clone()

    def forward(self, x):
        return self.constant + self.features(x) @ self.weights


def trend_mean(exponents, train_x, train_y=None, x_min=None, x_range=None):
    """A :class:`HydraulicMean`, or ``None`` when there are too few training runs for its coefficients.

    ``None`` tells the emulator to keep its constant mean, see :data:`MIN_RUNS_PER_COEFFICIENT`.
    """
    if exponents is None:
        return None
    n_runs, n_params = int(train_x.shape[0]), int(np.asarray(exponents).size)
    if n_runs < MIN_RUNS_PER_COEFFICIENT * (n_params + 1):
        _warn_too_few_runs(n_runs, n_params)
        return None
    return HydraulicMean(exponents, train_x, train_y=train_y, x_min=x_min, x_range=x_range)


_REPORTED_TOO_FEW_RUNS = set()


def _warn_too_few_runs(n_runs, n_params):
    if (n_runs, n_params) in _REPORTED_TOO_FEW_RUNS:
        return
    _REPORTED_TOO_FEW_RUNS.add((n_runs, n_params))
    logger_warn.warning(
        f"The trend mean of the emulator has {n_params + 1} coefficients, which {n_runs} training runs cannot "
        f"estimate next to the Gaussian process (at least {MIN_RUNS_PER_COEFFICIENT * (n_params + 1)} are needed). "
        f"The emulator keeps the constant mean until the design has grown that far.")
