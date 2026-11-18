"""Physics-informed mean functions of the Gaussian process emulators.

The emulators of :mod:`hydroBayesCal.surrogate.gpe_gpytorch` model every output column as a Gaussian process with a zero (single output) or constant (multitask) mean on the normalized outputs. Far from the training runs, the prediction then returns to the average of the training runs. A *trend* mean, m(θ) = a + Σ_k b_k f_k(θ), lets the Gaussian process learn only the departure of the model response from that trend, which pays off with few training runs and toward the bounds of the prior.

``gp_mean="linear"`` uses f_k(θ) = θ_k for every calibration parameter (universal kriging with a linear trend). ``gp_mean="hydraulic"`` takes the shape of the features from the physics of the calibration parameter and the calibration target:

* **Bed friction, normal flow of a wide channel at a fixed unit discharge** (TELEMAC, Delft3D). The water depth (and water-surface elevation) scales with n^(3/5), k_St^(-3/5) or C^(-2/3) (Manning, Strickler, Chezy), the depth-averaged velocity with the inverse power. An equivalent sand roughness k_s (Nikuradse, Haaland, log wall, Colebrook-White) enters the log law, U ∝ ln(12 h / k_s), so its feature is ln(k_s).
* **Rough-wall turbulence, k-ε model** (OpenFOAM). The wall roughness ``ks`` enters every velocity and turbulence target through the log law, as ln(k_s). In the equilibrium (log) layer the turbulent kinetic energy is k = u*² / √C_μ, so the feature of C_μ is C_μ^(-1/2) for the turbulent kinetic energy and C_μ^(-1/4) for a velocity fluctuation. The friction velocity u* depends on k_s as well, so k is a product of a function of C_μ and one of k_s; a product feature was tested on a k-ε channel model and made the emulator less accurate, so the trend stays additive.

A parameter without a known role (GAIA, a fortran constant, the other k-ε coefficients) and a target of another kind get the linear trend. The coefficients a and b_k are trained with the kernel hyperparameters, so their signs are free: a zone whose roughness raises the velocity at a point in another zone gets a negative coefficient with the same curvature.

A feature is a tuple: ``("power", j, p)`` for θ_j^p (``p = 1``: linear) or ``("log", j)`` for ln θ_j.
"""
import gpytorch
import numpy as np
import torch

from hydroBayesCal.utils.config_logging import logger, logger_warn

#: Values of ``sampling['gp_mean']``.
GP_MEANS = ("constant", "linear", "hydraulic")

#: Marks a sand roughness in :data:`DEPTH_EXPONENTS`: its feature is the logarithm.
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

#: Role of the k-ε coefficient C_μ, see :meth:`~hydroBayesCal.hysim.HydroSimulations.turbulence_constants`.
CMU = "CMU"

#: Exponent of C_μ per target kind: k = u*²/√C_μ in the log layer, a velocity fluctuation scales with √k.
CMU_EXPONENTS = {"tke": -0.5, "fluctuation": -0.25}

#: Calibration targets that scale like the water depth, with underscores read as blanks (TELEMAC, Delft3D). A water-surface elevation is the depth plus the bed level, a constant the coefficient a absorbs.
DEPTH_LIKE = ("WATER DEPTH", "FREE SURFACE", "WATER LEVEL", "WATER SURFACE")
#: Calibration targets that scale like a velocity (TELEMAC, Delft3D, OpenFOAM names). Any name that starts with VELOCITY counts as well.
VELOCITY_LIKE = ("SCALAR VELOCITY", "U MAG", "U X", "U Y", "U Z")
#: Turbulent kinetic energy (TELEMAC ``TURBULENT ENERG.``, OpenFOAM ``TKE``).
TKE_LIKE = ("TKE", "TURBULENT ENERG", "TURBULENT ENERG.", "TURBULENT ENERGY")
#: Velocity fluctuations (OpenFOAM).
FLUCTUATION_LIKE = ("U FLUCT", "V FLUCT", "W FLUCT")

#: The trend has one coefficient per feature plus a constant. With fewer than this many training runs per coefficient it nearly interpolates the runs and leaves the Gaussian process without residual to learn from, which makes the emulator overconfident; the emulator then keeps the constant mean.
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
    """``"depth"``, ``"velocity"``, ``"tke"``, ``"fluctuation"`` or ``None`` for a calibration target name (case-insensitive)."""
    name = " ".join(str(quantity).replace("_", " ").upper().split())
    if name in DEPTH_LIKE:
        return "depth"
    if name in VELOCITY_LIKE or name.startswith("VELOCITY"):
        return "velocity"
    if name in TKE_LIKE:
        return "tke"
    if name in FLUCTUATION_LIKE:
        return "fluctuation"
    return None


def friction_feature(j, law, kind):
    """Trend feature of parameter ``j`` with a friction law for a target kind, ``None`` if there is no physical shape."""
    if law is None or kind is None:
        return None
    depth_exponent = DEPTH_EXPONENTS.get(str(law).strip().upper()[:4])
    if depth_exponent is None:
        return None
    if depth_exponent == LOG_FEATURE:
        return ("log", j)
    # Normal flow: u* = sqrt(g h S), so the turbulent kinetic energy scales like h, a fluctuation like sqrt(h).
    exponent = {"depth": depth_exponent, "velocity": -depth_exponent, "tke": depth_exponent,
                "fluctuation": 0.5 * depth_exponent}[kind]
    return ("power", j, exponent)


def _is_shaped(feature):
    """True for a power other than 1 or a logarithm, which need a positive parameter."""
    return feature[0] == "log" or feature[2] != 1.0


def target_features(quantity, parameters, friction_laws=None, turbulence_constants=None):
    """Trend features of one calibration target, see the module documentation.

    Parameters
    ----------
    quantity : str
        Calibration target name.
    parameters : list of str
        Calibration parameter names.
    friction_laws : dict, optional
        ``{parameter: law}``, see :meth:`~hydroBayesCal.hysim.HydroSimulations.friction_laws`.
    turbulence_constants : dict, optional
        ``{parameter: role}`` with :data:`CMU` for C_μ, see :meth:`~hydroBayesCal.hysim.HydroSimulations.turbulence_constants`.

    Returns
    -------
    list of tuple
        One feature per calibration parameter, in parameter order.
    """
    kind = target_kind(quantity)
    laws, roles = dict(friction_laws or {}), dict(turbulence_constants or {})
    features = []
    for j, parameter in enumerate(parameters):
        feature = friction_feature(j, laws.get(parameter), kind)
        if feature is None and roles.get(parameter) == CMU and kind in CMU_EXPONENTS:
            feature = ("power", j, CMU_EXPONENTS[kind])
        features.append(feature or ("power", j, 1.0))
    return features


def trend_features(gp_mean, parameters, quantities, nloc, parameter_ranges=None, friction_laws=None,
                   turbulence_constants=None):
    """Trend features of every output column of a calibration.

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
    friction_laws, turbulence_constants : dict, optional
        Roles of the parameters, see :func:`target_features`. Only used by ``"hydraulic"``.

    Returns
    -------
    list of list of tuple, or None
        One feature list per output column, ``nloc * n_quantities`` in the interleaved order; ``None`` for ``"constant"``.
    """
    gp_mean = check_gp_mean(gp_mean)
    if gp_mean == "constant":
        return None
    parameters, quantities = list(parameters), list(quantities)
    if gp_mean == "linear":
        per_target = [[("power", j, 1.0) for j in range(len(parameters))] for _ in quantities]
    else:
        per_target = [target_features(q, parameters, friction_laws, turbulence_constants) for q in quantities]
        if parameter_ranges is not None:
            nonpositive = {j for j in range(len(parameters)) if min(parameter_ranges[j]) <= 0.0}
            dropped = sorted({parameters[f[1]] for features in per_target for f in features
                              if _is_shaped(f) and f[1] in nonpositive})
            if dropped:
                per_target = [[("power", f[1], 1.0) if _is_shaped(f) and f[1] in nonpositive else f
                               for f in features] for features in per_target]
                logger_warn.warning(
                    f"The prior range of {dropped} reaches zero or below, where the power or logarithm of the "
                    f"physical trend is undefined. These parameters get a linear trend in the hydraulic mean.")
        shaped = {q: [describe(f, parameters) for f in features if _is_shaped(f)]
                  for q, features in zip(quantities, per_target)}
        shaped = {q: f for q, f in shaped.items() if f}
        if shaped:
            logger.info(f"Hydraulic GP mean, physical trend features per calibration target (all other parameters "
                        f"linear): {shaped}")
        else:
            logger_warn.warning(
                f"gp_mean='hydraulic': none of the calibration parameters {parameters} has a known physical role "
                f"(bed friction, wall roughness, C_mu) for the targets {quantities}. The emulator uses the linear "
                f"trend of gp_mean='linear'.")
    return [list(features) for _ in range(int(nloc)) for features in per_target]


def describe(feature, parameters):
    """Readable form of a feature, e.g. ``ln(ks)`` or ``Cmu^-0.5``."""
    if feature[0] == "log":
        return f"ln({parameters[feature[1]]})"
    return parameters[feature[1]] if feature[2] == 1.0 else f"{parameters[feature[1]]}^{feature[2]:.3g}"


def _column_features(mean_features, n_columns, n_params):
    """Check one feature list per output column; ``None`` stays ``None``."""
    if mean_features is None:
        return None
    mean_features = [list(features) for features in mean_features]
    if len(mean_features) != n_columns:
        raise ValueError(f"mean_features has {len(mean_features)} feature lists for {n_columns} output columns "
                         f"(one per column, in the interleaved order of the outputs).")
    for features in mean_features:
        for feature in features:
            _check_feature(feature, n_params)
    return mean_features


def _check_feature(feature, n_params):
    valid = isinstance(feature, tuple) and ((len(feature) == 3 and feature[0] == "power")
                                            or (len(feature) == 2 and feature[0] == "log"))
    if not valid:
        raise ValueError(f"Unknown trend feature {feature!r}; use ('power', j, p) or ('log', j).")
    if not 0 <= int(feature[1]) < n_params:
        raise ValueError(f"Trend feature {feature!r} names parameter {feature[1]} of {n_params}.")


class HydraulicMean(gpytorch.means.Mean):
    """Trend mean m(θ) = a + Σ_k b_k (f_k(θ) - c_k) / s_k of one output of a Gaussian process.

    The features f_k are given as tuples, see the module documentation. They are standardized with their mean c_k and spread s_k over the training runs, so that a and b_k live on the scale of the normalized outputs. The emulator may see the parameters scaled to [0, 1] (``tp_normalization``); ``x_min`` and ``x_range`` map them back, since the physics acts on the physical value.

    Parameters
    ----------
    features : list of tuple
        Trend features.
    train_x : torch.Tensor
        Training inputs as the emulator sees them, shape ``[n_runs, n_params]``.
    train_y : torch.Tensor, optional
        Normalized training outputs of this column, shape ``[n_runs]``. When given, a and b_k start from their ridge least-squares estimate instead of zero, and the training refines them together with the kernel hyperparameters. Kept fixed at that estimate, the trend made the emulators of a k-ε test model less accurate in most cases.
    x_min, x_range : array_like, optional
        Input scaling of the emulator; physical θ = x_min + x * x_range.
    """

    def __init__(self, features, train_x, train_y=None, x_min=None, x_range=None):
        super().__init__()
        train_x = torch.as_tensor(train_x)
        dtype = train_x.dtype
        n_params = train_x.shape[-1]
        self.features = [tuple(feature) for feature in features]
        for feature in self.features:
            _check_feature(feature, n_params)
        n_features = len(self.features)
        offset = torch.zeros(n_params, dtype=dtype) if x_min is None else torch.as_tensor(np.asarray(x_min, dtype=float), dtype=dtype)
        scale = torch.ones(n_params, dtype=dtype) if x_range is None else torch.as_tensor(np.asarray(x_range, dtype=float), dtype=dtype)
        self.register_buffer("x_offset", offset)
        self.register_buffer("x_scale", scale)
        self.register_buffer("f_center", torch.zeros(n_features, dtype=dtype))
        self.register_buffer("f_scale", torch.ones(n_features, dtype=dtype))
        with torch.no_grad():
            raw = self._raw_features(train_x)
            spread = raw.std(dim=0) if raw.shape[0] > 1 else torch.ones(n_features, dtype=dtype)
            self.f_center.copy_(raw.mean(dim=0))
            self.f_scale.copy_(torch.where(torch.isfinite(spread) & (spread > 0), spread, torch.ones_like(spread)))
        constant, weights = torch.zeros(1, dtype=dtype), torch.zeros(n_features, dtype=dtype)
        if train_y is not None:
            constant, weights = self._least_squares(train_x, torch.as_tensor(train_y, dtype=dtype).reshape(-1))
        self.register_parameter("constant", torch.nn.Parameter(constant))
        self.register_parameter("weights", torch.nn.Parameter(weights))

    @staticmethod
    def _evaluate(feature, theta):
        kind = feature[0]
        value = theta[..., int(feature[1])]
        if kind == "power" and feature[2] == 1.0:
            return value
        # A power or logarithm needs a positive value; trend_features keeps such features to positive prior ranges, the clamp only guards against round-off at a bound.
        value = value.clamp_min(torch.finfo(value.dtype).tiny)
        return torch.log(value) if kind == "log" else value ** float(feature[2])

    def _raw_features(self, x):
        theta = self.x_offset + x * self.x_scale
        return torch.stack([self._evaluate(feature, theta) for feature in self.features], dim=-1)

    def standardized_features(self, x):
        """Standardized trend features, shape ``[..., n_points, n_features]``."""
        return (self._raw_features(x) - self.f_center) / self.f_scale

    def _least_squares(self, train_x, train_y):
        with torch.no_grad():
            design = torch.cat([torch.ones(train_x.shape[0], 1, dtype=train_y.dtype),
                                self.standardized_features(train_x)], dim=1)
            gram = design.T @ design + RIDGE * train_x.shape[0] * torch.eye(design.shape[1], dtype=train_y.dtype)
            gram[0, 0] -= RIDGE * train_x.shape[0]  # the intercept is not shrunk
            coefficients = torch.linalg.solve(gram, design.T @ train_y)
        if not torch.all(torch.isfinite(coefficients)):
            return torch.zeros(1, dtype=train_y.dtype), torch.zeros(design.shape[1] - 1, dtype=train_y.dtype)
        return coefficients[:1].clone(), coefficients[1:].clone()

    def forward(self, x):
        return self.constant + self.standardized_features(x) @ self.weights


def trend_mean(features, train_x, train_y=None, x_min=None, x_range=None):
    """A :class:`HydraulicMean`, or ``None`` when there are too few training runs for its coefficients.

    ``None`` tells the emulator to keep its constant mean, see :data:`MIN_RUNS_PER_COEFFICIENT`.
    """
    if features is None:
        return None
    n_runs, n_features = int(train_x.shape[0]), len(features)
    if n_runs < MIN_RUNS_PER_COEFFICIENT * (n_features + 1):
        _warn_too_few_runs(n_runs, n_features)
        return None
    return HydraulicMean(features, train_x, train_y=train_y, x_min=x_min, x_range=x_range)


_REPORTED_TOO_FEW_RUNS = set()


def _warn_too_few_runs(n_runs, n_features):
    if (n_runs, n_features) in _REPORTED_TOO_FEW_RUNS:
        return
    _REPORTED_TOO_FEW_RUNS.add((n_runs, n_features))
    logger_warn.warning(
        f"The trend mean of the emulator has {n_features + 1} coefficients, which {n_runs} training runs cannot "
        f"estimate next to the Gaussian process (at least {MIN_RUNS_PER_COEFFICIENT * (n_features + 1)} are "
        f"needed). The emulator keeps the constant mean until the design has grown that far.")
