"""
Global constant and variable definitions
"""
import os
import re
import numpy as np
import pandas as _pd

# get telemac and gaia control parameters to enable differentiated writing of steering files
TM_TEMPLATE_DIR = os.path.abspath(os.path.join(__file__, "..")) + os.sep + "templates" + os.sep
#print(TM_TEMPLATE_DIR)
GAIA_PARAMETERS = _pd.read_csv(TM_TEMPLATE_DIR+"parameters-gaia.csv", names=["parameter", "type"])
#print(GAIA_PARAMETERS)
TM2D_PARAMETERS = _pd.read_csv(TM_TEMPLATE_DIR+"parameters-telemac2d.csv", names=["parameter", "type"])
#print(TM2D_PARAMETERS)
# Dictionary mapping each variable name to its source model
classification_tm_gaia_dict = {
    "WATER DEPTH": "telemac",
    "SCALAR VELOCITY": "telemac",
    "FREE SURFACE": "telemac",
    "VELOCITY U": "telemac",
    "VELOCITY V": "telemac",
    "VELOCITY W": "telemac",
    "FROUDE NUMBER": "telemac",
    "TURBULENT ENERG": "telemac",
    "TURBULENT ENERG.": "telemac",  # real SELAFIN name (16-char, trailing period)
    "BOTTOM SHEAR STRESS": "telemac",
    "FRICTION VELOCI": "telemac",
    "DISSIPATION": "telemac",
    "3D VELOCITY MAGNITUDE": "telemac",

    "BED ELEVATION": "gaia",
    "CUMUL BED EVOL": "gaia",
    "SUSPENDED LOAD CONC.": "gaia",
    "BED LOAD": "gaia",
    "TOTAL SEDIMENT DISCHARGE": "gaia",
    "SEDIMENT DIAMETER": "gaia",
    "CRITICAL SHEAR STRESS": "gaia",
}


# ============================================================
# GAIA BED-LAYER SAND RATIOS
# ============================================================
def gaia_sand_ratio_variable(layer, sand_class):
    """Name of the GAIA result variable that holds the ratio of one sand class in one bed layer.

    GAIA writes one variable per sand class and bed layer and names it ``LAY<layer>SAND RAT<class>``, each number left-aligned in a two-character field (subroutine ``nomvar_gaia``). The number after ``LAY`` is the bed layer and the trailing number is the sand class: ``LAY1 SAND RAT2`` is the ratio of sand class 2 in layer 1, ``LAY2 SAND RAT1`` the ratio of sand class 1 in layer 2.

    Parameters
    ----------
    layer : int
        Bed layer number, 1 is the active layer at the bed surface.
    sand_class : int
        Sand class number, in the order of ``CLASSES SEDIMENT DIAMETERS`` in the GAIA steering file.

    Returns
    -------
    str
        Variable name with single spaces, as the extraction reads it from the SELAFIN file.
    """
    layer, sand_class = int(layer), int(sand_class)
    if not (1 <= layer <= 99 and 1 <= sand_class <= 99):
        raise ValueError(
            f"GAIA numbers bed layers and sand classes from 1 to 99, got layer={layer} and sand_class={sand_class}.")
    return " ".join(f"LAY{layer:<2d}SAND RAT{sand_class}".split())


#: A layer-average target named after one sand class, ``SAND RAT<class>``, see :func:`resolve_gaia_layer_average`.
SAND_RATIO_TARGET = re.compile(r"SAND RAT([1-9][0-9]*)")


def resolve_gaia_layer_average(gaia_layer_average, requested=None):
    """Check a ``gaia_layer_average`` configuration and name the GAIA variables it reads.

    A layer-averaged sand ratio is a derived calibration target: the ratio of one or several sand classes, averaged over several bed layers with the layer thicknesses as weights. A sediment sample taken over a known depth below the bed surface, for instance the fine sediment fraction of a freeze core, is compared with it. For every target the configuration gives::

        gaia_layer_average = {
            "FINE SEDIMENT FRACTION": {      # name of the target, also used in the calibration points file
                "sand_classes": [1],         # ratios of these classes are summed in every layer
                "layers": [1, 2],            # bed layers that are averaged
                "thicknesses": [0.08, 0.60], # weight of every layer, in meters
            }
        }

    A target named ``SAND RAT<class>`` (for instance ``SAND RAT2``) may leave out ``sand_classes``: the trailing number of the name is then the one sand class it averages. Every other name has to state ``sand_classes``.

    Parameters
    ----------
    gaia_layer_average : dict or None
        Configuration as above, from ``morphodynamic_simulation['gaia_layer_average']``.
    requested : iterable of str, optional
        Only resolve the targets named here. Default: all targets of the configuration.

    Returns
    -------
    dict
        Per target ``sand_classes``, ``layers``, ``thicknesses`` (array) and ``variables``, a list with one entry per layer holding the variable names of the classes in that layer.

    Raises
    ------
    ValueError
        If an entry is incomplete or inconsistent. An entry without ``sand_classes`` is rejected rather than guessed, because the variable name carries two numbers and reading the class for the layer yields a quantity that looks plausible and is wrong.
    """
    resolved = {}
    for target, spec in (gaia_layer_average or {}).items():
        if requested is not None and target not in requested:
            continue
        if not isinstance(spec, dict):
            raise ValueError(f"gaia_layer_average['{target}'] must be a dictionary.")

        layers, thicknesses, sand_classes = spec.get("layers"), spec.get("thicknesses"), spec.get("sand_classes")
        # The name 'SAND RAT<class>' carries only the class number, so it cannot be mistaken for a layer number.
        class_in_name = SAND_RATIO_TARGET.fullmatch(" ".join(str(target).split()).upper())
        if sand_classes is None and class_in_name is not None:
            sand_classes = [int(class_in_name.group(1))]
        if sand_classes is None:
            raise ValueError(
                f"gaia_layer_average['{target}'] has no 'sand_classes'. GAIA names the ratio of a sand class in a bed layer "
                f"LAY<layer> SAND RAT<class>: the number after LAY is the layer, the trailing number is the sand class. "
                f"State both, e.g. {{'sand_classes': [1], 'layers': [1, 2], 'thicknesses': [0.08, 0.60]}} for sand class 1 "
                f"averaged over the layers 1 and 2. Appending the layer number to '{target}' selects sand classes, not layers.")
        if layers is None or thicknesses is None:
            raise ValueError(f"gaia_layer_average['{target}'] must contain 'layers' and 'thicknesses'.")

        sand_classes = [int(c) for c in np.atleast_1d(sand_classes)]
        layers = [int(layer) for layer in np.atleast_1d(layers)]
        thicknesses = np.atleast_1d(np.asarray(thicknesses, dtype=float))
        if not sand_classes or not layers:
            raise ValueError(f"gaia_layer_average['{target}'] needs at least one sand class and one layer.")
        if len(layers) != len(thicknesses):
            raise ValueError(
                f"gaia_layer_average['{target}'] has {len(layers)} layers but {len(thicknesses)} thicknesses.")
        if np.any(thicknesses <= 0.0) or not np.all(np.isfinite(thicknesses)):
            raise ValueError(f"All layer thicknesses of gaia_layer_average['{target}'] must be finite and > 0.")

        resolved[target] = {
            "sand_classes": sand_classes,
            "layers": layers,
            "thicknesses": thicknesses,
            "variables": [[gaia_sand_ratio_variable(layer, c) for c in sand_classes] for layer in layers],
        }
    return resolved
#: Internal time mode of :meth:`~hydroBayesCal.telemac.control_telemac.TelemacModel._apply_time_mode`: the mean of the ``n`` output steps that end at ``time_index``. It is what ``"mean_last"`` becomes at a calibration point with its own time, see :func:`point_times`.
MEAN_WINDOW = "mean_window"


def point_times(calibration_pts_df, column):
    """Read the extraction time of every calibration point from a column of the calibration points file.

    A measurement taken at another time than the end of the simulation (a bed sample of an earlier survey, a gauge reading during a flood) is compared with the model output at that time. The times are given in seconds on the time axis of the SELAFIN result files, the axis TELEMAC and GAIA write (``0`` is the start of the simulation, a hot start carries on the time of its initial condition). A point without a value is extracted with the configured ``output_extraction_time``.

    Parameters
    ----------
    calibration_pts_df : pandas.DataFrame
        Calibration (or validation) points.
    column : str
        Name of the column with the times, ``extraction['point_time_column']`` in the configuration. Matched case-insensitively.

    Returns
    -------
    numpy.ndarray
        One time per point in seconds, ``nan`` where the point has none.

    Raises
    ------
    ValueError
        If the column is missing or holds a value that is not a number of seconds, for instance a date. The values are not guessed: a date read as a number would select a time step that has nothing to do with the measurement.
    """
    matches = [c for c in calibration_pts_df.columns if str(c).strip().lower() == str(column).strip().lower()]
    if not matches:
        raise ValueError(
            f"extraction['point_time_column'] is '{column}', but the calibration points file has no such column. "
            f"Columns: {list(calibration_pts_df.columns)}.")
    values = calibration_pts_df[matches[0]]
    text = values.astype(str).str.strip()
    empty = values.isna() | text.str.lower().isin(["", "nan", "none"])
    times = _pd.to_numeric(text.where(~empty), errors="coerce")
    invalid = times.isna() & ~empty
    if invalid.any():
        point_ids = calibration_pts_df.iloc[:, 0][invalid].tolist()
        raise ValueError(
            f"The column '{matches[0]}' of the calibration points file holds values that are not times in seconds "
            f"at the points {point_ids}: {values[invalid].tolist()}. Give the time in seconds on the time axis of the "
            "result files, or leave the cell empty to extract that point with output_extraction_time.")
    return times.to_numpy(dtype=float)


def point_time_index(model_times, point_time, label):
    """Index of the output step written at ``point_time``.

    Parameters
    ----------
    model_times : array-like
        Output times of a SELAFIN file in seconds.
    point_time : float
        Requested time in seconds.
    label : str
        Names the point and the result file in the error message.

    Raises
    ------
    ValueError
        If no output step was written at that time. The nearest step is not taken instead: a result file with a coarser output interval than the measurement times would otherwise go unnoticed.
    """
    model_times = np.asarray(model_times, dtype=float)
    # SELAFIN stores times in single precision, which resolves about 1e-7 of the value.
    matches = np.flatnonzero(np.isclose(model_times, point_time, rtol=1e-6, atol=1e-3))
    if not len(matches):
        nearest = model_times[np.argmin(np.abs(model_times - point_time))] if model_times.size else None
        raise ValueError(
            f"{label}: no output was written at t = {point_time} s; the nearest output time is {nearest} s "
            f"(outputs from {model_times.min() if model_times.size else None} to "
            f"{model_times.max() if model_times.size else None} s). Write outputs at the measurement times "
            "(GRAPHIC PRINTOUT PERIOD) or correct the time in the calibration points file.")
    return int(matches[0])


# ============================================================
# DEFAULT PHYSICAL CONSTANTS FOR WALL-LAW DIAGNOSTICS
# ============================================================
# These constants are used to compute TELEMAC-style friction
# velocity and y+ from the near-bed velocity and vertical spacing.
#
# They can be overwritten inside the extraction function if needed.
# ============================================================

von_Karman_constant = 0.40
nikuradse_log_factor = 30.0
kinematic_viscosity_water = 1.0e-6

# ============================================================
# 2D SLF VARIABLE NAMES FOR BOTTOM FRICTION / NIKURADSE ks
# ============================================================

slf_2d_variables_from_3d = [
    "FRICTION COEFFICIENT",
    "BOTTOM FRICTION COEFFICIENT",
    "FRICTION COEFFICIENT FOR THE BOTTOM",
    "BOTTOM FRICTION",
    "ROUGHNESS COEFFICIENT",
    "NIKURADSE ROUGHNESS",
    "NIKURADSE KS"
]