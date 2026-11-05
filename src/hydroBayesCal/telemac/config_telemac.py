"""
Global constant and variable definitions
"""
import os
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
    "SAND RAT1": "gaia",
    "SAND RAT2": "gaia",
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


def resolve_gaia_layer_average(gaia_layer_average, requested=None):
    """
    Resolve GAIA sand-ratio targets averaged over selected bed layers.

    Each target identifies exactly one sediment class:
        SAND RAT1 -> class 1
        SAND RAT2 -> class 2

    The configuration supplies only layers and thicknesses.
    """
    import re

    resolved = {}

    for target, spec in (gaia_layer_average or {}).items():

        if requested is not None and target not in requested:
            continue

        match = re.fullmatch(r"SAND RAT([1-9][0-9]*)", target)
        if match is None:
            raise ValueError(
                f"Invalid GAIA layer-average target '{target}'. "
                "Expected SAND RAT1, SAND RAT2, etc."
            )

        sand_class = int(match.group(1))

        if not isinstance(spec, dict):
            raise ValueError(
                f"gaia_layer_average['{target}'] must be a dictionary."
            )

        if "sand_classes" in spec:
            raise ValueError(
                f"Remove 'sand_classes' from {target}. "
                "The class is determined from the target name."
            )

        if "layers" not in spec or "thicknesses" not in spec:
            raise ValueError(
                f"{target} requires 'layers' and 'thicknesses'."
            )

        layers = [int(layer) for layer in np.atleast_1d(spec["layers"])]

        thicknesses = np.atleast_1d(
            np.asarray(spec["thicknesses"], dtype=float)
        )

        if not layers or len(layers) != len(thicknesses):
            raise ValueError(
                f"{target}: number of layers and thicknesses must match."
            )

        if len(set(layers)) != len(layers):
            raise ValueError(f"{target}: duplicate layers.")

        if any(layer < 1 or layer > 99 for layer in layers):
            raise ValueError(f"{target}: invalid GAIA layer number.")

        if not 1 <= sand_class <= 99:
            raise ValueError(f"{target}: invalid sand class.")

        if not np.all(np.isfinite(thicknesses)) or np.any(thicknesses <= 0):
            raise ValueError(
                f"{target}: thicknesses must be finite and positive."
            )

        resolved[target] = {
            "layers": layers,
            "thicknesses": thicknesses,
            "variables": [
                gaia_sand_ratio_variable(layer, sand_class)
                for layer in layers
            ],
        }

    return resolved

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