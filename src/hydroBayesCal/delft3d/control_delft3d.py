"""
Delft3D-FLOW binding for HydroBayesCal -- **planned, not yet implemented**.

This module is a placeholder that mirrors the TELEMAC
(:mod:`hydroBayesCal.telemac.control_telemac`) and OpenFOAM
(:mod:`hydroBayesCal.openfoam.control_openfoam`) bindings. It defines the
intended public interface for coupling HydroBayesCal to the structured-grid
**Delft3D-FLOW** engine (Deltares) so that the coupling can be implemented
incrementally without changing the surrogate / Bayesian-active-learning layer.

The :class:`Delft3DModel` class subclasses
:class:`hydroBayesCal.hysim.HydroSimulations`; the Python attribute names are
shared across solvers, while the *string and file conventions* below are
Delft3D-specific and must be preserved when the binding is filled in:

* ``<case>.mdf`` -- master definition FLOW file (the control file); the engine
  is launched through ``config_d_hydro.xml`` and the ``d_hydro`` executable.
* Bed roughness via Chézy / Manning / White-Colebrook (``.rgh`` file or
  ``Roughness`` keywords in the ``.mdf``); eddy viscosity/diffusivity
  ``Vicouv`` / ``Dicouv``.
* ``trim-<case>.dat`` / ``trim-<case>.def`` -- NEFIS map (field) output.
* ``trih-<case>.dat`` / ``trih-<case>.def`` -- NEFIS history (monitoring-point)
  output.

See the :doc:`usage-delft3d <usage-delft3d>` page for the planned workflow.
"""

from hydroBayesCal.hysim import HydroSimulations

#: Marker so callers / tests can detect that the binding is not ready yet.
DELFT3D_BINDING_IMPLEMENTED = False

_NOT_IMPLEMENTED_MSG = (
    "The Delft3D-FLOW binding is planned but not yet implemented. "
    "Use the TELEMAC (hydroBayesCal.telemac.control_telemac.TelemacModel) or "
    "OpenFOAM (hydroBayesCal.openfoam.control_openfoam.OpenFOAMModel) bindings, "
    "or contribute the Delft3D-FLOW implementation in "
    "hydroBayesCal.delft3d.control_delft3d."
)


class Delft3DModel(HydroSimulations):
    """
    Placeholder Delft3D-FLOW model wrapper (planned).

    Defines the intended constructor signature and interface but raises
    :class:`NotImplementedError`. Instantiating it documents the Delft3D-specific
    configuration the binding will need; it does not run a simulation.

    Parameters
    ----------
    control_file : str
        Master definition FLOW file, default ``"control.mdf"`` (Delft3D-FLOW
        convention ``<case>.mdf``).
    d_hydro_config : str
        Runtime configuration passed to the ``d_hydro`` launcher, default
        ``"config_d_hydro.xml"``.
    flow_executable : str
        Name of the Delft3D-FLOW launcher on ``PATH``, default ``"d_hydro"``.
    roughness_formulation : str
        Bed-roughness law used for the calibration parameters
        (``"Chezy"``, ``"Manning"`` or ``"WhiteColebrook"``).
    map_file_base, history_file_base : str
        Base names of the NEFIS map (``trim-<case>``) and history
        (``trih-<case>``) output files.
    **kwargs
        Common :class:`~hydroBayesCal.hysim.HydroSimulations` parameters
        (``model_dir``, ``res_dir``, ``calibration_pts_file_path``,
        ``calibration_parameters``, ``param_values``, ``calibration_quantities``,
        ``init_runs``, ``max_runs`` ...).

    Raises
    ------
    NotImplementedError
        Always -- the binding is not implemented yet.
    """

    def __init__(
        self,
        control_file="control.mdf",
        d_hydro_config="config_d_hydro.xml",
        flow_executable="d_hydro",
        roughness_formulation="Manning",
        map_file_base="trim",
        history_file_base="trih",
        *args,
        **kwargs,
    ):
        # Keep the Delft3D-specific configuration on the instance so the intended
        # interface is documented, then make the not-yet-implemented state
        # explicit instead of silently constructing a non-functional model.
        self.control_file = control_file
        self.d_hydro_config = d_hydro_config
        self.flow_executable = flow_executable
        self.roughness_formulation = roughness_formulation
        self.map_file_base = map_file_base
        self.history_file_base = history_file_base
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def run_multiple_simulations(self, *args, **kwargs):
        """Run the Delft3D-FLOW experimental-design simulations (planned)."""
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def output_processing(self, *args, **kwargs):
        """Extract calibration quantities from NEFIS map/history output (planned)."""
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)
