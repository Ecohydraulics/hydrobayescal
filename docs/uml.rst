.. UML / architecture

Code Structure
==============

HydroBayesCal is organised into three cooperating layers, tied together by a
small set of driver scripts:

1. **Full-complexity model bindings** — run the numerical model and extract its
   outputs at the calibration points. A solver-agnostic abstract base class,
   ``HydroSimulations`` (in :mod:`hydroBayesCal.hysim`), defines the contract;
   each solver provides a concrete subclass (``TelemacModel``,
   ``OpenFOAMModel``, ``Delft3DModel``).

2. **Surrogate model and Bayesian Active Learning** — the
   :mod:`hydroBayesCal.surrogate` package builds Gaussian Process Emulators
   (single- and multi-output) and performs Bayesian inference and sequential
   design (BAL).

3. **Post-processing** — the :mod:`hydroBayesCal.visualize` package plots
   calibration results (``BayesianPlotter``), and
   :func:`hydroBayesCal.extract_results` samples 2D/3D results at arbitrary
   points, inside and outside the calibration workflow (see
   :doc:`usage-telemac`).

The driver scripts ``src/hydroBayesCal/drivers/bal_telemac.py``, ``src/hydroBayesCal/drivers/bal_openfoam.py``
and ``src/hydroBayesCal/drivers/bal_delft3d.py`` wire these layers
together and read all user input from a configuration file. The rendered UML
class diagram is maintained in the repository under ``UML/BayesCal-UML.pdf``
(LaTeX/TikZ source next to it).

Architecture at a glance (UML)
------------------------------

The base class owns everything common to a calibration — calibration
parameters, observations and their variances, and the standard result-folder
layout — so each binding only implements the two solver-specific steps
(*running* the model and *processing* its output). This is what keeps the
TELEMAC, OpenFOAM and Delft3D-FLOW workflows aligned.

.. mermaid::

   classDiagram
       class HydroSimulations {
           <<abstract>>
           +model_dir, res_dir, control_file
           +calibration_parameters, param_values
           +observations, variances, measurement_errors
           +calibration_folder, restart_data_folder
           +run_multiple_simulations()*
           +output_processing()*
           +update_model_controls()
           +set_observations_and_variances()
       }
       class TelemacModel {
           +tm_xd  "Telemac2d / Telemac3d"
           +friction_file (.tbl)
           +gaia_steering_file
           +run_multiple_simulations()
           +output_processing()
           +extract_data_point()
       }
       class OpenFOAMModel {
           +solver_name  "interFoam"
           +alpha_water_name, control_points
           +run_multiple_simulations()
           +output_processing()
       }
       class OpenFOAMController {
           +decompose_parallel_case()
           +run_simulation()
           +extract_fields_from_vtk()
       }
       class Delft3DModel {
           +env_script  "env.sh"
           +roughness_formulation  "Manning / Chezy / ..."
           +run_multiple_simulations()
           +output_processing()
       }
       class Delft3DController {
           +update_mdf_parameter()
           +ensure_netcdf_output()
           +run_simulation()
           +extract_map_results()
       }
       class BayesianPlotter {
           +plot_posterior_updates()
           +plot_bme_re()
           +evaluate_calibration()
           +observed_vs_modeled_compare()
       }
       class extract {
           <<module>>
           +extract_results(result_file, variable, x, y, z)
       }
       HydroSimulations <|-- TelemacModel : implements
       HydroSimulations <|-- OpenFOAMModel : implements
       HydroSimulations <|-- Delft3DModel : implements
       OpenFOAMModel ..> OpenFOAMController : uses
       Delft3DModel ..> Delft3DController : uses
       extract ..> TelemacModel : standalone twin of extract_data_point()

The calibration pipeline
------------------------

The drivers orchestrate the surrogate-assisted calibration loop. Experimental
design and parameter sampling are delegated to `BayesValidRox
<https://pages.iws.uni-stuttgart.de/inversemodeling/bayesvalidrox/>`_; the GPE
training, Bayesian inference and training-point selection live in
:mod:`hydroBayesCal.surrogate`.

.. mermaid::

   flowchart TD
       CFG[Configuration file] --> ED["setup_experiment_design()<br/>BayesValidRox Input + ExpDesigns"]
       ED -->|initial collocation points| RUN["run_complex_model()<br/>HydroSimulations.run_multiple_simulations"]
       RUN -->|model_evaluations| TRAIN["Train GPE<br/>surrogate.gpe_gpytorch / gpe_skl"]
       TRAIN --> INFER["Bayesian inference<br/>surrogate.bal_functions.BayesianInference"]
       INFER --> CONV{max_runs reached<br/>or converged?}
       CONV -->|no| SD["Select new training point<br/>SequentialDesign (BME / relative entropy)"]
       SD -->|new collocation point| RUN
       CONV -->|yes| OUT[(auto-saved-results-HydroBayesCal:<br/>posteriors, GPEs, outputs)]

Package layout
--------------

.. code-block:: text

   src/hydroBayesCal/
   ├── hysim.py                 # HydroSimulations: abstract base class (the binding contract)
   ├── function_pool.py         # shared helpers (subprocess, IO, logging, mesh utilities)
   ├── extract.py               # extract_results(): standalone 2D/3D point extraction
   │                            #   (TELEMAC SELAFIN and OpenFOAM VTK)
   ├── telemac/
   │   ├── control_telemac.py   # TelemacModel(HydroSimulations)
   │   ├── config_telemac.py    # TELEMAC/GAIA keyword templates
   │   └── pputils/             # SELAFIN result-file IO (ppmodules)
   ├── openfoam/
   │   └── control_openfoam.py  # OpenFOAMModel(HydroSimulations) + OpenFOAMController
   ├── delft3d/
   │   └── control_delft3d.py   # Delft3DModel(HydroSimulations) + Delft3DController
   ├── drivers/                 # runnable drivers + configs, shipped as package data
   │   ├── bal_telemac.py / bal_openfoam.py / bal_delft3d.py    # entry-point drivers
   │   ├── config_Telemac.py / config_OpenFOAM.py / config_Delft3D.py
   │   └── telemac_extract.py, plot_posteriors.py, assess_calibration.py, ...
   ├── surrogate/               # owned GPE + BAL implementation
   │   ├── gpe_gpytorch.py      # GPyTraining / MultiGPyTraining (single- & multi-output GP)
   │   ├── gpe_skl.py           # scikit-learn GP training
   │   ├── bal_functions.py     # BayesianInference, SequentialDesign
   │   └── exploration.py       # parameter-space exploration for BAL
   ├── visualize/               # BayesianPlotter, assembled from thematic plot mixins
   │   ├── plotter.py           # BayesianPlotter (public API)
   │   ├── posterior_plots.py   # prior/posterior distributions
   │   ├── bal_plots.py         # BME/RE evolution, BME surfaces
   │   ├── model_comparison_plots.py  # surrogate vs complex model vs observations
   │   ├── metrics_plots.py     # metric evolution and per-location exports
   │   ├── calibration_assessment.py  # multi-model calibration assessment
   │   └── axis_utils.py        # shared axis/tick/limit helpers
   └── utils/, doepy/           # logging/config, design-of-experiments helpers

The drivers are scripts rather than importable modules, so they are shipped as
package *data* and are meant to be copied next to a configuration file before
being run: ``hydroBayesCal.copy_driver("bal_telemac.py", my_run_dir)``. Running a
source checkout's copy in place works just as well.

.. note::

   BayesValidRox is used **only** for the experimental design / parameter
   sampling (``Input`` and ``ExpDesigns``). The Gaussian-process emulators and
   the Bayesian active-learning logic are maintained in-tree under
   :mod:`hydroBayesCal.surrogate`.

A detailed description of each module's API is available under
:doc:`codedocs`.
