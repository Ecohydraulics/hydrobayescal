.. UML / architecture

Code Structure
==============

HydroBayesCal is organised into two cooperating layers, tied together by a small
set of driver scripts:

1. **Full-complexity model bindings** — run the numerical model and extract its
   outputs at the calibration points. A solver-agnostic abstract base class,
   ``HydroSimulations`` (in :mod:`hydroBayesCal.hysim`), defines the contract;
   each solver provides a concrete subclass (``TelemacModel``,
   ``OpenFOAMModel``).

2. **Surrogate model and Bayesian Active Learning** — the
   :mod:`hydroBayesCal.surrogate` package builds Gaussian Process Emulators
   (single- and multi-output) and performs Bayesian inference and sequential
   design (BAL).

The driver scripts ``bal_telemac.py`` and ``bal_openfoam.py`` wire these layers
together and read all user input from a configuration file.

Architecture at a glance (UML)
------------------------------

The base class owns everything common to a calibration — calibration
parameters, observations and their variances, and the standard result-folder
layout — so each binding only implements the two solver-specific steps
(*running* the model and *processing* its output). This is what keeps the
TELEMAC and OpenFOAM workflows aligned.

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
       HydroSimulations <|-- TelemacModel : implements
       HydroSimulations <|-- OpenFOAMModel : implements
       OpenFOAMModel ..> OpenFOAMController : uses

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
   ├── telemac/
   │   ├── control_telemac.py   # TelemacModel(HydroSimulations)
   │   ├── config_telemac.py    # TELEMAC/GAIA keyword templates
   │   └── pputils/             # SELAFIN result-file IO (ppmodules)
   ├── openfoam/
   │   └── control_openfoam.py  # OpenFOAMModel(HydroSimulations) + OpenFOAMController
   ├── surrogate/               # owned GPE + BAL implementation
   │   ├── gpe_gpytorch.py      # GPyTraining / MultiGPyTraining (single- & multi-output GP)
   │   ├── gpe_skl.py           # scikit-learn GP training
   │   ├── bal_functions.py     # BayesianInference, SequentialDesign
   │   └── exploration.py       # parameter-space exploration for BAL
   └── utils/, plots/, doepy/   # logging/config, plotting, design-of-experiments helpers

   bal_telemac.py / bal_openfoam.py   # entry-point drivers (read the config, run the loop)

.. note::

   BayesValidRox is used **only** for the experimental design / parameter
   sampling (``Input`` and ``ExpDesigns``). The Gaussian-process emulators and
   the Bayesian active-learning logic are maintained in-tree under
   :mod:`hydroBayesCal.surrogate`.

A detailed description of each module's API is available under
:doc:`codedocs`.
