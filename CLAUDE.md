# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

HydroBayesCal performs surrogate-assisted Bayesian calibration of expensive hydro- and morphodynamic
models. A Gaussian Process Emulator (GPE) is trained on a small initial design of full-complexity
simulations, then refined by Bayesian Active Learning (BAL), which iteratively picks the next
parameter set that maximises relative entropy (dkl) or Bayesian model evidence (bme). Solver
bindings exist for TELEMAC (2D/3D, incl. GAIA and multi-discharge), OpenFOAM and Delft3D-FLOW.

`src` layout, single `pyproject.toml`, no `setup.py`. Experimental design and prior sampling are
delegated to `bayesvalidrox`; the GPEs and the BAL logic are in-tree.

## Commands

```bash
pip install -e ".[dev,docs,mesh]"       # editable dev install (Python >= 3.10, tested 3.10-3.12)
pre-commit install                       # once per clone (20 MB file-size gate)
pytest                                   # test runner of record (no test files exist in-tree yet)
sphinx-build -b html -W docs docs/_build/html
ruff check src                           # linter declared in the dev extra
python -m build && twine check dist/*    # local release sanity check
```

Running a calibration (drivers ship WITH the installed package as data; copy one next to your config with `hydroBayesCal.copy_driver`, or run a source checkout's copy directly):

```bash
python src/hydroBayesCal/drivers/bal_telemac.py           --config src/hydroBayesCal/drivers/config_Telemac.py
python src/hydroBayesCal/drivers/bal_telemac_multiflow.py --config <config with a `multiflow` block>
python src/hydroBayesCal/drivers/bal_openfoam.py          --config src/hydroBayesCal/drivers/config_OpenFOAM.py
python src/hydroBayesCal/drivers/bal_delft3d.py           --config src/hydroBayesCal/drivers/config_Delft3D.py
python src/hydroBayesCal/drivers/prebal_telemac_error_analysis.py --config <config>   # pre-BAL diagnostics only
```

TELEMAC must be on `PATH` before a TELEMAC run: the binding shells out to `telemac2d.py` /
`telemac3d.py`. Source `/home/modelling/telemac-v911/telemac-mascaret/configs/pysource.debian12.sh`
(or adapt `env-scripts/activateHBCtelemac.sh`, which sources the TELEMAC config plus a venv).

Releases are automated: bump `version` in `pyproject.toml` (and `release`/`version` in
`docs/conf.py`), then publish a GitHub Release tagged `vX.Y.Z`; `.github/workflows/publish.yml`
uploads to PyPI via Trusted Publishing. Never re-use a version, PyPI uploads are immutable.

## Architecture

### The abstract binding layer

`src/hydroBayesCal/hysim.py` defines `HydroSimulations(ABC)`, the single contract every solver
binding implements. It owns everything solver-independent: parsing the calibration-points CSV into
`observations` / `variances` / `measurement_errors`, the parameter dictionary and `ndim`, and the
results-folder layout. Subclasses must implement `run_multiple_simulations()` and
`output_processing()`; `extract_data_point()`, `run_single_simulation()` and
`update_model_controls()` are optional overrides that raise `NotImplementedError` by default.

Concrete bindings: `telemac/control_telemac.py` (`TelemacModel`, by far the most complete),
`openfoam/control_openfoam.py`, `delft3d/control_delft3d.py`. `telemac/multiflow_telemac.py`
(`MultiflowTelemacModel`) is deliberately *not* a `HydroSimulations` subclass: it composes several
`TelemacModel` instances (one per steady discharge) and re-exposes the same attribute surface the
driver reads, concatenating per-flow observations and outputs into one combined space.

Key data contract, assumed everywhere downstream: `model_evaluations` is a 2D array of shape
`[num_runs, nloc * num_calibration_quantities]`, with quantities interleaved per location (two
quantities, two locations => columns 1-2 are location 1, columns 3-4 are location 2). Keep this
ordering when touching extraction, `rearrange_array()` or the surrogate code.

The calibration-points CSV must carry `Point, X, Y` plus a `<TARGET>_DATA` and `<TARGET>_ERROR`
column per calibration target (resolved case-insensitively). Total variance is
`measurement_error**2 + gpe_error**2 + model_structural_error**2 + site_specific_error**2`. The
three relative terms default to 0.10 / 0.0 / 0.0 as fractions of the measured value; the last comes
from the `_ERROR` column in physical units. `gpe_error` is 0.0 because the drivers default to
`include_surrogate_error=True`, which feeds the real GPE predictive standard deviation into the
likelihood as `model_error`; a non-zero `gpe_error` would count that twice.

Terminology in the docs: **calibration parameters** are the model parameters being adjusted,
**calibration targets** are the measured variables fitted against. The config keys keep the older
`calibration_quantities` / `extraction_quantities` names.

### The driver scripts

`src/hydroBayesCal/drivers/bal_*.py` are the orchestrators, not library code. Each one does the same four steps and
they are near-duplicates across solvers, so a change to the workflow usually has to be mirrored:

1. `load_config()` imports the `--config` Python file as a module (dicts `paths`,
   `hydrodynamic_simulation`, `morphodynamic_simulation`, `calibration`, `sampling`, `execution`).
2. `setup_experiment_design()` builds a `bayesvalidrox` `Input`/`ExpDesigns` and samples the initial
   collocation points (`sobol`, `latin_hypercube`, `halton`, `random`, ..., or user-supplied CSV).
3. `run_complex_model()` runs the initial full-complexity simulations (or reloads stored outputs in
   `only_bal_mode`).
4. `run_bal_model()` is the BAL loop: train GPE, pickle it, predict over the prior, run
   `BayesianInference` (rejection sampling for BME/RE/ELPD/IE), pick the next training point with
   `SequentialDesign`, run the model there, append, repeat.

`bal_telemac_multiflow.py` imports `setup_experiment_design`, `run_complex_model` and
`run_bal_model` from `bal_telemac.py` verbatim and only swaps in `MultiflowTelemacModel`, so
`bal_telemac.py` is the canonical driver, keep the two in sync.

### Execution modes

`complete_bal_mode` and `only_bal_mode` (from the `execution` config block) select the task:

| complete_bal_mode | only_bal_mode | task |
|---|---|---|
| True | False | full surrogate-assisted calibration |
| False | False | initial full-complexity runs only, outputs stored as JSON |
| True | True, `init_runs == max_runs` | rebuild surrogate from stored runs, no BAL |
| True | True, `init_runs < max_runs` | rebuild surrogate from stored runs, then continue BAL |

`only_bal_mode` reads `restart_data/initial-collocation-points.csv` and the stored
`initial-model-outputs.json`, which is why `extraction_quantities` should be a superset of
`calibration_quantities`: any extracted quantity can become the calibration target on restart
without re-running the solver.

### Surrogate and BAL internals

* `surrogate/gpe_gpytorch.py`: `GPyTraining` (single quantity, one independent GP per output column)
  and `MultiGPyTraining` (multitask). `multitask_selection` picks the task axis: `"variables"`
  (tasks = calibration quantities, the default), `"locations"`, or `"all"`.
* `surrogate/gpe_skl.py`: `SklTraining`, the scikit-learn alternative (`gp_library="skl"`).
* `surrogate/bal_functions.py`: `BayesianInference` (likelihood, rejection sampling, BME/RE/ELPD/IE)
  and `SequentialDesign` (candidate exploration plus the `dkl`/`bme` utility functions).
* `surrogate/exploration.py`, `doepy/`: candidate generation and design-of-experiments helpers.
* `surrogate/target_agreement.py`: the report-only closing step every `bal_*` driver runs
  after the BAL loop. Modeled vs. measured calibration targets before calibration (the
  initial-design ensemble) and after it (the posterior predictive, i.e.
  `BayesianInference.posterior_output` of the last iteration), the verdict that separates
  a systematic over-/underestimation from scatter, and the figure in
  `visualize/agreement_plots.py`. Stores its series in `BAL_dictionary.pkl` under
  `target_agreement`; `drivers/plot_target_agreement.py` replots them. Never raises.

### Output layout

Everything is written under `<res_dir>/auto-saved-results-HydroBayesCal/`:
`calibration-data/<quantities joined by _>/` (BAL_dictionary.pkl, model outputs),
`restart_data/` (collocation points, initial-model-outputs.json), `surrogate-gpe/<exploit>_<util>/`
(pickled GPEs per iteration), `plots/`. In multiflow runs each flow additionally gets its own
`<res_dir>/flow-<name>/` tree.

### Shared helpers

`function_pool.py` is imported with `from ... import *` by the bindings and drivers, so it also
supplies `os`, `subprocess`, `np`, `math`, `pickle` and `logger` to them. Adding or renaming a
symbol there can silently shadow names in those modules. It holds JSON/CSV output bookkeeping
(`update_json_file`, `update_collocation_pts_file`, `filter_model_outputs`, `rearrange_array`),
raster/mesh utilities, and the report-only roughness-identifiability diagnostic
(`diagnose_roughness_identifiability` / `log_roughness_identifiability`).

Logging goes through `utils/config_logging.py`, which exposes `logger`, `logger_warn` and
`logger_error` and writes `logfile.log`, `warnings.log`, `errors.log` into the current working
directory at import time.

`extract.py` is the only public API surface (`hbc.extract_results(...)`): standalone point
extraction from SELAFIN (2D/3D, with height-above-bed selection via `ELEVATION Z`) or OpenFOAM VTK,
with no model instance or folder conventions involved.

## Conventions

* Solver-facing strings are load-bearing and must survive refactors: TELEMAC `.cas` keywords (matched
  as both `KEYWORD = value` and `KEYWORD : value`), friction-zone names in the `.tbl` file (a
  calibration parameter naming a friction zone must be prefixed `zone`/`Zone`/`ZONE`), GAIA
  `CLASSES SHIELDS PARAMETERS n` (prefixed `gaia` in the config), SELAFIN variable names (16-char,
  e.g. `TURBULENT ENERG.`), OpenFOAM `system/controlDict`, Delft3D `.mdf` keywords. The Python
  attribute names are shared across bindings, the software-facing values are not.
  `telemac/config_telemac.py` maps variable names to TELEMAC vs GAIA and holds the parameter CSVs
  shipped as package data.
* NumPy-style docstrings on public classes and functions (rendered by Sphinx + napoleon).
* Do not commit files larger than 20 MB. Example folders are tracked but their large data files are
  excluded through the auto-generated tail of `.gitignore`; after adding example data run
  `bash env-scripts/update-large-file-ignores.sh` and commit the updated `.gitignore`. Enforced by
  pre-commit locally and by `.github/workflows/check-file-size.yml` server-side.
* Branch off `main` and open a pull request rather than pushing to `main`.
* Update `CHANGELOG.md` (Keep a Changelog format) for user-visible changes.
