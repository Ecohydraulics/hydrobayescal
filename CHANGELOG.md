# Changelog

All notable changes to HydroBayesCal are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2026-07-26

### Added
- **Roughness-identifiability diagnostic** at the pre-BAL stage. After the initial
  full-complexity design, `diagnose_roughness_identifiability()` /
  `log_roughness_identifiability()` (in `hydroBayesCal.function_pool`, invoked from
  `templates/prebal_telemac_error_analysis.py`) classify whether bottom roughness is
  the identifiable / efficient calibration knob from the sign pattern of the
  depth-vs-velocity residuals at the calibration points. Anti-correlated residuals
  (one quantity too high, the other too low) confirm a roughness error and indicate
  the direction to move it; **correlated** residuals (both too high or both too low)
  emit a **warning** that roughness alone is non-identifiable (its optimum will pin
  at a prior bound) and that a second parameter (e.g. velocity diffusivity, boundary
  friction, or the turbulence closure) is needed. The check is report-only
  (sampling and parameters are untouched), model-agnostic (TELEMAC, OpenFOAM, ...),
  and works for single- and multi-flow calibrations. Documented in the workflow docs.

## [1.1.0]

### Added
- Multi-flow TELEMAC calibration (`MultiflowTelemacModel`, `bal_telemac_multiflow.py`):
  calibrate one shared parameter set jointly against several steady discharges.

### Fixed
- `rewrite_steering_file` now matches both `KEYWORD = value` and `KEYWORD : value`.
- `update_json_file` normalises calibration-point keys to `str`, so per-run model
  outputs accumulate instead of overwriting.
- Replaced the NumPy 2-removed `np.in1d` with `np.isin` in `select_indexes`.

## [1.0.0]

- First tagged release of the surrogate-assisted Bayesian calibration framework
  (Gaussian Process Emulator + Bayesian Active Learning) for TELEMAC, OpenFOAM and
  Delft3D-FLOW.

[1.2.0]: https://github.com/Ecohydraulics/hydrobayescal/releases/tag/v1.2.0
[1.1.0]: https://github.com/Ecohydraulics/hydrobayescal/releases/tag/v1.1.0
[1.0.0]: https://github.com/Ecohydraulics/hydrobayescal/releases/tag/v1.0.0
