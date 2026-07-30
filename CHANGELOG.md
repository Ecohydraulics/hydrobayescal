# Changelog

All notable changes to HydroBayesCal are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.0] - 2026-07-29

### Added
- **Calibrated parameter sets derived from the BAL posterior**
  (`hydroBayesCal.surrogate.posterior_analysis`, driver
  `templates/derive_calibrated_parameters.py`). Bayesian active learning selects its
  training points by information gain, so the last training point of a calibration is
  not a calibration result, and nothing downstream previously turned the posterior
  into a parameter set. The new module reports, per calibration parameter, the peak of
  its own posterior marginal, the credible interval, the posterior-to-prior variance
  reduction, and flags for parameters that are pinned at a prior bound, multimodal or
  not identifiable at all. The peak is read directly off a histogram of the accepted
  samples with no smoothing, so it is the peak of the posterior rather than of a fitted
  curve; the bin count follows from the sample size and the posterior spread
  (`marginal_bin_count`: Freedman-Diaconis with a Sturges floor, capped at 25 samples
  per bin), chosen by measuring the located peak against known modes for narrow, broad,
  skewed and bound-pinned marginals at 400, 2000 and 20000 accepted samples. It
  additionally computes the
  **joint** posterior optimum, detects distinct posterior modes, and issues an explicit
  **equifinality verdict** on whether the vector assembled from the independent
  marginal peaks is jointly plausible: for correlated parameters that combination can
  sit at near-zero joint posterior density even though every component is individually
  optimal. Candidate parameter sets are exported to
  `restart_data/user-collocation-points.csv`, so the final full-complexity runs use the
  existing `user_param_values` path and `assess_calibration.py`.
- **Per-iteration tracking of the parameter optima** in the BAL loop of
  `bal_telemac.py`, `bal_openfoam.py` and `bal_delft3d.py` (and hence
  `bal_telemac_multiflow.py`), stored under additive `BAL_dictionary.pkl` keys
  (`marginal_optima`, `marginal_hdi`, `variance_reduction`, `identifiability_flags`,
  `marginal_joint_gap`), plus `calibration_parameters` and `param_values` so the result
  file is self-describing. The diagnostic never raises and never touches sampling.
- **Two plots** in `visualize.posterior_plots`:
  `plot_parameter_optimum_convergence` (each parameter's own optimum, credible band and
  bound pinning against the number of training points, plus the variance-reduction
  trajectory) and `plot_marginal_vs_joint` (joint posterior density percentile of the
  marginal-peak vector). Both reconstruct their series from the stored posteriors when
  a result file predates the new keys, so archived runs need no re-run.
- `best_estimate_value="posterior_marginal_peak"` in `plot_posterior_updates`, and
  `templates/plot_posteriors.py` now uses it. The previous estimate tied the reported
  optimum to the bin count used for *drawing*, resolving it only to one drawing bin
  width, a tenth of the calibration range at the shipped `bins=10`.
- An optional `extraction` block in the TELEMAC configuration
  (`output_extraction_time`, `n_last`, also accepting the `n` spelling), so the
  extraction timing no longer has to be edited into the driver. This upstreams a
  setting that previously existed only in the Ering example copy.
- `save_calibration_data` in the OpenFOAM and Delft3D bindings now also writes the
  per-iteration equifinality diagnostics into `bayesian_scores.csv` and the
  per-parameter optima into a new `marginal_optima.csv`. Result dictionaries without
  those keys are handled unchanged.
- **Opt-in surrogate uncertainty in the Bayesian inference**
  (`sampling['include_surrogate_error']`, default `False`). The inference previously
  ignored the GPE predictive variance while the BAL utility used it, making the
  posterior sharper than the surrogate supports. Enabling the flag passes
  `surrogate_output['std']` as `model_error`, backed by a new diagonal fast path in
  `BayesianInference` (`O(MC * n_obs)` instead of `O(MC * n_obs**2)` memory; the dense
  path needs several GB per array at `prior_samples=25000`). `calibration['gpe_error']`
  and the measurement-error fraction are now real constructor arguments of
  `HydroSimulations` instead of silent defaults, and the driver warns when the flat
  `gpe_error` and the actual surrogate variance would be counted twice. Both settings
  are threaded through the multi-discharge driver as well.
- **A test suite** (`tests/`, 65 tests), the first in the repository, covering the
  posterior analysis on synthetic posteriors, the likelihood paths, the multi-output
  GPE task layouts and the `user-collocation-points.csv` contract. It runs in minutes
  without a solver. `pyproject.toml` gains `[tool.pytest.ini_options] testpaths`, so
  the bare `pytest` that CONTRIBUTING has always advertised now works from the
  repository root.
- `CLAUDE.md`, a repository guide (commands, architecture, solver-facing conventions)
  for AI coding assistants.
- **The LaTeX system dependencies of the plotting code are now documented**
  (`docs/installation.rst`, cross-referenced from CONTRIBUTING and the workflow page).
  `BayesianPlotter` renders all text through LaTeX, which pip cannot install, so on a
  machine without `type1cm` (`texlive-latex-extra`), `cm-super` and `dvipng` every
  plotting call failed with a `RuntimeError` quoting a missing `.sty` file and nothing
  said why. The new section states the requirement functionally, then gives commands
  for Debian/Ubuntu, Fedora/RHEL, openSUSE, Arch, macOS, Windows and conda-forge, plus
  the universal `tlmgr` fallback, a one-line verification command, and how to switch
  the LaTeX mode off where it cannot be installed at all.

### Fixed
- **Multi-output GPE output columns could be silently mis-ordered.**
  `MultiGPyTraining.predict_` dispatched on `len(gp_list)`, which is ambiguous when the
  number of calibration points equals the number of quantities (or is 1): a model
  trained with `multitask_selection="variables"` then took the `"locations"` writer, so
  the predicted columns no longer matched `observations`/`variances` and the likelihood,
  BME, RE and posterior were all wrong with no error raised. Dispatch now uses an
  explicit `task_mode` set by each `train_tasks_*` method, surrogates pickled earlier
  fall back to the old heuristic with a warning, and an unwritten output column now
  raises instead of silently predicting zeros. **Results differ for affected runs.**
- **`multitask_selection` of `"all"` or `"locations"` could not run BAL at all.**
  `multitask_cov_list` was bound only in the `"variables"` branch of `predict_` but
  dereferenced unconditionally, so both raised `UnboundLocalError` at the first
  sequential-design step. All three task layouts now return the documented covariance;
  `"locations"` degrades to a diagonal per-location covariance with a warning, since
  cross-quantity correlation is not part of that task layout.
- **`train_tasks_locations` silently dropped every calibration quantity after the
  second** (hard-coded `Y[:, ::2]`, `range(2)` and stride-2 writes). Now generalised to
  any number of quantities, with a guard on the output width.
- **One shared `MultitaskGaussianLikelihood` (and kernel) across all sub-models** meant
  the noise and lengthscales fitted for the last location or quantity were applied to
  all of them. Each sub-model now trains and predicts with its own copy. **Results
  differ for existing multi-output runs.**
- `docs/conf.py` still declared version 1.1.0 while the package was at 1.2.0. The two
  are back in sync, as CONTRIBUTING requires.
- The `Source` project URL pointed at a non-canonical repository. It now matches the
  git remote and the Trusted Publishing configuration
  (`Ecohydraulics/hydrobayescal`), so the link shown on PyPI resolves correctly.
- **The documentation build was broken in two ways.** `.readthedocs.yaml` sat in
  `docs/`, but Read the Docs only searches the repository root, so the file was found
  only through a custom path setting; it now lives at the root. And `formats: [pdf]`
  ran the Sphinx LaTeX builder, which must rasterise the mermaid diagrams in
  `docs/uml.rst` with `mmdc` (mermaid-cli). That tool is absent from the build
  environment, so with `fail_on_warning: true` the PDF step failed the whole build.
  The documentation is now HTML only; mermaid renders in the browser and is
  unaffected. The comment claiming the PDF was linked from the docs was wrong: the
  only PDF referenced is `UML/BayesCal-UML.pdf`, a file committed in the repository.

### Changed
- `docs/workflow.rst` no longer states that the last BAL iteration "corresponds to the
  supposedly best solution", and gains a step on deriving the calibrated parameter sets
  from the posterior.
- **The documentation terminology is now consistent**: the model parameters being
  adjusted are always *calibration parameters*, and the measured variables the model is
  fitted against are always *calibration targets* (previously "calibration quantities",
  and in one place mislabelled a calibration parameter). Everything read out of the
  model results is an *extracted variable*, a superset of the calibration targets. A new
  terminology section in the introduction defines all three. The configuration keys keep
  their existing names (`calibration_quantities`, `extraction_quantities`), and the
  section states that mapping explicitly, so no configuration file needs changing.
- The Ering example driver and plotting script (`examples/Telemac/Hydromorphodynamic/
  Ering/bal_telemac.py`, `main_plots.py`) had drifted from the templates they were
  copied from. They are now byte-identical to `templates/bal_telemac.py` and
  `templates/plot_posteriors.py` apart from their default configuration file name.
- Separate posterior modes now require a genuine drop in posterior density between
  them, both for the per-parameter marginals and for the joint posterior. A continuous
  trade-off ridge is one connected family of solutions, not several modes, and a
  Gaussian mixture needs multiple components to follow a curved ridge; connectivity is
  therefore tested along the posterior samples themselves rather than along the
  straight line between candidates, which leaves a curved ridge.
- The equifinality diagnostic gained a `coupled` verdict for the case where the
  marginal-peak vector does lie in a dense region but the parameters are too tightly
  correlated for the marginals to determine the combination. Previously that case was
  reported as `inconsistent` with a message asserting a low posterior density that the
  same message contradicted.

### Removed
- Kernel-density estimation. The per-parameter optima are now read off the posterior
  samples directly (see above), and the joint posterior density used by the
  equifinality diagnostic is a nearest-neighbour estimate, which needs no bandwidth and
  does not spread mass across a calibration bound. A symmetric kernel biases exactly
  the case that matters most here: against a prior bound it pulls the apparent peak
  inward, so a parameter pinned at its limit stops looking pinned.
- Prominence-based counting of marginal modes, replaced by the valley criterion above.
  Two bins tied at the maximum were each assigned nearly the full height as their
  prominence, because neither is higher than the other, so a perfectly unimodal
  marginal was reported as bimodal; ties are common at realistic bin counts. The
  replacement also pads the histogram before searching for local maxima, so a peak in
  the first or last bin is visible at all, which is what a bound-pinned parameter and a
  bimodal marginal with a lobe at the edge of the sampled range look like.

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

[1.3.0]: https://github.com/Ecohydraulics/hydrobayescal/releases/tag/v1.3.0
[1.2.0]: https://github.com/Ecohydraulics/hydrobayescal/releases/tag/v1.2.0
[1.1.0]: https://github.com/Ecohydraulics/hydrobayescal/releases/tag/v1.1.0
[1.0.0]: https://github.com/Ecohydraulics/hydrobayescal/releases/tag/v1.0.0
