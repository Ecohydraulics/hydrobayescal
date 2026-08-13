# Changelog

All notable changes to HydroBayesCal are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.6.0] - 2026-08-13

An OpenFOAM release. Until now a k-epsilon calibration could only vary `Cmu`, because the
parameter dispatch named each supported coefficient in two hardcoded branches. The whole
`kEpsilonCoeffs` set is now available, the dispatch is table-driven so the next
coefficient needs no code, and a misspelled parameter name is caught when the model is
constructed rather than after the initial design has been sampled. TELEMAC and Delft3D
workflows are unaffected apart from the dependency declaration below.

### Added
- **The k-epsilon turbulence coefficients are calibration parameters in the OpenFOAM
  binding.** `sigmaEps` was contributed by Federica Scolari; the dispatch it was added
  to is now table-driven, so `Cmu`, `C1`, `C2`, `C3`, `sigmak` and `sigmaEps` are all
  supported and listed in `openfoam.control_openfoam.KEPSILON_COEFFS`. Config names are
  matched case-insensitively, but the key written into `constant/turbulenceProperties`
  comes from that table rather than from what the user typed, because OpenFOAM's
  dictionary lookup is case-sensitive and falls back to a built-in default for a
  coefficient it does not find. A coefficient absent from the case template's
  `kEpsilonCoeffs` subdictionary still stops the run, which is deliberate: writing
  nothing would leave every simulation on the same value while the surrogate was told
  the parameter had changed.

### Changed
- **Unknown OpenFOAM `calibration_parameters` are rejected when the model is
  constructed**, alongside the existing check on `calibration_quantities`. The dispatch
  in `run_multiple_simulations` still raises for a name it cannot route, but that fires
  only after the experimental design has been sampled and a case directory copied, so a
  typo used to cost a full setup before it was reported.
- The `Cmu` and `sigmaEps` branches in `OpenFOAMController.update_model_controls` and in
  `OpenFOAMModel.run_multiple_simulations` were verbatim copies of each other. Both
  dispatch sites now read the coefficient table, so a further coefficient needs no code.
  No behaviour change for `Cmu`, `ks` or `sigmaEps`.

### Fixed
- The Read the Docs build failed with `ModuleNotFoundError: No module named
  'threadpoolctl'`. That build installs only `docs/requirements-docs.txt` and mocks the
  heavy runtime stack through `autodoc_mock_imports`, and `surrogate/initial_design.py`
  imports `threadpoolctl` directly, which was in neither list. It is now mocked for the
  docs build and declared in `[project] dependencies`, where a direct import belongs: it
  arrives with scikit-learn in practice, but relying on that is implicit and the 1.5.0
  wheel therefore under-declares what it imports.

## [1.5.0] - 2026-08-11

Two ends of the same problem: whether a calibration finds the *global* maximum of the
joint posterior, or converges confidently onto a local one. At the front end, the initial
design is now sized from the number of calibration parameters and grown in Sobol blocks
until it is measurably good enough. At the back end, the maximum of the joint posterior is
located by mode-seeded refinement rather than by picking the best prior draw, and an
explicit rule decides whether the per-parameter marginal optima may be reported as a
parameter set at all.

### Added
- `hydroBayesCal.surrogate.initial_design`, a solver-agnostic, report-only module for the
  initial design:
  - `recommended_init_runs` sizes it as ten runs per calibration parameter, floored at 16
    and rounded up to a power of two, because the unscrambled Sobol sequence chaospy
    generates is balanced at `2**m`. It runs before the first simulation and warns when
    the configured `init_runs` is below the recommendation. It is never enforced and
    never silently raised: spending days of extra CPU time is the modeller's decision.
  - `sobol_block` extends a design along the same Sobol sequence, using the prefix
    property that makes the first `n` points of a length-`2n` sequence exactly the
    length-`n` sequence, so growing a design never discards a simulation. The prefix is
    verified rather than assumed, with a Latin hypercube fallback, because a silently
    reordered design would attach every stored model output to the wrong parameter set.
  - `initial_design_sufficiency` measures, after each block, GP leave-one-out
    predictivity, whether the emulator's error bars are calibrated, whether the implied
    posterior is resolved by enough accepted samples, whether its shape is driven by the
    data rather than by emulator uncertainty, and whether it stopped moving since the
    previous block. Verdict `sufficient` / `marginal` / `insufficient`; never raises.
  - `run_staged_initial_design` runs the ladder. It is capped by `init_runs`, so it can
    only ever save simulations, and the ones it saves become BAL iterations because
    `max_runs` is the total budget. New config keys `sampling['adaptive_init_runs']`
    (default `True`) and `sampling['init_runs_min']`.
- `posterior_analysis.refine_joint_optimum` maximises the surrogate's joint posterior
  over the continuous calibration ranges from several starting points, seeded by the
  highest-density samples **and one representative per detected posterior mode**. Without
  it the reported optimum is the best of finitely many prior draws, quantised to the prior
  sample and unstable between rejection samplings; without the mode seeding it is the top
  of whichever basin held the densest sample, which is not necessarily the deepest one.
  Exposed as `joint_optimum(..., refine=True)`, `analyze_posterior(..., refine=True)` and
  `derive_calibrated_parameters.py --refine`.
- `posterior_analysis.select_calibrated_parameters` decides which parameter set is the
  calibration result. The joint maximum is the default and the marginal peaks are promoted
  only when the posterior says the independence assumption behind stacking them holds:
  parameters effectively uncorrelated *and* the marginal-peak vector within 0.25 posterior
  standard deviations of the joint maximum. A multimodal posterior reports no single set
  and hands every mode representative to the solver to arbitrate. The decision reaches the
  log, the `selected` column of `calibrated-parameter-candidates.csv` and the driver's
  headline output.
- `sampling['bal_exploration_tradeoff']` (default `'auto'`): the sequential design exploits
  the posterior until the per-iteration diagnostic finds more than one well-separated mode,
  then adds exploration for the remaining iterations. Pure exploitation refines the mode it
  started in, which is how a local maximum survives to the end of a calibration.
- `ITERATION_KEYS` gained `joint_optimum`, `joint_log_density` and `posterior_modes`, so
  the convergence of the joint maximum is recoverable from an archived `BAL_dictionary.pkl`
  and is plotted alongside the marginal traces in `parameter_optimum_convergence`. The keys
  are additive; older result files still plot, reconstructed from their stored posteriors.
- `validate_sampling_method` rejects an unusable `parameter_sampling_method` before the
  first simulation instead of several minutes into the run, from inside chaospy. The
  historical `"chebyshev(FT)"` and `"grid(FT)"` spellings are not chaospy rules at all;
  they are now accepted with a warning and mapped to `chebyshev` and `grid`.
- `tests/test_initial_design.py`, plus tests for the refinement and the decision rule in
  `tests/test_posterior_analysis.py`. All run on analytic responses, no solver involved.

### Changed
- `run_multiple_simulations` gained a `start_index` argument in every binding (TELEMAC,
  OpenFOAM, Delft3D, multiflow), so a staged design runs only its new block while run
  numbering, the collocation CSV and the accumulated outputs stay continuous over the
  whole design. Default `0`, i.e. unchanged behaviour.
- `MultiflowTelemacModel.init_runs` is now a property that propagates to the per-flow
  models. Each flow runs the design through its own `TelemacModel`, which reads its *own*
  `init_runs`, so a plain attribute would have left the flows running the first block
  while the combined model believed the design had grown.
- The docs' "Bayesian Calibration Workflow" gained **Step 1: Size and sample the initial
  design**; the former Steps 1 to 4 are now Steps 2 to 5. The `calibrated-parameters`
  label is unchanged, so every cross-reference still resolves. Step 5 now states that the
  calibrated parameter set is the maximum of the joint posterior PDF, how that maximum is
  located, and the rule that vets the marginal optima against it.
- `BayesianInference.rejection_sampling` records the accepted prior rows in `post_index`.
  Rejection sampling is stochastic, so the acceptance cannot be reconstructed from the
  likelihood afterwards, and callers needing per-sample quantities of the accepted set had
  no way to get them.

### Fixed
- `setup_experiment_design` assigned `exp_design.x = complex_model.user_collocation_points`
  unconditionally. Assigning `None` is harmless today, but bayesvalidrox switches the
  design to `'user'` as soon as `x` is set, so this was one library change away from
  silently discarding the configured sampling method. The assignment now happens only when
  user points exist.
- `sampling['gp_library'] = "skl"` could not run at all, in two independent places.
  `run_bal_model` never bound `surrogate_object` in the scikit-learn branch, so the first
  log line raised `UnboundLocalError`, and `SklTraining.predict_` called `predict_` on the
  raw `GaussianProcessRegressor`, which only has `predict(X, return_std=True)`. Both are
  fixed, and `tests/test_gpe_skl.py` now trains and predicts through that path and checks
  the binding in every driver's source. Found while exercising the staged design end to
  end; unrelated to the initial-design work but on the same code path.
- The initial-design gate pins the BLAS thread pool to one thread while fitting its
  Gaussian processes. On a many-core machine, spawning one thread per core for matrices a
  few dozen rows across cost 1.5 s per column instead of 30 ms, i.e. the gate would have
  been slower than useful.

## [1.4.4] - 2026-08-11

A one-line fix with a wide blast radius: the multi-flow TELEMAC driver ignored the
configured output-extraction window, so the *same* configuration calibrated
single-flow and multi-flow was fitted to different data.

### Fixed
- `bal_telemac_multiflow.py` did not read the config's `extraction` block. It calls
  `run_complex_model`, imported from `bal_telemac.py`, without passing
  `output_extraction_time` or `n_last`, so it silently took that function's
  `"mean_last"` default no matter what the configuration asked for. A single-flow run
  on the same config honoured the setting; a multi-flow run averaged the last frames.
  On a model marching to steady state from a dry or pre-wetted start, that folds the
  residual transient into the values the surrogate is trained on, so the two modes
  disagreed about what was being calibrated with nothing in the logs to say so.
  `main()` now reads `extraction` and passes both arguments, exactly as
  `bal_telemac.py` does.

  Only the TELEMAC multiflow driver was affected: the OpenFOAM and Delft3D drivers
  define their own `run_complex_model` and have no extraction window.

### Added
- `test_telemac_drivers_pass_the_configured_extraction_window` parses each TELEMAC
  driver's `main()` and fails if it does not read `config.extraction` and forward
  both arguments. The defect was invisible because omitting an argument that has a
  default is not an error, so a test that only imports or runs the driver cannot see
  it - this one reads the source.

## [1.4.3] - 2026-08-10

Two bugs that made Bayesian Active Learning unreachable for OpenFOAM and Delft3D, plus
the follow-ups to the 1.4.1 review notes. Reported and fixed by Federica Scolari.

### Fixed
- `only_bal_mode=True` raised `FileNotFoundError` for
  `restart_data/initial-collocation-points.csv` before a single BAL iteration could
  start. `HydroSimulations.__init__` has always read that file, but only the TELEMAC
  binding wrote it. It is now written by the shared `_save_all_results`, so OpenFOAM
  and Delft3D can restart too. Reported and fixed by Federica Scolari.
- BAL crashed on its first iteration with "truth value of an array is ambiguous" in
  `save_calibration_data`. The `bayesian_dict.get(key) or [None] * (it + 1)` idiom
  calls `bool()` on the stored value, which is undefined for the multi-element numpy
  arrays `BayesianInference` actually returns (`log_BME` among them). All seven
  occurrences now test `is not None`. Reported and fixed by Federica Scolari.
- `Cs` is no longer overwritten with a hardcoded `0.5` when a rough-wall patch already
  sets it. Only `Ks` is calibrated, so a roughness constant chosen by the case author
  was being discarded silently. Reported and fixed by Federica Scolari.
- A `value nonuniform` list closed by `);` rather than a standalone `;` no longer
  swallows every following patch in the field file. Reported and fixed by Federica
  Scolari.
- `_get_vtm_time` raises instead of falling back to `0.0` when a `.vtm` carries no
  parseable time attribute. The silent fallback degraded the timestep sort back to
  arbitrary order, which is the exact failure the sort exists to prevent. Reported and
  fixed by Federica Scolari.

### Changed
- `save_calibration_data` moved from the OpenFOAM and Delft3D bindings to
  `HydroSimulations`, where `_save_all_results` already lives. The two copies were
  verbatim duplicates, so the ambiguous-truth-value crash above existed twice and was
  fixed once; hoisting fixes Delft3D as well and removes the drift vector. A test now
  fails if either method is overridden in a binding again.

## [1.4.2] - 2026-08-07

The calibration drivers now ship **with the installed package**. Until now
`pip install hydroBayesCal` gave you the library but not the scripts you actually run,
so any downstream tool had to be told where a source checkout lived - an environment
variable pointing at a directory, which breaks as soon as the checkout moves or is
absent. The drivers are now package data, discoverable through a small API, so an
installed package is self-sufficient.

### Added
- `hydroBayesCal.copy_driver(name, dest_dir)` copies a driver next to a config and
  brings any sibling it imports with it (`bal_telemac_multiflow.py` needs
  `bal_telemac.py` beside it). This is the supported way to use a driver: they are
  scripts meant to run from a working directory holding the config, not importable
  modules.
- `hydroBayesCal.drivers_dir()`, `driver_path(name)` and `available_drivers()` for
  callers that only need the location. `driver_path` names what *is* available when
  asked for something that is not, since a typo would otherwise surface much later as
  a confusing missing-file error.

### Changed
- **The drivers moved from `templates/` to `src/hydroBayesCal/drivers/`** so setuptools
  ships them. Running a checkout's copy directly still works, at the new path; the
  documented commands were updated. Nothing about how a driver behaves changed.

## [1.4.1] - 2026-08-05

A correctness release for the OpenFOAM binding, from bugs Federica Scolari found while
running real calibrations. Two of them are silent: a hot-started case ignored the
calibrated roughness altogether, and the extraction could average the wrong timesteps,
so an OpenFOAM calibration could complete and report results that owed nothing to the
parameters it was varying. Anyone running OpenFOAM calibrations on 1.4.0 should upgrade
and re-run. Delft3D users get one shared fix; TELEMAC workflows are unaffected.

### Fixed
- **A hot-started OpenFOAM case ignored the calibrated `ks` entirely.** The roughness
  update only ever wrote `0/nut`, but when `startTime > 0` OpenFOAM reads boundary
  conditions from `{startTime}/nut`. Every run therefore simulated the template
  roughness, and the surrogate was trained on outputs that did not vary with the
  parameter at all. `update_model_controls` now propagates `Ks` to every numeric time
  directory in the case. Reported and fixed by Federica Scolari.
- **The VTK timestep list could be sorted into the wrong order.**
  `extract_fields_from_vtk` sorted by the step index in the folder name, but a
  hot-start step index (e.g. 78496) outranks the indices of a freshly written run, so
  the `t=600` restart field sorted last and was pulled into the averaging window as if
  it were the final state. Sorting now uses the real simulation time read from each
  `.vtm`. Reported and fixed by Federica Scolari.
- **A leftover `VTK/` folder in the case template leaked into the results.**
  `convert_to_vtk` did not clear it before running `foamToVTK`, so a stale timestep
  from whenever the template was built could enter the timestep list and be averaged
  into a run's output. Reported and fixed by Federica Scolari.
- **`update_boundary_condition` had stopped writing `Cs` and mishandled multi-line
  values.** Only single-line `value uniform ...` entries were rewritten; a
  `value nonuniform List<scalar>` block, which is what hot-started fields carry, had
  its first line replaced and its body left behind as stray tokens. The list is now
  consumed as a unit and `Cs` is written alongside `Ks` again. Reported and fixed by
  Federica Scolari.
- **`results-detailed-<quantities>.npy` held only the most recent batch of rows.**
  The `.csv` is appended to across BAL iterations while the `.npy` is rewritten, and
  the array was rebuilt from the current call's rows rather than from the accumulated
  file, so the two copies of the same table disagreed after the first iteration. The
  `.npy` is now rebuilt from the full `.csv`. Reported and fixed by Federica Scolari
  in the OpenFOAM binding; **this also fixes Delft3D**, which carried a verbatim copy
  of the same method and so the same bug.
- **`Ks`/`Cs` could be written into boundary conditions that are not rough walls.**
  They are inserted ahead of a patch's `value` line, which fired for any patch updated
  with a non-`None` value instead of only for `nutkRoughWallFunction` patches. Not
  reachable from the shipped configs, whose dispatch builds only `Cmu` and `ks`
  entries, but wrong for code driving `OpenFOAMController` directly.

### Changed
- **`_save_all_results` now lives on `HydroSimulations`** instead of being duplicated
  verbatim in the OpenFOAM and Delft3D bindings, which is how the two copies came to
  carry the same `.npy` bug. It uses only base-class attributes, so every binding
  shares one implementation and TELEMAC gains the method. No call-site or output
  changes.

## [1.4.0] - 2026-08-03

A correctness release for the OpenFOAM and Delft3D bindings. Both drivers were unable
to start against the current `bayesvalidrox`, and OpenFOAM calibrations on velocity
magnitude were training the surrogate on `NaN` without saying so. Users on 1.3.0 with
an OpenFOAM or Delft3D workflow should upgrade; TELEMAC workflows are unaffected.

### Added
- **`OpenFOAMModel.EXTRACTABLE_QUANTITIES`**, the explicit list of field names the
  OpenFOAM extraction can produce (`U_x`, `U_y`, `U_z`, `U_MAG`, `TKE`, the
  fluctuation components, and the `U_magnitude` legacy alias). Both return paths of
  `_extract_at_control_points` are built from it, so they can no longer drift apart.

### Changed
- **Minimum `bayesvalidrox` is now 2.2** (was 2.1). 2.2 renamed the API the drivers
  depend on and the old names were removed, so 2.1 cannot run any driver. Upgrade with
  `pip install -U "bayesvalidrox>=2.2"` if your environment pinned the older release.
- **Unknown OpenFOAM calibration targets now raise instead of being silently
  substituted with `NaN`.** `OpenFOAMModel` validates `calibration_quantities` in its
  constructor and raises `ValueError` naming both the offending entry and the valid
  ones, before any simulation starts. This is deliberately breaking: a config that
  previously appeared to run while producing an all-`NaN` output column now fails
  immediately. It also rejects `WATER_DEPTH` and `FREE_SURFACE`, which
  `docs/usage-openfoam.rst` and `config_OpenFOAM.py` advertised for OpenFOAM even
  though the binding never extracted them; both are now documented as
  Delft3D/TELEMAC-only. The `NaN` row written when a run genuinely fails is unchanged.

### Fixed
- **Velocity magnitude was never extracted in OpenFOAM calibrations.**
  `_extract_at_control_points` exposed the magnitude only as `U_magnitude`, while the
  documented field name, the one used in `config_OpenFOAM.py` and in the Delft3D
  binding, is `U_MAG`. The extraction filter in `run_multiple_simulations` matched on
  `qty in results` and silently substituted `NaN` when it did not match, so a
  calibration on `U_MAG` trained the GPE on an all-`NaN` output column instead of
  failing. `U_MAG` is now the primary key, with `U_magnitude` retained as an alias so
  result files written by earlier versions stay readable. Reported and fixed by
  Federica Scolari.
- **The OpenFOAM and Delft3D drivers now run against bayesvalidrox 2.2.** That release
  renamed `Input.Marginals` to `marginals`, `ExpDesigns.X` to `x`, `ExpDesigns.JDist`
  to `j_dist`, and `generate_ED(n_samples=...)` to `generate_ed()` without the sample
  count, which it now reads from `n_init_samples`. The old names no longer exist, so
  `bal_openfoam.py` and `bal_delft3d.py` raised `AttributeError` before the first
  simulation. The TELEMAC drivers had already been migrated. OpenFOAM side reported
  and fixed by Federica Scolari.

## [1.3.0] - 2026-07-30

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
- **The Bayesian inference now accounts for the surrogate's own predictive variance**
  (`sampling['include_surrogate_error']`, **default `True`**). The inference previously
  ignored the GPE predictive variance while the BAL utility used it, so the posterior
  came out sharper than the surrogate supports. It now passes `surrogate_output['std']`
  as `model_error`, backed by a new diagonal fast path in `BayesianInference`
  (`O(MC * n_obs)` instead of `O(MC * n_obs**2)` memory; the dense path needs several
  GB per array at `prior_samples=25000`).
- **The observation-error budget is now three named terms instead of one fudge factor.**
  `calibration['measurement_error']` (0.10, instrument imprecision),
  `calibration['gpe_error']` (**now 0.0**, a flat stand-in for emulator uncertainty that
  is redundant while `include_surrogate_error` is on) and the new
  `calibration['model_structural_error']` (0.0, the solver being an imperfect
  description of the site, which `include_surrogate_error` does **not** supply). All
  three are real constructor arguments of `HydroSimulations` and are read from the
  configuration by every driver, including the multi-discharge one; `measurement_error`
  previously existed but no driver read it. **Expect posteriors to become sharper, not
  broader**: the old defaults added a flat 10 % emulator term on top of the 10 %
  measurement term, and a trained GPE is usually tighter than that. To reproduce the
  old behaviour exactly, set `include_surrogate_error = False` **and**
  `gpe_error = 0.10`; setting only the first represents the emulator uncertainty
  nowhere at all, which the drivers now warn about. The effective settings are logged
  on every run.
- **A test suite** (`tests/`, 89 tests), the first in the repository, covering the
  posterior analysis on synthetic posteriors, the likelihood paths and their numerical
  limits, the multi-output GPE task layouts, the `user-collocation-points.csv` contract,
  and the shipped default values themselves (including the `.get()` fallbacks in the
  drivers, which is what an existing configuration file actually hits). It runs in minutes
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
- **Resuming a calibration ran zero BAL iterations.** The documented
  `complete_bal_mode=True` + `only_bal_mode=True` mode is meant to rebuild the surrogate
  from the stored runs and then *continue* BAL, but `bal_openfoam.py` and
  `bal_delft3d.py` zeroed `n_iter` on `only_bal_mode` alone, so a resume silently did
  nothing beyond re-fitting the emulator, and their simulation call was additionally
  guarded by `not only_bal_mode`. The three drivers had drifted apart here:
  `bal_telemac.py` (and the Ering example copy) had no `n_iter` guard at all and instead
  ran new simulations in pure re-analysis mode (`only_bal_mode` without
  `complete_bal_mode`), which is the opposite failure. All of them now share one
  contract: iterations are skipped only for pure re-analysis, and `complete_bal_mode`
  alone decides whether new simulations run. `bal_telemac_multiflow.py` imports
  `run_bal_model` from `bal_telemac.py`, so it inherits the fix.
- **The OpenFOAM `ks` calibration parameter wrote to a hardcoded `bottom` patch.**
  Case templates name the bed patch differently (`base`, `bed`, ...), so calibrating
  `ks` against any template that does not happen to call it `bottom` wrote the roughness
  to a patch that does not exist. `OpenFOAMModel` now reads the patch name out of the
  case template's `0/nut`, selecting whichever patch actually declares
  `nutkRoughWallFunction`, and warns at construction time when there is none.
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
- **The Bayesian model evidence was computed in linear space and broke at both ends.**
  `BME = mean(exp(log_likelihood))` **underflowed to exactly 0.0** on the ordinary
  default path once a calibration had a few hundred outputs and a mediocre fit (300
  outputs at a 2.2 sigma mean residual, 600 at 1.7 sigma). `BME == 0` gave `RE = nan`,
  the nan was caught in `bayesian_active_learning` and replaced by `0.0` **for every
  candidate**, and the training point was then chosen arbitrarily rather than by
  information gain. Affected runs are identifiable: `bayesian_dict['util_func']` is
  relabelled `'global_mc'` for those iterations. The evidence is now computed with
  `logsumexp` and exposed as a new `log_BME`, from which `RE` and `IE` are derived, so
  neither end can poison the scores. `BME` is retained for backward compatibility and
  may still read `0.0` or `inf`.
- **The likelihood conventions of the two paths disagreed.** The default path dropped
  the normalising constant while the model-error path kept it, so the log-likelihood
  went large and *positive* and `exp()` overflowed to `inf` at realistic sizes. Both
  paths are now normalised against the observation covariance, which cancels the `2*pi`,
  keeps the sample-dependent `log(v/e)` term in full, makes the result non-positive **by
  construction**, and reduces exactly to the old default path at zero model error. BME
  is therefore comparable across the setting.
- **`calculate_likelihood_manual` crashed with `ZeroDivisionError`** on any calibration
  with a few hundred outputs: it formed `1/sqrt(np.linalg.det(cov_mat))`, and the
  determinant of a few hundred small variances underflows to exactly `0.0`. The value
  was computed only to be discarded, since the line consuming it had been commented out,
  and it is now gone. The dense `calculate_likelihood_with_error` uses `slogdet` for the
  same reason.
- Non-positive observation variances now raise a clear `ValueError` naming the offending
  entries, instead of an opaque `LinAlgError` from a matrix inverse.
- `bayesian_scores.csv` silently misaligned every column when an appended row carried a
  different column set, which already happened within a single run because the
  per-iteration diagnostics only appear once a posterior exists. The file is now
  rewritten with the union of columns.
- `plot_bme_re` fits its trend line on the finite points only; a single `inf` or `nan`
  previously made `linregress` return `nan` and the trend line vanished silently. It
  also prefers `log_BME` when the result file has it.
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

[1.6.0]: https://github.com/Ecohydraulics/hydrobayescal/releases/tag/v1.6.0
[1.5.0]: https://github.com/Ecohydraulics/hydrobayescal/releases/tag/v1.5.0
[1.4.4]: https://github.com/Ecohydraulics/hydrobayescal/releases/tag/v1.4.4
[1.4.3]: https://github.com/Ecohydraulics/hydrobayescal/releases/tag/v1.4.3
[1.4.2]: https://github.com/Ecohydraulics/hydrobayescal/releases/tag/v1.4.2
[1.4.1]: https://github.com/Ecohydraulics/hydrobayescal/releases/tag/v1.4.1
[1.4.0]: https://github.com/Ecohydraulics/hydrobayescal/releases/tag/v1.4.0
[1.3.0]: https://github.com/Ecohydraulics/hydrobayescal/releases/tag/v1.3.0
[1.2.0]: https://github.com/Ecohydraulics/hydrobayescal/releases/tag/v1.2.0
[1.1.0]: https://github.com/Ecohydraulics/hydrobayescal/releases/tag/v1.1.0
[1.0.0]: https://github.com/Ecohydraulics/hydrobayescal/releases/tag/v1.0.0
