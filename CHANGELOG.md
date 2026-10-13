# Changelog

All notable changes to HydroBayesCal are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.11.0] - 2026-10-13

Two changes to the Gaussian process emulators, double precision and a prediction in pieces, and the corrections of ten findings of a code review of `surrogate/`. Each finding was reproduced with the released code before it was changed, and each has a test that fails without the correction. Five of them can be reached with the shipped drivers; they come first among the corrections. A second, independent review of all changes since 1.10.1 found one more defect, in an option the drivers do not use; its correction is part of the entry on `SequentialDesign`.

Every result of a gpytorch emulator changes slightly with this release, and a calibration whose posterior turns multimodal selects other training points than before; see the first and the third entry under Changed.

### Changed
- **The gpytorch emulators compute in double precision.** Up to 1.10.1 they computed in single precision, which resolves about seven digits. A Gaussian process needs more in two places. Its training factorizes the covariance matrix of the training runs, whose condition number grows with the number of runs and with the smoothness of the model response. With the emulator settings of the drivers and an analytic model with two targets at three points, training or prediction stopped with `NotPSDError: Matrix not positive definite after repeatedly adding jitter` in 1 of 20 attempts with 20 training runs, in 6 of 20 with 40, in all 20 with 80, and in 5 of 20 with 30 runs of a linear response. An emulator is trained in every BAL iteration, so a calibration of that size was unlikely to reach its end. Second, the posterior covariance of an emulator is a small difference of large terms, which was lost in round-off where the emulator was nearly certain; this is the defect of 1.10.0 that 1.10.1 safeguards. In double precision none of the 80 attempts failed, no covariance had to be clipped, and the error at the training runs fell from between 0.7 and 1.9 % to between 0.1 and 0.3 % of the spread of the outputs. Every emulator result changes slightly. `precision="single"` in `GPyTraining`, `MultiGPyTraining` and `SOSequentialGPyTraining.from_training_data` reproduces the earlier arithmetic; the drivers have no key for it. An emulator pickled by an earlier release keeps predicting, in single precision. These numbers come from an analytic model whose outputs are free of noise and whose parameters are scaled to [0, 1]. They were not measured on a TELEMAC, OpenFOAM or Delft3D case.
- **Predictions are evaluated in pieces, and their memory no longer grows with the square of `prior_samples`.** gpytorch returns the joint distribution of all points it is given, with a covariance matrix between them, while the emulators use the mean, the variance and the covariance between the targets at each point only. A prediction at 25,000 prior samples, the value of the shipped configuration, needed 15 GB and 8 s in 1.10.1 for six outputs, and would have needed 30 GB in double precision. It now needs under 1 GB and 0.6 s, with the same results, and 100,000 prior samples take a few seconds. `PREDICTION_BUDGET` in `gpe_gpytorch.py` sets the size of a piece.
- **The exploration/exploitation trade-off of BAL takes effect.** All drivers pass the prior sample as the candidate pool, and for that case the exploration score was a vector of zeros. With `sampling['bal_exploration_tradeoff']` on, which `'auto'` (the default) switches on once the posterior has more than one mode, the candidates were therefore ranked by exploitation alone, while the log and `exploration_tradeoff_active` in `BAL_dictionary.pkl` reported the trade-off as active. An archived result of 1.10.1 or earlier with that flag set holds points that were chosen without exploration. The candidates now get a space-filling score (`SequentialDesign.space_filling_scores`): the intersite-projected distance to the simulated parameter sets, taken in the unit hypercube of the prior sample so that parameters with different units count alike. A calibration whose posterior turns multimodal now selects different points than before. The zeros also had the length `mc_samples_al`, which raised a broadcast error when the prior sample held fewer unused parameter sets than that.
- **A resampled posterior is analyzed by what its rows are worth** (`posterior_sampling_method='bayesian_weighting'`). Weighted resampling returns as many rows as the prior sample has, most of them repeats, and the posterior analysis treated every row as a sample. The joint density of a row that occurred more often than there are neighbors was set to a cap that most rows shared, so that the joint optimum was a tie-break and the vector of marginal peaks ranked near the bottom: `equifinality_diagnostic` returned "inconsistent" for unimodal test posteriors that rejection sampling called consistent. The marginal histograms were resolved by the number of rows and broke up into 8 to 14 spurious modes. The joint density now uses the distinct rows with their multiplicity, and the histogram resolution, `min_post_size` (`marginal_optima`) and `min_samples` (`detect_posterior_modes`) use the effective sample size after Kish (`effective_sample_size`). A posterior of 25,000 rows that is worth 50 samples is reported as `low_posterior_sample`, with `effective_post_size` next to `post_size`. Nothing changes for rejection sampling, the default.
- **The refinement of the joint optimum maximizes the likelihood of the calibration** (`derive_calibrated_parameters.py --refine`). It evaluated the emulator likelihood with the observation variances alone, although the calibration adds the variance of the emulator by default (`include_surrogate_error`). The setting is read from `BAL_dictionary.pkl`. `BayesianInference.calculate_prior_likelihood` is the one place that chooses the form of the likelihood.

### Fixed
- **An output that is the same in every run stopped the emulator** with `NanError: cholesky_cpu: ... elements ... are NaN`, after the initial design had been simulated. The emulators normalize every output by its standard deviation over the training runs, which is zero for a constant output. A calibration point that stays dry in all runs produces this, and so does any target that does not respond to the calibration parameters at a point. Such an output is now returned with its value and without emulator uncertainty, in `GPyTraining` and in all task layouts of `MultiGPyTraining`, and the log names calibration point and target once. Predictions for outputs that vary are bit-identical to 1.10.1. Training outputs that contain NaN or infinite values are refused up front, with the runs and columns concerned.
- **`--refine` did not refine on a gpytorch emulator.** The optimizer took the slope of the likelihood over its default step of 1e-8 in the units of each parameter, which is below the resolution of an emulator that computes in single precision: on a trained emulator the difference of two likelihoods was exactly zero at 23 of 40 points and round-off elsewhere. Five of six starts did not move, and the unmoved starting points were reported as a refined optimum and as several local maxima. The search now runs in the unit hypercube of the calibration ranges with a step of 1e-3 of each range.
- **A staged initial design changed its sampling rule without saying so.** With `adaptive_init_runs=True`, every block after the first was drawn as a Latin hypercube for any rule that is not an extensible sequence, while log and `initial-design.json` named the configured rule. `random` and `latin_hypercube` designs are now grown by a block of their own rule; `chebyshev` and `grid` cannot be continued and fall back to a Latin hypercube with a warning. Every stage in `initial-design.json` records its rule.
- **One candidate without a usable utility made BAL select an arbitrary parameter set.** Such a candidate carries the score `-inf`. The scores were normalized by the sum of their absolute values, which is then infinite, so every other score became 0 and the sort returned a candidate that no criterion had chosen (the 150th of 299 by utility in the reproduction). The normalization runs over the finite scores, and a failed candidate ranks last. If no candidate has a usable utility, the selection is space-filling and recorded as `global_mc`. No way was found to produce a failed candidate with a healthy emulator and the utilities `dkl` or `bme`.
- **`SequentialDesign` with options the drivers do not use.** The covariance between the targets was read whenever `multitask` was set but requested for two-dimensional observations only, which raised `UnboundLocalError` with `gaussian_assumption=True`. An exploitation method that contains but is not `bal` left `run_al_functions` without scores. A secondary (error) emulator added its standard deviation to the one of the emulator, and its uncertainty was dropped with a multitask emulator; its variance is added now, also to the covariance between the targets. The closed-form utility (`gaussian_assumption=True`, `analytical_bal`) inverted the emulator variances and stopped with `LinAlgError: Singular matrix` for an output without emulator uncertainty, which the emulators return since this release for an output that is the same in every run. Such an output is left out of the closed form: its observation cannot change it, so it adds nothing to the utility.
- **Training options of `GPyTraining` that the drivers do not use.** With `n_restarts=2`, the second restart repeated the first, because the hyperparameters were re-initialized from the third restart on. `optimizer='lbfgs'` named a class that does not exist in torch; it is `torch.optim.LBFGS` with a line search now. A training without a finite loss in any restart ended in `load_state_dict(None)` and now says what happened.
- **`validation_error` raised `TypeError` with scikit-learn 1.6 and later**, which removed `mean_squared_error(squared=False)`. The function and `save_valid_criteria` existed twice, word for word, in `gpe_gpytorch.py` and `gpe_skl.py`; they are in `surrogate/validation_metrics.py` now and stay importable from both modules. A prediction without confidence bounds no longer raises `KeyError`.
- **The space-filling criteria of `Exploration`.** The projected distance entered the intersite-projected score squared, next to an Euclidean distance that was not, and the threshold criterion removed candidates with `np.delete(candidates, [mask])`, which NumPy 2 refuses and older versions misread as rows 0 and 1.

## [1.10.1] - 2026-10-12

### Fixed
- **BAL could stop with `ValueError: The input matrix must be symmetric positive semidefinite` in a calibration on several targets** with a multitask emulator (`multitask_selection` `"variables"`, the default, or `"all"`). 1.10.0 hands BAL the exact posterior covariance between the calibration targets at every candidate parameter set. The emulator computes that covariance in single precision, as the difference between the prior covariance and the part explained by the training runs. Where the emulator is nearly certain, the result is orders of magnitude smaller than both, its error reaches the size of the result, and the 2 x 2 or 3 x 3 block can come out with a negative eigenvalue, which the sampler of the BAL utility refuses. This is not confined to candidates next to a training run: an emulator that finds the model response smooth over the whole parameter ranges is that certain everywhere. In a reproduction with the emulator settings of the drivers and nine training runs of an analytic model, one of eight trained emulators produced such blocks, for 3 to 8 % of its candidates. Its correlation lengths at one calibration point were about ten times the parameter ranges, its posterior variance there was six orders of magnitude below its prior variance, and a double-precision evaluation of the same emulator gave a valid covariance at every candidate. A calibration with such an emulator stopped in a BAL iteration, after the initial design had been simulated. The blocks are now converted to physical units in double precision and their negative eigenvalues are clipped as the last step (`surrogate.bal_functions.project_to_psd`); a positive semidefinite block is unchanged. The clipping yields a valid covariance, it does not restore the digits lost in single precision; in the reproduction these correspond to a standard deviation of about 1 % of the spread of the training outputs. The emulator reports once per run that it had to clip. The BAL utility applies the same projection if any emulator hands it an indefinite covariance, and reports that once as well. The defect entered with 1.10.0, the first release that hands BAL the posterior covariance instead of the prior one, and was found by the new end-to-end test of the TELEMAC driver, which failed in one of about ten runs of the test workflow. A calibration on one target, or with `multitask_selection` `"locations"` or `"SO_sequential"`, was not affected: BAL samples those emulators with their standard deviations.

### Added
- **Every push and every pull request is tested on GitHub** (`.github/workflows/tests.yml`): a lint for definite errors, the test suite on Python 3.10 and 3.12, and the documentation build with the Read the Docs settings. The workflow and the file-size check run with a read-only token.

## [1.10.0] - 2026-10-11

An integration release. It merges the branches `of-dev` of Federica Scolari and `hbc-telemac-andres` of Andrés Heredia into `main`. Several of the problems the two branches solved for one solver binding or one example case turned out to be defects of the shared code, and are corrected there, so that all bindings and all drivers get the correction. Two of the changes alter the numbers of a calibration and are listed first under Changed: the covariance that Bayesian active learning (BAL) draws from for a multitask emulator, and the time window over which TELEMAC runs selected by BAL are extracted.

### Added
- **OpenFOAM and Delft3D: the calibration targets can be chosen after the runs.** Both bindings store every extracted quantity for every run and calibration point in `restart_data/detailed-model-outputs.csv`. A restart with `only_bal_mode` and other `calibration_quantities` rebuilds the surrogate training data from that table instead of repeating the simulations. Contributed for OpenFOAM by Federica Scolari. The rebuild now lives in `HydroSimulations._load_stored_outputs`, which is why Delft3D shares it, and the table is written by `HydroSimulations._save_all_results` itself, see Fixed. Result folders written by earlier versions remain usable.
- **OpenFOAM: `ALPHA_WATER`**, the `interFoam` water volume fraction at the calibration point, is an extractable quantity. Contributed by Federica Scolari.
- **`sampling['multitask_selection']` sets the layout of the emulator in every driver.** The setting existed on the model, but no driver passed it, so it could be changed only by editing a driver. `"variables"` (default), `"locations"` and `"all"` are the multitask layouts that existed. Two layouts are new, both contributed by Andrés Heredia in his example drivers and now part of all drivers: `"SO_sequential"` trains independent single-output emulators for several calibration targets (class `SOSequentialGPyTraining`), and `"locations"` with a single calibration target trains one multitask emulator over the calibration points instead of one emulator per point. An unknown value is rejected when the model is constructed.
- **TELEMAC and GAIA: a sand ratio averaged over bed layers as a calibration target.** `morphodynamic_simulation['gaia_layer_average']` derives the ratio of one or several sand classes, averaged over several bed layers with the layer thicknesses as weights, which is what a sediment sample taken over a known depth measures. Contributed by Andrés Heredia, with the GAIA variable naming corrected, see Fixed. Documented in the TELEMAC usage guide.
- **TELEMAC: numbered validation attempts.** Every validation experiment writes its own `collocation-points-validation_<n>.csv` and `model-results-validation_<n>.json`. Contributed by Andrés Heredia. `TelemacModel.validation_paths()` returns the files of the latest attempt or of a given one, and still reads the unnumbered files of earlier versions.
- **`sampling['bal_n_jobs']`** sets the number of processes that score the BAL candidates (default `-1`, all cores) in all drivers.
- **The log-likelihood of the posterior samples is stored per iteration** in `BAL_dictionary.pkl` under `post_loglikelihood` (additive key). `plot_posterior_updates` can mark the joint posterior maximum with it (`best_estimate_value="joint_posterior_MAP"`). Contributed by Andrés Heredia.
- **New figures**, contributed by Andrés Heredia: `plot_collocation_points` (initial design and BAL-selected points in parameter space), `plot_posterior_model_comparison` (posteriors of two calibrations side by side), `summarize_validation_metrics` (emulator error over several validation attempts), a trend window in `plot_bme_re` and fixed limits in `plot_residuals`. A tilt correction for Vectrino velocity records is added in `utils/VectrinoPostproc/tilt_correction.py`.
- **Examples** for the Ering reach (TELEMAC-2D hydrodynamics, with emulator validation and a comparison of multitask and single-output emulators) and for a laboratory flume with a cylinder (TELEMAC-3D), contributed by Andrés Heredia. The examples contain scripts and configurations; the model and measurement files are not part of the repository.
- **`env-scripts/sync-example-drivers.py`** regenerates the example copies of `bal_telemac.py` from the template.
- **End-to-end tests of the drivers.** `tests/test_driver_end_to_end.py` runs the BAL loop of the TELEMAC, OpenFOAM and Delft3D drivers on an analytic model, once per emulator layout. Until now the drivers were checked by parsing their sources only.

### Changed
- **BAL draws a multitask emulator from its posterior covariance.** For a calibration on several targets with a multitask emulator, the TELEMAC drivers hand BAL the covariance between the targets at every candidate parameter set. Up to 1.9.0 this was the prior term, task kernel times input kernel, which is the same at every candidate for the kernels in use and does not depend on the training runs. It is now the posterior predictive covariance, which is small next to a training run and large far from the design. This changes which training points BAL selects. Found and corrected by Andrés Heredia. The evaluation was rewritten to return the exact covariance, without the jitter of the first version, and to evaluate only the blocks BAL uses, which keeps the `"all"` layout within memory (it needed 4.9 GB for 36 outputs and 250 candidates, and did not finish for 111 outputs).
- **TELEMAC: runs selected by BAL are extracted like the initial design.** `run_bal_model` extracted every BAL run with `"mean_last"` over 40 time steps, whatever the `extraction` block of the configuration said, while the initial design used the configured window (default: 80 time steps). The surrogate was therefore trained on two differently defined quantities. Both TELEMAC drivers now pass the configured window to the BAL loop. Noticed by Andrés Heredia.
- **User-supplied designs and validation runs are no longer staged.** The staged initial design could not handle either, see Fixed. Both are run in one block, and a user design whose size differs from `init_runs` is rejected before the first simulation.
- **TELEMAC: result files are moved to `saved_slf_files`, not copied.** The copy left one result file per run in the model directory, which `delete_complex_outputs` never removed. The result file of a module no quantity was read from is moved as well.
- **An extraction quantity unknown to both TELEMAC and GAIA is refused** at the first extraction, with its name. It used to be left out silently, and the run then failed on a missing key.
- `CalibrationAssessment.evaluate_calibration` takes the measurement errors as `err_split` after `obs_split`. `plot_posterior_updates` marks the peak of each marginal by default under its proper name `"posterior_marginal_peak"`. The former default name `"posterior_MAP"` still returns that peak when no log-likelihoods are passed, and the joint maximum when they are.
- The shipped example configuration `config_Ering.py` describes the current hydro-morphodynamic Ering case of Andrés Heredia. It keeps the shipped defaults `measurement_error = 0.10` and `gpe_error = 0.0`; with `include_surrogate_error = True`, a non-zero `gpe_error` counts the emulator uncertainty twice.

### Fixed
- **`results-detailed-*.csv` and `.npy` held every run several times** (OpenFOAM and Delft3D bindings). `HydroSimulations._save_all_results` appended the rows it was given, and the bindings give it all rows of the current block after every run and once more at the end, so run 0 of an N-run block was stored N+1 times. The table is now merged by run. This also made the re-selection of calibration targets work on real output, which matches rows to calibration points by count.
- **The table for the re-selection of calibration targets lost runs when BAL continued after a change of targets.** It was a copy of the per-target table, and a restart on other targets starts a new, empty per-target table. One further BAL run replaced the copy by a table holding that run alone, and the next re-selection would have trained the surrogate on it without any message.
- **A design supplied by the user stopped the calibration before its first simulation** when the staged initial design was on, which is the default. The staged design asked the sampler for a rule named `user`, which does not exist. Andrés Heredia worked around this in his example drivers; the correction is in `run_staged_initial_design`.
- **TELEMAC, initial runs only (`complete_bal_mode = False`): every block of a staged design simulated all earlier runs again.** The loop ignored the index of the first new run. Fixed by Andrés Heredia.
- **TELEMAC: a repeated validation mixed two experiments.** The parameter sets were overwritten while the results were appended. Fixed by the numbered validation attempts of Andrés Heredia, which now also resolve on a model that has not just run a validation; the first version raised `AttributeError` there, which is the way the shipped `validate_simulation.py` uses them.
- **GAIA layer average read sand classes for layers.** GAIA names the ratio of a sand class in a bed layer `LAY<layer> SAND RAT<class>`. The first version of the layer average appended the layer number to `LAY1 SAND RAT` and thus read the sand classes 1 and 2 of the top layer, weighted with the layer thicknesses. For a file holding class 1 at ratios of 0.20 and 0.50 in the layers 1 and 2 it returned 0.73 instead of 0.46. A configuration entry now names `sand_classes`, `layers` and `thicknesses`, and one without `sand_classes` is rejected with the naming rule. Results extracted with the first version have to be extracted again.
- **`bal_delft3d.py` raised `NameError` in its first BAL iteration**, after the whole initial design had been simulated: `do_tradeoff` was read but never assigned. 1.9.0 had corrected this in `bal_openfoam.py` only.
- **Regressions of the merged branch `hbc-telemac-andres` that did not reach a release**, corrected during the merge: a deleted comma that turned an argument of `run_staged_initial_design` into `validation ** run_kwargs`; shared defaults changed for one case (`measurement_error` 0.10 to 0.0, BAL `n_jobs` -1 to 1, plot decimals 2 to 1); configuration keys read without fallback in `bal_telemac.py`, so that configurations without an `extraction` block or without `posterior_sampling_method` failed; `plot_collocation_points` calling a helper and two modules it had not defined or imported; a plotting block in the shipped `config_Telemac.py` with five labels for seven calibration parameters; paths on one user's disk in shipped configurations and scripts; a stray line that made an example configuration unparsable; example copies of `bal_telemac.py` that carried their own version of the workflow.

## [1.9.0] - 2026-09-04

An OpenFOAM release. The binding shipped since 1.7 could not run against **OpenFOAM
Foundation** at all, and three further defects surfaced only *after* an experimental
design had already been simulated - i.e. at the point where the runs they invalidate
have already been paid for.

### Added
- **OpenFOAM Foundation support.** The turbulence dictionary is resolved by which file
  EXISTS rather than by name, so one binding serves both lines: ESI calls it
  `constant/turbulenceProperties`, Foundation 8+ calls it `constant/momentumTransport`.
  The caller's `param["file"]` is honoured only when it is really there, because the
  config's default names the ESI spelling.
- `OpenFOAMModel(write_vtk=...)`, default **off**. Field extraction no longer needs
  `foamToVTK`, and with `delete_complex_outputs` the converted copy was deleted moments
  after being written.
- `HBC_MPI_LAUNCHER` for installs whose MPI launcher is not `mpirun`.

### Fixed
- **`do_tradeoff` was read before assignment in `bal_openfoam.py`**, so the first BAL
  iteration raised `NameError` - after the whole initial design had run.
  `bal_telemac.py` initialises it; this driver did not.
- **`OpenFOAMModel.__init__` dropped `gpe_error`, `measurement_error` and
  `model_structural_error`.** They were accepted and never forwarded to
  `HydroSimulations`, so observation variances were always built from the base-class
  defaults and a configured `measurement_error` was silently ignored.
- **`update_model_controls` hard-coded the turbulence file** while reading
  `param["file"]` one line above, so every k-epsilon perturbation raised
  `FileNotFoundError` on a Foundation case.
- **`extract_fields_from_vtk` required the ESI `foamToVTK` layout** (`VTK/*.vtm` plus
  `VTK/<stem>/internal.vtu`). Foundation writes legacy `VTK/<case>_<n>.vtk`, so the
  extractor could never run against it. It now reads the case directory through
  pyvista's OpenFOAM reader - both lines, ASCII and binary `writeFormat`, with and
  without `alpha.water` (a single-phase or rigid-lid case has none, which correctly
  means every cell is water). Times come from the reader, so a hot-started run's step
  indices cannot reorder the averaging window.
- **`run_simulation` sourced a hard-coded `$HOME/OpenFOAM/OpenFOAM-v2412/etc/bashrc`**,
  which made the binding usable on exactly one machine.

### Verified
Against a real OpenFOAM Foundation 9 build: parameter routing into the dictionaries,
the MPI launch, field extraction and per-run accumulation, over a 3-run plumbing test
and a multi-run campaign on the Inn KB15 reach.

## [1.8.1] - 2026-08-28

### Fixed
- **The roughness-identifiability check can no longer end a calibration.** 1.8.0 moved it
  onto the default path, where it runs after an initial design that has already cost
  hours of solver time - but with mismatched output/observation shapes it raised out of
  its own residual broadcast, which would have discarded those completed runs. It now
  reports `unavailable` for a shape mismatch or all-non-finite input, uses `nanmedian`
  so one dry or failed calibration point cannot void the verdict, and the driver call
  sites catch anything else and log it as a warning. Report-only means report-only.

## [1.8.0] - 2026-08-28

A TELEMAC release, from a 14-day multi-discharge calibration that ended with a posterior
of 35 samples and a result the tooling could have predicted on its first day. Both are
fixed here: the posterior sample size no longer collapses when a calibration goes well,
and the roughness-identifiability check now runs on the path calibrations actually take.

### Added
- **`sampling['posterior_sampling_method']` selects how the posterior is drawn.**
  `bayesian_weighting` now resamples the prior with replacement in proportion to the
  likelihood weights (sampling-importance resampling) and returns a full-size posterior
  from the same likelihoods, at no extra model cost. It reports the Kish effective
  sample size, `1/sum(w^2)`, and warns when that falls below 5% of the drawn size, so a
  posterior resting on a few prior samples stays visible instead of hiding behind a
  full-looking array. `rejection_sampling` remains the default and is unchanged.
- **The roughness-identifiability diagnostic runs in `bal_telemac.py` and
  `bal_telemac_multiflow.py`**, on the initial design and before the BAL loop. It was
  already written and documented but only reachable from the opt-in
  `prebal_telemac_error_analysis.py` driver, so an ordinary calibration never saw it.
  Correlated depth and velocity residuals mean bottom roughness cannot close the misfit
  and its optimum will pin against whichever prior bound it is given - worth knowing
  before the active-learning iterations, not after. It is report-only and changes no
  sampling and no parameter.

### Fixed
- **`bayesian_weighting` produced no posterior at all.** The branch computed the
  evidence and the weights, then discarded the weights: `posterior`, `posterior_output`
  and `post_index` were set only by `rejection_sampling`, so selecting the documented
  alternative left every downstream consumer - posterior analysis, plots,
  `derive_calibrated_parameters` - with `None`. This mattered because rejection sampling
  is the one that starves: its acceptance rate is `mean(L)/max(L)`, which collapses
  precisely when the likelihood is sharp, i.e. when the calibration has succeeded.
- **`tbl_creator` raised `IndexError` on a blank line in a friction table.** The comment
  branch guarded against an empty split, the zone comparison on the next line did not.
  Since the friction table is rewritten before every model evaluation, a hand-edited
  table with a trailing or separating blank line stopped the calibration at its first
  run, after the design had been sampled and before any usable output existed.

## [1.7.0] - 2026-08-18

### Added
- **Modeled vs. measured calibration targets as the closing step of every calibration.**
  Each `bal_*` driver now writes `plots/<targets>/calibration-target-agreement.png|svg`
  and logs the matching verdicts: one panel per calibration target and calibration state,
  measured values on the x-axis, modeled values on the y-axis, open black circles with the
  measurement uncertainty as error bars, against the 45-degree line of perfect agreement.
  "Before calibration" is the initial-design ensemble of full-complexity runs, "after
  calibration" the posterior predictive of the final surrogate, so the figure shows what
  the calibration bought in observation space, which the parameter-space results of
  `posterior_analysis` cannot. New modules `surrogate/target_agreement.py` (analysis,
  report-only) and `visualize/agreement_plots.py` (figure, also available on
  `BayesianPlotter`).
- **A verdict per calibration target that separates a systematic deviation from scatter.**
  A mean residual above both a 2 % deadband and the standard error of the mean measurement
  uncertainty, with at least 60 % of the calibration points on the same side of the 1:1
  line, is reported as a systematic over- or underestimation, i.e. a bias a calibration
  parameter can still remove. Residuals that change sign across the calibration points are
  reported as scatter, which no global parameter value can fix. The
  roughness-identifiability diagnostic runs on both states and its reading appears in the
  column titles, so the figure states whether roughness was the identifiable calibration
  parameter, whether the calibration moved it, or whether roughness was never the issue.
- **`drivers/plot_target_agreement.py`** reproduces the figure from an archived
  `BAL_dictionary.pkl` with different thresholds, units or labels. `--config` uses the
  TELEMAC result-folder layout; `--bal-dictionary` needs neither a configuration nor a
  model instance and works for TELEMAC, OpenFOAM and Delft3D results alike.
- The series behind the figure are stored in `BAL_dictionary.pkl` under the new
  `target_agreement` key (additive, read with `.get`, absent from older result files).

### Changed
- `hydroBayesCal.visualize` imports `BayesianPlotter` lazily (PEP 562 module
  `__getattr__`). `from hydroBayesCal.visualize import BayesianPlotter` is unchanged, but
  a single plot module no longer drags in the dependencies of every other one, so the
  closing agreement figure survives a missing dependency of an unrelated plot.

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

[1.11.0]: https://github.com/Ecohydraulics/hydrobayescal/releases/tag/v1.11.0
[1.10.1]: https://github.com/Ecohydraulics/hydrobayescal/releases/tag/v1.10.1
[1.10.0]: https://github.com/Ecohydraulics/hydrobayescal/releases/tag/v1.10.0
[1.9.0]: https://github.com/Ecohydraulics/hydrobayescal/releases/tag/v1.9.0
[1.8.1]: https://github.com/Ecohydraulics/hydrobayescal/releases/tag/v1.8.1
[1.8.0]: https://github.com/Ecohydraulics/hydrobayescal/releases/tag/v1.8.0
[1.7.0]: https://github.com/Ecohydraulics/hydrobayescal/releases/tag/v1.7.0
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
