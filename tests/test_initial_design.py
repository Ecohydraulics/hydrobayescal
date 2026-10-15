"""Tests for :mod:`hydroBayesCal.surrogate.initial_design`.

Everything here runs on a cheap analytic response in seconds: no solver, no TELEMAC and
no GPyTorch. The load-bearing cases are ``test_sobol_block_continues_the_sequence``,
which pins the property that makes a staged design free of waste, and
``test_ladder_stops_early_and_never_exceeds_the_ceiling``, which pins that the ladder can
only ever save simulations, never spend more than the configuration authorised.
"""
import json
import os

import numpy as np
import pytest

from hydroBayesCal.surrogate.initial_design import (
    TENSOR_RULES,
    generate_initial_design,
    initial_design_ladder,
    initial_design_sufficiency,
    loo_predictivity,
    recommended_init_runs,
    run_staged_initial_design,
    sobol_block,
    tensor_rule_samples,
    validate_sampling_method,
)

RANGES = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
TRUTH = np.array([0.4, 0.7, 0.25])


def _response(points, nloc=10):
    """Smooth, parameter-sensitive analytic stand-in for a solver."""
    points = np.atleast_2d(points)
    stations = np.linspace(0.2, 1.0, nloc)
    return np.array([[np.sin(3 * p[0] * s) + p[1] * s ** 2 + 0.5 * p[2] * np.cos(2 * s)
                      for s in stations] for p in points])


class _ExpDesign:
    """The slice of the ``bayesvalidrox`` ExpDesigns surface used by this module."""

    def __init__(self, ranges, sampling_method="sobol"):
        self.ranges = np.asarray(ranges, dtype=float)
        self.sampling_method = sampling_method
        self.n_init_samples = None
        self.x = None

    def generate_samples(self, n_samples, sampling_method=None):
        import chaospy

        method = sampling_method or "random"
        distribution = chaospy.J(*[chaospy.Uniform(low, high)
                                   for low, high in self.ranges])
        return chaospy.generate_samples(int(n_samples), domain=distribution,
                                        rule=method).T

    @property
    def j_dist(self):
        import chaospy

        return chaospy.J(*[chaospy.Uniform(low, high) for low, high in self.ranges])

    def generate_ed(self):
        self.x = self.generate_samples(self.n_init_samples, self.sampling_method)
        self.n_init_samples = self.x.shape[0]


class _StubModel:
    """Minimal stand-in for a binding, recording how the ladder called it."""

    def __init__(self, restart_folder, init_runs=64, ndim=3):
        self.init_runs = init_runs
        self.ndim = ndim
        self.param_values = RANGES
        self.complete_bal_mode = True
        self.validation = False
        self.restart_data_folder = str(restart_folder)
        self.observations = _response(TRUTH[None, :])
        self.measurement_errors = 0.05 * np.abs(self.observations).ravel()
        self.variances = self.measurement_errors ** 2
        self.model_evaluations = None
        self.calls = []
        self.keywords = []

    def run_multiple_simulations(self, collocation_points=None, start_index=0, **kwargs):
        self.calls.append((int(start_index), int(np.atleast_2d(collocation_points).shape[0])))
        self.keywords.append(kwargs)
        self.model_evaluations = _response(collocation_points)
        return self.model_evaluations


# ---------------------------------------------------------------------------
# a-priori sizing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ndim, expected", [(1, 16), (2, 32), (5, 64), (7, 128),
                                            (10, 128), (13, 256)])
def test_recommended_init_runs_is_ten_per_dimension_rounded_to_a_power_of_two(ndim, expected):
    report = recommended_init_runs(ndim)
    assert report["recommended"] == expected
    assert report["floor"] == max(10 * ndim, 16)


def test_undersized_init_runs_is_reported_but_not_enforced():
    report = recommended_init_runs(ndim=7, init_runs=20, max_runs=60)
    assert report["verdict"] == "undersized"
    assert "128" in report["recommendation"]
    # Report-only: the configured value is echoed back, never replaced.
    assert report["configured"] == 20


def test_a_budget_without_room_for_bal_is_flagged():
    report = recommended_init_runs(ndim=3, init_runs=50, max_runs=50)
    assert report["verdict"] == "no_bal_budget"


def test_ndim_below_one_is_rejected():
    with pytest.raises(ValueError, match="at least one parameter"):
        recommended_init_runs(0)


def test_ladder_doubles_and_ends_at_the_ceiling():
    ladder = initial_design_ladder(ceiling=100, ndim=5)
    assert ladder[-1] == 100
    assert ladder == sorted(set(ladder))
    # Every block but the last doubles the one before it; the last is the ceiling.
    doubling = ladder[:-1]
    assert all(doubling[i + 1] == 2 * doubling[i] for i in range(len(doubling) - 1))


def test_ladder_of_a_small_ceiling_is_a_single_block():
    assert initial_design_ladder(ceiling=12, ndim=3) == [12]


# ---------------------------------------------------------------------------
# sampling method
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("given, expected", [
    ("sobol", "sobol"), ("SOBOL", "sobol"), ("Latin_Hypercube", "latin_hypercube"),
    ("chebyshev(FT)", "chebyshev"), ("grid(FT)", "grid"), ("lhs", "latin_hypercube")])
def test_sampling_methods_are_canonicalised(given, expected):
    assert validate_sampling_method(given) == expected


@pytest.mark.parametrize("bad", ["gaussian", "", 3, None])
def test_unusable_sampling_methods_are_rejected_before_any_run(bad):
    with pytest.raises(ValueError):
        validate_sampling_method(bad)


# ---------------------------------------------------------------------------
# extensible Sobol blocks
# ---------------------------------------------------------------------------
def test_sobol_block_continues_the_sequence():
    """The property the staged design rests on: no simulated point is ever discarded."""
    design = _ExpDesign(RANGES)
    first = sobol_block(design, 0, 16)
    second = sobol_block(design, 16, 32, existing=first)

    assert first.shape == (16, 3)
    assert second.shape == (16, 3)
    combined = np.vstack([first, second])
    assert np.allclose(combined[:16], first)
    assert len(np.unique(combined, axis=0)) == 32
    # And the union is exactly the order-32 Sobol design, not first plus something.
    assert np.allclose(combined, design.generate_samples(32, "sobol"))


def test_sobol_block_falls_back_when_the_prefix_does_not_match():
    """A design that cannot be continued must not be silently reordered."""
    design = _ExpDesign(RANGES)
    stale = design.generate_samples(16, "random")  # not the Sobol prefix
    block = sobol_block(design, 16, 32, existing=stale)
    assert block.shape == (16, 3)
    # The fallback block is a fresh Latin hypercube, not the Sobol continuation.
    assert not np.allclose(block, design.generate_samples(32, "sobol")[16:])


def test_a_block_needs_to_grow_the_design():
    with pytest.raises(ValueError, match="n_to > n_from"):
        sobol_block(_ExpDesign(RANGES), 32, 32)


# ---------------------------------------------------------------------------
# the sufficiency gate
# ---------------------------------------------------------------------------
def test_leave_one_out_predictivity_is_high_for_a_smooth_response():
    design = _ExpDesign(RANGES)
    points = design.generate_samples(48, "sobol")
    result = loo_predictivity(points, _response(points), RANGES)
    assert result["q2_median"] > 0.9
    assert 0.0 <= result["coverage"] <= 1.0


def test_leave_one_out_predictivity_collapses_for_noise():
    """Q2 measures prediction at unseen points, so pure noise cannot pass it."""
    rng = np.random.default_rng(0)
    points = rng.random((40, 3))
    result = loo_predictivity(points, rng.normal(size=(40, 6)), RANGES)
    assert result["q2_median"] < 0.5


def test_a_tiny_design_is_insufficient_and_a_larger_one_is_not():
    design = _ExpDesign(RANGES)
    observations = _response(TRUTH[None, :])
    variances = (0.10 * np.abs(observations).ravel()) ** 2
    def prior(n):
        return design.generate_samples(n, "random")

    small = design.generate_samples(8, "sobol")
    poor = initial_design_sufficiency(small, _response(small), observations, variances,
                                      parameter_ranges=RANGES, prior=prior)
    assert poor["verdict"] in ("insufficient", "marginal")

    large = design.generate_samples(64, "sobol")
    first = initial_design_sufficiency(large, _response(large), observations, variances,
                                       parameter_ranges=RANGES, prior=prior)
    grown = design.generate_samples(128, "sobol")
    second = initial_design_sufficiency(grown, _response(grown), observations, variances,
                                        parameter_ranges=RANGES, prior=prior,
                                        previous=first)
    assert second["verdict"] == "sufficient"
    assert second["q2_median"] > 0.9
    assert second["failed"] == []


def test_stability_cannot_be_decided_without_a_previous_block():
    design = _ExpDesign(RANGES)
    points = design.generate_samples(32, "sobol")
    observations = _response(TRUTH[None, :])
    report = initial_design_sufficiency(
        points, _response(points), observations,
        (0.10 * np.abs(observations).ravel()) ** 2, parameter_ranges=RANGES,
        prior=lambda n: design.generate_samples(n, "random"))
    # Undecided is not the same as passed: a first block can never be 'sufficient'.
    assert report["criteria"]["stability"]["passed"] is None
    assert report["verdict"] != "sufficient"


def test_the_gate_never_raises_on_malformed_input():
    report = initial_design_sufficiency(np.zeros((4, 3)), np.zeros((7, 5)),
                                        np.zeros((1, 5)), np.ones(5))
    assert report["verdict"] == "unavailable"
    assert "could not be measured" in report["message"]


# ---------------------------------------------------------------------------
# the staged ladder
# ---------------------------------------------------------------------------
def test_ladder_stops_early_and_never_exceeds_the_ceiling(tmp_path):
    model = _StubModel(tmp_path, init_runs=256)
    design = _ExpDesign(RANGES)

    points, outputs = run_staged_initial_design(model, design)

    achieved = points.shape[0]
    assert achieved <= 256
    assert outputs.shape[0] == achieved
    # Every block continued the previous one: start_index equals the rows already run.
    running = 0
    for start_index, total in model.calls:
        assert start_index == running
        running = total
    assert running == achieved
    # The design that was run is what the BAL budget is computed from.
    assert design.n_init_samples == achieved
    assert model.init_runs == achieved
    assert design.x.shape[0] == achieved


def test_ladder_records_the_achieved_count_for_a_restart(tmp_path):
    model = _StubModel(tmp_path, init_runs=128)
    design = _ExpDesign(RANGES)
    points, _ = run_staged_initial_design(model, design)

    with open(os.path.join(str(tmp_path), "initial-design.json")) as handle:
        record = json.load(handle)
    assert record["achieved_init_runs"] == points.shape[0]
    assert record["configured_init_runs"] == 128
    assert record["stages"]


def test_adaptive_off_runs_the_configured_design_in_one_block(tmp_path):
    model = _StubModel(tmp_path, init_runs=24)
    design = _ExpDesign(RANGES)
    design.x = design.generate_samples(24, "sobol")

    points, _ = run_staged_initial_design(model, design, adaptive=False)

    assert points.shape[0] == 24
    assert model.calls == [(0, 24)]
    assert model.init_runs == 24


def test_adaptive_off_forwards_the_solver_keywords(tmp_path):
    """Keywords such as TELEMAC's extraction window reach the binding without the ladder too."""
    model = _StubModel(tmp_path, init_runs=8)
    design = _ExpDesign(RANGES)
    design.x = design.generate_samples(8, "sobol")

    run_staged_initial_design(model, design, adaptive=False,
                              output_extraction_time="mean_last", n=80)

    assert model.keywords[0]["output_extraction_time"] == "mean_last"
    assert model.keywords[0]["n"] == 80
    assert model.keywords[0]["validation"] is False


# ---------------------------------------------------------------------------
# designs the ladder must not touch
# ---------------------------------------------------------------------------
def test_a_user_supplied_design_is_run_as_given_and_in_one_block(tmp_path):
    """No sampling rule regenerates a user design, so the ladder cannot extend it.

    With the ladder on, which is the default, the first block asked the sampler for rule ``user`` and the calibration stopped before its first simulation.
    """
    user_points = _ExpDesign(RANGES).generate_samples(12, "latin_hypercube")
    model = _StubModel(tmp_path, init_runs=12)
    model.user_collocation_points = user_points
    design = _ExpDesign(RANGES, sampling_method="user")
    design.x = user_points

    points, outputs = run_staged_initial_design(model, design)

    assert model.calls == [(0, 12)]
    assert np.array_equal(points, user_points)
    assert outputs.shape[0] == 12


def test_a_user_supplied_design_of_the_wrong_size_is_rejected_before_any_run(tmp_path):
    """Running fewer or more simulations than configured must not happen silently."""
    user_points = _ExpDesign(RANGES).generate_samples(12, "latin_hypercube")
    model = _StubModel(tmp_path, init_runs=16)
    model.user_collocation_points = user_points
    design = _ExpDesign(RANGES, sampling_method="user")
    design.x = user_points

    with pytest.raises(ValueError, match="init_runs"):
        run_staged_initial_design(model, design)
    assert model.calls == []


def test_a_validation_set_is_not_staged(tmp_path):
    """A validation set has the size that was asked for, whatever an emulator of it would say."""
    model = _StubModel(tmp_path, init_runs=64)
    model.validation = True
    design = _ExpDesign(RANGES)
    design.x = design.generate_samples(64, "sobol")

    points, _ = run_staged_initial_design(model, design)

    assert model.calls == [(0, 64)]
    assert points.shape[0] == 64
    assert model.keywords[0]["validation"] is True



# ---------------------------------------------------------------------------
# blocks of rules that are not extensible sequences
# ---------------------------------------------------------------------------
class _RecordingDesign(_ExpDesign):
    """Records which rule every block was requested with."""

    def __init__(self, ranges, sampling_method="sobol"):
        super().__init__(ranges, sampling_method)
        self.requests = []

    def generate_samples(self, n_samples, sampling_method=None):
        self.requests.append((int(n_samples), sampling_method))
        return super().generate_samples(n_samples, sampling_method)


def _warnings_of(monkeypatch):
    from hydroBayesCal.surrogate import initial_design

    messages = []
    monkeypatch.setattr(initial_design.logger_warn, "warning", lambda message, *a, **k: messages.append(str(message)))
    return messages


@pytest.mark.parametrize("rule", ["random", "latin_hypercube"])
def test_a_rule_without_a_sequence_is_continued_with_a_fresh_block_of_the_same_rule(monkeypatch, rule):
    """A random design extended by random points is a random design, and the record says so.

    Every block after the first was drawn as a Latin hypercube whatever the configured rule, while the log and ``initial-design.json`` kept naming the configured one.
    """
    messages = _warnings_of(monkeypatch)
    design = _RecordingDesign(RANGES)
    existing = design.generate_samples(8, rule)
    design.requests.clear()

    block, used = sobol_block(design, 8, 12, existing=existing, sampling_method=rule, return_rule=True)

    assert block.shape == (4, 3)
    assert design.requests == [(4, rule)]
    assert used == rule
    assert not messages


@pytest.mark.parametrize("rule", ["grid", "chebyshev"])
def test_a_rule_that_cannot_be_continued_falls_back_to_a_latin_hypercube_and_says_so(monkeypatch, rule):
    """The nodes of a grid depend on its size, so a second grid would repeat or miss the first. The fallback was silent."""
    messages = _warnings_of(monkeypatch)
    design = _RecordingDesign(RANGES)

    block, used = sobol_block(design, 8, 12, existing=np.zeros((8, 3)), sampling_method=rule, return_rule=True)

    assert block.shape == (4, 3)
    assert design.requests == [(4, "latin_hypercube")]
    assert used == "latin_hypercube"
    assert len(messages) == 1 and rule in messages[0] and "Latin hypercube" in messages[0]


def test_a_block_is_returned_alone_unless_the_rule_is_asked_for():
    design = _ExpDesign(RANGES)
    block = sobol_block(design, 0, 16)
    assert isinstance(block, np.ndarray) and block.shape == (16, 3)
    _, used = sobol_block(design, 0, 16, return_rule=True)
    assert used == "sobol"


@pytest.mark.parametrize("rule", ["sobol", "random"])
def test_the_record_names_the_rule_every_block_was_drawn_with(tmp_path, rule):
    model = _StubModel(tmp_path, init_runs=128)
    design = _ExpDesign(RANGES, sampling_method=rule)

    run_staged_initial_design(model, design)

    with open(os.path.join(str(tmp_path), "initial-design.json")) as handle:
        record = json.load(handle)
    assert record["sampling_method"] == rule
    assert [stage["sampling_method"] for stage in record["stages"]] == [rule] * len(record["stages"])


# ---------------------------------------------------------------------------
# grid and Chebyshev rules
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rule", TENSOR_RULES)
@pytest.mark.parametrize("n_runs", [8, 27, 30])
def test_a_tensor_rule_returns_the_requested_number_of_runs(monkeypatch, rule, n_runs):
    """chaospy returned n ** ndim parameter sets for n requested ones: 27,000 instead of 30 over three parameters."""
    _warnings_of(monkeypatch)
    points = tensor_rule_samples(_ExpDesign(RANGES), n_runs, rule)

    assert points.shape == (n_runs, 3)
    assert np.unique(points.round(12), axis=0).shape[0] == n_runs
    assert np.all((points > 0.0) & (points < 1.0))


def test_a_complete_tensor_product_keeps_every_node_of_the_rule(monkeypatch):
    messages = _warnings_of(monkeypatch)
    design = _ExpDesign(RANGES)

    points = tensor_rule_samples(design, 27, "chebyshev")
    nodes = design.generate_samples(3, "chebyshev")

    assert np.allclose(np.sort(points, axis=0), np.sort(nodes, axis=0))
    assert not messages


def test_an_incomplete_tensor_product_stays_on_the_nodes_and_says_how_to_complete_it(monkeypatch):
    messages = _warnings_of(monkeypatch)
    design = _ExpDesign(RANGES)

    points = tensor_rule_samples(design, 30, "chebyshev")
    nodes = design.generate_samples(4, "chebyshev")

    assert all(np.any(np.all(np.isclose(nodes, point), axis=1)) for point in points)
    assert len(messages) == 1 and "64" in messages[0]
    assert np.allclose(points, tensor_rule_samples(design, 30, "chebyshev"))


def test_a_tensor_product_that_cannot_be_built_is_refused():
    with pytest.raises(ValueError, match="sobol"):
        tensor_rule_samples(_ExpDesign([[0.0, 1.0]] * 20), 40, "grid")


@pytest.mark.parametrize("rule", ["chebyshev", "sobol"])
def test_the_initial_design_has_the_configured_size(monkeypatch, rule):
    _warnings_of(monkeypatch)
    design = _ExpDesign(RANGES, sampling_method=rule)
    design.n_init_samples = 20

    generate_initial_design(design)

    assert design.x.shape == (20, 3)
    assert design.n_init_samples == 20


def test_the_first_block_of_a_staged_chebyshev_design_has_the_size_of_the_block(monkeypatch):
    _warnings_of(monkeypatch)
    block, used = sobol_block(_ExpDesign(RANGES), 0, 16, sampling_method="chebyshev", return_rule=True)

    assert block.shape == (16, 3)
    assert used == "chebyshev"
