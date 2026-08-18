"""Tests for the modeled-vs-measured calibration-target diagnostic and its plot.

What is pinned here is the verdict logic, because it is what a modeller acts on: a
systematic offset says "keep calibrating, the parameter is not there yet", scatter says
"stop, no parameter value can fix this", and confusing the two sends a calibration in the
wrong direction. The interleaved ``[nloc * nq]`` column layout is pinned alongside it,
since a wrong split would mix two calibration targets into one verdict. The figure itself
is only smoke-tested (LaTeX is a system dependency).
"""
import numpy as np
import pytest

from hydroBayesCal.surrogate.target_agreement import (
    AGREEMENT_KEY,
    diagnose_target_agreement,
    finalize_target_agreement,
    log_target_agreement,
    measurement_error_bars,
    target_agreement_data,
)

RNG = np.random.default_rng(20260818)
QUANTITIES = ["WATER DEPTH", "SCALAR VELOCITY"]
N_LOCATIONS = 24


def _interleave(depth, velocity):
    """Two per-target series into the [nloc * nq] layout the whole package assumes."""
    stacked = np.empty(depth.size + velocity.size)
    stacked[0::2] = depth
    stacked[1::2] = velocity
    return stacked


def _measured():
    depth = RNG.uniform(0.4, 2.5, N_LOCATIONS)
    velocity = RNG.uniform(0.2, 1.4, N_LOCATIONS)
    return _interleave(depth, velocity)


def _ensemble(measured, depth_factor=1.0, velocity_factor=1.0, noise=0.0, n_members=8):
    """An ensemble whose median is the measurement scaled per calibration target."""
    depth = measured[0::2] * depth_factor
    velocity = measured[1::2] * velocity_factor
    center = _interleave(depth, velocity)
    members = np.tile(center, (n_members, 1))
    if noise:
        members = members + RNG.normal(0.0, noise, members.shape)
    return members


def _data(pre=None, post=None, error_fraction=0.05):
    measured = _measured()
    return target_agreement_data(
        observations=measured.reshape(1, -1),
        calibration_quantities=QUANTITIES,
        errors=np.abs(measured) * error_fraction,
        initial_model_outputs=(_ensemble(measured, **pre) if pre else None),
        posterior_predictions=(_ensemble(measured, **post) if post else None),
    )


def test_a_uniform_offset_is_reported_as_systematic_overestimation():
    data = _data(post={"depth_factor": 1.25, "velocity_factor": 1.0})
    diagnosis = diagnose_target_agreement(data)

    depth = diagnosis["states"]["post"]["targets"]["WATER DEPTH"]
    assert depth["verdict"] == "overestimation"
    assert depth["systematic"] is True
    assert depth["relative_bias"] == pytest.approx(0.25, abs=0.02)
    assert depth["sign_consistency"] == 1.0

    # The second calibration target is untouched and must not inherit the verdict.
    assert diagnosis["states"]["post"]["targets"]["SCALAR VELOCITY"]["verdict"] == "agreement"
    assert diagnosis["verdict"] == "systematic_deviation"


def test_a_uniform_deficit_is_reported_as_systematic_underestimation():
    data = _data(post={"depth_factor": 0.7, "velocity_factor": 0.75})
    diagnosis = diagnose_target_agreement(data)

    for quantity in QUANTITIES:
        target = diagnosis["states"]["post"]["targets"][quantity]
        assert target["verdict"] == "underestimation"
        assert target["bias"] < 0.0
    assert diagnosis["verdict"] == "systematic_deviation"


@pytest.mark.parametrize("offset_high, offset_low", [(0.3, -0.3), (0.4, -0.2)])
def test_sign_changing_residuals_are_scatter_and_not_a_bias(offset_high, offset_low):
    """A mismatch that changes sign across the calibration points is not a parameter
    problem, and reporting it as one would send the modeller after the wrong knob.

    The first case cancels in the mean and the second does not: neither may come out as
    agreement, and neither as a systematic deviation.
    """
    measured = _measured()
    alternating = measured * (1.0 + np.where(np.arange(measured.size) % 4 < 2,
                                             offset_high, offset_low))
    data = target_agreement_data(
        observations=measured,
        calibration_quantities=QUANTITIES,
        errors=np.abs(measured) * 0.02,
        posterior_predictions=np.tile(alternating, (5, 1)),
    )
    diagnosis = diagnose_target_agreement(data)

    for quantity in QUANTITIES:
        target = diagnosis["states"]["post"]["targets"][quantity]
        assert target["verdict"] == "scatter"
        assert target["systematic"] is False
        assert target["sign_consistency"] < 0.6
    assert diagnosis["verdict"] == "scatter_dominated"


def test_a_small_offset_inside_the_measurement_uncertainty_is_agreement():
    data = _data(post={"depth_factor": 1.005, "velocity_factor": 0.995},
                 error_fraction=0.10)
    diagnosis = diagnose_target_agreement(data)

    assert diagnosis["verdict"] == "calibrated"
    for quantity in QUANTITIES:
        assert diagnosis["states"]["post"]["targets"][quantity]["verdict"] == "agreement"


def test_both_states_are_diagnosed_and_the_improvement_is_reported():
    data = _data(pre={"depth_factor": 1.4, "velocity_factor": 0.6},
                 post={"depth_factor": 1.01, "velocity_factor": 0.99})
    diagnosis = diagnose_target_agreement(data)

    assert set(diagnosis["states"]) == {"pre", "post"}
    assert diagnosis["states"]["pre"]["targets"]["WATER DEPTH"]["verdict"] == "overestimation"
    assert diagnosis["states"]["post"]["targets"]["WATER DEPTH"]["verdict"] == "agreement"
    # The overall verdict describes the calibrated state, not the initial design.
    assert diagnosis["verdict"] == "calibrated"
    assert diagnosis["improved"] is True
    assert diagnosis["rmse_post"] < diagnosis["rmse_pre"]


def test_a_calibration_that_did_not_help_is_not_reported_as_an_improvement():
    data = _data(pre={"depth_factor": 1.05, "velocity_factor": 1.0},
                 post={"depth_factor": 1.30, "velocity_factor": 1.0})
    diagnosis = diagnose_target_agreement(data)

    assert diagnosis["improved"] is False


def test_the_roughness_diagnostic_is_called_per_state(monkeypatch):
    """Wiring test: the diagnostic has to see one state's ensemble against the
    measurements, in the interleaved layout it assumes. Runs without the mesh stack that
    the real function_pool import needs."""
    import sys
    import types

    calls = []
    stub = types.ModuleType("hydroBayesCal.function_pool")

    def _fake(model_outputs, observations, calibration_quantities, **kwargs):
        calls.append((np.asarray(model_outputs), np.asarray(observations),
                      list(calibration_quantities)))
        return {"verdict": "roughness_too_high", "message": "stubbed"}

    stub.diagnose_roughness_identifiability = _fake
    monkeypatch.setitem(sys.modules, "hydroBayesCal.function_pool", stub)

    data = _data(pre={"depth_factor": 1.3, "velocity_factor": 0.7},
                 post={"depth_factor": 1.0, "velocity_factor": 1.0})
    diagnosis = diagnose_target_agreement(data)

    assert len(calls) == 2
    for outputs, observations, quantities in calls:
        assert outputs.shape[-1] == observations.size == N_LOCATIONS * len(QUANTITIES)
        assert quantities == QUANTITIES
    assert diagnosis["states"]["pre"]["roughness"]["verdict"] == "roughness_too_high"
    assert "stubbed" in diagnosis["message"]


def test_the_roughness_reading_travels_with_each_state():
    """Simulated too deep and too slow is the anti-correlated pattern that makes
    roughness the identifiable calibration parameter."""
    pytest.importorskip("h5py")
    pytest.importorskip("pyvista")
    data = _data(pre={"depth_factor": 1.3, "velocity_factor": 0.7},
                 post={"depth_factor": 1.0, "velocity_factor": 1.0})
    diagnosis = diagnose_target_agreement(data)

    assert diagnosis["states"]["pre"]["roughness"]["verdict"] == "roughness_too_high"
    assert diagnosis["states"]["pre"]["roughness"]["identifiable"] is True
    assert diagnosis["states"]["post"]["roughness"]["verdict"] == "inconclusive"


def test_the_states_keep_the_calibration_targets_apart():
    """A wrong column split would average two calibration targets into one verdict."""
    data = _data(post={"depth_factor": 1.5, "velocity_factor": 1.0})

    assert data["n_locations"] == N_LOCATIONS
    modeled = data["states"]["post"]["modeled"]
    assert np.allclose(modeled[1::2] / data["measured"][1::2], 1.0)
    assert np.allclose(modeled[0::2] / data["measured"][0::2], 1.5)


def test_mismatched_input_lengths_are_rejected():
    measured = _measured()
    with pytest.raises(ValueError):
        target_agreement_data(observations=measured,
                              calibration_quantities=QUANTITIES,
                              errors=measured[:-1],
                              posterior_predictions=np.tile(measured, (3, 1)))
    with pytest.raises(ValueError, match="initial-design outputs"):
        target_agreement_data(observations=measured,
                              calibration_quantities=QUANTITIES)


def test_a_state_of_the_wrong_width_is_skipped_rather_than_fatal():
    measured = _measured()
    data = target_agreement_data(
        observations=measured,
        calibration_quantities=QUANTITIES,
        posterior_predictions=np.tile(measured, (4, 1)),
        initial_model_outputs=np.tile(measured[:-2], (4, 1)),   # wrong number of columns
    )
    assert set(data["states"]) == {"post"}


class _Model:
    """The attribute surface finalize_target_agreement reads off a model binding."""

    def __init__(self, tmp_path, measured, errors, dataframe=None):
        self.observations = measured.reshape(1, -1)
        self.measurement_errors = errors
        self.calibration_quantities = QUANTITIES
        self.calibration_pts_df = dataframe
        self.asr_dir = str(tmp_path)
        self.calibration_folder = str(tmp_path)


def test_measurement_error_bars_add_the_measured_fluctuations():
    pandas = pytest.importorskip("pandas")
    measured = _measured()
    relative = np.abs(measured) * 0.1
    fluctuations = np.full(measured.size, 0.05)
    dataframe = pandas.DataFrame({
        "WATER DEPTH_ERROR": fluctuations[0::2],
        "SCALAR VELOCITY_ERROR": fluctuations[1::2],
    })
    model = _Model(".", measured, relative, dataframe)

    assert np.allclose(measurement_error_bars(model),
                       np.sqrt(relative ** 2 + fluctuations ** 2))


def test_measurement_error_bars_fall_back_to_the_stored_errors():
    """Without the _ERROR columns the vector the likelihood used is the honest one."""
    measured = _measured()
    relative = np.abs(measured) * 0.1
    assert np.allclose(measurement_error_bars(_Model(".", measured, relative)), relative)


def test_finalize_stores_the_block_and_writes_the_figure(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    matplotlib.rcParams["text.usetex"] = False

    measured = _measured()
    model = _Model(tmp_path, measured, np.abs(measured) * 0.08)
    bayesian_dict = {"N_tp": np.arange(3)}

    diagnosis = finalize_target_agreement(
        complex_model=model,
        bayesian_dict=bayesian_dict,
        initial_model_outputs=_ensemble(measured, depth_factor=1.4, velocity_factor=0.7),
        posterior_predictions=_ensemble(measured, depth_factor=1.02, noise=0.01,
                                        n_members=40),
    )

    assert diagnosis["verdict"] in {"calibrated", "systematic_deviation", "scatter_dominated"}
    stored = bayesian_dict[AGREEMENT_KEY]
    assert set(stored["data"]["states"]) == {"pre", "post"}
    # The bulky ensembles stay out of the archived dictionary.
    assert "ensemble" not in stored["data"]["states"]["pre"]
    assert (tmp_path / "BAL_dictionary.pkl").is_file()

    figures = sorted((tmp_path / "plots" / "_".join(QUANTITIES)).glob("*"))
    assert [path.name for path in figures] == ["calibration-target-agreement.png",
                                               "calibration-target-agreement.svg"]

    # An archived block must re-diagnose without the ensembles it no longer carries.
    assert diagnose_target_agreement(stored["data"])["verdict"] == diagnosis["verdict"]


def test_finalize_never_raises_on_unusable_input(tmp_path):
    """A finished calibration must not be lost to a post-processing failure."""
    measured = _measured()
    model = _Model(tmp_path, measured, np.abs(measured) * 0.08)
    bayesian_dict = {}

    assert finalize_target_agreement(complex_model=model,
                                     bayesian_dict=bayesian_dict) is None
    assert AGREEMENT_KEY not in bayesian_dict


def test_log_target_agreement_returns_the_diagnosis():
    diagnosis = diagnose_target_agreement(_data(post={"depth_factor": 1.3}))
    assert log_target_agreement(diagnosis) is diagnosis
