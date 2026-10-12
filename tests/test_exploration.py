"""Tests for the space-filling scores of candidate parameter sets (:class:`Exploration`).

The scores follow the Monte Carlo criteria of Crombecq et al. (2011): a candidate is the better, the farther it lies from every parameter set that has been simulated. They are the exploration half of the exploration/exploitation trade-off of Bayesian active learning.
"""
import numpy as np
import pytest

from hydroBayesCal.surrogate.exploration import Exploration


class _Design:
    """What :class:`Exploration` reads from the experimental design."""
    explore_method = "random"

    def generate_samples(self, n_samples, sampling_method="random"):
        return np.random.default_rng(5).uniform(size=(n_samples, 2))


def test_the_projected_distance_enters_the_score_as_a_distance():
    """Both terms of the intersite-projected score are distances.

    The Chebyshev distance was requested without ``do_sqrt`` and came back squared, so that the two terms were on different scales: 0.3 counted as 0.09 next to an Euclidean distance of 0.32.
    """
    training_points = np.array([[0.0, 0.0]])
    candidates = np.array([[0.3, 0.1], [0.2, 0.2]])
    exploration = Exploration(n_candidate=2, old_tp=training_points, exp_design=_Design(), mc_criterion="mc-intersite-proj")

    _, scores = exploration.get_mc_samples(all_candidates=candidates)

    weight_euclidean, weight_projected = 0.5 * (3 ** 0.5 - 1), 0.5 * 3  # two candidates, two parameters
    expected = np.array([weight_euclidean * np.hypot(0.3, 0.1) + weight_projected * 0.3,
                         weight_euclidean * np.hypot(0.2, 0.2) + weight_projected * 0.2])
    assert scores == pytest.approx(expected / expected.sum())


def test_the_threshold_criterion_drops_the_candidates_that_are_too_close_in_projection():
    """``mc-intersite-proj-th`` removes candidates below the projected-distance threshold and scores the rest.

    The mask of the candidates to remove was passed to ``np.delete`` inside a list. NumPy 2 refuses that; older versions removed rows 0 and 1 instead of the masked rows, which left candidates and scores misaligned.
    """
    training_points = np.array([[0.5, 0.5]])
    candidates = np.array([[0.60, 0.55],   # projected distance 0.10: too close
                           [0.90, 0.90],   # 0.40
                           [0.10, 0.50],   # 0.40
                           [0.50, 0.70]])  # 0.20: too close (threshold 2 * 0.5 / 4 = 0.25)
    exploration = Exploration(n_candidate=4, old_tp=training_points, exp_design=_Design(), mc_criterion="mc-intersite-proj-th")

    kept, scores = exploration.get_mc_samples(all_candidates=candidates)

    assert np.array_equal(kept, candidates[[1, 2]])
    distances = np.array([np.hypot(0.4, 0.4), 0.4])
    assert scores == pytest.approx(distances / distances.sum())


def test_the_threshold_criterion_keeps_all_candidates_that_are_far_enough():
    training_points = np.array([[0.0, 0.0]])
    candidates = np.array([[0.9, 0.9], [0.5, 0.8]])
    exploration = Exploration(n_candidate=2, old_tp=training_points, exp_design=_Design(), mc_criterion="mc-intersite-proj-th")

    kept, scores = exploration.get_mc_samples(all_candidates=candidates)

    assert np.array_equal(kept, candidates)
    assert scores.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("criterion", ["mc-intersite-proj", "mc-intersite-proj-th"])
def test_voronoi_sampling_returns_one_candidate_per_cell(criterion):
    training_points = np.array([[0.2, 0.2], [0.8, 0.3], [0.5, 0.8]])
    exploration = Exploration(n_candidate=1, old_tp=training_points, exp_design=_Design(), mc_criterion=criterion, w=40)

    new_samples, scores = exploration.get_vornoi_samples()

    assert new_samples.ndim == 2 and new_samples.shape[1] == 2
    assert 1 <= len(new_samples) <= len(training_points)
    assert len(scores) == len(training_points)
    assert scores.sum() == pytest.approx(1.0), "the cell volumes are fractions of the domain"
