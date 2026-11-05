"""Tests for the per-location validation metrics written by ``MetricsPlots.location_metrics``."""
import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from hydroBayesCal.visualize.metrics_plots import MetricsPlots


def _entry(train_points, quantity, surrogate_type, rmse):
    """One entry of ``metrics_per_location`` as ``validate_simulation.py`` builds it, for two locations."""
    return {"TrainPoints": train_points, "Quantity": quantity, "SurrogateType": [surrogate_type] * 2,
            "LocationIdx": [0, 1], "MSE": [r ** 2 for r in rmse], "RMSE": rmse, "MAE": rmse,
            "Correlation": [0.9, 0.8], "CI": [0.1, 0.2]}


def test_only_the_evaluated_surrogates_need_per_location_entries(tmp_path):
    """validate_simulation.py fills SurrogateType for every analyzed surrogate, metrics_per_location only for the evaluated ones."""
    plotter = MetricsPlots.__new__(MetricsPlots)
    plotter.save_folder = str(tmp_path)
    metrics = {"TrainPoints": [20, 50, 100] * 2, "Quantity": ["WATER DEPTH"] * 6,
               "SurrogateType": ["SO"] * 3 + ["MO"] * 3,
               "metrics_per_location": [_entry(100, "WATER DEPTH", "SO", [0.1, 0.2]),
                                        _entry(100, "WATER DEPTH", "MO", [0.3, 0.4])]}

    plotter.location_metrics(metrics, pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 0.0]}))

    raw = pd.read_csv(tmp_path / "metrics-locations-tp-raw.csv")
    assert raw.groupby("SurrogateType")["RMSE"].apply(list).to_dict() == {"MO": [0.3, 0.4], "SO": [0.1, 0.2]}
    summary = pd.read_csv(tmp_path / "metrics-locations-tp-mean-std.csv")
    assert "LocationIdx_Mean" not in summary.columns and "RMSE_Mean" in summary.columns
