"""Tests for the labels of parameter values in the posterior plots."""
import pytest

pytest.importorskip("matplotlib")

from hydroBayesCal.visualize.posterior_plots import _format_number


@pytest.mark.parametrize("value, label", [
    (512.347, "512.3"), (0.0315, "0.0315"), (0.05, "0.05"), (12.5, "12.5"), (1500.0, "1500"), (0.0, "0"),
    (-2.25, "-2.25"), (4.5e-4, r"$4.5\times10^{-4}$"), (2.5e-5, r"$2.5\times10^{-5}$"),
    (3.2e5, r"$3.2\times10^{5}$"), (9.99996e-4, "0.001"), (9.99996e-5, r"$1\times10^{-4}$")])
def test_an_annotated_value_keeps_four_significant_digits(value, label):
    """The MAP annotation was '.3f' up to 1.11: 512.347 must not become 5.1x10^2."""
    assert _format_number(value, significant=4) == label


def test_tick_labels_keep_three_significant_digits_in_one_style():
    """Ticks of linspace(0.01, 0.05, 4) were '0.01', '2.3x10^-2', '3.7x10^-2', '0.05'."""
    assert [_format_number(v) for v in (0.01, 0.0233333, 0.0366667, 0.05)] == ["0.01", "0.0233", "0.0367", "0.05"]
    assert _format_number(0.5, 0) == "0.5", "matplotlib's FuncFormatter passes the tick position"
