"""Tests for writing the value of one GAIA sediment class into its steering-file line."""
import pytest

from hydroBayesCal.function_pool import update_gaia_class_line


@pytest.mark.parametrize("value, written", [
    (0.00025, "0.00025"),      # a fine sand diameter in m: 3 decimals wrote 0.000
    (0.000125, "0.000125"),    # 5 decimals wrote 0.00013
    (0.0472, "0.0472"),
    (2.65e-6, "2.65e-06"),
])
def test_the_value_keeps_its_significant_digits(value, written):
    line = "CLASSES SEDIMENT DIAMETERS = 0.0005;0.002"

    assert update_gaia_class_line(line, 1, value) == f"CLASSES SEDIMENT DIAMETERS = 0.0005;{written}"


def test_the_colon_form_and_the_index_are_kept():
    assert update_gaia_class_line("CLASSES SHIELDS PARAMETERS : 0.047;0.05", 0, 0.03) == \
        "CLASSES SHIELDS PARAMETERS : 0.03;0.05"
    with pytest.raises(IndexError):
        update_gaia_class_line("CLASSES SHIELDS PARAMETERS = 0.047;0.05", 2, 0.03)
