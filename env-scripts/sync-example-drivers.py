#!/usr/bin/env python3
"""Regenerate the example copies of ``bal_telemac.py`` from the template.

The TELEMAC example folders carry their own copy of the driver, so that a case can be started from its folder. A copy differs from ``src/hydroBayesCal/drivers/bal_telemac.py`` in two places only: one header line that names it as a copy, and the default of ``--config``. Everything else has to be the template, because a change of the calibration workflow made in a copy reaches no other case and is lost with the next update of the template. ``tests/test_shipped_defaults.py`` compares every copy with what this script generates.

Usage::

    python env-scripts/sync-example-drivers.py           # rewrite the copies
    python env-scripts/sync-example-drivers.py --check   # report differences, exit 1 if there are any
"""
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
TEMPLATE = REPO / "src" / "hydroBayesCal" / "drivers" / "bal_telemac.py"
TEMPLATE_CONFIG = "config_Telemac.py"

#: Example copy -> (name of the case, default configuration file in that folder).
EXAMPLE_DRIVERS = {
    "examples/Telemac/Hydromorphodynamic/Ering/bal_telemac.py": ("Ering", "config_Ering.py"),
    "examples/Telemac/Telemac2d/Ering/hydrodynamic_calibration/bal_telemac.py": ("Ering hydrodynamic", "config_Telemac.py"),
    "examples/Telemac/Telemac3d/cylinderFlume/bal_telemac.py": ("Cylinder flume", "config.py"),
}


def expected_copy(case, config, template_text=None):
    """Text of the example copy for one case, generated from the template."""
    text = TEMPLATE.read_text() if template_text is None else template_text
    if not text.startswith('"""\n'):
        raise ValueError(f"{TEMPLATE.name} no longer starts with a docstring; adapt this script.")
    if text.count(TEMPLATE_CONFIG) != 2:
        raise ValueError(f"expected the default configuration {TEMPLATE_CONFIG} twice in {TEMPLATE.name}, "
                         f"found it {text.count(TEMPLATE_CONFIG)} times; adapt this script.")
    text = text.replace(TEMPLATE_CONFIG, config)
    header = (f"{case} case driver: copy of src/hydroBayesCal/drivers/bal_telemac.py with {config} as the default "
              f"configuration. Keep in sync with the template (python env-scripts/sync-example-drivers.py).\n")
    return '"""\n' + header + text[len('"""\n'):]


def main(argv):
    check = "--check" in argv
    stale = []
    for relative_path, (case, config) in EXAMPLE_DRIVERS.items():
        path = REPO / relative_path
        expected = expected_copy(case, config)
        if path.is_file() and path.read_text() == expected:
            continue
        stale.append(relative_path)
        if not check:
            path.write_text(expected)
    for relative_path in stale:
        print(f"{'differs from the template' if check else 'rewritten'}: {relative_path}")
    if not stale:
        print("all example drivers match the template")
    return 1 if (check and stale) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
