"""The turbulence dictionary is named differently in the two OpenFOAM lines.

ESI (openfoam.com) keeps ``constant/turbulenceProperties``; OpenFOAM Foundation
8+ renamed it to ``constant/momentumTransport``. The OpenFOAM config's default -
and therefore what callers pass in ``param["file"]`` - is the ESI spelling, so
honouring it unconditionally targets a file that does not exist on a Foundation
case. That fails on the FIRST perturbed run, i.e. after the whole experimental
design has already been simulated.

Resolution therefore keys on which file actually exists, not on its name.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from hydroBayesCal.openfoam.control_openfoam import OpenFOAMController

ESI = "constant/turbulenceProperties"
FOUNDATION = "constant/momentumTransport"


def _case(*dictionaries: str) -> str:
    root = tempfile.mkdtemp()
    os.makedirs(os.path.join(root, "constant"), exist_ok=True)
    for name in dictionaries:
        open(os.path.join(root, name), "w").close()
    return root


def _resolve(controller, requested: str) -> str:
    """The resolution `update_model_controls` performs for a coefficient."""
    if not os.path.isfile(controller._case_path(requested)):
        return controller._turbulence_dictionary()
    return requested


def test_foundation_case_resolves_away_from_the_esi_name():
    controller = OpenFOAMController(_case(FOUNDATION))
    assert controller._turbulence_dictionary() == FOUNDATION
    # the caller asks for the ESI spelling, as config_OpenFOAM.py's default does
    assert _resolve(controller, ESI) == FOUNDATION


def test_esi_case_keeps_the_callers_file():
    controller = OpenFOAMController(_case(ESI))
    assert controller._turbulence_dictionary() == ESI
    assert _resolve(controller, ESI) == ESI


def test_neither_present_names_the_conventional_path():
    """A missing-file error should name the conventional dictionary rather than
    whichever spelling happened to be guessed."""
    controller = OpenFOAMController(_case())
    assert controller._turbulence_dictionary() == ESI


@pytest.mark.parametrize("present,requested,expected", [
    (FOUNDATION, ESI, FOUNDATION),
    (FOUNDATION, FOUNDATION, FOUNDATION),
    (ESI, ESI, ESI),
    (ESI, FOUNDATION, ESI),
])
def test_resolution_always_lands_on_a_file_that_exists(present, requested, expected):
    controller = OpenFOAMController(_case(present))
    resolved = _resolve(controller, requested)
    assert resolved == expected
    assert os.path.isfile(controller._case_path(resolved))
