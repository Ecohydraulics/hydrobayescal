"""Tests for the resolved turbulent kinetic energy of the OpenFOAM binding, 0.5 tr(UPrime2Mean), and the TKE parts it reports."""
import os

import numpy as np
import pytest

from hydroBayesCal.openfoam.control_openfoam import OpenFOAMModel, _resolved_tke

#: Header of an OpenFOAM dictionary.
HEADER = 'FoamFile\n{{\n    version     2.0;\n    format      ascii;\n    class       {cls};\n    location    "{loc}";\n    object      {obj};\n}}\n'


def _write_case(root, tensor=(1, 2, 3, 4, 5, 6)):
    """A one-cell OpenFOAM case whose UPrime2Mean holds the components (xx, xy, xz, yy, yz, zz) = tensor."""
    mesh = os.path.join(root, "constant", "polyMesh")
    os.makedirs(mesh, exist_ok=True); os.makedirs(os.path.join(root, "system"), exist_ok=True); os.makedirs(os.path.join(root, "0"), exist_ok=True)
    def w(path, cls, loc, obj, body):
        with open(os.path.join(root, path), "w") as f:
            f.write(HEADER.format(cls=cls, loc=loc, obj=obj) + body)
    pts = [(0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1)]
    w("constant/polyMesh/points", "vectorField", "constant/polyMesh", "points",
      "8\n(\n" + "".join(f"({x} {y} {z})\n" for x, y, z in pts) + ")\n")
    faces = [(0,3,2,1),(4,5,6,7),(0,1,5,4),(2,3,7,6),(0,4,7,3),(1,2,6,5)]   # outward normals
    w("constant/polyMesh/faces", "faceList", "constant/polyMesh", "faces",
      "6\n(\n" + "".join("4(" + " ".join(map(str, f)) + ")\n" for f in faces) + ")\n")
    w("constant/polyMesh/owner", "labelList", "constant/polyMesh", "owner", "6\n(\n0\n0\n0\n0\n0\n0\n)\n")
    w("constant/polyMesh/neighbour", "labelList", "constant/polyMesh", "neighbour", "0\n(\n)\n")
    w("constant/polyMesh/boundary", "polyBoundaryMesh", "constant/polyMesh", "boundary",
      "1\n(\n    walls\n    {\n        type wall;\n        nFaces 6;\n        startFace 0;\n    }\n)\n")
    w("system/controlDict", "dictionary", "system", "controlDict",
      "application icoFoam;\nstartFrom startTime;\nstartTime 0;\nstopAt endTime;\nendTime 1;\ndeltaT 1;\nwriteControl timeStep;\nwriteInterval 1;\n")
    w("0/UPrime2Mean", "volSymmTensorField", "0", "UPrime2Mean",
      "dimensions [0 2 -2 0 0 0 0];\ninternalField uniform (" + " ".join(map(str, tensor)) + ");\nboundaryField\n{\n    walls\n    {\n        type zeroGradient;\n    }\n}\n")
    open(os.path.join(root, "case.foam"), "w").close()
    return os.path.join(root, "case.foam")


def test_the_reader_keeps_the_openfoam_order_of_a_symmetric_tensor(tmp_path):
    """pyvista returns (xx, xy, xz, yy, yz, zz); the normal stresses are the components 0, 3 and 5."""
    pv = pytest.importorskip("pyvista")
    reader = pv.OpenFOAMReader(_write_case(str(tmp_path)))
    reader.set_active_time_value(0.0)

    stored = reader.read()["internalMesh"].cell_data["UPrime2Mean"]

    assert stored[0].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert _resolved_tke(stored)[0] == pytest.approx(0.5 * (1.0 + 4.0 + 6.0))


def test_positive_shear_stresses_are_not_read_as_normal_stresses():
    """With xy and xz positive, reading the first three components gives xx + xy + xz."""
    R = np.array([[0.04, 0.01, 0.02, 0.09, 0.0, 0.16]])

    assert _resolved_tke(R)[0] == pytest.approx(0.5 * (0.04 + 0.09 + 0.16))


def test_a_full_tensor_uses_its_diagonal_and_round_off_is_clipped():
    R = np.array([[1.0, 9.0, 9.0, 9.0, 2.0, 9.0, 9.0, 9.0, -1e-12]])

    assert _resolved_tke(R)[0] == pytest.approx(1.5)
    with pytest.raises(ValueError, match="6 or 9"):
        _resolved_tke(np.ones((2, 3)))


def _model_with_points(points):
    model = OpenFOAMModel.__new__(OpenFOAMModel)
    model.control_points = np.asarray(points, dtype=float)
    return model


def test_the_tke_parts_add_up_to_the_total():
    model = _model_with_points([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    U = np.ones((2, 3))
    k_resolved = np.array([0.01, 0.02])
    k_total = np.array([0.05, 0.07])

    results = model._extract_at_control_points(coords, U, k_total, None, k_resolved=k_resolved)

    assert results["TKE"].tolist() == pytest.approx([0.05, 0.07])
    assert results["TKE_RESOLVED"].tolist() == pytest.approx([0.01, 0.02])
    assert results["TKE_MODELED"].tolist() == pytest.approx([0.04, 0.05])
    assert set(results) == set(OpenFOAMModel.EXTRACTABLE_QUANTITIES)


def test_without_uprime2mean_the_resolved_part_is_unknown():
    model = _model_with_points([[0.0, 0.0, 0.0]])

    results = model._extract_at_control_points(np.zeros((1, 3)), np.ones((1, 3)), np.array([0.05]), None)

    assert results["TKE_MODELED"].tolist() == pytest.approx([0.05])
    assert np.isnan(results["TKE_RESOLVED"]).all()
