"""Locating the calibration driver scripts that ship with the package.

The drivers (``bal_telemac.py``, ``bal_telemac_multiflow.py``, ``bal_openfoam.py``,
``bal_delft3d.py``, and the analysis helpers beside them) are **scripts, not
importable modules**: ``bal_telemac_multiflow.py`` imports its single-flow sibling by
file name, and each expects to run next to a configuration file. They are shipped as
package data so that ``pip install hydroBayesCal`` gives you the drivers as well as
the library - a downstream package should not have to be told where a source checkout
lives in order to run a calibration.

The intended use is to **copy** a driver next to your config and run it there::

    from hydroBayesCal.drivers import copy_driver
    driver = copy_driver("bal_telemac.py", my_run_dir)      # brings siblings it needs
    # then:  python <driver> --config config_Telemac.py

``drivers_dir()`` and ``driver_path()`` are available when only the location is
wanted, but note that the packaged copy may live inside a zip or a read-only
site-packages tree, so write nothing there.
"""

from __future__ import annotations

import shutil
from pathlib import Path

#: drivers that import a sibling driver, and the sibling(s) they need beside them
_COMPANIONS: dict[str, tuple[str, ...]] = {
    "bal_telemac_multiflow.py": ("bal_telemac.py",),
}


def drivers_dir() -> Path:
    """Directory holding the packaged driver scripts."""
    return Path(__file__).resolve().parent


def available_drivers() -> list[str]:
    """Names of the packaged driver/analysis scripts, sorted."""
    return sorted(p.name for p in drivers_dir().glob("*.py")
                  if p.name != "__init__.py")


def driver_path(name: str) -> Path:
    """Full path of a packaged driver, e.g. ``driver_path("bal_telemac.py")``.

    Raises :class:`FileNotFoundError` naming what *is* available, since a typo here
    is otherwise reported much later as a confusing missing-file error.
    """
    if not name.endswith(".py"):
        name = f"{name}.py"
    path = drivers_dir() / name
    if not path.is_file():
        raise FileNotFoundError(
            f"no packaged driver named {name!r}; available: "
            f"{', '.join(available_drivers())}"
        )
    return path


def copy_driver(name: str, dest_dir, *, overwrite: bool = True) -> Path:
    """Copy a driver (and any sibling it imports) into *dest_dir*; return its path.

    This is the supported way to use a driver: they are written to be run as scripts
    from a working directory that also holds the configuration file, and
    ``bal_telemac_multiflow.py`` imports ``bal_telemac.py`` from beside itself.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    source = driver_path(name)
    targets = [source] + [driver_path(c) for c in _COMPANIONS.get(source.name, ())]
    for src in targets:
        dst = dest_dir / src.name
        if dst.exists() and not overwrite:
            continue
        shutil.copyfile(src, dst)
    return dest_dir / source.name


__all__ = ["available_drivers", "copy_driver", "driver_path", "drivers_dir"]
