.. _installation:

Installation
============

HydroBayesCal is a Python package bound to an external numerical model
(TELEMAC is fully supported; OpenFOAM and Delft3D-FLOW bindings are in
progress).
It is developed and tested
on **Debian/Ubuntu Linux**, which we recommend: Linux gives the flexibility to
configure the environment and TELEMAC together, and a command-line interface
well suited to batch simulations. Windows has not been tested.

This page covers (1) the Python environment and the package itself, and
(2) the numerical-model bindings.

.. contents::
   :local:
   :depth: 2

Python environment
------------------

Dependencies are declared once in ``pyproject.toml``; the legacy
``requirements*.txt`` files simply install the project. Use a dedicated
environment.

.. important::

   HydroBayesCal requires **Python >= 3.10** and is tested on **CPython
   3.10, 3.11 and 3.12**. No upper bound is enforced, but newer interpreters
   are only supported once the dependency stack provides compatible wheels.

Option A — from PyPI (recommended)
++++++++++++++++++++++++++++++++++

Install the released package with pip (preferably into a fresh virtual
environment):

.. code-block:: bash

   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install hydroBayesCal

The core install pulls the full surrogate/UQ stack (NumPy, SciPy, pandas,
scikit-learn, PyTorch, GPyTorch, emcee, chaospy, PyVista and BayesValidRox).

Option B — conda / mamba (development)
++++++++++++++++++++++++++++++++++++++

A ready-made environment file is provided. From the repository root:

.. code-block:: bash

   mamba env create -f environment.yml   # or: conda env create -f environment.yml
   mamba activate hbenv

This creates a Python 3.11 environment and installs HydroBayesCal in editable
mode with its documentation, mesh and model-server extras.

Option C — editable install from a clone
++++++++++++++++++++++++++++++++++++++++

HydroBayesCal uses a ``src`` layout and is installed like any modern Python
package:

.. code-block:: bash

   git clone https://github.com/sschwindt/hydrobayescal.git
   cd hydrobayescal
   pip install -e ".[dev,docs,mesh]"

Optional feature sets (*extras*) can be added in brackets (e.g.
``pip install "hydroBayesCal[mesh]"``):

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Extra
     - Purpose
   * - ``mesh``
     - Extra mesh / geospatial IO (``vtk``, ``meshio``, ``rasterio``) for some
       post-processing utilities.
   * - ``server``
     - UM-Bridge model-server coupling (``umbridge``).
   * - ``mpi``
     - Parallel execution on HPC clusters (``mpi4py``; needs an MPI toolchain).
   * - ``docs``
     - Build this documentation (Sphinx + RTD theme + Mermaid).
   * - ``dev``
     - Development tooling (``pytest``, ``build``, ``twine``, ``ruff``).

For example, a developer install with documentation and mesh tools:

.. code-block:: bash

   pip install -e ".[dev,docs,mesh]"

.. warning::

   **Large download — multi-GB.** Installing HydroBayesCal pulls in PyTorch,
   which on Linux ``x86_64`` brings a full set of **NVIDIA CUDA wheels**
   (``nvidia-cublas``, ``nvidia-cudnn``, ``nvidia-cusolver``, ``triton`` …).
   Together these amount to **several gigabytes**, and they are downloaded *even
   on machines without an NVIDIA GPU*. On metered, disk-constrained or
   CPU-only systems, install the CPU build of PyTorch first, from its dedicated
   index, then install HydroBayesCal:

   .. code-block:: bash

      pip install torch --index-url https://download.pytorch.org/whl/cpu
      pip install hydroBayesCal

   (``pip`` keeps the already-installed CPU ``torch`` instead of fetching the
   CUDA build.) The ``vtk`` wheel pulled in via PyVista is also large
   (~hundreds of MB). On clusters without a working MPI toolchain, omit the
   ``mpi`` extra.

.. _latex-for-plots:

LaTeX for the plots (system packages)
-------------------------------------

Every figure is rendered with matplotlib's LaTeX text mode: ``BayesianPlotter``
sets ``text.usetex = True`` and a Times serif font. This is a **system**
dependency, not a Python one, so ``pip`` cannot supply it. Without it every
plotting call fails with a ``RuntimeError`` from matplotlib quoting a LaTeX
error such as ``File 'type1cm.sty' not found``, no matter which Python
environment is active.

What is required, independently of the platform:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Component
     - Why it is needed
   * - a LaTeX engine
     - the ``latex`` binary that typesets the text, plus the base packages
       ``fix-cm``, ``inputenc`` and ``textcomp``
   * - ``type1cm``
     - lets matplotlib scale the fonts to arbitrary sizes; **missing this is by
       far the most common failure**, reported as
       ``File 'type1cm.sty' not found``
   * - ``cm-super``
     - scalable Type 1 versions of the Computer Modern fonts, needed at the
       axis-label sizes used here
   * - ``psnfss`` / Times
     - the ``mathptmx`` Times font selected by the plotting style
   * - ``dvipng``
     - converts the LaTeX output to raster images; required for the ``Agg``
       backend used in headless and batch runs
   * - Ghostscript
     - used by matplotlib for some output formats

Platform-specific commands
++++++++++++++++++++++++++

**Debian / Ubuntu**

.. code-block:: bash

   sudo apt install texlive-latex-base texlive-latex-extra \
                    texlive-fonts-recommended cm-super dvipng ghostscript

(``type1cm`` lives in ``texlive-latex-extra``, which is *not* pulled in by
``texlive-latex-base``.)

**Fedora / RHEL / Rocky / AlmaLinux**

.. code-block:: bash

   sudo dnf install texlive-scheme-medium texlive-type1cm texlive-cm-super \
                    texlive-dvipng ghostscript

**openSUSE**

.. code-block:: bash

   sudo zypper install texlive-latex texlive-type1cm texlive-cm-super \
                       texlive-dvipng ghostscript

**Arch Linux**

.. code-block:: bash

   sudo pacman -S texlive-latexextra texlive-fontsrecommended \
                  texlive-fontsextra ghostscript
   # dvipng ships with texlive-bin on current Arch; install the dvipng package
   # separately on older installations

**macOS**

The simplest route is the full MacTeX distribution, which already contains
everything above:

.. code-block:: bash

   brew install --cask mactex          # ~4 GB

For a small install, use BasicTeX and add the pieces:

.. code-block:: bash

   brew install --cask basictex
   sudo tlmgr update --self
   sudo tlmgr install type1cm cm-super psnfss dvipng underscore

**Windows**

Install `MiKTeX <https://miktex.org/>`_ or `TeX Live
<https://tug.org/texlive/>`_, and make sure the installation directory is on
``PATH``. MiKTeX is configured by default to fetch missing packages on demand,
so ``type1cm`` and ``cm-super`` are installed automatically the first time a
figure is drawn; leave that setting enabled. With TeX Live, install the pieces
explicitly as in the ``tlmgr`` command below.

**conda / mamba (any platform, no root access)**

.. code-block:: bash

   conda install -c conda-forge texlive-core dvipng

``texlive-core`` provides a large TeX Live subset. If a ``.sty`` file is still
reported missing, add it with ``tlmgr`` as below.

**Any TeX Live installation (universal fallback)**

TeX Live ships its own package manager, which works the same on Linux, macOS
and Windows:

.. code-block:: bash

   tlmgr install type1cm cm-super psnfss dvipng underscore

Prefix with ``sudo`` for a system-wide installation. This does **not** apply to
Debian/Ubuntu, where TeX Live is managed by ``apt`` and ``tlmgr`` is disabled
for system packages; use the ``apt`` command above instead.

Verifying the installation
++++++++++++++++++++++++++

To check before starting a long calibration:

.. code-block:: bash

   python -c "import matplotlib; matplotlib.use('Agg'); \
              import matplotlib.pyplot as plt; plt.rcParams['text.usetex']=True; \
              plt.plot([0,1]); plt.xlabel(r'test \$\\alpha\$'); \
              plt.savefig('/tmp/tex-check.png'); print('LaTeX rendering OK')"

.. tip::

   On a machine where the LaTeX stack cannot be installed (no root access on a
   cluster, for instance), the calibration itself still runs; only the plotting
   is affected. Disable the LaTeX text mode after creating the plotter:

   .. code-block:: python

      import matplotlib.pyplot as plt
      plotter = BayesianPlotter(results_folder_path=..., variable_name=...)
      plt.rcParams.update({"text.usetex": False})

   The figures then use matplotlib's built-in font renderer. Set it *after*
   constructing the plotter, because ``BayesianPlotter`` enables the LaTeX mode
   in its constructor.

Numerical-model bindings
------------------------

TELEMAC
+++++++

*Time requirement: ~60 min.*

The calibration routines drive the open-source modelling system
**TELEMAC-Mascaret** (TELEMAC 2D solves the depth-averaged shallow-water
equations; TELEMAC 3D solves the Reynolds-averaged Navier–Stokes equations).
Install it following the developers' instructions at `opentelemac.org
<http://www.opentelemac.org/index.php/installation>`_, or the
HydroBayesCal-tailored guide at `hydro-informatics.com/install-telemac
<https://hydro-informatics.com/install-telemac/>`_. We recommend
the ``pysource.gfortranHPC.sh`` configuration (or your preferred ``pysource``
file).

Loading HydroBayesCal **and** TELEMAC together
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Running a calibration needs both the HydroBayesCal environment and the TELEMAC
environment variables active at once. The repository ships an activation
template in ``env-scripts/activateHBCtelemac.sh``.

.. note::

   **One-time setup** — open ``env-scripts/activateHBCtelemac.sh`` in a text
   editor and set:

   * ``TELEMAC_CONFIG_DIR`` — your TELEMAC installation's config directory.
   * ``TELEMAC_CONFIG_NAME`` — your TELEMAC configuration file name.
   * the path/name of your HydroBayesCal environment.

   **Each session** — from the repository root, either source the script
   directly or use the convenience launcher:

   .. code-block:: bash

      source env-scripts/activateHBCtelemac.sh
      # or
      python env-scripts/activate_HBC_Telemac.py

   A successful activation reports that both the HydroBayesCal and TELEMAC
   environments were loaded.

Verify the TELEMAC Python API is reachable:

.. code-block:: bash

   python -c "import telapy; print(telapy.__version__)"

OpenFOAM
++++++++

.. note::

   The OpenFOAM (interFoam) binding is under active development. It requires a
   working `OpenFOAM <https://www.openfoam.com/>`_ installation on ``PATH``
   (``decomposePar``, ``reconstructPar``, the solver, and ``foamToVTK``); see
   the installation guide at `hydro-informatics.com/install-openfoam
   <https://hydro-informatics.com/install-openfoam/>`_. The ``mesh`` extra
   (PyVista/VTK) is used to read the VTK output. See :doc:`usage-openfoam` for
   running a calibration.

Delft3D-FLOW
++++++++++++

.. note::

   The Delft3D-FLOW (Deltares, Delft3D 4 suite) binding
   (:class:`hydroBayesCal.delft3d.control_delft3d.Delft3DModel`) requires a
   Delft3D-FLOW installation compiled from source with its ``env.sh``
   environment script and the ``run_dflow2d3d.sh`` /
   ``run_dflow2d3d_parallel.sh`` launchers, following the guide at
   `hydro-informatics.com/delft3d
   <https://hydro-informatics.com/delft3d>`_ (default prefix
   ``~/opt/delft3d-flow``). Reading the NetCDF map output additionally needs
   the ``delft3d`` extra:

   .. code-block:: bash

      pip install "hydroBayesCal[delft3d]"

   See :doc:`usage-delft3d` for running a calibration.

Check your installation
------------------------

With the environment active, confirm the package imports:

.. code-block:: bash

   python -c "import hydroBayesCal; print('HydroBayesCal OK')"

You are now ready to set up a calibration — continue with :doc:`workflow`.

Building and releasing (maintainers)
------------------------------------

The project is configured for PyPI via :file:`pyproject.toml` (``src`` layout,
SPDX license, build backend ``setuptools``). The full contributor guide —
including coding conventions and the release checklist — lives in
:file:`CONTRIBUTING.md`.

Versioning
++++++++++

HydroBayesCal follows `Semantic Versioning <https://semver.org/>`_
(``MAJOR.MINOR.PATCH``), a subset of `PEP 440
<https://peps.python.org/pep-0440/>`_:

* **MAJOR** — incompatible API changes,
* **MINOR** — new, backward-compatible features (e.g. a new solver binding),
* **PATCH** — backward-compatible bug fixes.

The version is declared **once**, as ``version`` in :file:`pyproject.toml`; keep
``release``/``version`` in :file:`docs/conf.py` in sync. Pre-1.0, the API may
still change between minor versions.

.. important::

   PyPI versions are **immutable**: an uploaded ``X.Y.Z`` can never be
   re-uploaded or overwritten (even after yanking). Always bump the version for
   a new release.

Releasing
+++++++++

Releases are **automated through GitHub Actions and PyPI Trusted Publishing**
(OIDC) — no API token is stored. The workflow
(:file:`.github/workflows/publish.yml`) runs when a GitHub *Release* is
published:

#. Bump ``version`` in :file:`pyproject.toml` (and :file:`docs/conf.py`); commit
   and push to ``main``.
#. On GitHub, draft a new **Release**, create the tag ``vX.Y.Z``, and publish
   it.
#. The workflow builds the sdist + wheel, runs ``twine check``, and publishes to
   PyPI via the trusted publisher.

Building locally
++++++++++++++++

To reproduce what CI does (without publishing):

.. code-block:: bash

   pip install -e ".[dev]"          # provides build + twine
   python -m build                  # creates dist/*.whl and dist/*.tar.gz
   twine check dist/*               # validate the metadata/long description

A manual ``twine upload`` is only needed as a fallback when Trusted Publishing
is unavailable (use a project-scoped token); see :file:`CONTRIBUTING.md`.
