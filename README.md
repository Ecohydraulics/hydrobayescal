# HydroBayesCal

**Surrogate-assisted Bayesian calibration for computationally expensive
hydro- and morphodynamic models.**

[![Documentation](https://img.shields.io/badge/docs-readthedocs-blue)](https://hydrobayescal.readthedocs.io)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-green)](LICENSE)

HydroBayesCal calibrates expensive numerical models without running them
thousands of times. It trains a **Gaussian Process Emulator (GPE)** as a fast
surrogate from a small set of strategically sampled simulations, then refines it
with **Bayesian Active Learning (BAL)** — iteratively adding the training points
that maximise the information gain (relative entropy) and Bayesian model
evidence for the calibration. Single- and multi-output GPEs are supported.

The package couples to open-source modelling software through a common binding
layer:

* **TELEMAC** (2D/3D) — fully supported
* **OpenFOAM** (interFoam) — binding under active development

Experimental design and parameter sampling are delegated to
[BayesValidRox](https://pages.iws.uni-stuttgart.de/inversemodeling/bayesvalidrox/);
the GP emulators and the Bayesian active-learning logic are implemented in-tree.

## Installation

HydroBayesCal targets **Python 3.10–3.11** (the upper bound is imposed by the
`bayesvalidrox` dependency). It is developed and tested on Linux.

```bash
pip install hydroBayesCal
```

or, for a development/editable install from a clone:

```bash
git clone https://github.com/sschwindt/hydrobayescal.git
cd hydrobayescal
pip install -e ".[dev,docs,mesh]"
```

A calibration additionally requires a working installation of the numerical
solver (e.g. TELEMAC) on the system. See the
[installation guide](https://hydrobayescal.readthedocs.io/en/latest/installation.html)
for the full environment setup, including coupling HydroBayesCal with TELEMAC.

## Quick start

Configure a calibration in a Python config file and run the TELEMAC driver:

```bash
python bal_telemac.py --config config.py
```

See the [documentation](https://hydrobayescal.readthedocs.io) for the
end-to-end workflow, the configuration parameters, the code architecture, and
worked examples.

## Development & releases

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for the
development setup, coding conventions, and the documentation build.

For maintainers, a few essentials:

* **Editable install:** `pip install -e ".[dev,docs,mesh]"` (Python 3.10–3.11).
* **Versioning:** [Semantic Versioning](https://semver.org/) /
  [PEP 440](https://peps.python.org/pep-0440/); the version lives only in
  `pyproject.toml` (keep `docs/conf.py` in sync). PyPI versions are immutable —
  always bump for a new release.
* **Releases are automated:** publishing a GitHub *Release* (tag `vX.Y.Z`)
  triggers `.github/workflows/publish.yml`, which builds the distributions and
  uploads them to PyPI via **Trusted Publishing** (OIDC, no stored token). No
  manual `twine upload` is needed. Build locally to sanity-check with
  `python -m build && twine check dist/*`.

## Citing / scientific background

HydroBayesCal builds on the Bayesian active-learning framework of Oladyshkin et
al. (2020) and on Gaussian-process regression (Rasmussen & Williams, 2006). Its
application to reservoir sedimentation and 3D reservoir hydrodynamics is
documented in Mouris et al. (2023) and Schwindt et al. (2023). Full references
with DOIs are on the
[references page](https://hydrobayescal.readthedocs.io/en/latest/references.html).

## License

Distributed under the BSD 3-Clause License. See [LICENSE](LICENSE).
