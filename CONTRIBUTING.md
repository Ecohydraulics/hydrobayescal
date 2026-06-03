# Contributing to HydroBayesCal

Thanks for contributing! This guide covers the development setup, coding
conventions, the documentation build, and — importantly — **how versioning and
PyPI releases work**.

## Development setup

HydroBayesCal targets **Python 3.10–3.11** (the upper bound is imposed by the
`bayesvalidrox` dependency, which declares `python < 3.12`). It uses a `src`
layout and a single `pyproject.toml`; there is no `setup.py`.

```bash
git clone https://github.com/Ecohydraulics/hydrobayescal.git
cd hydrobayescal
python3.11 -m venv .venv && source .venv/bin/activate   # or conda/mamba
pip install -e ".[dev,docs,mesh]"
```

The `dev` extra provides `pytest`, `build`, `twine` and `ruff`; `docs` provides
Sphinx and the Read the Docs theme; `mesh` adds the geospatial/VTK IO used by
some bindings.

## Making changes

* Branch off `main`; open a pull request rather than pushing to `main`.
* Match the surrounding code style. Public classes and functions use
  **NumPy-style docstrings** (rendered by Sphinx + napoleon).
* When you add or change a numerical-solver binding, **preserve the
  solver-specific strings and file conventions** (e.g. TELEMAC `.cas`/SELAFIN
  keywords, OpenFOAM `system/controlDict`, planned Delft3D-FLOW `.mdf`/NEFIS).
  The Python attribute names are shared across bindings; the software-facing
  values are not.
* Run the test suite and the docs build before opening a PR:

  ```bash
  pytest
  sphinx-build -b html -W docs docs/_build/html
  ```

## Versioning

We follow **Semantic Versioning** (`MAJOR.MINOR.PATCH`, see
[semver.org](https://semver.org/)), which is a subset of Python's version rules
([PEP 440](https://peps.python.org/pep-0440/)):

* **MAJOR** — incompatible API changes.
* **MINOR** — new, backward-compatible functionality (e.g. a new solver binding).
* **PATCH** — backward-compatible bug fixes.

While the project is pre-1.0, the API may still change between minor versions.

The version is declared in **one place**, `pyproject.toml`:

```toml
[project]
version = "0.1.0"
```

When you bump it, also update `release`/`version` in `docs/conf.py` so the
documentation header matches.

> **PyPI versions are immutable.** Once `X.Y.Z` is uploaded it can never be
> re-uploaded or overwritten (even if yanked). Always bump the version for any
> new release.

## Releasing to PyPI (maintainers)

Releases are **automated via GitHub Actions and PyPI Trusted Publishing**
(OIDC) — no API token is stored anywhere. The workflow lives in
`.github/workflows/publish.yml` and runs when a GitHub *Release* is published.

To cut a release:

1. Bump `version` in `pyproject.toml` (and `docs/conf.py`); commit and push to
   `main`.
2. On GitHub, go to **Releases → Draft a new release**, create the tag
   `vX.Y.Z`, write release notes, and **Publish release**.
3. The workflow builds the sdist + wheel, runs `twine check`, and publishes to
   PyPI through the trusted publisher. Watch progress under the **Actions** tab.

That's it — there is no manual `twine upload` in the normal flow.

### Building locally (sanity check)

You can reproduce what CI does without publishing:

```bash
pip install -e ".[dev]"      # provides build + twine
python -m build              # writes dist/*.whl and dist/*.tar.gz
twine check dist/*           # validates metadata + long description
```

### One-time Trusted Publishing setup

This is configured already, but for reference, a maintainer with PyPI access
registered a GitHub publisher at
`https://pypi.org/manage/project/hydrobayescal/settings/publishing/` with:

| Field | Value |
| --- | --- |
| Owner | `Ecohydraulics` |
| Repository | `hydrobayescal` |
| Workflow | `publish.yml` |
| Environment | `pypi` |

### Manual fallback

Only needed if Trusted Publishing is unavailable (e.g. bootstrapping a new
project name). Build as above, then upload with a token:

```bash
TWINE_USERNAME=__token__ TWINE_PASSWORD=pypi-<token> twine upload dist/*
```

Prefer a **project-scoped** token, and delete account-scoped tokens after use.
