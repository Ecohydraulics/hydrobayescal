# Configuration file for the Sphinx documentation builder.
#
# Full reference: https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
# Make the ``hydroBayesCal`` package importable for autodoc without installing
# it. The package lives in the ``src`` layout, so the import root is ``src``.
import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------

project = "HydroBayesCal"
copyright = "2022-2026, the HydroBayesCal authors"
author = "Andrés Heredia, Federica Scolari, Sebastian Schwindt"
release = "0.1.2"
version = "0.1"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinxcontrib.mermaid",
]

# Render ``.. todo::`` admonitions in the output.
todo_include_todos = True

# Napoleon (NumPy/Google style docstrings)
napoleon_numpy_docstring = True
napoleon_google_docstring = True
napoleon_include_init_with_doc = True

# Autodoc / autosummary
autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

# The heavy / compiled scientific and coupling dependencies are *mocked* so the
# documentation can be built (e.g. on Read the Docs) without installing the
# full runtime stack. Only lightweight libraries (numpy, pandas) are expected
# to be present; everything below is replaced by a stub at import time.
autodoc_mock_imports = [
    "scipy",
    "sklearn",
    "torch",
    "gpytorch",
    "linear_operator",
    "pyvista",
    "vtk",
    "h5py",
    "matplotlib",
    "seaborn",
    "corner",
    "joblib",
    "tqdm",
    "chaospy",
    "numpoly",
    "emcee",
    "bayesvalidrox",
    "umbridge",
    "openpyxl",
    "yaml",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # ``create-tm-model.rst`` is an unfinished TELEMAC-meshing draft kept out of
    # the build until it is revised; remove this entry once it is wired into a
    # toctree.
    "create-tm-model.rst",
]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = "HydroBayesCal"
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
}

# -- Options for EPUB output -------------------------------------------------
epub_show_urls = "footnote"
