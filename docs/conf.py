# Configuration file for the Sphinx documentation builder.

import os
import sys
from datetime import datetime

# Add project root to PYTHONPATH so autodoc can import signxai2
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = "SIGN-XAI2"
author = "TimeXAI Team"
year = datetime.now().year
copyright = f"{year}, {author}"

# Try to obtain the version from the package
from importlib.metadata import version, PackageNotFoundError

try:
    release = version("signxai2")   # must match the distribution name in pyproject.toml
except PackageNotFoundError:
    release = ""

version = release

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "myst_parser",          # optional Markdown support
    "nbsphinx"              # support for jupyter notebooks
]

extlinks = {
    "repo": (
        "https://github.com/TimeXAIgroup/signxai2/blob/main/%s",
        ""
    )
}

# Do not execute notebooks during the Sphinx build
nbsphinx_execute = "never"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

html_static_path = ["_static"]
html_favicon = "_static/favicon.svg"
html_logo = "_static/logo.png"
html_theme = "sphinx_rtd_theme"

# GitHub Pages canonical URL
html_baseurl = "https://timexaigroup.github.io/signxai2/"

# GitHub integration (for "Edit on GitHub" links, etc.)
html_context = {
    "display_github": True,
    "github_user": "TimeXAIgroup",
    "github_repo": "signxai2",
    "github_version": "main",
    # Path in the repo where this conf.py lives:
    "conf_py_path": "/docs/",
}

# -- Autodoc settings --------------------------------------------------------

autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
