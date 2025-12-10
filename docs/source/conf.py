# docs/conf.py
import os
import sys

# Add project root to sys.path so autodoc can import signxai2
sys.path.insert(0, os.path.abspath(".."))

project = "signxai2"
author = "TimeXAIgroup"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "sphinx_rtd_theme"
