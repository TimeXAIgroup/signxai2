# Configuration file for the Sphinx documentation builder.

import os
import sys
from subprocess import run, CalledProcessError
import inspect

from pybtex.style.formatting.plain import Style as PlainStyle
from pybtex.style.labels import BaseLabelStyle
from pybtex.plugin import register_plugin


# --- Ensure our package is importable for autodoc --------------------------------
# conf.py is in docs/, so ".." is the repo root
sys.path.insert(0, os.path.abspath(".."))


# --- Custom BibTeX style ---------------------------------------------------------

class AuthorYearLabelStyle(BaseLabelStyle):
    def format_labels(self, sorted_entries):
        for entry in sorted_entries:
            yield f'[{entry.persons["author"][0].last_names[0]} et al., {entry.fields["year"]}]'


class AuthorYearStyle(PlainStyle):
    default_label_style = AuthorYearLabelStyle


register_plugin("pybtex.style.formatting", "author_year_style", AuthorYearStyle)


# --- Git revision + linkcode URL -------------------------------------------------

def getrev():
    try:
        revision = run(
            ["git", "describe", "--tags", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
        ).stdout[:-1]
    except CalledProcessError:
        # fall back to main, not master
        revision = "main"

    return revision


REVISION = getrev()

# For signxai2 we don't have a "src/" layout, so use repo root directly
LINKCODE_URL = (
    "https://github.com/TimeXAIgroup/signxai2/blob/{revision}/{filepath}"
    "#L{linestart}-L{linestop}"
).format


def linkcode_resolve(domain, info):
    if domain != "py" or not info["module"]:
        return None

    modname = info["module"]
    topmodulename = modname.split(".")[0]
    fullname = info["fullname"]

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            obj = getattr(obj, part)
        except Exception:
            return None

    try:
        module = sys.modules.get(topmodulename)
        if module is None:
            return None
        # module.__file__ is something like /path/to/repo/signxai2/__init__.py
        # so ".." brings us to the repo root
        modpath = os.path.abspath(os.path.join(os.path.dirname(module.__file__), ".."))
        filepath = os.path.relpath(inspect.getsourcefile(obj), modpath)
        if filepath is None:
            return None
    except Exception:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except OSError:
        return None
    else:
        linestart, linestop = lineno, lineno + len(source) - 1

    return LINKCODE_URL(
        revision=REVISION,
        filepath=filepath,
        linestart=linestart,
        linestop=linestop,
    )


def config_inited_handler(app, config):
    os.makedirs(os.path.join(app.srcdir, app.config.generated_path), exist_ok=True)


def setup(app):
    app.add_config_value("REVISION", "main", "env")
    app.add_config_value("generated_path", "_generated", "env")
    app.connect("config-inited", config_inited_handler)


# --- Project information ---------------------------------------------------------

project = "signxai2"
author = "TimeXAIgroup"
copyright = "2025, TimeXAIgroup"


# --- General configuration -------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinxcontrib.datatemplates",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
    "sphinx_rtd_theme",
    "nbsphinx",
]

templates_path = ["_templates"]
html_static_path = ["_static"]
html_favicon = "_static/favicon.svg"

exclude_patterns = []

# autodoc
autodoc_class_signature = "separated"
autodoc_member_order = "bysource"
autodoc_typehints = "both"
autodoc_preserve_defaults = True

# --- nbsphinx badges (adjusted to this repo & docs layout) ----------------------

nbsphinx_prolog = r"""
{% set docname = 'docs/' + env.doc2path(env.docname, base=False) %}

.. raw:: html

    <div class="admonition note">
      This page was generated from
      <a class="reference external" href="https://github.com/TimeXAIgroup/signxai2/blob/{{ env.config.REVISION }}/{{ docname|e }}">{{ docname|e }}</a>
      <br />
      Interactive online version:
      <span style="white-space: nowrap;">
        <a href="https://mybinder.org/v2/gh/TimeXAIgroup/signxai2/{{ env.config.REVISION|e }}?filepath={{ docname|e }}">
            <img alt="launch binder" src="https://mybinder.org/badge_logo.svg" style="vertical-align:text-bottom">
        </a>
      </span>
      <span style="white-space: nowrap;">
        <a href="https://colab.research.google.com/github/TimeXAIgroup/signxai2/blob/{{ env.config.REVISION|e }}/{{ docname|e }}">
            <img alt="Open in Colab" src="https://colab.research.google.com/assets/colab-badge.svg" style="vertical-align:text-bottom">
        </a>
      </span>
    </div>
"""

# copybutton
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_line_continuation_character = "\\"
copybutton_here_doc_delimiter = "EOT"

# bibtex
bibtex_bibfiles = ["bibliography.bib"]
bibtex_default_style = "author_year_style"
bibtex_reference_style = "author_year"

# intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "torchvision": ("https://pytorch.org/vision/stable", None),
    "click": ("https://click.palletsprojects.com/en/stable/", None),
    "Pillow": ("https://pillow.readthedocs.io/en/stable/", None),
}
# avoid accidental cross-project ref resolution
intersphinx_disabled_reftypes = ["*"]

# extlinks
extlinks = {
    "repo": (
        f"https://github.com/TimeXAIgroup/signxai2/blob/{REVISION}/%s",
        "%s",
    ),
}

# HTML options
html_theme = "sphinx_rtd_theme"
html_context = {
    "display_github": True,
    "github_user": "TimeXAIgroup",
    "github_repo": "signxai2",
    # path inside the repo where docs live
    "github_version": f"{REVISION}/docs/",
}
