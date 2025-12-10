# Configuration file for the Sphinx documentation builder.
#
# For a full list of options see:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import sys
import os
from subprocess import run, CalledProcessError
import inspect

# Make the package importable when building locally without installation
sys.path.insert(0, os.path.abspath('..'))

from pybtex.style.formatting.plain import Style as PlainStyle
from pybtex.style.labels import BaseLabelStyle
from pybtex.plugin import register_plugin


class AuthorYearLabelStyle(BaseLabelStyle):
    def format_labels(self, sorted_entries):
        for entry in sorted_entries:
            yield f'[{entry.persons["author"][0].last_names[0]} et al., {entry.fields["year"]}]'


class AuthorYearStyle(PlainStyle):
    default_label_style = AuthorYearLabelStyle


register_plugin('pybtex.style.formatting', 'author_year_style', AuthorYearStyle)


def getrev():
    """Return the current git revision or fall back to 'main'."""
    try:
        revision = run(
            ['git', 'describe', '--tags', 'HEAD'],
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
    except CalledProcessError:
        revision = 'main'
    return revision


# revision of the documentation
REVISION = getrev()

# path for the linkcode plugin
# NOTE: in this repo the sources live directly under the repo root (e.g. signxai2/...)
LINKCODE_URL = (
    'https://github.com/TimeXAIgroup/signxai2/blob/'
    f'{REVISION}'
    '/{filepath}#L{linestart}-L{linestop}'
)


# revised from https://gist.github.com/nlgranger/55ff2e7ff10c280731348a16d569cb73
def linkcode_resolve(domain, info):
    if domain != 'py' or not info['module']:
        return None

    modname = info['module']
    topmodulename = modname.split('.')[0]
    fullname = info['fullname']

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split('.'):
        try:
            obj = getattr(obj, part)
        except Exception:
            return None

    try:
        module = sys.modules.get(topmodulename)
        if module is None:
            return None
        # go from top-level module dir (signxai2/...) up to repo root
        modpath = os.path.abspath(os.path.join(os.path.dirname(module.__file__), '..'))
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

    return LINKCODE_URL.format(
        filepath=filepath,
        linestart=linestart,
        linestop=linestop,
    )


def config_inited_handler(app, config):
    os.makedirs(os.path.join(app.srcdir, app.config.generated_path), exist_ok=True)


def setup(app):
    app.add_config_value('REVISION', 'main', 'env')
    app.add_config_value('generated_path', '_generated', 'env')
    app.connect('config-inited', config_inited_handler)


# -- Project information -----------------------------------------------------

project = 'signxai2'
copyright = '2025, TimeXAIgroup'
author = 'TimeXAIgroup'

# Try to get version from the installed package
try:
    from signxai2 import __version__  # type: ignore
    release = __version__
    version = __version__
except Exception:
    # Fallback if package is not importable
    release = '0.0.0'
    version = '0.0.0'


# -- General configuration ---------------------------------------------------

templates_path = ['_templates']
html_static_path = ['_static']
html_favicon = '_static/favicon.svg'  # if you add one; otherwise you can comment this out

exclude_patterns = []

extensions = [
    # Sphinx built-ins
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.extlinks',
    'sphinx.ext.linkcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',

    # Third-party
    'sphinxcontrib.bibtex',
    'sphinx_copybutton',
    'myst_parser',
    'nbsphinx',
]

# autodoc configuration
autodoc_class_signature = 'separated'
autodoc_member_order = 'bysource'
autodoc_typehints = 'both'
autodoc_preserve_defaults = True
# autosummary_generate = True  # can be enabled if you want auto-generated summary pages


# interactive badges for binder and colab
# Our docs root is "docs/", so we prepend "docs/" instead of "docs/source/"
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

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_line_continuation_character = "\\"
copybutton_here_doc_delimiter = "EOT"

# BibTeX configuration
# NOTE: you'll want to add docs/bibliography.bib or adjust this path
bibtex_bibfiles = ['bibliography.bib']
bibtex_default_style = 'author_year_style'
bibtex_reference_style = 'author_year'

# Intersphinx: avoid accidentally resolving local refs to external docs
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'torch': ('https://pytorch.org/docs/stable', None),
    'tensorflow': ('https://www.tensorflow.org/api_docs/python/', None),
}
intersphinx_disabled_reftypes = ["*"]

# extlinks: helper for linking into the repo
extlinks = {
    'repo': (
        f'https://github.com/TimeXAIgroup/signxai2/blob/{REVISION}/%s',
        '%s',
    )
}

# MyST Parser settings
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_image",
]

# nbsphinx behaviour
nbsphinx_execute = 'never'
nbsphinx_allow_errors = True
nbsphinx_kernel_name = 'python3'


# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'

# Custom CSS (already present in your existing conf)
html_css_files = [
    'custom.css',
]

html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',
    'logo_only': False,
    # 'display_version': True,  # can be re-enabled if desired
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#2980B9',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
}

pygments_style = 'sphinx'

# HTML context for "Edit on GitHub" etc.
html_context = {
    'display_github': True,
    'github_user': 'TimeXAIgroup',
    'github_repo': 'signxai2',
    # docs live in /docs at the given revision
    'github_version': f'{REVISION}/docs/',
    'conf_py_path': '/docs/',
}


# -- LaTeX / man / texinfo / epub (optional, kept minimal) -------------------

latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
    'preamble': '',
    'figure_align': 'htbp',
}

latex_documents = [
    ('index', 'signxai2.tex', 'signxai2 Documentation', 'TimeXAIgroup', 'manual'),
]

man_pages = [
    ('index', 'signxai2', 'signxai2 Documentation', [author], 1),
]

texinfo_documents = [
    (
        'index',
        'signxai2',
        'signxai2 Documentation',
        author,
        'signxai2',
        'SIGNed XAI for Image and Time Series Models.',
        'Miscellaneous',
    ),
]

epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright
epub_exclude_files = ['search.html']
