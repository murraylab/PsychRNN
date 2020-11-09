# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('../psychrnn'))


# -- Project information -----------------------------------------------------

project = 'PsychRNN'
copyright = '2020, Daniel B. Ehrlich*, Jasmine T. Stone*, David Brandfonbrener, Alex Atanasov, John D. Murray (* indicates equal contribution)'
author = 'Daniel B. Ehrlich*, Jasmine T. Stone*, David Brandfonbrener, Alex Atanasov, John D. Murray (* indicates equal contribution)'

exec(open("../psychrnn/_version.py", "r").read()) # get __version__ variable

# The short X.Y version
version = ".".join(__version__.split(".")[0:-1])
# The full version, including alpha/beta/rc tags
release = __version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
	'sphinx.ext.autodoc',
  'sphinx.ext.coverage',
  'sphinx.ext.imgmath',
  'sphinx.ext.viewcode',
  'nbsphinx',
  'sphinx.ext.mathjax',
  'sphinx_copybutton',
  "sphinx_rtd_theme",
  'sphinxcontrib.napoleon',
  'autodocsumm',
]

#include autosummary by defualt
autodoc_default_options = {
    'autosummary': True,
}
autodata_content = 'both'
autosummary_generate = True


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The master toctree document.
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'notebooks/.ipynb_checkpoints']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

html_logo = "images/logo.svg"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}
html_sidebars = { '**': ['customtoc.html', 'localtoc.html','relations.html', 'searchbox.html', 'sourcelink.html'], }

nbsphinx_prolog = r"""
{% set docname = 'docs/' + env.doc2path(env.docname, base=None) %}

.. raw:: html

    <div class="admonition note">
      This page was generated from
      <a class="reference external" href="https://github.com/murraylab/PsychRNN/blob/v{{ env.config.release|e }}/{{ docname|e }}">{{ docname|e }}</a>.
      Interactive online version:
      <a href="https://colab.research.google.com/github/murraylab/PsychRNN/blob/v{{ env.config.release|e }}/{{ docname|e }}"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>.
      <script>
        if (document.location.host) {
          $(document.currentScript).replaceWith(
            '<a class="reference external" ' +
            'href="https://nbviewer.jupyter.org/url' +
            (window.location.protocol == 'https:' ? 's/' : '/') +
            window.location.host +
            window.location.pathname.slice(0, -4) +
            'ipynb">View in <em>nbviewer</em></a>.'
          );
        }
      </script>
    </div>
"""

# Taken from https://stackoverflow.com/questions/8821511/substitutions-in-sphinx-code-blocks
def ultimateReplace(app, docname, source):
    result = source[0]
    for key in app.config.ultimate_replacements:
        result = result.replace(key, app.config.ultimate_replacements[key])
    source[0] = result

ultimate_replacements = {
    "{release}" : release
}

def setup(app):
   app.add_config_value('ultimate_replacements', {}, True)
   app.connect('source-read', ultimateReplace)

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = False
napoleon_use_rtype = True
napoleon_use_keyword = False
napoleon_custom_sections = None
