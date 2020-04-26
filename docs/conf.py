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
import os
import sys
sys.path.insert(0, os.path.abspath('../flare/'))
import sphinx_rtd_theme


# -- Project information -----------------------------------------------------

project = 'flare'
copyright = '2020, Jacob Pettit'
author = 'Jacob Pettit'

# The full version, including alpha/beta/rc tags
release = '0.1.6'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
'sphinxcontrib.napoleon',
'sphinx_rtd_theme',
'sphinx.ext.intersphinx'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
'style_nav_header_background' : 'linear-gradient(to bottom right, rgba(121,4,199,1) 0%, rgba(251,26,26,1) 100%);',
'collapse_navigation': False,
}
html_context = {
    "display_github": True,
    "github_user" : "jfpettit",
    "github_repo" : "flare",
    "github_version" : "master",
    "conf_py_path": "/docs/"
}


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
