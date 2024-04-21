# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
from ragger_duck import __version__

project = "Ragger Duck"
copyright = "2023, G. Lemaitre"
author = "G. Lemaitre"
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx_gallery.gen_gallery",
    "numpydoc",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "_templates", "Thumbs.db", ".DS_Store"]

# The reST default role (used for this markup: `text`) to use for all
# documents.
default_role = "literal"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_style = "css/ragger_duck.css"
html_logo = "_static/img/logo.png"
html_favicon = "_static/img/favicon.ico"
html_css_files = [
    "css/ragger_duck.css",
]
html_sidebars = {
    "install": [],
    "auto_examples/index": [],
    "whats_new": [],
    "about": [],
}

html_theme_options = {
    "external_links": [],
    "github_url": "https://github.com/glemaitre/sklearn-ragger-duck",
    # "twitter_url": "https://twitter.com/pandas_dev",
    "use_edit_page_button": True,
    "show_toc_level": 1,
    # "navbar_align": "right",  # For testing that the navbar items align properly
}

html_context = {
    "github_user": "glemaitre",
    "github_repo": "sklearn-ragger-duck",
    "github_version": "main",
    "doc_path": "doc",
}

# -- Options for autodoc ------------------------------------------------------

autodoc_default_options = {
    "members": True,
    "inherited-members": True,
}

# generate autosummary even if no references
autosummary_generate = True

# -- Options for numpydoc -----------------------------------------------------

# this is needed for some reason...
# see https://github.com/numpy/numpydoc/issues/69
numpydoc_show_class_members = False

# -- Options for sphinx-gallery -----------------------------------------------

# Generate the plot for the gallery
plot_gallery = True

sphinx_gallery_conf = {
    "doc_module": "ragger_duck",
    "backreferences_dir": os.path.join("references/generated"),
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    "reference_url": {"ragger_duck": None},
}

# -- Additional temporary hacks -----------------------------------------------

# Temporary work-around for spacing problem between parameter and parameter
# type in the doc, see https://github.com/numpy/numpydoc/issues/215. The bug
# has been fixed in sphinx (https://github.com/sphinx-doc/sphinx/pull/5976) but
# through a change in sphinx basic.css except rtd_theme does not use basic.css.
# In an ideal world, this would get fixed in this PR:
# https://github.com/readthedocs/sphinx_rtd_theme/pull/747/files


# def setup(app):
#     app.add_css_file("basic.css")
