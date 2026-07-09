# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "pooling-windows"
copyright = "2026, Flatiron NeuroRSE"
author = "Flatiron NeuroRSE"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "matplotlib.sphinxext.plot_directive",
    "myst_nb",
    "jupytext",
]

nb_custom_formats = {
    ".md": ["jupytext.reads", {"fmt": "mystnb"}],
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The suffix(es) of source filenames.
# You can specify multiple suffixes as a list of strings:
source_suffix = [".rst", ".md"]

# The master toctree document.
master_doc = "index"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

# Path for static files (custom stylesheets or JavaScript)
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Enable automatic stub page generation
autosummary_generate = True
