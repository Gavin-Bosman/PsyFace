# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PsyFace'
copyright = '2024, Gavin Bosman'
author = 'Gavin Bosman'
version = '0.4'
release = '0.4.1'
import os, sys
sys.path.insert(0, os.path.abspath('..'))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',   # Automatically document from docstrings
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.napoleon',  # Support for Google style docstrings
    'sphinx.ext.viewcode',  # Add links to highlighted source code
    'sphinx.ext.intersphinx',  # Link to other projects' documentation
    'sphinx.ext.todo',  # Support for todo and todolist directives
    'sphinx.ext.mathjax',  # Render math via MathJax
]

napoleon_google_doctrings = False
autodoc_mock_imports = ['numpy']
templates_path = ['_templates']
exclude_patterns = ['.\\data', '.\\source\\processing_script.py', '.\\source\\OpenCV_References']
root_doc = 'index'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = []
