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
    'sphinx.ext.autodoc',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.napoleon',  
    'sphinx.ext.viewcode',  
    'sphinx.ext.intersphinx',  
    'sphinx.ext.todo',
    'numpy',
    'mediapipe',
    'opencv-python',
    'pandas'  
]

napoleon_google_doctrings = False
templates_path = ['_templates']
exclude_patterns = ['.\\data', '.\\source\\processing_script.py', '.\\source\\OpenCV_References']
root_doc = 'index'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = []
