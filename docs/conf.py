# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------

project = 'PsyFace'
author = 'Gavin Bosman'
release = '0.4.1'  # The full version, including alpha/beta/rc tags
version = '0.4'    # The short X.Y version

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. 
# These extensions are built into Sphinx or can be installed separately.
extensions = [
    'sphinx.ext.autodoc',   # Automatically document from docstrings
    'sphinx.ext.napoleon',  # Support for Google style docstrings
    'sphinx.ext.viewcode',  # Add links to highlighted source code
    'sphinx.ext.intersphinx',  # Link to other projects' documentation
    'sphinx.ext.todo',  # Support for todo and todolist directives
    'sphinx.ext.mathjax',  # Render math via MathJax
]

exclude_patterns = ['./data', './src/processing_script.py']

# The master document is the root document where the "table of contents" lives.
master_doc = 'README.rst'

# The suffix(es) of source filenames.
source_suffix = ['.rst', '.md']

# The theme to use for HTML and HTML Help pages.
html_theme = 'sphinx_rtd_theme'

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '10pt',
    # 'preamble': '',
}

# -- Options for EPUB output -------------------------------------------------

epub_show_urls = 'footnote'

# -- Options for intersphinx extension ---------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master', None),
}

# -- Options for autodoc extension -------------------------------------------

autodoc_default_options = {
    'members': True,  # Include class members
    'undoc-members': True,  # Include undocumented members
    'private-members': True,  # Include private members (_foo)
    'show-inheritance': True,  # Show class inheritance
}