import os
import sys
sys.path.insert(0, os.path.abspath('..'))  # make your package importable

project = 'Fast-NGP'
author = 'Your Name'
release = '0.1'

extensions = [
    'sphinx.ext.autodoc',    # Generate docs from docstrings
    'sphinx.ext.napoleon',   # Support Google/NumPy style docstrings
]

templates_path = ['_templates']  # optional
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'  # Recommended for ReadTheDocs
html_static_path = ['_static']   # optional