
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
# import sphinx_rtd_theme

# -- Project information -----------------------------------------------------

project = 'PolSARtools'
copyright = '2025, PolSARtools team'
author = 'Narayanarao Bhogapurapu'


release = '0.10.1'


# -- General configuration ---------------------------------------------------


latex_elements = {
	'papersize':'letterpaper',
	'pointsize':'10pt',
	'preamble':'',
	'figure_align': 'htbp'
}

templates_path = ['_templates']
exclude_patterns = []

# https://pydata-sphinx-theme.readthedocs.io/en/stable/index.html
html_theme = "pydata_sphinx_theme"
html_favicon =  'files/figures/icon.png'

html_theme_options = {
   "logo": {
       "text": "PolSARtools",
      "image_light": "files/figures/icon.png",
      "image_dark": "files/figures/icon.png",
   }
}


extensions = [
	# 'sphinx_rtd_theme',
    # "sphinxawesome_theme",
    "sphinxcontrib.bibtex",
    'sphinx.ext.autodoc',    # Autodocumentation
    'sphinx.ext.intersphinx',  # Links to other documentation
    'sphinx.ext.viewcode',    # View source links
    'sphinx.ext.napoleon',
]


html_static_path = ['_static']
bibtex_bibfiles = ['files/ref.bib']