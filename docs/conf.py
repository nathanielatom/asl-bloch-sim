# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'gigablochs'
copyright = '2025, Adam Suban-Loewen'
author = 'Adam Suban-Loewen'
release = '0.2.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinxcontrib.bibtex',
    'sphinxcontrib.video',
    'bokeh.sphinxext.bokeh_plot',
    'sphinx_copybutton',
    'sphinx_design',
    'sphinx_pyscript',
]

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'cupy': ('https://docs.cupy.dev/en/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'pandas': ('http://pandas.pydata.org/pandas-docs/stable/', None),
    'bokeh': ('https://docs.bokeh.org/en/latest/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'manim': ('https://docs.manim.community/en/stable/', None),
}

bibtex_bibfiles = ['references.bib']
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'conf.py']

# Generate autosummary stubs automatically, instead of using sphinx-autogen script.
autodoc_default_options = {'members': None}
autosummary_generate = True
default_role = 'py:obj'

# # TODO: fix latex rendering of \gammabar, physics package not working
# r"""
# % define symbol \gammabar
# \usepackage{stackengine}
# \usepackage{scalerel}
# \newcommand\gammabar{\ThisStyle{\ensurestackMath{%
# \stackengine{-1.5\LMpt}{$$\SavedStyle\gamma$$}{$$\SavedStyle-$$}{O}{c}{F}{F}{L}}}}
# """

# # MathJax Latex equation settings
mathjax3_config = {
    'tex': {
        'macros': {
            'gammabar': r"""\ThisStyle{\ensurestackMath{\stackengine{-1.5\LMpt}{$$\SavedStyle\gamma$$}{$$\SavedStyle-$$}{O}{c}{F}{F}{L}}}""",
        },
        'packages': {'[+]': ['ams', 'newcommand', 'configmacros', 'stackengine', 'scalerel', 'physics']},
    },
    'loader': {
        'load': ['[tex]/physics'],
    }
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_logo = "_static/gigablochs_logo_transparent.png"
html_title = "GigaBlochs"
html_theme_options = {
    "repository_url": "https://github.com/nathanielatom/gigablochs",
    "use_repository_button": True,
}
html_favicon = "_static/gigablochs_logo_small_icon_transparent.png"
html_static_path = ['_static']
html_css_files = ["custom.css"]
