"""Sphinx configuration for the Silogic documentation.

Build locally with:  sphinx-build -b html docs docs/_build/html
"""
import silogic

# -- Project information -----------------------------------------------------
project = "Silogic"
author = "Silogic contributors"
copyright = "2026, Silogic contributors"
release = silogic.__version__
version = release

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",      # Google/NumPy-style docstrings
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",              # Markdown support
    "sphinx_copybutton",
    "sphinx_design",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_rtype = False

myst_enable_extensions = ["colon_fence", "deflist", "smartquotes"]
myst_heading_anchors = 3        # so in-page [#anchor] links in the guide resolve

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

# -- HTML output (pydata-sphinx-theme, the NumPy look) -----------------------
html_theme = "pydata_sphinx_theme"
html_title = "Silogic"
html_short_title = "Silogic"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "github_url": "https://github.com/gaspardol/silogic",
    "icon_links": [
        {"name": "PyPI", "url": "https://pypi.org/project/silogic/",
         "icon": "fa-brands fa-python"},
    ],
    "use_edit_page_button": False,
    "show_prev_next": True,
    "navbar_align": "left",
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "secondary_sidebar_items": ["page-toc", "sourcelink"],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
    "header_links_before_dropdown": 5,
}

html_context = {
    "github_user": "gaspardol",
    "github_repo": "silogic",
    "github_version": "main",
    "doc_path": "docs",
    "default_mode": "light",
}

# Cleaner signatures (drop the long module prefix on class/func names).
add_module_names = False
python_use_unqualified_type_names = True
