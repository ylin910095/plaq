# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from pathlib import Path

# Add the source directory to the path for autodoc
sys.path.insert(0, str(Path("../src").resolve()))

# -- Project information -----------------------------------------------------
project = "plaq"
copyright = "2024, Plaq Contributors"
author = "Plaq Contributors"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Create _static directory if it doesn't exist (for custom CSS/JS)
(Path(__file__).parent / "_static").mkdir(parents=True, exist_ok=True)

# -- Options for autodoc -----------------------------------------------------
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

autodoc_typehints = "description"
autodoc_typehints_format = "short"

# -- Options for Napoleon (Google/NumPy style docstrings) --------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# -- Options for MathJax -----------------------------------------------------
mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
        "processEscapes": True,
        "processEnvironments": True,
    },
    "options": {
        "ignoreHtmlClass": "tex2jax_ignore",
        "processHtmlClass": "tex2jax_process",
    },
}

# -- Options for intersphinx -------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

# -- Options for MyST parser -------------------------------------------------
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "colon_fence",
]

# Suppress the mathjax override warning from myst-parser
suppress_warnings = ["myst.mathjax"]
