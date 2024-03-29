from pycufsm._version import __version__

# -- Project information

project = "pyCUFSM"
copyright = "2023, Brooks H. Smith"
author = "Brooks H. Smith"

release = __version__
version = __version__
# sys.path.insert(0, os.path.abspath('../../pycufsm'))  # Source code dir relative to this file

# -- General configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "autoapi.extension",
]
autoapi_dirs = ["../../pycufsm"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for HTML output

html_theme = "sphinx_rtd_theme"

# -- Options for EPUB output
epub_show_urls = "footnote"
