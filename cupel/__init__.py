"""cupel — fire-assay your local LLMs"""
from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("cupel")
except PackageNotFoundError:
    __version__ = "dev"
