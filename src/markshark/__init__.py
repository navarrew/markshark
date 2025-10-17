# src/markshark/__init__.py
try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:  # Python <3.8 backport, but you require 3.9+, so not needed.
    from importlib_metadata import version, PackageNotFoundError  # pragma: no cover

try:
    __version__ = version("markshark")
except PackageNotFoundError:
    __version__ = "0.0.0+local"