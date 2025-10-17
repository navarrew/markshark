# Prefer the generated _version.py written by hatch-vcs during build
try:
    from ._version import __version__  # created at build/editable install time
except Exception:
    # Fallback during editable runs before build hook fires
    try:
        from importlib.metadata import version, PackageNotFoundError
        try:
            __version__ = version("markshark")
        except PackageNotFoundError:
            __version__ = "0.0.0+local"
    except Exception:
        __version__ = "0.0.0+local"