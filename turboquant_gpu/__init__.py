__version__ = "0.1.4"

def __getattr__(name):
    if name == "TurboQuantEngine":
        from .host import TurboQuantEngine
        return TurboQuantEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["__version__", "TurboQuantEngine"]
