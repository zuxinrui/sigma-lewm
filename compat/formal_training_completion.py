"""Compatibility alias for trim.protocols.formal_training_completion."""
import sys
from importlib import import_module
_implementation = import_module("trim.protocols.formal_training_completion")
globals().update({k: v for k, v in vars(_implementation).items() if not k.startswith("__")})
if __name__ != "__main__":
    sys.modules[__name__] = _implementation
