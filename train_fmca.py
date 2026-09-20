"""Compatibility alias for slewm.training.fmca."""
import sys
from importlib import import_module
_implementation = import_module("slewm.training.fmca")
globals().update({k: v for k, v in vars(_implementation).items() if not k.startswith("__")})
if __name__ != "__main__":
    sys.modules[__name__] = _implementation

if __name__ == "__main__":
    _implementation.run()
