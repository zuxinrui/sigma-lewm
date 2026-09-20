"""Compatibility alias for trim.protocols.demo_goal_formal_specs."""
import sys
from importlib import import_module
_implementation = import_module("trim.protocols.demo_goal_formal_specs")
globals().update({k: v for k, v in vars(_implementation).items() if not k.startswith("__")})
if __name__ != "__main__":
    sys.modules[__name__] = _implementation
