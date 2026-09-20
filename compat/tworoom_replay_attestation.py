"""Compatibility alias for trim.protocols.tworoom_replay_attestation."""
import sys
from importlib import import_module
_implementation = import_module("trim.protocols.tworoom_replay_attestation")
globals().update({k: v for k, v in vars(_implementation).items() if not k.startswith("__")})
if __name__ != "__main__":
    sys.modules[__name__] = _implementation
