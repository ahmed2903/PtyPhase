# Moved to ptyphase.projection.algorithms_3d; this alias keeps old imports working.
import sys
from .projection import algorithms_3d as _m
sys.modules[__name__] = _m
