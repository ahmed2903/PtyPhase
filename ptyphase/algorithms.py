# Moved to ptyphase.projection.algorithms; this alias keeps old imports working.
import sys
from .projection import algorithms as _m
sys.modules[__name__] = _m
