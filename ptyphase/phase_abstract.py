# Moved to ptyphase.projection.phase_abstract; this alias keeps old imports working.
import sys
from .projection import phase_abstract as _m
sys.modules[__name__] = _m
