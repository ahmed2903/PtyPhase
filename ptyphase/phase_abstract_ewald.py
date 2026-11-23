# Moved to ptyphase.projection.phase_abstract_ewald; this alias keeps old imports working.
import sys
from .projection import phase_abstract_ewald as _m
sys.modules[__name__] = _m
