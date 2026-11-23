# Moved to ptyphase.projection.epry_abstract; this alias keeps old imports working.
import sys
from .projection import epry_abstract as _m
sys.modules[__name__] = _m
