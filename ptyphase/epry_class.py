# Moved to ptyphase.projection.epry_class; this alias keeps old imports working.
import sys
from .projection import epry_class as _m
sys.modules[__name__] = _m
