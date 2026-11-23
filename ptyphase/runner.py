# Moved to ptyphase.projection.runner; this alias keeps old imports working.
import sys
from .projection import runner as _m
sys.modules[__name__] = _m
