# Moved to ptyphase.projection.divide_and_concur; this alias keeps old imports working.
import sys
from .projection import divide_and_concur as _m
sys.modules[__name__] = _m
