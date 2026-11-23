# Moved to ptyphase.optimisation.legacy.finnp_class; this alias keeps old imports working.
import sys
from .optimisation.legacy import finnp_class as _m
sys.modules[__name__] = _m
