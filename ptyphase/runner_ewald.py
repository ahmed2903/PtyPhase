# Moved to ptyphase.projection.runner_ewald; this alias keeps old imports working.
import sys
from .projection import runner_ewald as _m
sys.modules[__name__] = _m
