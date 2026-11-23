# Moved to ptyphase.optimisation.legacy.finnp_class_zernike; this alias keeps old imports working.
import sys
from .optimisation.legacy import finnp_class_zernike as _m
sys.modules[__name__] = _m
