from src.utils.instantiators import instantiate_callbacks, instantiate_loggers
from src.utils.logging_utils import log_hyperparameters
from src.utils.pylogger import RankedLogger
from src.utils.rich_utils import enforce_tags, print_config_tree
from src.utils.utils import extras, get_metric_value, package_available, lib_available, task_wrapper


# Duplicate of tests/helpers/package_available.py
# Due to the way the code is structured, we need to redefine the package availability checks here
_SPICE_AVAILABLE = package_available("PySpice")
_XYCE_AVAILABLE = lib_available("Xyce")
_SCIPY_AVAILABLE = package_available("scipy")
_QPSOLVERS_AVAILABLE = package_available("qpsolvers")
_PROXSUITE_AVAILABLE = package_available("proxsuite")
