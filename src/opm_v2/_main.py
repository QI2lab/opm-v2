
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)
import logging

numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)

logger = logging.getLogger("pymmcore-plus")
logger.setLevel(logging.DEBUG)  # Allow DEBUG and INFO messages in principle

# Create and add a filter to suppress WARNING and above
class BelowWarningFilter(logging.Filter):
    def filter(self, record):
        return record.levelno < logging.WARNING

logger.addFilter(BelowWarningFilter())

# Set the logging level to WARNING to suppress DEBUG messages
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
from opm_v2._opm_app import main

if __name__ == "__main__":
    main()