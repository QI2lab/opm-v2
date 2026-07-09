
# ------------------------------------------------------------------------------------ #
# Manage warnings and logging levels
# ------------------------------------------------------------------------------------ #

import logging
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)

numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)

logger = logging.getLogger("pymmcore-plus")
logger.setLevel(logging.DEBUG)  # Allow DEBUG and INFO messages in principle

logging.raiseExceptions = False

# Create and add a filter to suppress WARNING and above
class BelowWarningFilter(logging.Filter):
    def filter(self, record):
        return record.levelno < logging.WARNING

logger.addFilter(BelowWarningFilter())

# Set the logging level to WARNING to suppress DEBUG messages
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)


# ------------------------------------------------------------------------------------ #
# Start the GUI
# ------------------------------------------------------------------------------------ #

from _app_v2 import main

if __name__ == "__main__":
    main()