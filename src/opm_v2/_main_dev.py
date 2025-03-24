
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)
import logging
import matplotlib.pyplot as plt

# Set the logging level to WARNING to suppress DEBUG messages
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
from opm_v2._opm_app_dev import main


if __name__ == "__main__":
    main()