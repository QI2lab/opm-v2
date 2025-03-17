
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)
import logging

# Set the logging level to WARNING to suppress DEBUG messages
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
from opm_v2._opm_app import main


if __name__ == "__main__":
    main()