# External packages
import sys
import os
import warnings
warnings.filterwarnings("ignore") # suppress warnings
# Set DataFrame display options
import pandas as pd
# pd.options.display.float_format = lambda x: f'{int(x)}' if x == int(x) else f'{x:.3f}'
# pd.options.display.float_format = lambda x: f'{int(x):,}' if x == int(x) else f'{x:,.3f}'
pd.options.display.float_format = lambda x: (
    f'{int(x):,}' if x == int(x) else 
    f'{x:.2e}' if abs(x) >= 1e3 or (abs(x) < 1e-3 and x != 0) else 
    f'{x:.3f}'
)

# pyutils classes
from pyutils.pylogger import Logger
from pyutils.pyprint import Print

# mu2e-cosmic classes
sys.path.extend(["../../../src/core", "../../../src/utils"])
from io_manager import Load
from draw import Draw

# Make everything available when using "from preamble import *"
__all__ = ["Logger", "Print", "Load", "Draw"] 