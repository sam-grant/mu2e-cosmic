"""
Example demonstrating reading/write awkward arrays from/to parquet
"""
######################################################
# Setup tools and parameters
######################################################

import os
# Define paths
in_path = "input/ana_alpha_CRY_offspill-LH_as/"
out_path = "output/example_parquet"
# Create directory if it doesn't exist
os.makedirs(out_path, exist_ok=True)

# pyutils tools
from pyutils.pylogger import Logger
logger = Logger()
from pyutils.pyprint import Print
printer = Print()

# IO tools 
import sys
sys.path.append("../src/core")
from io_manager import Load, Write
loader = Load(in_path = in_path)
writer = Write(out_path = out_path)

######################################################
# Load data
######################################################

results = loader.load_all() # parquet, h5, and txt
backup_results = loader.load_all_pkl() # pickle

logger.log(f"Result keys: {results.keys()}", "info")
logger.log(f"Backup result keys: {backup_results.keys()}", "info")

# Loaded event info
# logger.log(f"Filtered events from parquet ({len(results["events"])} total)", "info")
# results["events"].type.show()

######################################################
# Check if arrays from parquet and pickle are equal
######################################################

import io

def arrays_equal(arr1, arr2, printer):
    # Quick length check first
    if len(arr1) != len(arr2):
        return False
    
    # Capture printer output for first array
    old_stdout = sys.stdout
    sys.stdout = buffer1 = io.StringIO()
    try:
        printer.print_n_events(arr1, n_events=len(arr1))
        output1 = buffer1.getvalue()
    finally:
        sys.stdout = old_stdout
    
    # Capture printer output for second array
    sys.stdout = buffer2 = io.StringIO()
    try:
        printer.print_n_events(arr2, n_events=len(arr2))
        output2 = buffer2.getvalue()
    finally:
        sys.stdout = old_stdout
    
    # Compare the string outputs
    return output1 == output2

equal = arrays_equal(results["events"], backup_results["events"], printer)
logger.log(f"Arrays are equal? {equal}", "info")


######################################################
# Write back to parquet
######################################################
writer.save_events_parquet(results["events"])