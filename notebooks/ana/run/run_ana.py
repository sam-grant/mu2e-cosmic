#!/usr/bin/env python3
"""Analysis function that can be called with ana_label as argument"""

import os
# === Cell 5 ===
import sys
sys.path.append("..")
from preamble import *

def run_analysis(ana_label):
    """Run analysis for the given ana_label"""
    
    # === Cell 3 ===
    # ana_label is now passed as parameter
    
    # === Cell 5 ===
    
    # === Cell 7 ===
    import subprocess
    result = subprocess.run(['ls', f'../../../output/results/{ana_label}'], 
                          capture_output=True, text=True)
    print(result.stdout)
    
    # === Cell 8 ===
    loader = Load(
        in_path = f"../../../output/results/{ana_label}"
    )
    
    # === Cell 9 ===
    results = loader.load_pkl() 
    results_from_persistent = loader.load_all() # Persistent backup 
    
    # === Cell 12 ===
    # display(results["cut_flow"])
    
    # === Cell 14 ===
    # display(results["analysis"])
    
    # === Cell 16 ===
    # Setup draw for this cutset
    on_spill = "offspill" not in ana_label
    draw = Draw(cutset_name=ana_label.split('_')[0], on_spill=on_spill)
    # Define image directory
    img_dir = f"../../../output/images/{ana_label}"
    os.makedirs(img_dir, exist_ok=True)
    
    # === Cell 17 ===
    draw.plot_mom_windows(results["hists"], out_path=f"{img_dir}/h1o_1x3_mom_windows.png") 
    
    # === Cell 18 ===
    draw.plot_summary(results["hists"], out_path=f"{img_dir}/h1o_3x3_summary.png")
    
    # === Cell 19 ===
    draw.plot_crv_z(results["hists"], out_path=f"{img_dir}/h1o_crv_.png") 
    
    # === Cell 21 ===
    # print(results["event_info"])
    
    # === Cell 22 ===
    # if results["events"] is not None:
    #     from pyutils.pyprint import Print
    #     printer = Print()
    #     printer.print_n_events(results["events"], n_events = len(results["events"]))

def main():
    """Main function to handle command line arguments"""
    if len(sys.argv) != 2:
        print("Usage: python analysis.py <ana_label>")
        sys.exit(1)
    
    ana_label = sys.argv[1]
    
    if not ana_label.strip():
        print("Error: ana_label cannot be empty")
        sys.exit(1)
    
    run_analysis(ana_label)

if __name__ == "__main__":
    main()