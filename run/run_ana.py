"""
Run core analysis
Samuel Grant 2025
"""
# External
import sys
import os
import yaml
import shutil
import argparse
from pathlib import Path
# Mu2e
from pyutils.pylogger import Logger

# Internal (core)
sys.path.append("../src/core")
from process import CosmicProcessor 
from postprocess import PostProcess 

# Internal (utils)
sys.path.append("../src/utils")
from config_manager import ConfigManager
from io_manager import Write
from run_logger import RunLogger

# Create logger once at module level
logger = Logger(print_prefix="[run]")

def main():
    """Run core analysis"""
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Configuration YAML file")
    args = parser.parse_args()
    
    # Generate log filename from config filename
    config_stem = Path(args.config).stem  # Remove .yaml extension
    log_file_path = f"{config_stem}.log"
    
    # Set up logging (always enabled)
    tee = RunLogger(log_file_path)
    sys.stdout = tee
    
    try:
        # Load configuration from manager
        config_manager = ConfigManager(args.config)
        config = config_manager.config
        actual_config_path = config_manager.config_path
        
        # Initialise processor with config parameters
        cosmic_processor = CosmicProcessor(**config["process"]) 
        
        # Run main processing and analysis
        results_per_file = cosmic_processor.execute()

        # Propagate on_spill parameter from process to postprocess
        # It is needed by both. Use True default.
        config["postprocess"]["on_spill"] = config["process"].get("on_spill", True)
        
        # Initialise postprocessor
        postprocessor = PostProcess(**config.get("postprocess", {}))
        
        # Run postprocessing
        results = postprocessor.execute(results_per_file)
        
        # Save results
        writer = Write(**config["output"])
        writer.write_all(results)
        
        # Create output directory
        out_path = Path(config["output"]["out_path"])
        out_path.mkdir(parents=True, exist_ok=True)
        
        # Copy config file to output directory for reproducibility
        shutil.copy2(actual_config_path, out_path / Path(args.config).name)
      
        # Copy log file to output directory
        if os.path.exists(log_file_path):
            log_filename = Path(log_file_path).name
            os.makedirs(out_path, exist_ok=True)
            shutil.move(log_file_path, out_path / log_filename)
            logger.log(f"Moved log file to {out_path / log_filename}", "info")
        
        logger.log(f"Analysis complete, wrote results to {config['output']['out_path']}", "success")
        
    except Exception as e:
        logger.log(f"Analysis failed: {e}", "error")
        sys.exit(1)
    finally:
        # Clean up logging
        sys.stdout = tee.terminal
        sys.stderr = tee.terminal
        tee.close()

if __name__ == "__main__":
    main()
