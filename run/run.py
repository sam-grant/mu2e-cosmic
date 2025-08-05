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
# Internal
sys.path.append("../src/core")
from process import CosmicProcessor 
from postprocess import PostProcess 
from io_manager import Write
sys.path.append("../src/utils")
from run_logger import RunLogger

# Create logger once at module level
logger = Logger(print_prefix="[run]")

def load_config(config_path):
    """Load config with fallback locations"""
    
    # List of paths to try (in order)
    paths = [
        config_path,                                    # User-provided path
        os.path.join("../config/ana", config_path),     # Relative fallback
        os.path.join("config", "ana", config_path),     # From repo root
        os.path.join(os.path.dirname(__file__), "..", "..", "config", "ana", config_path)  # Script-relative
    ]
    
    # Iterate through paths
    for path in paths:
        if os.path.exists(path):
            logger.log(f"Loading config from: {path}", "info")
            with open(path, "r") as f:
                config = yaml.safe_load(f)
                # Return both config and the actual path used
                return config, os.path.abspath(path)
    
    # If we get here, file wasn't found anywhere
    raise FileNotFoundError(f"Config file '{config_path}' not found in any of: {paths}")

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
    # sys.stderr = tee  # Also capture tqdm progress bars
    
    try:
        # Load configuration, returns both config and path
        config, actual_config_path = load_config(args.config)
        
        config_str = yaml.dump(config, default_flow_style=False, indent=2)
        logger.log(f"Starting analysis with configuration:\n{config_str}", "info")
        
        # Initialise processor with config parameters
        cosmic_processor = CosmicProcessor(**config["processor"]) 
        
        # Run main processing and analysis
        results_per_file = cosmic_processor.execute()
        
        # Initialise postprocessor
        postprocessor = PostProcess(**config.get("postprocess", {}))
        
        # Run postprocessing
        results = postprocessor.execute(results_per_file)
        
        # Save results
        writer = Write(**config["output"])
        writer.save_all(results)
        
        # Create output directory
        out_path = Path(config["output"]["out_path"])
        out_path.mkdir(parents=True, exist_ok=True)
        
        # Copy config file to output directory for reproducibility
        shutil.copy2(actual_config_path, out_path / Path(args.config).name)
        
        # Copy log file to output directory
        if os.path.exists(log_file_path):
            log_filename = Path(log_file_path).name
            shutil.copy2(log_file_path, out_path / log_filename)
            logger.log(f"Copied log file to {out_path / log_filename}", "info")
        
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