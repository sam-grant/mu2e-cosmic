"""Run analysis."""
import sys
import yaml
import shutil
import argparse
from pathlib import Path

sys.path.append("../common")
from process import CosmicProcessor 
from postprocess import PostProcess 
from io_manager import Write

from pyutils.pylogger import Logger

def load_config(config_file):
    """Load processing configuration."""
    with open(config_file, "r") as f:
        return yaml.safe_load(f)

def main():
    # Start logger
    logger = Logger(print_prefix="[ana]")

    # Parse arguments
    parser = argparse.ArgumentParser(description="Mu2e cosmic ray analysis")
    parser.add_argument("--config", required=True, help="Configuration YAML file")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)

    logger.log(f"Starting analysis with {config}", "info")
    
    # Initialise processor with config parameters
    cosmic_processor = CosmicProcessor(**config["processor"])
    results_per_file = cosmic_processor.execute()
    
    # Run postprocessing
    postprocessor = PostProcess(**config.get("postprocess", {}))
    results = postprocessor.execute(results_per_file)
    
    # Save results
    writer = Write(**config["output"])
    writer.save_all(results)
    
    # Copy config file to output directory for reproducibility
    out_path = Path(config["output"]["out_path"])
    out_path.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, out_path / Path(args.config).name)
    
    print(f"Results saved to: {config["output"]["out_path"]}")

    logger.log(f"Analysis complete, results saved to analysis with {config["output"]["out_path"]}", "success")

if __name__ == "__main__":
    main()