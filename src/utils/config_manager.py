import os
import yaml

from pyutils.pylogger import Logger

class ConfigManager:
    """ Configuration manager
    """
    def __init__(self, config_file_name=None, verbosity=1):
        """Initialise
    
        Args:
            config_file_name (str, opt): Analysis config YAML file. Defaults to None.
            verbosity (int, optional): Level of output detail (0: critical errors only, 1: info, 2: debug, 3: deep debug)
        """
        # Parameters
        self.config_file_name = config_file_name
        self.verbosity = verbosity

        # Tools
        self.logger = Logger(
            print_prefix="[ConfigManager]",
            verbosity=self.verbosity
        )
        
        if self.config_file_name is None:
            self.logger.log("No configuration YAML file provided (see config/ana!)", "error")
            return            

        # Get config (self.config and self.config_path)
        self._load_config() 

    def _load_config(self):
        """Load config with fallback locations"""
        
        # List of paths to try (in order)
        paths = [
            self.config_file_name, # Provided file name
            os.path.join("../../config/ana", self.config_file_name), # Relative fallback
            os.path.join("config", "ana", self.config_file_name), # From repo root
            os.path.join(os.path.dirname(__file__), "..", "..", "config", "ana", self.config_file_name)  # Script-relative
        ]
        
        # Iterate through paths
        for path in paths:
            if os.path.exists(path):
                self.logger.log(f"Loading config from: {path}", "success")
                with open(path, "r") as f:
                    self.config = yaml.safe_load(f)
                    self.config_path = os.path.abspath(path)
                    # Print config
                    config_str = yaml.dump(self.config , default_flow_style=False, indent=2)
                    self.logger.log(f"Analysis configuration:\n{config_str}", "info")
                    # Return 
                    return 
        
        # If we get here, file wasn't found anywhere
        self.logger.log(f"Config file '{self.config_path}' not found in any of: {paths}")
        raise

            