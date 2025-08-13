# External
import sys
import gc

# pyutils 
from pyutils.pyprocess import Processor, Skeleton
from pyutils.pylogger import Logger

# mu2e_cosmic_ana
from analyse import Analyse

class CosmicProcessor(Skeleton):
    """Class for core data processing
    
    This class inherits from the pyutils Skeleton template class

     pyutils.Skeleton (base template)
     ├── Provides: init(), process_file(), execute()
     └── Expects: You to override member varibles and process_file() 
     
     CosmicProcessor(Skeleton)  
     ├── Inherits: execute(), file handling, parallel processing
     ├── Customises: __init__() with cosmic-specific parameters
     └── Overrides: process_file() with cosmic analysis logic
    """
    def __init__(
        self,
        defname = None, 
        file_list_path = None,
        file_name = None,
        cutset_name = "alpha",
        branches = { 
            "evt" : ["run", "subrun", "event"],
            "crv" : ["crvcoincs.time", "crvcoincs.nHits", "crvcoincs.pos.fCoordinates.fZ"],
            "trk" : ["trk.nactive", "trk.pdg", "trkqual.valid", "trkqual.result"],
            "trkfit" : ["trksegs", "trksegpars_lh"],
            "trkmc" : ["trkmcsim"]
        },
        on_spill=False,
        cuts_to_toggle = None,
        groups_to_toggle = None,
        use_remote = True,
        location = "disk",
        max_workers = None,
        use_processes = True,
        verbosity = 1,
        worker_verbosity = 0
        
    ):
        """Initialise your processor with specific configuration
        
        This method sets up all the parameters needed for this specific analysis.

        Args:
            defname (str, opt): Dataset definition name. Defaults to None.
            file_list_path (str, opt): Path to file list. Defaults to None.
            file_name (str, opt): Single file to process. Defaults to None.
            cutset_name (str, opt): Which cutset to use. Defaults to "alpha".
            branches (dict or list, opt): EventNtuple branches to extract.
            on_spill (bool, opt): Apply on-spill timing cuts. Defaults to False.
            cuts_to_toggle (dict, opt): Cuts to enable/disable. Defaults to None.
            groups_to_toggle (dict, opt): Cut groups to enable/disable. Defaults to None.
            use_remote (bool, opt): Use remote file access. Defaults to True.
            location (str, opt): File location ("disk", etc.). Defaults to "disk".
            max_workers (int, opt): Number of parallel workers. Defaults to 50.
            use_processes (bool, opt): Use processes rather than threads. Defaults to True.
            verbosity (int, opt): Logging verbosity level. Defaults to 2 (max).
            worker_verbosity (int, opt): Verbosity for worker processes (debug only!). Defaults to 0.

        Note: 
            Could also use kwargs**, but this is self documenting 
        """
        # Call the Skeleton's __init__ method first
        super().__init__()

        # Override parameters from Skeleton

        # Dataset, file list, or file name/path
        self.defname = defname 
        self.file_list_path = file_list_path
        self.file_name = file_name 

        # EventNtuple branches
        self.branches = { 
            "evt" : ["run", "subrun", "event"],
            "crv" : ["crvcoincs.time", "crvcoincs.nHits", "crvcoincs.pos.fCoordinates.fZ"],
            "trk" : ["trk.nactive", "trk.pdg", "trkqual.valid", "trkqual.result"],
            "trkfit" : ["trksegs", "trksegpars_lh"],
            "trkmc" : ["trkmcsim"]
        }

        # Process parameters
        self.on_spill = on_spill                    # Apply t0 cut for onspill
        self.use_remote = use_remote                # Use remote file via mdh
        self.location = location                    # File location
        self.max_workers = max_workers              # Limit the number of workers
        self.use_processes = use_processes          # Use processes rather than threads
        self.verbosity = verbosity                  # Set verbosity 
        self.worker_verbosity = worker_verbosity    # Set worker verbosity 

        # Analysis methods
        self.analyse = Analyse(
            cutset_name=cutset_name,
            on_spill=self.on_spill,
            verbosity=self.worker_verbosity # Reduce verbosity for workers
        )

        self.logger = Logger(
            print_prefix = "[CosmicProcessor]"
        )
            
        # Toggle cuts
        self.cuts_to_toggle = cuts_to_toggle
        self.groups_to_toggle = groups_to_toggle

        init_summary = f"""
        \tdefname         = {self.defname}
        \tfile_list_path  = {self.file_list_path}
        \tfile_name       = {self.file_name}
        \tcutset_name     = {self.analyse.cutset_name}
        \ton_spill        = {self.on_spill}
        \tcuts_to_toggle  = {self.cuts_to_toggle}
        \tgroups_to_toggle = {self.groups_to_toggle}
        \tbranches        = {list(self.branches.keys()) if isinstance(self.branches, dict) else self.branches}
        \tuse_remote      = {self.use_remote}
        \tlocation        = {self.location}
        \tmax_workers     = {self.max_workers}
        \tverbosity       = {self.verbosity}
        \tuse_processes   = {self.use_processes}"""

        # init_summary = f"""
        # \tdefname         = {self.defname}
        # \tfile_list_path  = {self.file_list_path}
        # \tfile_name       = {self.file_name}
        # \ton_spill        = {self.on_spill}
        # \tcuts_to_toggle  = {self.cuts_to_toggle}
        # \tbranches        = {list(self.branches.keys()) if isinstance(self.branches, dict) else self.branches}
        # \tuse_remote      = {self.use_remote}
        # \tlocation        = {self.location}
        # \tmax_workers     = {self.max_workers}
        # \tverbosity       = {self.verbosity}
        # \tuse_processes   = {self.use_processes}"""
        # Confirm init
        self.logger.log(f"Initialised with:{init_summary}", "success")
        
    # ==========================================
    # Define the core processing logic
    # ==========================================
    # This method overrides Skeleton's process_file method
    # It will be called automatically for each file by the execute method
    def process_file(self, file_name): 
        """Process a single ROOT file
        
        This method will be called for each file in our list.
        It extracts data, processes it, and returns a result.
        
        Args:
            file_name: Path to the ROOT file to process
            
        Returns:
            A tuple containing the histogram (counts and bin edges)
        """        
        try:
            # Create processor for this file
            this_processor = Processor(
                use_remote=self.use_remote,    
                location=self.location,  
                verbosity=self.worker_verbosity,
                worker_verbosity=self.worker_verbosity
            )
            
            # Extract the data 
            this_data = this_processor.process_data(
                file_name = file_name, 
                branches = self.branches
            )
            
            # Run analysis          
            results = self.analyse.execute(
                this_data, 
                file_name,
                self.cuts_to_toggle,
                self.groups_to_toggle
            )

            # Clean up memory
            gc.collect()

            return results 
        
        except Exception as e:
            # Report any errors that occur during processing
            self.logger.log(f"Error processing {file_name}: {e}", "error")
            raise e

    # def process_file(self, file_name):
    #     """Minimal test version"""
    #     import time
    #     import random
        
    #     print(f"Processing {file_name}", flush=True)
    #     time.sleep(random.random())  # Simulate work
    #     return {"file": file_name, "result": "done"}

    # def process_file(self, file_name):
    #     """Test 2: Create Processor and call process_data"""
    #     import sys
    #     print(f"Creating Processor for {file_name}", file=sys.stderr, flush=True)
        
    #     this_processor = Processor(
    #         use_remote=self.use_remote,
    #         location=self.location,
    #         verbosity=0
    #     )
        
    #     print(f"Calling process_data for {file_name}", file=sys.stderr, flush=True)
        
    #     this_data = this_processor.process_data(
    #         file_name=file_name,
    #         branches=self.branches
    #     )
        
    #     print(f"process_data completed for {file_name}, data type: {type(this_data)}", file=sys.stderr, flush=True)
        
    #     return {"file": file_name, "data_length": len(this_data) if this_data else 0}