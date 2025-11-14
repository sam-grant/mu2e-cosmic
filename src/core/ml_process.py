"""
Prepare cosmic data for ML work
Sam Grant 2025
"""

# External
import sys
import gc
import os
import awkward as ak

# pyutils 
from pyutils.pyprocess import Processor, Skeleton
from pyutils.pylogger import Logger
from pyutils.pyselect import Select

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, "..", "utils"))

from analyse import Analyse
from postprocess import PostProcess
from cut_manager import CutManager

# from pyutils.pylogger import Logger

class MLProcessor(Skeleton):
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
        cutset_name = "MLPreprocess",
        branches = { 
            "evt" : ["run", "subrun", "event"],
            "crv" : ["crvcoincs.time", "crvcoincs.PEs", "crvcoincs.nHits", 
                     "crvcoincs.pos.fCoordinates.fX", "crvcoincs.pos.fCoordinates.fY", "crvcoincs.pos.fCoordinates.fZ"],
            "trk" : ["trk.nactive", "trk.pdg", "trkqual.valid", "trkqual.result"],
            "trkfit" : ["trksegs", "trksegpars_lh"],
            "trkmc" : ["trkmcsim"]
        },
        feature_set = "trk", # "crv" # TODO: better implementation? 
        on_spill=True,
        cuts_to_toggle = None,
        groups_to_toggle = None,
        use_remote = True,
        location = "disk",
        max_workers = 50,
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
            on_spill (bool, opt): Apply on-spill timing cuts. Defaults to True.
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
            "crv" : ["crvcoincs.time", "crvcoincs.PEs", "crvcoincs.nHits", "crvcoincs.pos.fCoordinates.fZ"],
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
        self.feature_set = feature_set              # Feature set (tracker or CRV)
        
        # Analysis methods
        self.analyse = Analyse(
            cutset_name=cutset_name,
            on_spill=self.on_spill,
            verbosity=self.worker_verbosity # Reduce verbosity for workers
        )
        
        # Cut manager
        self.cut_manager = CutManager(verbosity=self.worker_verbosity)
        
        # Selector
        self.selector = Select(verbosity=self.worker_verbosity)

        # Postprocess
        # BE VERY CAREFUL HERE SINCE THERE IS A METHOD IN SKELETON CALLED POSTPROCESS
        # THIS ATTRIBUTE MUST BE CALLED SOMETHING DIFFERENT 
        self.postprocessor = PostProcess(on_spill=self.on_spill)
        
        # Create unique logger
        self.logger = Logger( 
            print_prefix="[MLPreprocess]",
            verbosity=self.verbosity
        )

        # Init printout
        self.logger.log(
            f"Initialised with:\n"
            f"  on_spill        = {self.on_spill}\n"
            f"  verbosity       = {self.verbosity}",
            "info"
        )

        # Create unique logger
        self.logger = Logger(
            print_prefix = "[MLPreprocessor]"
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

        # Confirm init
        self.logger.log(f"Initialised with:{init_summary}", "success")
        
    def _process_array(self, data, file_id):
        """Helper to execute processing on array

        Args:
            data: The data to analyse
            file_id: Identifier for the file
            
        Returns:
            dict: filtered and flattened arrays for ML
        """
        
        self.logger.log(f"Beginning preprocessing on file: {file_id}", "max") 
        
        try:
            # Define cuts
            self.logger.log(f"Defining cuts", "max")
            data = self.analyse.define_cuts(data, self.cut_manager)

            # Create main cut flow
            self.logger.log("Creating cut flow", "max")
            cut_flow = self.cut_manager.create_cut_flow(data)

            ##########################################
            # Apply cuts
            ##########################################  

            # Apply Preselect cuts
            self.logger.log("Applying cuts", "max")
            data_cut = ak.copy(data)
            data_cut = self.analyse.apply_cuts(data_cut, self.cut_manager)

            # Create verification plots (before flattening)
            # pdg, pz, trk / event, t0, trkqual
            hists = self.analyse.create_histograms(
                datasets = {
                    "Before cuts" : data,
                    "After cuts" : data_cut
                }
            )
            
            # Container for processed data 
            processed_data = {}

            if self.feature_set == "trk":
                # Tracker surfaces on trkfit level
                at_trk_front = self.selector.select_surface(data_cut["trkfit"], surface_name="TT_Front") 
                # Select
                processed_data["d0"] = data_cut["trkfit"]["trksegpars_lh"]["d0"][at_trk_front]
                processed_data["tanDip"] = self.analyse.get_pitch_angle(data_cut["trkfit"][at_trk_front])
                processed_data["maxr"] = data_cut["trkfit"]["trksegpars_lh"]["maxr"][at_trk_front]
                # Calculate momentum magnitude
                mom_mag = self.analyse.vector.get_mag(data_cut["trkfit"][at_trk_front]["trksegs"], "mom")
                processed_data["mom_mag"] = mom_mag

            elif self.feature_set == "crv":

                # Note that you MUST have has_trk_mid switched on for this 
                
                # Event parameters (nice to have)
                processed_data["event"] = data_cut["evt"]["event"]
                processed_data["subrun"] = data_cut["evt"]["subrun"]

                # Work with event-level parameters
                # Conditional coincidence selection to avoid dT bias:
                # - Signal samples (CeEndpoint*): Use random coincidence (avoid bias)
                # - Background samples (CosmicCRY*): Use central dT (physics-motivated)
                
                # Check if we're processing a signal CE sample by looking for "CeEndpoint" in defname
                # is_signal = self.defname and "CeEndpoint" in self.defname
                
                # if False: # is_signal:

                    # This is really complicated as usual
                    # For signal samples (CeEndpoint*): Select random coincidence from available dT indices
                    # import numpy as np
                    # dT_idx = data_cut["dev"]["dT_idx"]
                    # print("dT_idx ",dT_idx)
                    # n_coinc = ak.num(dT_idx, axis=-1)
                    # print("n_coinc ",n_coinc)
                    # random_positions = np.random.RandomState(42).randint(
                    #     0, 
                    #     ak.max(n_coinc, -1)-1, # n-1
                    #     size=len(dT_idx)
                    # )
                    # print("random_positions ",random_positions)
                    # random_positions = ak.Array(random_positions)[:, None] # broadcast back to coincidence level
                    # print("random_positions ",random_positions.type, random_positions)
                    # print("cent_dT_idx", data_cut["dev"]["cent_dT_idx"].type, data_cut["dev"]["cent_dT_idx"])
                    # valid = ~ak.is_none(data_cut["dev"]["cent_dT_idx"], axis=-1)
                    # # Select the actual index value from dT_idx using the random position
                    # selected_idx = random_positions[valid]
                
                    
                    # selected_idx = random_positions
                    # print("selected_idx ",selected_idx)

                    # import numpy as np
                    
                    # Get all valid dT indices for each event
                    # dT_idx = data_cut["dev"]["dT_idx"]  # Shape: [event, coincidence]
                    
                    # # Set random seed for reproducibility
                    # random_state = np.random.RandomState(42)
                    
                    # # Get number of coincidences per event
                    # n_coinc_per_event = ak.num(dT_idx, axis=-1)
                    
                    # # Generate random indices for each event
                    # # This gives an integer in [0, n_coinc-1] for each event
                    # random_positions = random_state.randint(
                    #     0, 
                    #     ak.to_numpy(n_coinc_per_event),  # High is exclusive, so this gives [0, n_coinc)
                    #     dtype=int
                    # )
                    
                    # # Handle edge case: events with 0 coincidences
                    # random_positions = np.where(
                    #     ak.to_numpy(n_coinc_per_event) > 0, 
                    #     random_positions, 
                    #     0
                    # )
                    
                    # # Select the random coincidence from each event
                    # # Need to pair each event with its random position
                    # selected_idx = dT_idx[ak.local_index(dT_idx) == random_positions]

                    # Just take the first one
                    # selected_idx = ak.firsts(data_cut["dev"]["dT_idx"], axis=-1)[:, None]
                    
                    # self.logger.log(f"Using random coincidence selection for signal sample (CeEndpoint): {self.defname}", "info")
                # else:
                    # For background samples (CosmicCRY): Use central dT (physics-motivated selection)
                # selected_idx = data_cut["dev"]["cent_dT_idx"]
                    # self.logger.log(f"Using central dT selection for background sample (CosmicCRY): {self.defname}", "info")

                # One coinc / event
                # Based on central ∆t
                coinc_idx = data_cut["dev"]["cent_dT_idx"]
                
                if ak.all(data_cut["dev"]["dT"][coinc_idx]!=data_cut["dev"]["cent_dT"]):
                    self.logger.log(f"Central ∆T mismatch", "error")
                    raise ValueError()

                
                # CRV parameters using selected index
                processed_data["crv_z"] = data_cut["crv"]["crvcoincs.pos.fCoordinates.fZ"][coinc_idx]
                processed_data["PEs"] = data_cut["crv"]["crvcoincs.PEs"][coinc_idx]
                processed_data["dT"] = data_cut["dev"]["dT"][coinc_idx]
                processed_data["nHits"] = data_cut["crv"]["crvcoincs.nHits"][coinc_idx]
                
                # Calculate PEs/nHits ratio (avoid division by zero)
                processed_data["PEs_per_hit"] = ak.where(
                    processed_data["nHits"] > 0, 
                    processed_data["PEs"] / processed_data["nHits"], 
                    0
                )
                
                # Tracker parameters
                at_trk_front = self.selector.select_surface(data_cut["trkfit"], surface_name="TT_Front") 
                at_trk_mid = self.selector.select_surface(data_cut["trkfit"], surface_name="TT_Mid")
                processed_data["t0"] = data_cut["trkfit"]["trksegs"]["time"][at_trk_mid]
                processed_data["d0"] = data_cut["trkfit"]["trksegpars_lh"]["d0"][at_trk_front]
                processed_data["tanDip"] = self.analyse.get_pitch_angle(data_cut["trkfit"][at_trk_front])
                processed_data["maxr"] = data_cut["trkfit"]["trksegpars_lh"]["maxr"][at_trk_front]
                # Calculate momentum magnitude
                mom_mag = self.analyse.vector.get_mag(data_cut["trkfit"][at_trk_front]["trksegs"], "mom")
                processed_data["mom_mag"] = mom_mag

                # Broadcast all features to match CRV coincidence shape [event, coincidence]
                # Get target shape from any CRV array
                coinc_shape = ak.ones_like(processed_data["crv_z"])

                # Broadcast event-level parameters: [event] -> [event, coincidence]
                for key in ["event", "subrun"]:
                    processed_data[key] = processed_data[key][:, None] * coinc_shape
                
                # Broadcast track-level parameters: [event, track, segment] -> [event, coincidence]
                # Flatten segments, broadcast to coincidences, then flatten again
                for key in ["t0", "d0", "tanDip", "maxr", "mom_mag"]:
                    arr = processed_data[key]
                    arr = ak.flatten(arr, axis=-1)  # [event, track, segment] -> [event, track]
                    arr = arr[:, None] * coinc_shape  # [event, track] -> [event, track, coincidence]
                    arr = ak.flatten(arr, axis=-1)  # [event, track, coincidence] -> [event, coincidence] 
                    # Override 
                    processed_data[key] = arr
        
            else: 
                self.logger.log(f"Feature set '{self.feature_set}' invalid: {e}", "error")  
                raise
                
            # Store results
            self.logger.log("Creating result", "max")
            result = {
                "id": file_id,
                "cut_flow": cut_flow,
                "hists": hists, 
                "events": processed_data 
            }
            
            self.logger.log("Preprocessing completed", "max")
            gc.collect()
            return result
            
        except Exception as e:
            self.logger.log(f"Error during preprocessing execution: {e}", "error")  
            raise

    def process_file(self, file_name): 
        """Process a single file
        
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
            
            # Run processing on array
            results = self._process_array(
                this_data, 
                file_name
            )

            # Clean up memory
            gc.collect()

            return results 
        
        except Exception as e:
            # Report any errors that occur during processing
            self.logger.log(f"Error processing {file_name}: {e}", "error")
            raise e
    
    def postprocess(self, results):
        """ Combine histograms and arrays 
            Borrow some methods from postprocess
        """
        # Ensure results is always a list
        if not isinstance(results, list):
            results = [results]
            
        postprocessed = {} 
        postprocessed["cut_flow"] = self.postprocessor.combine_cut_flows(results)
        postprocessed["hists"] = self.postprocessor.combine_hists(results)
        postprocessed["events"] = self.postprocessor.combine_arrays(results)
        
        # Flatten (after combination!)
        flattened_events = {}
        lengths = {}
        for field in postprocessed["events"].fields: 
            flattened_events[field] = ak.flatten(postprocessed["events"][field], axis=None)
            lengths[field] = len(flattened_events[field])
        
        if len(set(lengths.values())) > 1:
            self.logger.log(f"Length mismatch detected: {lengths}", "warning")
            # Pad and retry 
            # postprocess(results)
            print(postprocessed["events"].type)

        else:
            self.logger.log(f"All field lengths match: {lengths}", "success")
        
        postprocessed["events"] = ak.Array(flattened_events) # enforce awkward type
        self.logger.log(f"Postprocessing complete", "success")
        
        return postprocessed