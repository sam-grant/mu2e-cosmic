import uproot
import awkward as ak
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import hist
import gc
import sys 
import os
import yaml 
import copy

from pyutils.pyselect import Select
from pyutils.pyvector import Vector
from pyutils.pylogger import Logger
from pyutils.pyprint import Print

# Add relative path to utils
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, "..", "utils"))
from cut_manager import CutManager
from hist_manager import HistManager

class Analyse:
    """Class to handle analysis method and flow
    """
    def __init__(self, on_spill=False, cut_config_path="../../config/common/cuts.yaml", cutset_name="alpha", verbosity=1):
        """Initialise the analysis handler
        
        Args:
            on_spill (bool, opt): Include on-spill cuts
            cut_config_path (str, opt): Path to cut definition YAML file
            cutset (str, opt): The cutset name
            verbosity (int, optional): Level of output detail (0: critical errors only, 1: info, 2: debug, 3: deep debug)
        """
        # Member variables
        self.on_spill = on_spill
        # self.event_subrun = event_subrun # event selection (for debugging)
        self.cut_config_path = cut_config_path
        self.cutset_name = cutset_name
        self.verbosity = verbosity

        # Helper sets cut parameters, self.track_pdg, self.lo_nactive, etc.
        # TODO: put load_cuts in config_manager.py or cut_manager.py, whichever makes more sense.
        self.cut_config = {"description": "FAILED TO LOAD"}  # Default fallback
        self._load_cuts()
        
        # Setup tools
        self.selector = Select(verbosity=self.verbosity)
        self.vector = Vector(verbosity=self.verbosity) 
        self.hist_manager = HistManager(self) # Needs to come after _load_cuts
        self.logger = Logger(
            print_prefix="[Analyse]",
            verbosity=self.verbosity
        )
        
        self.logger.log(
            f"Initialised with:\n"
            f"  on_spill        = {self.on_spill}\n"
            f"  cut_config_path = {self.cut_config_path}\n"
            f"  cutset_name     = {self.cutset_name} ({self.cut_config["description"]})\n"
            f"  verbosity       = {self.verbosity}",
            "info"
        )

    def _load_cuts(self):
        """Load cut configuration from YAML. Supports baseline and derived cutsets."""

        # Resolve path to cut config file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cut_config_path = os.path.join(script_dir, self.cut_config_path or "../../config/common/cuts.yaml")
        
        if not os.path.exists(cut_config_path):
            self.logger.log(f"Cut config file not found: {cut_config_path}", "error")
            sys.exit(1) # Stop process
        else:
            self.cut_config_path = cut_config_path

        # Load YAML
        with open(self.cut_config_path, "r") as f:
            config = yaml.safe_load(f)

        # Check if baseline exists
        if "defaults" not in config:
            self.logger.log("No 'baseline' configuration found in YAML", "error")
            sys.exit(1) # Stop process

        # Start with baseline config
        if self.cutset_name == "defaults":
            final_config = config["defaults"]
        else:
            # Check if cutset exists
            if "cutsets" not in config or self.cutset_name not in config["cutsets"]:
                self.logger.log(f"Cutset '{self.cutset_name}' not found in config", "error")
                sys.exit(1) # Stop process
            
            # Deep copy baseline to avoid modifying original
            final_config = copy.deepcopy(config["defaults"])
            override_config = config["cutsets"][self.cutset_name]

            # Merge overrides
            for section in ["thresholds", "active"]:
                if section in override_config:
                    final_config[section].update(override_config[section])
            
            # Update description if provided
            if "description" in override_config:
                final_config["description"] = override_config["description"]

        # Assign internal variables
        self.thresholds = final_config["thresholds"]
        self.active_cuts = final_config["active"]
        self.cut_config = final_config

        # return True

    # except Exception as e:
    #     self.logger.log(f"Error loading cuts: {e}", "error")
    #     return False

    def get_pitch_angle(self, trkfit):
        """Helper to calculate pitch angle"""
        pvec = self.vector.get_vector(trkfit["trksegs"], "mom")
        pt = np.sqrt(pvec["x"]**2 + pvec["y"]**2) 
        pz = trkfit["trksegs"]["mom"]["fCoordinates"]["fZ"]
        return pz/pt 
            
    def define_cuts(self, data, cut_manager):
        """Define analysis cuts
        
        Args:
            data (ak.Array): Array to cut 
            cut_manager: CutManager instance

        Raises: 
            Exception per cut definition

        * All cuts need to be defined at track level
        * Tracking algorthm fits upstream/downstream muon/electrons and then uses trkqual find the correct fit hypthosis
        * For track-level masks I use ak.all(~at_trk_front | trksegs_condition, axis=-1), which requires that all
        segments at the tracker entrance fulfill this condition. I think this is more robust than ak.any(at_trk_front &
        trksegs_condition, axis=-1), since it implicitly handles reflections. 
        
        """
        ###################################################
        # Tracks intersect tracker entrance
        ###################################################
        try:
            # Track segments level definition
            at_trk_front = self.selector.select_surface(data["trkfit"], surface_name="TT_Front") 
            # at_trk_mid = self.self.selector.select_surface(data["trkfit"], surface_name="TT_Mid")
            # at_trk_back = self.self.selector.select_surface(data["trkfit"], surface_name="TT_Back")
            # in_trk = (at_trk_front | at_trk_mid | at_trk_back)
            
            # Track level definition 
            has_trk_front = ak.any(at_trk_front, axis=-1)
            # Add cut 
            cut_manager.add_cut(
                name="has_trk_front", 
                description="Tracks intersect tracker entrance", 
                mask=has_trk_front,
                active=self.active_cuts["has_trk_front"],
                group="Preselection"
            )
            # Append for debugging
            data["at_trk_front"] = at_trk_front
            data["has_trk_front"] = has_trk_front
        except Exception as e:
            self.logger.log(f"Error defining 'has_trk_front': {e}", "error") 
            raise e

        ###################################################
        # Select electron track fit hypothesis  
        ###################################################
        try:
            # Track level definition
            is_reco_electron = self.selector.is_electron(data["trk"])
            # Add cut 
            cut_manager.add_cut(
                name="is_reco_electron", 
                description="Electron track fits", 
                mask=is_reco_electron,
                active=self.active_cuts["is_reco_electron"],
                group="Preselection"
            )
            # Append mask for debugging
            data["is_reco_electron"] = is_reco_electron
        except Exception as e:
            self.logger.log(f"Error defining 'is_reco_electron': {e}", "error") 
            raise e

        ###################################################
        # Downstream tracks at the tracker entrance 
        ###################################################
        try:
            # Track segments level mask
            is_downstream = self.selector.is_downstream(data["trkfit"]) 
            # Track level mask
            is_downstream = ak.all(~at_trk_front | is_downstream, axis=-1) 
            # Add cut 
            cut_manager.add_cut(
                name="is_downstream",
                description="Downstream tracks (p_z > 0 at tracker entrance)",
                mask=is_downstream,
                active=self.active_cuts["is_downstream"],
                group="Preselection"
            )
            # Append for debugging
            data["is_downstream"] = is_downstream
        except Exception as e:
            self.logger.log(f"Error defining 'is_downstream': {e}", "error") 
            raise e

        ###################################################
        # One electron track fit / event 
        # This deals with split reflected tracks
        # TODO: make this dependent on a deltaT
        ###################################################
        try: 
            # Event-level definition
            one_reco_electron_per_event = ak.sum(is_reco_electron, axis=-1) == 1
            # Broadcast to track level
            one_reco_electron, _ = ak.broadcast_arrays(one_reco_electron_per_event, is_reco_electron) # this returns a tuple
            # Add cut 
            cut_manager.add_cut(
                name="one_reco_electron",
                description="One reco electron / event",
                mask=one_reco_electron,
                active=self.active_cuts["one_reco_electron"],
                group="Preselection"
            )
            # Append for debugging 
            data["one_reco_electron"] = one_reco_electron
            data["one_reco_electron_per_event"] = one_reco_electron_per_event
        except Exception as e:
            self.logger.log(f"Error defining 'one_reco_electron': {e}", "error") 
            raise e

        ###################################################
        # Track "PID" from truth track parents 
        ###################################################
        try:
            # Truth track parent is electron 
            is_electron = data["trkmc"]["trkmcsim"]["pdg"] == self.thresholds["track_pdg"]
            is_trk_parent = data["trkmc"]["trkmcsim"]["nhits"] == ak.max(data["trkmc"]["trkmcsim"]["nhits"], axis=-1)
            is_trk_parent_electron = is_electron & is_trk_parent 
            has_trk_parent_electron = ak.any(is_trk_parent_electron, axis=-1) 
            # Add cut 
            cut_manager.add_cut(
                name="is_truth_electron", 
                description="Track parents are electrons (truth PID)", 
                mask=has_trk_parent_electron,
                active=self.active_cuts["is_truth_electron"] ,
                group="Preselection"
            )
            # Append for debugging
            data["is_truth_electron"] = has_trk_parent_electron
        except Exception as e:
            self.logger.log(f"Error defining 'is_truth_electron': {e}", "error") 
            raise e 

        ###################################################
        # Track fit hypothesis quality
        ###################################################
        try: 
            # Track level mask
            good_trkqual = self.selector.select_trkqual(data["trk"], quality=self.thresholds["lo_trkqual"])
            # Add cut 
            cut_manager.add_cut(
                name="good_trkqual",
                description=f"Track fit quality > {self.thresholds["lo_trkqual"]}",
                mask=good_trkqual,
                active=self.active_cuts["good_trkqual"],
                group="Tracker"
            )
            # Append for debugging
            data["good_trkqual"] = good_trkqual
        except Exception as e:
            self.logger.log(f"Error defining 'good_trkqual': {e}", "error") 
            raise e
        
        ###################################################
        # Time at tracker entrance (t0)
        ###################################################
        try:
            if self.on_spill:
                # Track segments level definition
                within_t0 = ((self.thresholds["lo_t0_ns"] < data["trkfit"]["trksegs"]["time"]) & 
                             (data["trkfit"]["trksegs"]["time"] < self.thresholds["hi_t0_ns"]))
                # Track level definition
                within_t0 = ak.all(~at_trk_front | within_t0, axis=-1)
                # Add cut 
                cut_manager.add_cut( 
                    name="within_t0",
                    description=f"t0 at tracker entrance ({self.thresholds["lo_t0_ns"]}"\
                                f" < t_0 < {self.thresholds["hi_t0_ns"]} ns)",
                    mask=within_t0,
                    active=self.active_cuts["within_t0"],
                    group="Tracker"
                )
                # Append for debugging
                data["within_t0"] = within_t0
        except Exception as e:
            self.logger.log(f"Error defining 'within_t0': {e}", "error") 
            raise e

        ###################################################
        # Minimum active hits
        ###################################################
        try:
            # Track level definition 
            has_hits = self.selector.has_n_hits(data["trk"], n_hits=self.thresholds["lo_nactive"])
            # Add cut
            cut_manager.add_cut(
                name="has_hits",
                description=f">{self.thresholds["lo_nactive"]-1} active tracker hits",
                mask=has_hits,
                active=self.active_cuts["has_hits"],
                group="Tracker"
            )
            # Append for debugging
            data["has_hits"] = has_hits
        except Exception as e:
            self.logger.log(f"Error defining 'has_hits': {e}", "error") 
            raise e

        ###################################################
        # t0 uncertainty distance of closest approach
        ###################################################
        try:
            # Track segments level definition
            within_t0err = (data["trkfit"]["trksegpars_lh"]["t0err"] < self.thresholds["hi_t0err"])

            # Track level definition
            within_t0err = ak.all(~at_trk_front | within_t0err, axis=-1) 

            # Add cut 
            cut_manager.add_cut(
                name="within_t0err",
                description=f"Track fit t0 uncertainty (t0err < {self.thresholds["hi_t0err"]} ns)",
                mask=within_t0err,
                active=self.active_cuts["within_t0err"],
                group="Tracker"
            )
            # Append for debugging
            data["within_t0err"] = within_t0err
        except Exception as e:
            self.logger.log(f"Error defining 'within_t0err': {e}", "error") 
            raise e
            
        ###################################################
        # Extrapolated distance of closest approach
        ###################################################
        try:
            # Track segments level definition
            within_d0 = (data["trkfit"]["trksegpars_lh"]["d0"] < self.thresholds["hi_d0_mm"])

            # Track level definition
            within_d0 = ak.all(~at_trk_front | within_d0, axis=-1) 

            # Add cut 
            cut_manager.add_cut(
                name="within_d0",
                description=f"Distance of closest approach (d_0 < {self.thresholds["hi_d0_mm"]} mm)",
                mask=within_d0,
                active=self.active_cuts["within_d0"],
                group="Tracker"
            )
            # Append for debugging
            data["within_d0"] = within_d0
        except Exception as e:
            self.logger.log(f"Error defining 'within_d0': {e}", "error") 
            raise e

        ###################################################
        # Pitch angle
        ###################################################
        try:
            # # Define pitch angle
            # pvec = self.vector.get_vector(data["trkfit"]["trksegs"], "mom")
            # pt = np.sqrt(pvec["x"]**2 + pvec["y"]**2) 
            # pz = data["trkfit"]["trksegs"]["mom"]["fCoordinates"]["fZ"]
            # pitch_angle = pz/pt 
            # data["pitch_angle"] = pitch_angle
            pitch_angle = self.get_pitch_angle(data["trkfit"])
            # Old track segments definition (tanDip has a bug in EventNtuple)
            # within_pitch_angle = ((self.thresholds["lo_tanDip"] < data["trkfit"]["trksegpars_lh"]["tanDip"]) & 
            #                       (data["trkfit"]["trksegpars_lh"]["tanDip"] < self.thresholds["hi_tanDip"]))
            # Correct definition
            within_pitch_angle = ((self.thresholds["lo_tanDip"] < pitch_angle) & (pitch_angle < self.thresholds["hi_tanDip"]))
            # Track level definition
            within_pitch_angle = ak.all(~at_trk_front | within_pitch_angle, axis=-1)
            # Add cut 
            cut_manager.add_cut(
                name="within_pitch_angle",
                description=f"Extrapolated pitch angle ({self.thresholds["lo_tanDip"]}"\
                            f" < tan(theta_Dip) < {self.thresholds["hi_tanDip"]})",
                mask=within_pitch_angle,
                active=self.active_cuts["within_pitch_angle"],
                group="Tracker"
            )
            # Append 
            data["within_pitch_angle"] = within_pitch_angle
        except Exception as e:
            self.logger.log(f"Error defining 'within_pitch_angle': {e}", "error") 
            raise e

        ###################################################
        # Loop helix maximimum radius lower bound 
        ###################################################
        try:
            # Track segmentslevel definition 
            within_lhr_max_lo = (self.thresholds["lo_maxr_mm"] < data["trkfit"]["trksegpars_lh"]["maxr"]) 
            # Track level definition 
            within_lhr_max_lo = ak.all(~at_trk_front | within_lhr_max_lo, axis=-1)
            # Add cut 
            cut_manager.add_cut(
                name="within_lhr_max_lo",
                description=f"Loop helix maximum radius (R_max > {self.thresholds["lo_maxr_mm"]} mm)",
                mask=within_lhr_max_lo,
                active=self.active_cuts["within_lhr_max_lo"],
                group="Tracker"
            )
            # Append for debugging
            data["within_lhr_max_lo"] = within_lhr_max_lo
        except Exception as e:
            self.logger.log(f"Error defining 'within_lhr_max_lo': {e}", "error") 
            raise e

        ###################################################
        # Loop helix maximimum radius upper bound (main)
        ###################################################
        try:
            # Track segmentslevel definition 
            within_lhr_max_hi = (data["trkfit"]["trksegpars_lh"]["maxr"] < self.thresholds["hi_maxr_mm"])
            # Track level definition 
            within_lhr_max_hi = ak.all(~at_trk_front | within_lhr_max_hi, axis=-1)
            # Add cut 
            cut_manager.add_cut(
                name="within_lhr_max_hi",
                description=f"Loop helix maximum radius (R_max < {self.thresholds["hi_maxr_mm"]} mm)",
                mask=within_lhr_max_hi,
                active=self.active_cuts["within_lhr_max_hi"],
                group="Tracker"
            )
            # Append for debugging
            data["within_lhr_max_hi"] = within_lhr_max_hi
        except Exception as e:
            self.logger.log(f"Error defining 'within_lhr_max_hi': {e}", "error") 
            raise e

        ###################################################
        # CRV veto: |dt| < 150 ns 
        # Check if EACH track is within 150 ns of ANY coincidence 
        ###################################################
        try:
            # Get track and coincidence times
            trk_times = data["trkfit"]["trksegs"]["time"][at_trk_front]  # events × tracks × segments
            coinc_times = data["crv"]["crvcoincs.time"]                  # events × coincidences
            # Broadcast CRV times to match track structure, so that we can compare element-wise
            # Could use ak.broadcast?
            coinc_broadcast = coinc_times[:, None, None, :]  # Add dimensions for tracks and segments
            trk_broadcast = trk_times[:, :, :, None]         # Add dimension for coincidences
            # coinc_broadcast shape is [E, 1, 1, C] 
            # trk_broadcast shape is [E, T, S, 1]
            # Calculate time differences
            dt = abs(trk_broadcast - coinc_broadcast)
            # Check if within threshold
            within_threshold = dt < self.thresholds["veto_dt_ns"]
            # Reduce one axis at a time 
            # First reduce over coincidences (axis=3)
            any_coinc = ak.any(within_threshold, axis=3)
            # Then reduce over trks (axis=2) 
            veto = ak.any(any_coinc, axis=2)
            # Add cut 
            cut_manager.add_cut(
                name="unvetoed",
                description=f"No veto: |dt| >= {self.thresholds["veto_dt_ns"]} ns",
                mask=~veto,
                active=self.active_cuts["unvetoed"],
                group="CRV"
            )
            # Append for debugging
            data["unvetoed"] = ~veto
        except Exception as e:
            self.logger.log(f"Error defining 'unvetoed': {e}", "error") 
            raise e

        ###################################################
        # Extended window 
        ###################################################
        try:
            # Get momentum magnitude 
            mom = self.vector.get_mag(data["trkfit"]["trksegs"], "mom") 
            # Track segments level definition 
            within_ext_win = (mom > self.thresholds["lo_ext_win_mevc"]) & (mom < self.thresholds["hi_ext_win_mevc"])
            # Track level definition
            within_ext_win = ak.all(~at_trk_front | within_ext_win, axis=-1)
            # Add cut 
            cut_manager.add_cut(
                name="within_ext_win",
                description=f"Extended window ({self.thresholds["lo_ext_win_mevc"]}"\
                            f" < p < {self.thresholds["hi_ext_win_mevc"]} MeV/c)",
                mask=within_ext_win,
                active=self.active_cuts["within_ext_win"],
                group="Momentum"
            )
            # Append 
            data["within_ext_win"] = within_ext_win
        except Exception as e:
            self.logger.log(f"Error defining 'within_ext_win': {e}", "error") 
            raise e

        ###################################################
        # Signal window 
        ###################################################
        try:
            # Get momentum magnitude 
            # FIXME: should probably only do this once...
            mom = self.vector.get_mag(data["trkfit"]["trksegs"], "mom") 
            # Track segments level definition 
            within_sig_win = (mom > self.thresholds["lo_sig_win_mevc"]) & (mom < self.thresholds["hi_sig_win_mevc"])
            # Track level definition
            within_sig_win = ak.all(~at_trk_front | within_sig_win, axis=-1)
            # Add cut 
            cut_manager.add_cut(
                name="within_sig_win",
                description=f"Signal window ({self.thresholds["lo_sig_win_mevc"]}"\
                            f" < p < {self.thresholds["hi_sig_win_mevc"]} MeV/c)",
                mask=within_sig_win,
                active=self.active_cuts["within_sig_win"],
                group="Momentum"
            )
            # Append 
            data["within_sig_win"] = within_sig_win
        except Exception as e:
            self.logger.log(f"Error defining 'within_sig_win': {e}", "error") 
            raise e

        # Done 
        self.logger.log("All cuts defined", "success")
        
    def apply_cuts(self, data, cut_manager, group=None, active_only=True):

        ## data_cut needs to be an awkward array 
    
        """Apply all trk-level mask to the data
        
        Args:
            data: Data to apply cuts to
            mask: Mask to apply 
            
        Returns:
            ak.Array: Data after cuts applied
        """
        self.logger.log("Applying cuts to data", "info")
        
        try:
            # Copy the array 
            # This is memory intensive but the easiest solution for what I'm trying to do
            data_cut = ak.copy(data) 
            
            # Combine cuts
            self.logger.log(f"Combining cuts", "info") 

            # Track-level mask
            trk_mask = cut_manager.combine_cuts(active_only=active_only)
            
            # Select tracks
            self.logger.log("Selecting tracks", "max")
            data_cut["trk"] = data_cut["trk"][trk_mask]
            data_cut["trkfit"] = data_cut["trkfit"][trk_mask]
            data_cut["trkmc"] = data_cut["trkmc"][trk_mask]
            
            # Then clean up events with no tracks after cuts
            self.logger.log(f"Cleaning up events with no tracks after cuts", "max") 
            data_cut = data_cut[ak.any(trk_mask, axis=-1)] 
            
            self.logger.log(f"Cuts applied successfully", "success")
            
            return data_cut
            
        except Exception as e:
            self.logger.log(f"Error applying cuts: {e}", "error") 
            raise

    def create_histograms(self, datasets):
        """Create histograms from the data before and after applying cuts
        
        Args:
            datasets (dict): Dictionary of arrays 
            
        Returns:
            dict: Dictionary of histograms
        """
        self.logger.log("Creating histograms", "info")
        return self.hist_manager.create_histograms(datasets)
        
    def execute(self, data, file_id, cuts_to_toggle=None, groups_to_toggle=None):
        """Perform complete analysis on an array
        
        Args:
            data: The data to analyse
            file_id: Identifier for the file
            cuts_to_toggle: Dict of cut name with active state  
            groups_to_toggle: Dict of cut group names with active state
            
        Returns:
            dict: Complete analysis results
        """
        
        self.logger.log(f"Beginning analysis execution for file: {file_id}", "info") 
        
        try:
            # Initialise cut manager and apply any prefiltering
            cut_manager = CutManager(verbosity=self.verbosity)
            
            # Define cuts
            self.logger.log(f"Defining cuts", "max")
            self.define_cuts(data, cut_manager)

            # Toggle cuts
            if cuts_to_toggle: 
                self.logger.log(f"Toggling cuts", "max")
                cut_manager.toggle_cut(cuts_to_toggle) 
            if groups_to_toggle: 
                self.logger.log(f"Toggling cut groups", "max")
                cut_manager.toggle_group(groups_to_toggle) 

            # Save cut state
            cut_manager.save_state("original")
            # Retrieve initial veto status (for cut plots)
            veto = cut_manager.cuts["unvetoed"]["active"]
            
            # Create main cut flow
            self.logger.log("Creating cut flow", "max")
            cut_flow = cut_manager.create_cut_flow(data)

            ##########################################
            # Apply cuts
            ##########################################  
            
            # # Apply preselection cuts
            # self.logger.log("Applying preselection cuts", "max")
            # # Turn all cut groups except preselection
            # cut_manager.toggle_group(
            #     {
            #         "Tracker": False,
            #         "CRV": False,
            #         "Momentum": False
                    
            #     }
            # )           
            # data["preselect"] = cut_manager.combine_cuts(active_only=True)
            # data_preselect = self.apply_cuts(data, cut_manager)
            
            # # Apply CE-like cuts (without veto & momentum windows)
            # self.logger.log("Applying CE-like cuts", "max")
            # # Turn tracker cut group back on
            # # cut_manager.toggle_group({"Tracker": True})
            # # # Turn off veto and momentum windows
            # cut_manager.toggle_group(
            #     {
            #         "Tracker": True,
            #         "CRV": False,
            #         "Momentum": False
                    
            #     }
            # ) 
            # # cut_manager.toggle_cut(
            # #     {
            # #         "unvetoed": False,
            # #         "within_ext_win": False,
            # #         "within_sig_win": False
                    
            # #     }
            # # )
            # data["CE_like"] = cut_manager.combine_cuts(active_only=True)
            # data_CE = self.apply_cuts(data, cut_manager)

            # Apply preselection cuts
            self.logger.log("Applying preselection cuts", "max")
            cut_manager.toggle_group({
                "Tracker": False,
                "CRV": False, 
                "Momentum": False
            })  
            # cut_manager.save_state("preselect")
            data["preselect"] = cut_manager.combine_cuts(active_only=True)
            data_preselect = self.apply_cuts(data, cut_manager)

            # Apply CE-like cuts (without veto & momentum windows)
            self.logger.log("Applying CE-like cuts", "max")
            # Restore state
            # Toggling Tracker to True will activate tracker cuts from different
            # cutsets that we may not want
            cut_manager.restore_state("original")
            cut_manager.toggle_group({
                "CRV": False,       # Keep CRV off (this turns off "unvetoed")
                "Momentum": False   # Keep Momentum off (this turns off both momentum windows)
            })
            # cut_manager.save_state("CE_like")
            # Print active cuts
            cut_manager.list_groups()
            # Save CE-like selection
            data["CE_like"] = cut_manager.combine_cuts(active_only=True)
            data_CE = self.apply_cuts(data, cut_manager)

            # Apply CRV cuts
            data_CE_unvetoed = {} # Default with length zero
            if veto:  
                self.logger.log("Applying veto cuts", "max")  
                # Turn veto on
                cut_manager.toggle_group({"CRV": True}) # same thing
                # cut_manager.toggle_cut({"unvetoed": True}) # same thing
                data["unvetoed_CE_like"] = cut_manager.combine_cuts(active_only=True)
                data_CE_unvetoed = self.apply_cuts(data, cut_manager) 

            print(f"!!!!! N CE-like: {len(data_CE)}")
            
            # Create histograms and compile results
            self.logger.log("Creating histograms", "max")
            histograms = self.create_histograms(
                datasets = { 
                    "All": data,
                    "Preselect": data_preselect,
                    "CE-like": data_CE,
                    "Unvetoed": data_CE_unvetoed
                }
            )

            # Store results
            self.logger.log("Creating result", "max")
            result = {
                "id": file_id,
                "cut_flow": cut_flow,
                "hists": histograms,
                "events": data_CE_unvetoed if veto else data_CE
            }
            
            self.logger.log("Analysis completed", "success")
            gc.collect()
            return result
            
        except Exception as e:
            self.logger.log(f"Error during analysis execution: {e}", "error")  
            raise