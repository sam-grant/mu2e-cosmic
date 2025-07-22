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

from pyutils.pyselect import Select
from pyutils.pyvector import Vector
from pyutils.pylogger import Logger
from pyutils.pyprint import Print
from cut_manager import CutManager

class Analyse:
    """Class to handle analysis functions
    """
    def __init__(self, on_spill=False, event_subrun=None, cut_params_path="cuts.yaml", verbosity=1):
        """Initialise the analysis handler
        
        Args:
            on_spill (bool, optional): Include on-spill cuts
            event_subrun (tuple of ints, optional): Select a specific event and subrun
            verbosity (int, optional): Level of output detail (0: critical errors only, 1: info, 2: debug, 3: deep debug)
        """

        # Verbosity 
        self.verbosity = verbosity
        
        # Start logger
        self.logger = Logger(
            print_prefix="[Analyse]",
            verbosity=self.verbosity
        )
        
        # Initialise tools
        self.selector = Select(verbosity=self.verbosity)
        self.vector = Vector(verbosity=self.verbosity) 
        
        # Analysis configuration
        self.on_spill = on_spill  # Default to off-spill 
        self.event_subrun = event_subrun # event selection (for debugging)

        # Get cut params
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cut_params_path = os.path.join(script_dir, cut_params_path) 
        with open(cut_params_path, "r") as file:
            params = yaml.safe_load(file)
                
            # Load cut parameters
            cuts = params["cuts"]
            self.track_pdg = cuts["track_pdg"]
            self.lower_nactive = cuts["lower_nactive"]
            self.lower_trkqual = cuts["lower_trkqual"]
            self.lower_t0_ns = cuts["lower_t0_ns"]
            self.upper_t0_ns = cuts["upper_t0_ns"]
            self.lower_maxr_mm = cuts["lower_maxr_mm"]
            self.upper_maxr_mm = cuts["upper_maxr_mm"]
            self.upper_d0_mm = cuts["upper_d0_mm"]
            self.lower_tanDip = cuts["lower_tanDip"]
            self.upper_tanDip = cuts["upper_tanDip"]
            self.veto_dt_ns = cuts["veto_dt_ns"]

        self.logger.log(f"Initialised with on_spill={self.on_spill}", "info")
    
    def define_cuts(self, data, cut_manager):
        """Define analysis cuts

        Note that all cuts here need to be defined at trk level. 

        Also note that the tracking algorthm produces cut for upstream/downstream muon/electrons and then uses trkqual to guess the right one
        trkqual needs to be good before making a selection 
        this is particulary important for the pileup cut, since it needs to be selected from tracks which are above 90% or whatever 
        
        Args:
            data (ak.Array): data to apply cuts to
            cut_manager: The CutManager instance to use
            on_spill (bool, optional): Whether to apply on-spill specific cuts

        Note 1: for track-level masks I typically use ak.all(~at_trk_front | trksegs_condition, axis=-1), which requires that all
        segments at the tracker entrance fulfill this condition. This is more robust than ak.any(at_trk_front & trksegs_condition, axis=-1), 
        since it implicitly handles reflections. 

        Note 2: the cut order should not matter, but I have 
        """

        ###################################################
        # 1. Select electron track fit hypothesis  
        ###################################################
        try:
            # Track level definition
            is_reco_electron = self.selector.is_electron(data["trk"])
            # Add cut 
            cut_manager.add_cut(
                name="is_reco_electron", 
                description="Electron track fits", 
                mask=is_reco_electron 
            )
            # Append mask for debugging
            data["is_reco_electron"] = is_reco_electron
        except Exception as e:
            self.logger.log(f"Error defining 'is_reco_electron': {e}", "error") 
            return None  

        ###################################################
        # 2. Tracks intersect tracker entrance
        ###################################################
        try:
            # Track segments level definition
            at_trk_front = self.selector.select_surface(data["trkfit"], sid=0) 
            # at_trk_mid = self.self.selector.select_surface(data["trkfit"], sid=1)
            # at_trk_back = self.self.selector.select_surface(data["trkfit"], sid=2)
            # in_trk = (at_trk_front | at_trk_mid | at_trk_back)
            # Track level definition 
            has_trk_front = ak.any(at_trk_front, axis=-1)
            # Add cut 
            cut_manager.add_cut(
                name="has_trk_front", 
                description="Tracks intersect tracker entrance", 
                mask=has_trk_front 
            )
            # Append for debugging
            data["at_trk_front"] = at_trk_front
            data["has_trk_front"] = has_trk_front
        except Exception as e:
            self.logger.log(f"Error defining 'has_trk_front': {e}", "error") 
            return None  
            
        ###################################################
        # 3. Track fit hypothesis quality
        ###################################################
        try: 
            # Track level mask
            good_trkqual = self.selector.select_trkqual(data["trk"], quality=self.lower_trkqual)
            # Add cut 
            cut_manager.add_cut(
                name="good_trkqual",
                description=f"Track fit quality > {self.lower_trkqual})",
                mask=good_trkqual 
            )
            # Append for debugging
            data["good_trkqual"] = good_trkqual
        except Exception as e:
            self.logger.log(f"Error defining 'good_trkqual': {e}", "error") 
            return None  
            

        ###################################################
        # 4. Time at tracker entrance (t0)
        ###################################################
        try:
            if self.on_spill:
                # Track segments level definition
                within_t0 = ((self.lower_t0_ns < data["trkfit"]["trksegs"]["time"]) & 
                             (data["trkfit"]["trksegs"]["time"] < self.upper_t0_ns))
                # Track level definition
                within_t0 = ak.all(~at_trk_front | within_t0, axis=-1)
                # Add cut 
                cut_manager.add_cut( 
                    name="within_t0",
                    description=f"t0 at tracker entrance ({self.lower_t0_ns} < t_0 < {self.upper_t0_ns} ns)",
                    mask=within_t0 
                )
                # Append for debugging
                data["within_t0"] = within_t0
        except Exception as e:
            self.logger.log(f"Error defining 'within_t0': {e}", "error") 
            return None  

        ###################################################
        # 5. Downstream tracks at the tracker entrance 
        ###################################################
        try:
            # Track segments level mask
            is_downstream = self.selector.is_downstream(data["trkfit"]) 
            # Track level mask
            is_downstream = ak.all(~at_trk_front | is_downstream, axis=-1) 
            # Add cut 
            cut_manager.add_cut(
                name="downstream",
                description="Downstream tracks (p_z > 0 at tracker entrance)",
                mask=is_downstream 
            )
            # Append for debugging
            data["is_downstream"] = is_downstream
        except Exception as e:
            self.logger.log(f"Error defining 'is_downstream': {e}", "error") 
            return None  

        ###################################################
        # 6. Minimum active hits
        ###################################################
        try:
            # Track level definition 
            has_hits = self.selector.has_n_hits(data["trk"], n_hits=self.lower_nactive)
            # Add cut
            cut_manager.add_cut(
                name="has_hits",
                description=f">= {self.lower_nactive} active tracker hits",
                mask=has_hits 
            )
            # Append for debugging
            data["has_hits"] = has_hits
        except Exception as e:
            self.logger.log(f"Error defining 'has_hits': {e}", "error") 
            return None  
            
        ###################################################
        # 7. Extrapolated distance of closest approach
        ###################################################
        try:
            # Track segments level definition
            within_d0 = (data["trkfit"]["trksegpars_lh"]["d0"] < self.upper_d0_mm)

            # Track level definition
            within_d0 = ak.all(~at_trk_front | within_d0, axis=-1) 

            # Add cut 
            cut_manager.add_cut(
                name="within_d0",
                description=f"Distance of closest approach (d_0 < {self.upper_d0_mm} mm)",
                mask=within_d0 
                
            )
            # Append for debugging
            data["within_d0"] = within_d0
        except Exception as e:
            self.logger.log(f"Error defining 'within_d0': {e}", "error") 
            return None  

        ###################################################
        # 8. Pitch angle
        ###################################################
        try:
            # Define pitch angle
            pvec = self.vector.get_vector(data["trkfit"]["trksegs"], "mom")
            pt = np.sqrt(pvec["x"]**2 + pvec["y"]**2) 
            pz = data["trkfit"]["trksegs"]["mom"]["fCoordinates"]["fZ"]
            pitch_angle = pz/pt 
            # Old track segments definition (tanDip has a bug in EventNtuple)
            within_pitch_angle = ((self.lower_tanDip < data["trkfit"]["trksegpars_lh"]["tanDip"]) & 
                                  (data["trkfit"]["trksegpars_lh"]["tanDip"] < self.upper_tanDip))
            # Correct definition
            # within_pitch_angle = ((self.lower_tanDip < pitch_angle) & (pitch_angle < self.upper_tanDip))
            # Track level definition
            within_pitch_angle = ak.all(~at_trk_front | within_pitch_angle, axis=-1)
            # Add cut 
            cut_manager.add_cut(
                name="within_pitch_angle",
                description=f"Extrapolated pitch angle ({self.lower_tanDip} < tan(theta_Dip) < {self.upper_tanDip})",
                mask=within_pitch_angle
            )
            # Append 
            data["within_pitch_angle"] = within_pitch_angle
        except Exception as e:
            self.logger.log(f"Error defining 'within_pitch_angle': {e}", "error") 
            return None  

        ###################################################
        # 9. Loop helix maximimum radius 
        ###################################################
        try:
            # Track segmentslevel definition 
            within_lhr_max = ((self.lower_maxr_mm < data["trkfit"]["trksegpars_lh"]["maxr"]) & 
                              (data["trkfit"]["trksegpars_lh"]["maxr"] < self.upper_maxr_mm))
            # Track level definition 
            within_lhr_max = ak.all(~at_trk_front | within_lhr_max, axis=-1)
            # Add cut 
            cut_manager.add_cut(
                name="within_lhr_max",
                description=f"Loop helix maximum radius ({self.lower_maxr_mm} < R_max < {self.upper_maxr_mm} mm)",
                mask=within_lhr_max
            )
            # Append for debugging
            data["within_lhr_max"] = within_lhr_max
        except Exception as e:
            self.logger.log(f"Error defining 'within_lhr_max': {e}", "error") 
            return None  
            
        ###################################################
        # 10. Track "PID" from truth track parents 
        ###################################################
        try:
            # Truth track parent is electron 
            is_electron = data["trkmc"]["trkmcsim"]["pdg"] == self.track_pdg
            is_trk_parent = data["trkmc"]["trkmcsim"]["nhits"] == ak.max(data["trkmc"]["trkmcsim"]["nhits"], axis=-1)
            is_trk_parent_electron = is_electron & is_trk_parent 
            has_trk_parent_electron = ak.any(is_trk_parent_electron, axis=-1) 
            # Add cut 
            cut_manager.add_cut(
                name="is_truth_electron", 
                description="Track parents are electrons (truth PID)", 
                mask=has_trk_parent_electron 
            )
            # Append for debugging
            data["is_truth_electron"] = has_trk_parent_electron
        except Exception as e:
            self.logger.log(f"Error defining 'is_truth_electron': {e}", "error") 
            return None  

        ###################################################
        # 11. One electron track fit / event 
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
                mask=one_reco_electron 
            )
            # Append for debugging 
            data["one_reco_electron"] = one_reco_electron
            data["one_reco_electron_per_event"] = one_reco_electron_per_event
        except Exception as e:
            self.logger.log(f"Error defining 'one_reco_electron': {e}", "error") 
            return None  

        ###################################################
        # 12. CRV veto: |dt| < 150 ns 
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
            within_threshold = dt < self.veto_dt_ns
            # Reduce one axis at a time 
            # First reduce over coincidences (axis=3)
            any_coinc = ak.any(within_threshold, axis=3)
            # Then reduce over trks (axis=2) 
            veto = ak.any(any_coinc, axis=2)
            # Add cut 
            cut_manager.add_cut(
                name="unvetoed",
                description=f"No veto: |dt| >= {self.veto_dt_ns} ns",
                mask=~veto
            )
            # Append for debugging
            data["unvetoed"] = ~veto
        except Exception as e:
            self.logger.log(f"Error defining 'unvetoed': {e}", "error") 
            return None  

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

            # OPTIONAL: select track segments in tracker 
            # This is handled by 
            # This makes it easier to study background events in the end
            # This seems to make a massive difference 
            # data_cut["trkfit"] = data_cut["trkfit"][data_cut["at_trk_front"]]
            
            # Then clean up events with no tracks after cuts
            self.logger.log(f"Cleaning up events with no tracks after cuts", "max") 
            data_cut = data_cut[ak.any(trk_mask, axis=-1)] 
            
            self.logger.log(f"Cuts applied successfully", "success")
            
            return data_cut
            
        except Exception as e:
            self.logger.log(f"Error applying cuts: {e}", "error") 
            return None
            
    def create_histograms(self, data, data_CE, data_CE_unvetoed):
        
        """Create histograms from the data before and after applying cuts
        
        Args:
            data: Data before cuts
            data_cut: Data after cuts
            
        Returns:
            dict: Dictionary of histograms
        """
        self.logger.log("Creating histograms", "info")
        
        # Tools 
        selector = self.selector 
        vector = self.vector

        hist_labels = ["All", "CE-like", "Unvetoed CE-like"]

        try: 

            #### Create histogram objects:
            # Full momentum range histogram
            h1_mom_full_range = hist.Hist(
                hist.axis.Regular(30, 0, 300, name="momentum", label="Momentum [MeV/c]"),
                hist.axis.StrCategory(hist_labels, name="selection", label="Selection")
            )
            # Signal region histogram (fine binning)
            h1_mom_signal_region = hist.Hist(
                hist.axis.Regular(13, 103.6, 104.9, name="momentum", label="Momentum [MeV/c]"),
                hist.axis.StrCategory(hist_labels, name="selection", label="Selection")
            )

            # Z-position histograms
            h1_crv_z = hist.Hist(
                hist.axis.Regular(100, -15e3, 10e3, name="crv_z", label="CRV z-position [mm]"),
                hist.axis.StrCategory(hist_labels, name="selection", label="Selection")
            )


            # Process and fill histograms in batches
            def _fill_hist(data, label): 
                """ Nested helper function to fill hists """
                
                # Tracks must be electron candidates at the tracker entrance
                is_electron = selector.is_electron(data["trk"])  
                data["trkfit"] = data["trkfit"][is_electron]
                at_trk_front = selector.select_surface(data["trkfit"], sid=0)              
                mom = vector.get_mag(data["trkfit"]["trksegs"][at_trk_front], "mom")
                crv_z = ak.flatten(data["crv"]["crvcoincs.pos.fCoordinates.fZ"], axis=None)
                
                # Flatten 
                if mom is None:
                    mom = ak.Array([])
                else:
                    mom = ak.flatten(mom, axis=None)
                    
                # Fill h1ogram for wide range
                h1_mom_full_range.fill(momentum=mom, selection=np.full(len(mom), label))

                # Fill signal region
                mom_sig = mom[(mom >= 103.6) & (mom <= 104.9)]
                h1_mom_signal_region.fill(momentum=mom_sig, selection=np.full(len(mom_sig), label))

                # Fill position
                h1_crv_z.fill(crv_z=crv_z, selection=np.full(len(crv_z), label))
                
                # Clean up 
                del mom, mom_sig, crv_z
                import gc
                gc.collect()
                    
            # Fill histograms
            _fill_hist(data, "All") 
            _fill_hist(data_CE, "CE-like")
            if data_CE_unvetoed:
                _fill_hist(data_CE_unvetoed, "Unvetoed CE-like")
    
            self.logger.log("Histograms filled", "success")

            # Create a copy in results and explicitly delete large arrays after use
            result = {
                "Wide range": h1_mom_full_range.copy(), 
                "Signal region": h1_mom_signal_region.copy(),
                "CRV z-position": h1_crv_z.copy()
            }

            return result 

        except Exception as e:
            # Handle any errors that occur during processing
            self.logger.log(f"Error filling histograms: {e}", "error")
            return None

    def execute(self, data, file_id, cuts_to_toggle=None):
        """Perform complete analysis on an array
        
        Args:
            data: The data to analyse
            file_id: Identifier for the file
            cuts_to_toggle: Dict of cut name with active state  
            
        Returns:
            dict: Complete analysis results
        """
        self.logger.log(f"Beginning analysis execution for file: {file_id}", "info") 
        
        try:
            # Initialise cut manager and apply any prefiltering
            cut_manager = CutManager(verbosity=self.verbosity)
            
            if self.event_subrun is not None: 
                mask = ((data["evt"]["event"] == self.event_subrun[0]) & 
                       (data["evt"]["subrun"] == self.event_subrun[1]))
                data = data[mask]
            
            # Define cuts
            self.logger.log("Defining cuts", "max")
            self.define_cuts(data, cut_manager)

            # Toggle cuts
            if cuts_to_toggle: 
                cut_manager.toggle_cut(cuts_to_toggle) 

            # Retrieve initial veto status (for cut plots)
            veto = cut_manager.cuts["unvetoed"]["active"]
            
            # Calculate cut statistics
            self.logger.log("Getting cut stats", "max")
            cut_stats = cut_manager.calculate_cut_stats(data, progressive=True, active_only=True)
            
            # Apply CE-like cuts (without veto)
            self.logger.log("Applying cuts", "max")
            cut_manager.toggle_cut({"unvetoed": False})
            data["CE_like"] = cut_manager.combine_cuts(active_only=True)
            data_CE = self.apply_cuts(data, cut_manager)
            
            # Apply vetoed cuts if needed
            data_CE_unvetoed = None
            if veto:  
                self.logger.log("Applying veto cuts", "max")  
                cut_manager.toggle_cut({"unvetoed": True})
                data["unvetoed_CE_like"] = cut_manager.combine_cuts(active_only=True)
                data_CE_unvetoed = self.apply_cuts(data, cut_manager)
            
            # Create histograms and compile results
            self.logger.log("Creating histograms", "max")
            histograms = self.create_histograms(data, data_CE, data_CE_unvetoed)

            # Store results
            result = {
                "file_id": file_id,
                "cut_stats": cut_stats,
                "filtered_data": data_CE_unvetoed if veto else data_CE, 
                "histograms": histograms
            }
            
            self.logger.log("Analysis completed", "success")
            gc.collect()
            return result
            
        except Exception as e:
            self.logger.log(f"Error during analysis execution: {e}", "error")  
            return None
            
    # def execute(self, data, file_id, cuts_to_toggle=None):
    #     """Perform complete analysis on an array
        
    #     Args:
    #         data: The data to analyse
    #         file_id: Identifier for the file
    #         cuts_to_toggle: Dict of cut name with active state  
            
    #     Returns:
    #         dict: Complete analysis results
    #     """

    #     self.logger.log(f"Beginning analysis execution for file: {file_id}", "info") 
        
    #     try:

    #         # Create a unique cut manager for this file
    #         cut_manager = CutManager(verbosity=self.verbosity)

    #         # Optional prefiltering 
    #         if self.event_subrun is not None: 
    #             mask = ((data["evt"]["event"] == self.event_subrun[0]) & (data["evt"]["subrun"] == self.event_subrun[1]))
    #             data = data[mask]

    #         # Define cuts
    #         self.logger.log("Defining cuts", "max")
    #         self.define_cuts(data, cut_manager)

    #         # Set activate cuts
    #         if cuts_to_toggle: 
    #             cut_manager.toggle_cut(cuts_to_toggle) 

    #         # Get initial veto status
    #         veto = cut_manager.cuts["unvetoed"]["active"]
            
    #         # Calculate cut stats
    #         self.logger.log("Getting cut stats", "max")
    #         cut_stats = cut_manager.calculate_cut_stats(data, progressive=True, active_only=True)
            
    #         # Apply CE-like cuts
    #         self.logger.log("Applying cuts", "max")
    #         # Turn off veto 
    #         cut_manager.toggle_cut({"unvetoed" : False})
    #         # Mark CE-like tracks (useful for debugging 
    #         data["CE_like"] = cut_manager.combine_cuts(active_only=True)
    #         # Apply cuts
    #         data_CE = self.apply_cuts(data, cut_manager) # Just CE-like tracks 
            
    #         # Only create vetoed dataset if needed
    #         data_CE_unvetoed = None
    #         if veto:  
    #             self.logger.log("Applying veto cuts", "max")  
    #             # Turn on veto 
    #             cut_manager.toggle_cut({"unvetoed" : True})
    #             # Mark CE-like tracks (useful for debugging 
    #             data["unvetoed_CE_like"] = cut_manager.combine_cuts(active_only=True)
    #             # Apply cuts
    #             data_CE_unvetoed = self.apply_cuts(data, cut_manager)
            
    #         # Create histograms
    #         self.logger.log("Creating histograms", "max")
    #         histograms = self.create_histograms(data, data_CE, data_CE_unvetoed)

    #         # Store results
    #         result = {
    #             "file_id": file_id,
    #             "cut_stats": cut_stats,
    #             "filtered_data": data_CE_unvetoed if veto else data_CE, 
    #             "histograms": histograms
    #         }
            
    #         # Print sucesss
    #         self.logger.log("Analysis completed", "success")

    #         # Force garbage collection
    #         gc.collect()
            
    #         return result
            
    #     except Exception as e:
    #         self.logger.log(f"Error during analysis execution: {e}", "error")  
    #         return None

class Utils():
    """
    Utils class for misc helper methods 

    These do not fit into the main analysis process
    """
    def __init__(self, verbosity=1):
        """Initialise
        """
        # Verbosity 
        self.verbosity = verbosity
        # Start logger
        self.logger = Logger(
            print_prefix="[PostProcess]",
            verbosity=self.verbosity
        )
        
        # Printer
        self.printer = Print(verbose=True)
        # Confirm
        self.logger.log(f"Initialised", "info")

    def get_kN(self, df_stats, numerator_name=None, denominator_name=None):
        """
        Retrieve efficiency data from DataFrame
        
        Args:
            df_stats (pandas.DataFrame): DataFrame with cut statistics
            numerator_name: The row to use as numerator.
            denominator_name: The row to use as denominator_name.
            
        Returns:
            tuple: k, N
        """
    
        if numerator_name is None or denominator_name is None:
            logger.log("Please provide a numerator_name and a denominator_name", "error")
            return None
    
        # Get numerator
        numerator_row = df_stats[df_stats["Cut"] == numerator_name]
        k = numerator_row["Events Passing"].iloc[0]
    
        # Get denominator
        denominator_row = df_stats[df_stats["Cut"] == denominator_name]
        N = denominator_row["Events Passing"].iloc[0]
        
        return k, N
    
    def get_eff(self, df_stats, ce_row_name="within_pitch_angle", veto=True):
        
        """
        Report efficiency results
        """

        self.logger.log(f"Getting efficiency with ce_row_name = {ce_row_name} and veto = {veto}", "info")
    
        # 
        results = []
        
        k_sig, N_sig = self.get_kN(
            df_stats, 
            numerator_name = ce_row_name, 
            denominator_name = "No cuts"
        ) 
    
        # Calculate efficiency
        eff_sig = (k_sig / N_sig) if N_sig > 0 else 0
        # Calculate poisson uncertainty 
        eff_sig_err = np.sqrt(k_sig) / N_sig
    
        results.append({
            "Type": "Signal",
            "Events Passing (k)": k_sig,
            "Total Events (N)": N_sig,
            "Efficiency [%]": eff_sig * 100,
            "Efficiency Error [%]": eff_sig_err * 100
        })
    
        if veto:
        
                k_veto, N_veto = None, None 
        
                k_veto, N_veto = self.get_kN(
                    df_stats, 
                    numerator_name = "unvetoed", 
                    denominator_name = ce_row_name
                ) 
        
                # Calculate efficiency
                eff_veto = 1 - (k_veto / N_veto) if N_veto > 0 else 0
                # Calculate poisson uncertainty 
                eff_veto_err = np.sqrt(k_veto) / N_veto
        
                results.append({
                    "Type": "Veto",
                    "Events Passing (k)": k_veto,
                    "Total Events (N)": N_veto,
                    "Efficiency [%]": eff_veto * 100,
                    "Efficiency Error [%]": eff_veto_err * 100
                })

        self.logger.log(f"Returning efficiency information", "success")
        
        return pd.DataFrame(results)

    def write_background_events(self, info, printout=True, out_path=None): 
        """
        Write background event info

        Args: 
            info (str): info string from postprocessing 
            printout (bool, opt): print output to screen
            out_path: File path for txt output 
        """        
        # Print 
        if printout:
            self.logger.log(f"Background event info:", "info")
            print(info)
        
        # Write to file
        if out_path:
            with open(out_path, "w") as f:
                f.write(info)
        
            self.logger.log(f"\tWrote {out_path}", "success")
            
    def write_verbose_background_events(self, data, out_path, override=False):
        """
        Write verbose background event info

        Args: 
            data (ak.Array): the data array  output to screen
            out_path: File path for txt output 
        """  
        # semi-abitrary check to make sure the printout isn't too long
        max_len = 25
        if len(data) > max_len and not override: 
            self.logger.log(f">{max_len} events! Are you sure you want to proceed?\n\tIf yes, set 'override' to True", "warning")
            return 
            
        # Redirect stdout to file
        with open(out_path, "w") as f:
            old_stdout = sys.stdout
            sys.stdout = f
            self.printer.print_n_events(data, n_events=len(data))
            # Restore stdout
            sys.stdout = old_stdout
            self.logger.log(f"Wrote {out_path}", "success")
            
    # # This could be quite nice, but it needs the full analysis config including everything which could be a bit complex. 
    # # Need to think about this. 
            
    # def save_configuration(self, filename):
    #     """Save the current analysis configuration to a file
        
    #     Args:
    #         filename (str): Path to the file to save the configuration to
    #     """
    #     self._log(f"Saving configuration to {filename}", level=1)
        
    #     try:
            
    #         # Create a configuration dictionary
    #         config = {
    #             "verbosity": self.verbosity,
    #             "on_spill": self.on_spill,
    #             # Add other configuration parameters as needed
    #         }
            
    #         # Save to file
    #         with open(filename, 'w') as f:
    #             json.dump(config, f, indent=4)
                
    #         self._log("Configuration saved successfully", level=1)
            
    #     except Exception as e:
    #         self._log(f"Error saving configuration: {e}", level=0)
    #         raise
            
    # def load_configuration(self, filename):
    #     """Load an analysis configuration from a file
        
    #     Args:
    #         filename (str): Path to the configuration file
    #     """
    #     self._log(f"Loading configuration from {filename}", level=1)
        
    #     try:
    #         import json
            
    #         # Load from file
    #         with open(filename, 'r') as f:
    #             config = json.load(f)
                
    #         # Apply configuration
    #         self.verbosity = config.get("verbosity", self.verbosity)
    #         self.on_spill = config.get("on_spill", self.on_spill)
    #         # Apply other configuration parameters as needed
            
    #         # Update dependent objects
    #         if hasattr(self, 'selector'):
    #             self.selector.verbosity = self.verbosity
    #         if hasattr(self, 'vector'):
    #             self.vector.verbosity = self.verbosity
                
    #         self._log("Configuration loaded successfully", level=1)
            
    #     except Exception as e:
    #         self._log(f"Error loading configuration: {e}", level=0)
            # raise