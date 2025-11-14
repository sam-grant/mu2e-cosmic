import uproot
import awkward as ak
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import hist
import gc
import sys 
import os

from pyutils.pyselect import Select
from pyutils.pyvector import Vector
from pyutils.pylogger import Logger
from pyutils.pyprint import Print

# Add relative path to utils
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, "..", "utils"))
from io_manager import Load
from cut_manager import CutManager
from hist_manager import HistManager

class Analyse:
    """Class to handle analysis method and flow
    """
    def __init__(self, on_spill=False, cut_config_path="../../config/common/cuts.yaml", cutset_name="alpha", event_subrun=None, verbosity=1):
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
        self.event_subrun = event_subrun # event selection (for debugging)

        # Tools
        self.logger = Logger( # Needs to come before _load_cuts
            print_prefix="[Analyse]",
            verbosity=self.verbosity
        )
        self.selector = Select(verbosity=self.verbosity)
        self.vector = Vector(verbosity=self.verbosity) 

        # Helper sets cut parameters, self.track_pdg, self.lo_nactive, etc.
        loader = Load() # in_path="../config/common/") # relative to /run 
        self.cut_config = {"description": "FAILED TO LOAD"}  # Default fallback
        loader.load_cuts_yaml(self)
        
        # More tools
        self.hist_manager = HistManager(self) # Needs to come after _load_cuts

        # Init printout
        self.logger.log(
            f"Initialised with:\n"
            f"  on_spill        = {self.on_spill}\n"
            f"  cut_config_path = {self.cut_config_path}\n"
            f"  cutset_name     = {self.cutset_name} ({self.cut_config["description"]})\n"
            f"  verbosity       = {self.verbosity}",
            "info"
        )

    def get_pitch_angle(self, trkfit):
        """Helper to calculate pitch angle"""
        pvec = self.vector.get_vector(trkfit["trksegs"], "mom")
        pt = np.sqrt(pvec["x"]**2 + pvec["y"]**2) 
        pz = trkfit["trksegs"]["mom"]["fCoordinates"]["fZ"]
        # Avoid division by zero
        return ak.where(pt != 0, pz/pt, 0) 

    def get_trk_crv_dt(self, trkfit, crv):
        """Helper to get trk/crv time difference (dT).

        Calculates the time difference between every track segment in trkfit and
        every CRV coincidence in crv for the same event.

        Args:
            trkfit (ak.Array): Contains track fit information per segment. 
            crv (ak.Array): Contains CRV coincidence information.

        Returns:
            ak.Array: Time difference dT (track segment time - CRV coincidence time).
                      Shape: [event, track, segment, coincidence]
        """
        # Find dT
        # Extract track segment times. Shape is [E, T, S] (Event, Track, Segment).
        trk_times = trkfit["trksegs"]["time"] 
        
        # Apply start/end time mask to coincidences 
        # TODO: formalise this as a proper cut 
        # start_time_cond = ((crv["crvcoincs.timeStart"] > 450) & (crv["crvcoincs.timeStart"] < 1550))
        # end_time_cond = (crv["crvcoincs.timeEnd"] < 1690)
        # start_end_time_cond = (start_time_cond & end_time_cond)

        # this produces a lot of Nones which cause the event to veto, need to be 
        # an event level cut somehow. 
        
        # Extract CRV coincidence times. Shape is [E, C] (Event, Coincidence).
        coinc_times = crv["crvcoincs.time"] # [start_end_time_cond]

        # Broadcast CRV times to match track structure so we can element-wise subtract.

        # Shape transform: [E, C] -> [E, 1, 1, C].
        # The '1' dimensions will be automatically expanded to match the
        # T and S dimensions during subtraction.
        coinc_broadcast = coinc_times[:, None, None, :] 
        
        # Broadcast track times to match the CRV structure.
        # Shape transform: [E, T, S] -> [E, T, S, 1].
        # The final '1' dimension will be broadcast to match the C dimension.
        trk_broadcast = trk_times[:, :, :, None] 
        
        # coinc_broadcast shape is [E, 1, 1, C]
        # trk_broadcast shape is [E, T, S, 1]
        
        # The subtraction now yields the desired shape [E, T, S, C], 
        # containing dT for every possible track-segment/CRV-coincidence pairing.

        # Return time differences
        return { # Nice to have the times for debugging
            "dT" : (trk_broadcast - coinc_broadcast),
            "trk_times" : trk_times,
            "coinc_times" : coinc_times
        }
    
    def _append_array(self, data, arr, name):
        """Helper to append arrays/masks to dev field 
        This is useful for debugging and development
        Must be trk or trkfit level
        """
        try:
            if "dev" not in ak.fields(data):
                # Initialise dev field - structure will be determined by first array added
                data = ak.with_field(data, ak.zip({name: arr}, depth_limit=1), "dev")
            else: 
                # Add new field to existing 'dev' record
                new_dev = ak.with_field(data.dev, arr, name)
                data = ak.with_field(data, new_dev, "dev")
            return data
        except Exception as e:
            self.logger.log(f"Error appending '{name}' to data array: {e}", "error") 
            raise e
        
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
            at_trk_mid = self.selector.select_surface(data["trkfit"], surface_name="TT_Mid")
            at_trk_back = self.selector.select_surface(data["trkfit"], surface_name="TT_Back")
            in_trk = (at_trk_front | at_trk_mid | at_trk_back)
        except Exception as e:
            self.logger.log(f"Error defining tracker surface masks: {e}", "error") 
            raise e
            
        try:
            # Track level definition 
            has_trk_front = ak.any(at_trk_front, axis=-1)
            
            # Add cut 
            cut_manager.add_cut(
                name="has_trk_front", 
                description="Tracks intersect tracker entrance", 
                mask=has_trk_front,
                active=self.active_cuts["has_trk_front"],
                group="Preselect"
            )
            # Append for debugging
            data = self._append_array(data, at_trk_front, "at_trk_front")
            data = self._append_array(data, has_trk_front, "has_trk_front")
            # data["at_trk_front"] = at_trk_front
            # data["has_trk_front"] = has_trk_front
        except Exception as e:
            self.logger.log(f"Error defining 'has_trk_front' cut: {e}", "error") 
            raise e

        try:
            # Track level definition 
            has_trk_mid = ak.any(at_trk_mid, axis=-1)
            
            # Add cut 
            cut_manager.add_cut(
                name="has_trk_mid", 
                description="Tracks intersect tracker middle", 
                mask=has_trk_mid,
                active=self.active_cuts["has_trk_mid"],
                group="Preselect"
            )
            # Append for debugging
            data = self._append_array(data, at_trk_mid, "at_trk_mid")
            data = self._append_array(data, has_trk_mid, "has_trk_mid")
            # data["at_trk_mid"] = at_trk_mid
            # data["has_trk_mid"] = has_trk_mid
        except Exception as e:
            self.logger.log(f"Error defining 'has_trk_mid' cut: {e}", "error") 
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
                group="Preselect"
            )
            # Append mask for debugging
            data = self._append_array(data, is_reco_electron, "is_reco_electron")
            # data["is_reco_electron"] = is_reco_electron
        except Exception as e:
            self.logger.log(f"Error defining 'is_reco_electron' cut: {e}", "error") 
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
                group="Preselect"
            )
            # Append for debugging 
            data = self._append_array(data, one_reco_electron, "one_reco_electron")
            data = self._append_array(data, one_reco_electron_per_event, "one_reco_electron_per_event")
            # data["one_reco_electron"] = one_reco_electron
            # data["one_reco_electron_per_event"] = one_reco_electron_per_event
        except Exception as e:
            self.logger.log(f"Error defining 'one_reco_electron' cut: {e}", "error") 
            raise e

        ###################################################
        # Downstream tracks at the tracker entrance 
        ###################################################
        try:
            # Track segments level mask
            is_downstream = self.selector.is_downstream(data["trkfit"]) 
            # Track level mask
            # "all" method handles reflections 
            # is_downstream = ak.all(~at_trk_front | is_downstream, axis=-1) 
            is_downstream = ak.all(~in_trk | is_downstream, axis=-1) 

            # Updated definition: all downstream
            # should mean that NO downstream tracks are reconstructed 
            # Event-level definition
            all_downstream_per_event = ak.all(is_downstream, axis=-1)
            # Broadcast to track level
            all_downstream, _ = ak.broadcast_arrays(all_downstream_per_event, is_downstream)

            # "any" method does not handle reflections, but may be closer to C++
            # is_downstream = ak.any(at_trk_front & is_downstream, axis=-1) 
            # Add cut - use is_downstream (has any downstream track)
            cut_manager.add_cut(
                name="is_downstream",
                description="Has downstream track (p_z > 0 in tracker)",
                mask=is_downstream,
                active=self.active_cuts["is_downstream"],
                group="Preselect"
            )
            
            # Commented out: all_downstream is too restrictive
            # cut_manager.add_cut(
            #     name="all_downstream",
            #     description="All tracks are downstream (all p_z > 0 in tracker)",
            #     mask=all_downstream,
            #     active=self.active_cuts["all_downstream"],
            #     group="Preselect"
            # )
            
            # Append for debugging
            data = self._append_array(data, is_downstream, "is_downstream")
            data = self._append_array(data, all_downstream, "all_downstream")
            data = self._append_array(data, all_downstream_per_event, "all_downstream_per_event")
            # data["is_downstream"] = is_downstream
        except Exception as e:
            self.logger.log(f"Error defining 'is_downstream' cut: {e}", "error") 
            raise e

        ###################################################
        # Track "PID" from truth track parents 
        ###################################################
        try:
            # Truth track parent is electron 
            is_truth_electron = data["trkmc"]["trkmcsim"]["pdg"] == self.thresholds["track_pdg"]
            is_trk_parent = data["trkmc"]["trkmcsim"]["nhits"] == ak.max(data["trkmc"]["trkmcsim"]["nhits"], axis=-1)
            is_trk_parent_electron = is_truth_electron & is_trk_parent 
            has_trk_parent_electron = ak.any(is_trk_parent_electron, axis=-1) 
            # Add cut 
            cut_manager.add_cut(
                name="is_truth_electron", 
                description="Track parents are electrons (truth PID)", 
                mask=has_trk_parent_electron,
                active=self.active_cuts["is_truth_electron"],
                group="Preselect"
            )
            # Append for debugging
            data = self._append_array(data, has_trk_parent_electron, "has_trk_parent_electron")
            # data["has_trk_parent_electron"] = has_trk_parent_electron
        except Exception as e:
            self.logger.log(f"Error defining 'is_truth_electron' cut: {e}", "error") 
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
            data = self._append_array(data, good_trkqual, "good_trkqual")
            # data["good_trkqual"] = good_trkqual
        except Exception as e:
            self.logger.log(f"Error defining 'good_trkqual' cut: {e}", "error") 
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
                within_t0 = ak.all(~at_trk_mid | within_t0, axis=-1)
                # within_t0 = ak.all(~at_trk_front | within_t0, axis=-1)
                # Add cut 
                cut_manager.add_cut( 
                    name="within_t0",
                    description=f"t0 at tracker mid ({self.thresholds["lo_t0_ns"]}"\
                                f" < t_0 < {self.thresholds["hi_t0_ns"]} ns)",
                    mask=within_t0,
                    active=self.active_cuts["within_t0"],
                    group="Tracker"
                )
                # Append for debugging
                data = self._append_array(data, within_t0, "within_t0")
                # data["within_t0"] = within_t0
        except Exception as e:
            self.logger.log(f"Error defining 'within_t0' cut: {e}", "error") 
            raise e

        ###################################################
        # t0 uncertainty 
        ###################################################
        try:
            # Track segments level definition
            within_t0err = (data["trkfit"]["trksegpars_lh"]["t0err"] < self.thresholds["hi_t0err"])

            # Track level definition
            # within_t0err = ak.all(~at_trk_front | within_t0err, axis=-1)
            within_t0err = ak.all(~at_trk_mid | within_t0err, axis=-1) 

            # Add cut 
            cut_manager.add_cut(
                name="within_t0err",
                description=f"Track fit t0 uncertainty (t0err < {self.thresholds["hi_t0err"]} ns)",
                mask=within_t0err,
                active=self.active_cuts["within_t0err"],
                group="Tracker"
            )
            # Append for debugging
            data = self._append_array(data, within_t0err, "within_t0err")
            # data["within_t0err"] = within_t0err
        except Exception as e:
            self.logger.log(f"Error defining 'within_t0err' cut: {e}", "error") 
            raise e
            
        ###################################################
        # Minimum active hits
        ###################################################
        try:
            # Track level definition 
            # has_hits = self.selector.has_n_hits(data["trk"], n_hits=self.thresholds["lo_nactive"])
            has_hits = data["trk"]["trk.nactive"] > self.thresholds["lo_nactive"]
            # Add cut
            cut_manager.add_cut(
                name="has_hits",
                # description=f">{self.thresholds["lo_nactive"]-1} active tracker hits",
                description=f">{self.thresholds["lo_nactive"]} active tracker hits",
                mask=has_hits,
                active=self.active_cuts["has_hits"],
                group="Tracker"
            )
            # Append for debugging
            data = self._append_array(data, has_hits, "has_hits")
            # data["has_hits"] = has_hits
        except Exception as e:
            self.logger.log(f"Error defining 'has_hits' cut: {e}", "error") 
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
            data = self._append_array(data, within_d0, "within_d0")
            # data["within_d0"] = within_d0
        except Exception as e:
            self.logger.log(f"Error defining 'within_d0' cut: {e}", "error") 
            raise e

        ###################################################
        # Pitch angle lower bound
        ###################################################
        try:
            # Define pitch angle
            pitch_angle = self.get_pitch_angle(data["trkfit"])
            # Trk segments level
            within_pitch_angle_lo = (self.thresholds["lo_pitch_angle"] < pitch_angle) 
            # Track level definition
            within_pitch_angle_lo = ak.all(~at_trk_front | within_pitch_angle_lo, axis=-1)
            # Add cut 
            cut_manager.add_cut(
                name="within_pitch_angle_lo",
                description=f"Extrapolated pitch angle (pz/pt > {self.thresholds["lo_pitch_angle"]})",
                mask=within_pitch_angle_lo,
                active=self.active_cuts["within_pitch_angle_lo"],
                group="Tracker"
            )
            # Append 
            data = self._append_array(data, pitch_angle, "pitch_angle")
            data = self._append_array(data, within_pitch_angle_lo, "within_pitch_angle_lo")
            # data["pitch_angle"] = pitch_angle
            # data["within_pitch_angle_lo"] = within_pitch_angle_lo
        except Exception as e:
            self.logger.log(f"Error defining 'within_pitch_angle_lo' cut: {e}", "error") 
            raise e

        ###################################################
        # Pitch angle upper bound
        ###################################################
        try:
            # Define pitch angle
            pitch_angle = self.get_pitch_angle(data["trkfit"])
            # Trk segments level
            within_pitch_angle_hi = (self.thresholds["hi_pitch_angle"] > pitch_angle) 
            # Track level definition
            within_pitch_angle_hi = ak.all(~at_trk_front | within_pitch_angle_hi, axis=-1)
            # Add cut 
            cut_manager.add_cut(
                name="within_pitch_angle_hi",
                description=f"Extrapolated pitch angle (pz/pt < {self.thresholds["hi_pitch_angle"]})",
                mask=within_pitch_angle_hi,
                active=self.active_cuts["within_pitch_angle_hi"],
                group="Tracker"
            )
            # Append 
            data = self._append_array(data, within_pitch_angle_hi, "within_pitch_angle_hi")
            # data["within_pitch_angle_hi"] = within_pitch_angle_hi
        except Exception as e:
            self.logger.log(f"Error defining 'within_pitch_angle_hi' cut: {e}", "error") 
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
            data = self._append_array(data, within_lhr_max_lo, "within_lhr_max_lo")
            # data["within_lhr_max_lo"] = within_lhr_max_lo
        except Exception as e:
            self.logger.log(f"Error defining 'within_lhr_max_lo' cut: {e}", "error") 
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
            data = self._append_array(data, within_lhr_max_hi, "within_lhr_max_hi")
            # data["within_lhr_max_hi"] = within_lhr_max_hi
        except Exception as e:
            self.logger.log(f"Error defining 'within_lhr_max_hi' cut: {e}", "error") 
            raise e

        ###################################################
        # CRV veto: trk/crv time window cut  
        # Check if ANY track has mid segment within time of ANY coincidence 
        ###################################################

        try: 
            # Calculate time differences: [E, T, S, C] (Event, Track, Segment, Coincidence)
            # Measure time difference for the tracker middle using electron hypothesis 
            dt_trk_condition = at_trk_mid & is_reco_electron
            trk_crv_dt = self.get_trk_crv_dt(data["trkfit"][dt_trk_condition], data["crv"])
            dT = trk_crv_dt["dT"] # I do it this way so I can debug the trk/coinc times
            # Flatten to over coindidences (3) and segments (2) 
            dT = ak.flatten(ak.flatten(dT, axis=3), axis=2) # [E, T]

            # Time window cut 
            veto_condition = (
                (dT > self.thresholds["lo_veto_dt_ns"])
                & (dT < self.thresholds["hi_veto_dt_ns"])
            )

            # Event level veto 
            # A track is vetoed if *any* coincidence lies within the window
            vetoed = ak.any(veto_condition, axis=-1)
            unvetoed = ~vetoed # invert
            unvetoed_trk = unvetoed[:, None] # trk-level mask
            
            # Add cut
            description = (
                f"No veto: {self.thresholds['lo_veto_dt_ns']} < "
                f"dT < {self.thresholds['hi_veto_dt_ns']} ns"
            )
            cut_manager.add_cut(
                name="unvetoed",
                description=description, 
                mask=unvetoed, # _trk,
                active=self.active_cuts["unvetoed"],
                group="CRV"
            )

            # dT index, useful for ML
            dT_idx = ak.local_index(dT, axis=-1)
            
            # Find minimum "leading" dT 
            # Note that min dT is a bad representation, does not always represent the vetoed dT 
            min_dT_idx = ak.argmin(abs(dT), axis=-1, keepdims=True)
            min_dT = dT[min_dT_idx]

            # Central dT, the one most likely to be used in the veto 
            # Calculate the midpoint between your two thresholds
            mid_value = (self.thresholds['lo_veto_dt_ns'] + self.thresholds['hi_veto_dt_ns']) / 2
            # Find the dT closest to the midpoint
            cent_dT_idx = ak.argmin(abs(dT - mid_value), axis=-1, keepdims=True)
            cent_dT = dT[cent_dT_idx]

            # Store 
            data = self._append_array(data, veto_condition, "veto_condition")
            data = self._append_array(data, vetoed, "vetoed") 
            data = self._append_array(data, unvetoed, "unvetoed")
            data = self._append_array(data, trk_crv_dt["trk_times"], "trk_times") 
            data = self._append_array(data, trk_crv_dt["coinc_times"], "coinc_times")             
            data = self._append_array(data, dT_idx, "dT_idx") 
            data = self._append_array(data, dT, "dT") 
            data = self._append_array(data, min_dT, "min_dT")  
            data = self._append_array(data, min_dT_idx, "min_dT_idx") 
            data = self._append_array(data, cent_dT, "cent_dT") 
            data = self._append_array(data, cent_dT_idx, "cent_dT_idx") 

        except Exception as e:
            self.logger.log(f"Error defining 'unvetoed' cut: {e}", "error") 
            raise e

        ###################################################
        # Wide window 
        ###################################################
        try:
            # Get momentum magnitude 
            mom = self.vector.get_mag(data["trkfit"]["trksegs"], "mom") 
            # Track segments level definition 
            within_wide_win = (mom > self.thresholds["lo_wide_win_mevc"]) & (mom < self.thresholds["hi_wide_win_mevc"])
            # Track level definition
            within_wide_win = ak.all(~at_trk_front | within_wide_win, axis=-1)
            # Store momentum magnitude at this stage (array written to "data" before momentum cuts)
            data = self._append_array(data, mom, "mom_mag")
            # Add cut 
            cut_manager.add_cut(
                name="within_wide_win",
                description=f"Wide window ({self.thresholds["lo_wide_win_mevc"]}"\
                            f" < p < {self.thresholds["hi_wide_win_mevc"]} MeV/c)",
                mask=within_wide_win,
                active=self.active_cuts["within_wide_win"],
                group="Momentum"
            )
            # Append 
            data = self._append_array(data, within_wide_win, "within_wide_win") 
        except Exception as e:
            self.logger.log(f"Error defining 'within_wide_win' cut: {e}", "error") 
            raise e
            
        ###################################################
        # Extended window 
        ###################################################
        try:
            # Get momentum magnitude again
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
            data = self._append_array(data, within_ext_win, "within_ext_win")
        except Exception as e:
            self.logger.log(f"Error defining 'within_ext_win' cut: {e}", "error") 
            raise e

        ###################################################
        # Signal window 
        ###################################################
        try:
            # Get momentum magnitude again
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
            data = self._append_array(data, within_sig_win, "within_sig_win")
        except Exception as e:
            self.logger.log(f"Error defining 'within_sig_win' cut: {e}", "error") 
            raise e

        # Done 
        self.logger.log("All cuts defined", "success")

        return data # to pick up new fields
        
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

            # ##########
            # this would be nice, but too complicated
            # data_cut["dev"] = data_cut["dev"][trk_mask]
            # ##########
            
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

            # Optional prefiltering 
            if self.event_subrun is not None: 
                self.logger.log(f"Selection: Event={self.event_subrun[0]}, Subrun={self.event_subrun[1]}", "info")
                mask = ((data["evt"]["event"] == self.event_subrun[0]) & (data["evt"]["subrun"] == self.event_subrun[1]))
                data = data[mask]
            
            # Define cuts
            self.logger.log(f"Defining cuts", "max")
            data = self.define_cuts(data, cut_manager)

            # self.logger.log(20*"*" + " START DEBUG " + 20*"*" , "test")
            # from pyutils.pyprint import Print
            # Print().print_n_events(data["dev"])
            # self.logger.log(20*"*" + " END DEBUG " + 20*"*", "test")

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

            # Apply Preselect cuts
            self.logger.log("Applying Preselect cuts", "max")
            cut_manager.toggle_group({
                "Tracker": False,
                "CRV": False, 
                "Momentum": False
            })  
            # cut_manager.save_state("preselect")
            data = self._append_array(
                data, 
                cut_manager.combine_cuts(active_only=True), 
                "preselect"
            )
            data_preselect = self.apply_cuts(data, cut_manager)

            # Apply selection cuts (without veto & momentum windows)
            self.logger.log("Applying Select cuts", "max")
            # Restore state
            # Toggling Tracker to True will activate tracker cuts from different
            # cutsets that we may not want
            cut_manager.restore_state("original")
            cut_manager.toggle_group({
                "CRV": False,       # Keep CRV off (this turns off "unvetoed")
                "Momentum": False   # Keep Momentum off (this turns off both momentum windows)
            })
            # Print active cuts
            cut_manager.list_groups()
            # Save Select selection
            data = self._append_array(
                data, 
                cut_manager.combine_cuts(active_only=True), 
                "select"
            )
            data_CE = self.apply_cuts(data, cut_manager)

            # Apply CRV cuts
            data_CE_unvetoed = {} # Default with length zero
            if veto:  
                self.logger.log("Applying veto cuts", "max")  
                # Turn veto on
                cut_manager.toggle_group({"CRV": True}) 
                # cut_manager.toggle_cut({"unvetoed": True}) # same thing
                data = self._append_array(
                    data, 
                    cut_manager.combine_cuts(active_only=True), 
                    "unvetoed_combined"
                )
                data_CE_unvetoed = self.apply_cuts(data, cut_manager) 
            
            # Create histograms and compile results
            self.logger.log("Creating histograms", "max")
            histograms = self.create_histograms(
                datasets = { 
                    "All": data,
                    "Preselect": data_preselect,
                    "Select": data_CE,
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