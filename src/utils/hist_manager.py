import hist
import numpy as np
import awkward as ak
from pyutils.pylogger import Logger

# DEBUG
from pyutils.pyprint import Print

class HistManager:
    """
    Manages histogram creation and filling
    """
    
    def __init__(self, analyse):
        """
        Initialise histogram manager
        
        Args:
            analyse: Analyse instance (contains all needed attributes)
        """
        self.analyse = analyse
        
        # Get attributes from analyse instance
        self.thresholds = analyse.thresholds
        self.selector = analyse.selector
        self.vector = analyse.vector
        self.verbosity = analyse.verbosity
        #DEBUG
        self.printer = Print()

        # Seperate logger
        self.logger = Logger(
            print_prefix="[HistManger]",
            verbosity=self.verbosity
        )
            
        # Define all histogram configurations
        self._define_histogram_configs()
    
    def _define_histogram_configs(self):
        """Define histogram configurations"""

        # Could put this in a YAML, but that seems unnecessary
        self.histogram_configs = {
            # Momentum histograms
            "mom_full": { # 5 MeV/c binning
                "axis": hist.axis.Regular(200, 0, 1000, name="mom", label="Momentum [MeV/c]"),
                "param": "mom",
                "filter": None
            },
            "mom_wide": { # 1 MeV/c binning
                "axis": hist.axis.Regular(115, self.thresholds["lo_wide_win_mevc"], 
                                          self.thresholds["hi_wide_win_mevc"], 
                                          name="mom", label="Momentum [MeV/c]"),
                "param": "mom",
                "filter": "wide_window"
            },
            "mom_ext": { # 0.5 MeV/c binning
                "axis": hist.axis.Regular(20, self.thresholds["lo_ext_win_mevc"], 
                                        self.thresholds["hi_ext_win_mevc"], 
                                        name="mom", label="Momentum [MeV/c]"),
                "param": "mom",
                "filter": "ext_window"
            },
            "mom_sig": { # 0.05 MeV/c binning
                "axis": hist.axis.Regular(20, self.thresholds["lo_sig_win_mevc"], 
                                        self.thresholds["hi_sig_win_mevc"], 
                                        name="mom", label="Momentum [MeV/c]"),
                "param": "mom", 
                "filter": "sig_window"
            },

            "pdg": { # Integer binning
                "axis": hist.axis.Regular(41, -20.5, 20.5, name="pdg", label="PDG"),
                "param": "pdg", 
                "filter": None
            },

            "sid": { # Integer binning
                "axis": hist.axis.Regular(51, -0.5, 50.0, name="sid", label="SID"),
                "param": "sid", 
                "filter": None
            },
            
            # CRV histograms
            "crv_z": { # 25 cm binning
                "axis": hist.axis.Regular(100, -15e3, 10e3, name="crv_z", label="CRV z-position [mm]"),
                "param": "crv_z",
                "filter": None
            },
            
            # Track histograms
            "trkqual": { # 1% binning
                "axis": hist.axis.Regular(100, 0, 1, name="trkqual", label="Track quality"),
                "param": "trkqual",
                "filter": None
            },
            
            "nactive": { # Integer binning
                "axis": hist.axis.Regular(101, -0.5, 100.5, name="nactive", label="Active tracker hits"),
                "param": "nactive", 
                "filter": None
            },
            "t0": { # 1 ns binning
                "axis": hist.axis.Regular(1400, 400, 1800, name="t0err", 
                                        label=r"Track fit time [ns]"),
                "param": "t0",
                "filter": None
            },
            # Track-fit parameter histograms
            "t0err": { # 0.01 ns binning
                "axis": hist.axis.Regular(500, 0, 5.0, name="t0err", 
                                        label=r"Track $t_{0}$ uncertainty, $\sigma_{t_{0}}$ [ns]"),
                "param": "t0err",
                "filter": None
            },
            "d0": { # 5 mm binning
                "axis": hist.axis.Regular(40, 0, 200, name="d0", 
                                        label=r"Distance of closest approach, $d_{0}$ [mm]"),
                "param": "d0",
                "filter": None
            },
            "maxr": { # 5 mm binning
                "axis": hist.axis.Regular(170, 150, 1000, name="maxr", 
                                        label=r"Loop helix maximum radius, $R_{\text{max}}$ [mm]"),
                "param": "maxr",
                "filter": None
            },
            "pitch_angle": { # 1% binning
                "axis": hist.axis.Regular(400, -1, 3.0, name="pitch_angle", 
                                        label=r"Pitch angle, $p_{z}/p_{T}$"),
                "param": "pitch_angle",
                "filter": None
            },

            # Momentum histograms
            "mom_z": { # 1 MeV/c binning
                "axis": hist.axis.Regular(125, -25, 100, name="mom_z", label=r"$p_{z}$ [MeV/c]"),
                "param": "mom_z",
                "filter": None
            },

            "mom_T": { # 1 MeV/c binning
                "axis": hist.axis.Regular(200, 0, 200, name="mom_T", label=r"$p_{T}$ [MeV/c]"),
                "param": "mom_T",
                "filter": None
            },

            "mom_err": { # 1 MeV/c binning
                "axis": hist.axis.Regular(120, 0, 3, name="mom_err", label=r"$\sigma_{p}$ [MeV/c]"),
                "param": "mom_err",
                "filter": None
            }, 

            "mom_res": { # 50 keV/c binning
                "axis": hist.axis.Regular(140, -5, 2, name="mom_res", label=r"$\delta p = p_{\text{reco}} - p_{\text{truth}}$ [MeV/c]"),
                "param": "mom_res",
                "filter": None
            }, 

            "dT": { # 1 ns binning
                "axis": hist.axis.Regular(500, -200, 300, name="dT", label=r"Track time $-$ CRV time [ns]"),
                "param": "dT",
                "filter": None
            }, 

            "trk_per_event": { # Integer binning
                "axis": hist.axis.Regular(6, -0.5, 5.5, name="trk_per_event", label=r"Track / event"),
                "param": "trk_per_event",
                "filter": None
            },

            "coinc_start_time": { 
                "axis": hist.axis.Regular(100, 0, 2000, name="coinc_start_time", label=r"Coincidence start time [ns]"),
                "param": "coinc_start_time",
                "filter": None
            }, 

            "coinc_end_time": { 
                "axis": hist.axis.Regular(100, 0, 2000, name="coinc_end_time", label=r"Coincidence end time [ns]"),
                "param": "coinc_end_time",
                "filter": None
            }, 
            
            # Commented out - too complicated with cut data structure
            # Use draw_cosmic_parents_from_array() instead with events array
            # "cosmic_parent_pdg": { # Categorical binning for particle types
            #     "axis": hist.axis.IntCategory([], name="cosmic_parent_pdg", label="Cosmic parent PDG", growth=True),
            #     "param": "cosmic_parent_pdg",
            #     "filter": None
            # },
            # # Trkqual histograms (excluding t0err, momerr, and nactive which we already have)
            # "factive": {},
            # "fambig": {},
            # "fitcon": {},
            # "fstraws": {} 

            
            # # nactive=[]
            # # factive=[]
            # # t0err=[]
            # # momerr=[]
            # # fambig=[]
            # # fitcon=[]
            # # fstraws=[]
        
            # # Misc
            # "nnullambig": {
            # }, 
            # "nmatactive": {
            # },     
            # # Add more 

            # # Track/calo histograms
            # "calo_E": { 
            # }, 
            # "calo_dt": {
            # },
            # "calo_Ep": {
            # }
        }

    def _prepare_track_data(self, data, surface_name="TT_Front", debug=False):
        """
        Loose cuts

        We have to select 
        - Electron fits at a tracker surface (front for extrap and middle for t0)
        - MC track parents (for momentum resolution)

        and everything needs to align! 

        This is slightly problematic in that the histograms may not align with analysis histograms?
        
        Returns:
            tuple: (trk, trkfit, trkmcsim)
        """
        # Select electron tracks
        is_reco_electron = self.selector.is_electron(data["trk"])

        # Select tracker front
        at_trk_surf = self.selector.select_surface(data["trkfit"], surface_name=surface_name)
        has_trk_surf = ak.any(at_trk_surf, axis=-1)

        # MC truth selections
        # Truth track parent is electron 
        is_truth_electron = data["trkmc"]["trkmcsim"]["pdg"] == 11
        is_trk_parent = data["trkmc"]["trkmcsim"]["nhits"] == ak.max(data["trkmc"]["trkmcsim"]["nhits"], axis=-1)
        is_trk_parent_electron = is_truth_electron & is_trk_parent 
        has_trk_parent_electron = ak.any(is_trk_parent_electron, axis=-1)

        # Copy data
        data_cut = ak.copy(data)
    
        # Construct combined trk level mask 
        trk_mask = is_reco_electron & has_trk_surf # & has_trk_parent_electron
        
        # Event level mask 
        has_trks = ak.any(trk_mask, axis=-1)

        # # Apply trksegs-level selection
        data_cut["trkfit"] = data_cut["trkfit"][at_trk_surf]

        # # # Apply trkmcsim-level selection
        data_cut["trkmc"] = data_cut["trkmc"][is_trk_parent_electron]

        # Apply trk-level selection 
        data_cut["trk"]  = data_cut["trk"][trk_mask]
        data_cut["trkfit"] = data_cut["trkfit"][trk_mask]
        data_cut["trkmc"] = data_cut["trkmc"][trk_mask]

        # Apply event-level cleanup
        data_cut = data_cut[has_trks]

        return data_cut["trk"], data_cut["trkfit"], data_cut["trkmc"]
        
        # trk = data["trk"]
        # trkfit = data["trkfit"]
        # trkmc = data["trkmc"] 
        
        # # Select electron tracks
        # is_reco_electron = self.selector.is_electron(data["trk"])

        # # Select tracker front
        # at_trk_front = self.selector.select_surface(trkfit, surface_name="TT_Front")
        # has_trk_front = ak.any(at_trk_front, axis=-1)

        # # Apply trksegs-level selection
        # # This must come first!
        # trkfit = trkfit[at_trk_front]

        # # Construct combined trk reco mask 
        # trk_reco_mask = is_reco_electron & has_trk_front 

        # # Apply trk-level selection 
        # trk = trk[trk_reco_mask]
        # trkfit = trkfit[trk_reco_mask]
        # trkmc = trkmc[trk_reco_mask]

        # # MC truth selections
        # # Truth track parent is electron 
        # # We must construct these masks after the reco cuts, or 
        # # the resulting trkmc array will not align with trksegs
        # is_truth_electron = trkmc["trkmcsim"]["pdg"] == self.thresholds["track_pdg"]
        # is_trk_parent = trkmc["trkmcsim"]["nhits"] == ak.max(trkmc["trkmcsim"]["nhits"], axis=-1)
        # is_trk_parent_electron = is_truth_electron & is_trk_parent 

        # # Apply MC genealogy-level selections
        # trkmc = trkmc[is_trk_parent_electron]
        
        # return trk, trkfit, trkmc
    
    def _extract_data(self, data, param, selection):
        """
        Extract histogram data based on parameter name

        TODO: do we always need to prepare track data, this seems to be asking for trouble!
        
        Args:
            data: Input data array
            param: Name of param to extract (e.g., 'mom', 'trkqual', etc.)
            
        Returns:
            ak.Array: Flattened extracted data
        """        
        if param == "mom":
            trk, trkfit, _  = self._prepare_track_data(data)
            mom = self.vector.get_mag(trkfit["trksegs"], "mom")
            return ak.flatten(mom, axis=None) if mom is not None else ak.Array([])
            
        elif param == "crv_z":
            return ak.flatten(data["crv"]["crvcoincs.pos.fCoordinates.fZ"], axis=None)
            
        elif param == "trkqual":
            # trk, _, _  = self._prepare_track_data(data)
            trk = data["trk"]
            return ak.flatten(trk["trkqual.result"], axis=None)
            
        elif param == "nactive":
            # trk, _, _  = self._prepare_track_data(data)
            trk = data["trk"]
            return ak.flatten(trk["trk.nactive"], axis=None)

        elif param == "t0":
            _, trkfit, _  = self._prepare_track_data(data, surface_name="TT_Mid")
            return ak.flatten(trkfit["trksegs"]["time"], axis=None)
            
        elif param == "t0err":
            _, trkfit, _  = self._prepare_track_data(data, surface_name="TT_Mid")
            return ak.flatten(trkfit["trksegpars_lh"]["t0err"], axis=None)
            
        elif param == "d0":
            _, trkfit, _  = self._prepare_track_data(data)
            return ak.flatten(trkfit["trksegpars_lh"]["d0"], axis=None)
            
        elif param == "maxr":
            _, trkfit, _  = self._prepare_track_data(data)
            return ak.flatten(trkfit["trksegpars_lh"]["maxr"], axis=None)
            
        elif param == "pitch_angle":
            _, trkfit, _  = self._prepare_track_data(data)
            pitch_angle = self.analyse.get_pitch_angle(trkfit)  # Use existing method
            return ak.flatten(pitch_angle, axis=None)
            
        elif param == "pdg":
            # trk, _, _  = self._prepare_track_data(data)
            trk = data["trk"]
            return ak.flatten(trk["trk.pdg"], axis=None)
            
        elif param == "sid":
            _, trkfit, _  = self._prepare_track_data(data)
            return ak.flatten(trkfit["trksegs"]["sid"], axis=None)
            
        elif param == "mom_z":
            _, trkfit, _  = self._prepare_track_data(data)
            return ak.flatten(trkfit["trksegs"]["mom"]["fCoordinates"]["fZ"], axis=None)

        elif param == "mom_T":
            _, trkfit, _  = self._prepare_track_data(data)
            mom_vec = self.vector.get_vector(trkfit["trksegs"], "mom")
            mom_T = np.sqrt(mom_vec["x"]**2 + mom_vec["y"]**2)
            return ak.flatten(mom_T, axis=None)

        elif param == "mom_err":
            _, trkfit, _ = self._prepare_track_data(data)
            return ak.flatten(trkfit["trksegs"]["momerr"], axis=None)

        elif param == "mom_res":
            _, trkfit, trkmc = self._prepare_track_data(data, debug=True)
            mom_reco = self.vector.get_mag(trkfit["trksegs"], "mom")
            mom_truth = self.vector.get_mag(trkmc["trkmcsim"], "mom")
            mom_reco = ak.flatten(mom_reco,axis=None) 
            mom_truth = ak.flatten(mom_truth,axis=None)
            try:
                mom_res = mom_reco - mom_truth
                return mom_res
            except:
                self.logger.log(f"Misalignment in momentum resolution calculation for {selection} (returning mom_res=[]):\n\tmom_reco has length {len(mom_reco)}\n\tmom_truth has length {len(mom_truth)}", "warning")
                return []

        elif param == "dT":
            _, trkfit, _ = self._prepare_track_data(data, surface_name="TT_Mid")
            # Get time differences
            try:
                # mom_res = mom_reco - mom_truth
                dT = self.analyse.get_trk_crv_dt(trkfit, data["crv"])["dT"]
                return ak.flatten(dT, axis=None)
            except:
                self.logger.log(f"Misalignment in dT calculation for {selection} (returning dT=[])", "warning")
                return []

        elif param == "trk_per_event":
            trk, _, _ = self._prepare_track_data(data)
            trk_event = ak.count(trk["trk.pdg"], axis=-1)
            return ak.flatten(trk_event, axis=None)

        elif param == "coinc_start_time":
            crv = data["crv"]
            start_time = crv["crvcoincs.timeStart"]
            return ak.flatten(start_time, axis=None)

        elif param == "coinc_end_time":
            crv = data["crv"]
            end_time = crv["crvcoincs.timeEnd"]
            return ak.flatten(end_time, axis=None)
            
        else:
            error_message = f"Unknown parameter: {param}"
            self.logger.log(error_message, "error")
            raise ValueError(error_message)
    
    def _apply_filter(self, values, filter_name):
        """
        Apply filters to extracted data
        
        Args:
            values: Extracted data values
            filter_name: Name of filter to apply
            
        Returns:
            ak.Array: Filtered values
        """
        if filter_name is None:
            return values

        elif filter_name == "wide_window":
            return values[(values > self.thresholds["lo_wide_win_mevc"]) & 
                         (values < self.thresholds["hi_wide_win_mevc"])]
            
        elif filter_name == "ext_window":
            return values[(values > self.thresholds["lo_ext_win_mevc"]) & 
                         (values < self.thresholds["hi_ext_win_mevc"])]
                         
        elif filter_name == "sig_window":
            return values[(values > self.thresholds["lo_sig_win_mevc"]) & 
                         (values < self.thresholds["hi_sig_win_mevc"])]
                         
        else:
            error_message = f"Unknown filter: {filter_name}"
            self.logger.log(error_message, "error")
            raise ValueError(error_message)
            
    def create_histograms(self, datasets):
        """
        Create and fill histograms for multiple datasets
        
        Args:
            datasets: Dictionary with selection labels as keys and data as values
                     e.g., {"All": data, "Select": data_CE, "Unvetoed": data_CE_unvetoed}
            
        Returns:
            dict: Dictionary of filled histograms
        """
        self.logger.log("Creating histograms", "info")
        
        selection_labels = list(datasets.keys())
        
        # Create empty histograms
        histograms = {}
        for hist_name, config in self.histogram_configs.items():
            histograms[hist_name] = hist.Hist(
                config["axis"],
                hist.axis.StrCategory(selection_labels, name="selection", label="Selection")
            )
        
        # Fill histograms for each dataset
        for selection_label, data in datasets.items():
            if len(data) == 0:
                self.logger.log(f"Skipping empty dataset: {selection_label}", "warning")
                continue
                
            self.logger.log(f"Filling histograms for: {selection_label}", "max")
            
            for hist_name, config in self.histogram_configs.items():
                try:
                    # Extract the data
                    values = self._extract_data(data, config["param"], selection_label)
                    
                    # Apply any filters
                    filtered_values = self._apply_filter(values, config["filter"])
                    
                    # Skip if no data after filtering
                    if len(filtered_values) == 0:
                        if hist_name == "cosmic_parent_pdg":
                            self.logger.log(f"No data for {hist_name} after filtering for {selection_label}", "warning")
                        continue
                    
                    # Fill the histogram
                    fill_kwargs = {
                        config["axis"].name: filtered_values,
                        "selection": np.full(len(filtered_values), selection_label)
                    }
                    histograms[hist_name].fill(**fill_kwargs)
                    
                    if hist_name == "cosmic_parent_pdg":
                        self.logger.log(f"Filled {hist_name} for {selection_label} with {len(filtered_values)} entries", "info")
                    
                except Exception as e:
                    self.logger.log(f"Error filling {hist_name} for {selection_label}: {e}", "error")
                    raise
        
        self.logger.log("Histograms filled successfully", "success")
        
        # Return copies to avoid reference issues
        return {name: hist_obj.copy() for name, hist_obj in histograms.items()}