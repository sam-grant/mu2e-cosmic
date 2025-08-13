import hist
import numpy as np
import awkward as ak
from pyutils.pylogger import Logger

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

        # Seperate logger
        self.logger = Logger(
            print_prefix="[HistManger]",
            verbosity=self.verbosity
        )
            
        # Define all histogram configurations
        self._define_histogram_configs()
    
    def _define_histogram_configs(self):
        """Define histogram configurations"""

        # Could put this in a YAML, seems unnecessary
        self.histogram_configs = {
            # Momentum histograms
            "mom_full": { # 5 MeV/c binning
                "axis": hist.axis.Regular(200, 0, 1000, name="mom", label="Momentum [MeV/c]"),
                "param": "mom",
                "filter": None
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
            
            # Track-fit parameter histograms
            "t0err": { # 0.05 ns binning
                "axis": hist.axis.Regular(60, 0, 3.0, name="t0err", 
                                        label=r"Track $t_{0}$ uncertainty, $\sigma_{t_{0}}$ [ns]"),
                "param": "t0err",
                "filter": None
            },
            "d0": { # 5 mm binning
                "axis": hist.axis.Regular(60, 0, 300, name="d0", 
                                        label=r"Distance of closest approach, $d_{0}$ [mm]"),
                "param": "d0",
                "filter": None
            },
            "maxr": { # 5 mm binning
                "axis": hist.axis.Regular(130, 200, 850, name="maxr", 
                                        label=r"Loop helix maximum radius, $R_{\text{max}}$ [mm]"),
                "param": "maxr",
                "filter": None
            },
            "pitch_angle": { # 5% binning
                "axis": hist.axis.Regular(60, -0.5, 2.5, name="pitch_angle", 
                                        label=r"Pitch angle, $p_{z}/p_{T}$"),
                "param": "pitch_angle",
                "filter": None
            },
            
            # Debug histograms
            "pdg": { # Integer binning
                "axis": hist.axis.Regular(30, -15, 15, name="pdg", label="Track PDG"),
                "param": "pdg",
                "filter": None
            },
            "sid": { # Integer binning
                "axis": hist.axis.Regular(60, -10, 50, name="sid", label="Surface ID"),
                "param": "sid", 
                "filter": None
            },
            "pz": { # 1 MeV/c binning
                "axis": hist.axis.Regular(110, -10, 100, name="pz", label=r"$p_{z}$ [MeV/c]"),
                "param": "pz",
                "filter": None
            }
        }
    
    def _prepare_track_data(self, data):
        """
        Loose cuts
        
        Returns:
            tuple: (trk, trkfit)
        """
        # Select electron tracks
        is_electron = self.selector.is_electron(data["trk"])
        # # One electron fit
        # one_electron_per_event = ak.sum(is_electron, axis=-1) == 1
        # # Broadcast to track level
        # one_electron, _ = ak.broadcast_arrays(one_electron_per_event, is_electron)
        
        # Apply trk-level selections 
        # trk = data["trk"][(is_electron & one_electron)]
        # trkfit = data["trkfit"][(is_electron & one_electron)]

        trk = data["trk"][is_electron]
        trkfit = data["trkfit"][is_electron]
        
        # Select tracker front
        at_trk_front = self.selector.select_surface(trkfit, surface_name="TT_Front")
        
        # Apply trkseg-level selections
        trkfit = trkfit[at_trk_front]  

        return trk, trkfit
    
    def _extract_data(self, data, param):
        """
        Extract histogram data based on parameter name
        
        Args:
            data: Input data array
            param: Name of param to extract (e.g., 'mom', 'trkqual', etc.)
            
        Returns:
            ak.Array: Flattened extracted data
        """        
        if param == "mom":
            trk, trkfit = self._prepare_track_data(data)
            mom = self.vector.get_mag(trkfit["trksegs"], "mom")
            return ak.flatten(mom, axis=None) if mom is not None else ak.Array([])
            
        elif param == "crv_z":
            return ak.flatten(data["crv"]["crvcoincs.pos.fCoordinates.fZ"], axis=None)
            
        elif param == "trkqual":
            trk, _ = self._prepare_track_data(data)
            return ak.flatten(trk["trkqual.result"], axis=None)
            
        elif param == "nactive":
            trk, _ = self._prepare_track_data(data)
            return ak.flatten(trk["trk.nactive"], axis=None)
            
        elif param == "t0err":
            _, trkfit = self._prepare_track_data(data)
            return ak.flatten(trkfit["trksegpars_lh"]["t0err"], axis=None)
            
        elif param == "d0":
            _, trkfit = self._prepare_track_data(data)
            return ak.flatten(trkfit["trksegpars_lh"]["d0"], axis=None)
            
        elif param == "maxr":
            _, trkfit = self._prepare_track_data(data)
            return ak.flatten(trkfit["trksegpars_lh"]["maxr"], axis=None)
            
        elif param == "pitch_angle":
            _, trkfit = self._prepare_track_data(data)
            pitch_angle = self.analyse.get_pitch_angle(trkfit)  # Use existing method
            return ak.flatten(pitch_angle, axis=None)
            
        elif param == "pdg":
            trk, _ = self._prepare_track_data(data)
            return ak.flatten(trk["trk.pdg"], axis=None)
            
        elif param == "sid":
            _, trkfit = self._prepare_track_data(data)
            return ak.flatten(trkfit["trksegs"]["sid"], axis=None)
            
        elif param == "pz":
            _, trkfit = self._prepare_track_data(data)
            return ak.flatten(trkfit["trksegs"]["mom"]["fCoordinates"]["fZ"], axis=None)
            
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
                     e.g., {"All": data, "CE-like": data_CE, "Unvetoed": data_CE_unvetoed}
            
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
                    values = self._extract_data(data, config["param"])
                    
                    # Apply any filters
                    filtered_values = self._apply_filter(values, config["filter"])
                    
                    # Skip if no data after filtering
                    if len(filtered_values) == 0:
                        continue
                    
                    # Fill the histogram
                    fill_kwargs = {
                        config["axis"].name: filtered_values,
                        "selection": np.full(len(filtered_values), selection_label)
                    }
                    histograms[hist_name].fill(**fill_kwargs)
                    
                except Exception as e:
                    self.logger.log(f"Error filling {hist_name} for {selection_label}: {e}", "error")
                    raise
        
        self.logger.log("Histograms filled successfully", "success")
        
        # Return copies to avoid reference issues
        return {name: hist_obj.copy() for name, hist_obj in histograms.items()}