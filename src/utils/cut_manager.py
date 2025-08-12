import json
import awkward as ak
import csv
import pandas as pd
from pyutils.pylogger import Logger

class CutManager:
    """Class to manage analysis cuts"""
    
    def __init__(self, verbosity=1):
        """Initialise 
        
        Args:
            verbosity (int, optional): Printout level (0: minimal, 1: normal, 2: detailed)
        """
        # Init cut object
        self.cuts = {}
        # Start logger
        self.logger = Logger( 
            verbosity=verbosity,
            print_prefix="[CutManager]"
        )
    
    def add_cut(self, name, description, mask, active=True):
        """
        Add a cut to the collection.
        
        Args: 
            name (str): Name of the cut
            description (str): Description of what the cut does
            mask (awkward.Array): Boolean mask array for the cut
            active (bool, optional): Whether the cut is active by default
        """

        # Get the next available index
        next_idx = len(self.cuts)
    
        self.cuts[name] = {
            "description": description,
            "mask": mask,
            "active": active,
            "idx" : next_idx
        }

        self.logger.log(f"Added cut {name} with index {next_idx}", "info")
        # return self # This would allow method chaining, could be useful maybe?

    def toggle_cut(self, cut_dict):
        """Utility to set cut(s) as inactive or active based on input dictionary
        
        Args: 
            cut_dict (dict): Dictionary mapping cut names to their desired active state
                            e.g., {"cut_name_1": False, "cut_name_2": True}
        """
        # Validate input type
        if not isinstance(cut_dict, dict):
            self.logger.log(f"Invalid input type: expected dict, got {type(cut_dict)}", "error")
            return False
        
        # Process each cut name and state
        success = True
        bad_cuts = []
        activated_cuts = []
        deactivated_cuts = []
        
        for cut_name, active_state in cut_dict.items():
            if cut_name in self.cuts:
                self.cuts[cut_name]["active"] = active_state
                if active_state:
                    activated_cuts.append(cut_name)
                else:
                    deactivated_cuts.append(cut_name)
            else:
                bad_cuts.append(cut_name)
                success = False
        
        # Log results
        if len(bad_cuts) > 0:
            self.logger.log(f"Cut(s) not valid: {bad_cuts}", "error")
        
        if len(activated_cuts) > 0:
            self.logger.log(f"Successfully activated cut(s): {activated_cuts}", "info")
        
        if len(deactivated_cuts) > 0:
            self.logger.log(f"Successfully deactivated cut(s): {deactivated_cuts}", "info")
        
        return success
    
    # def get_active_cuts(self):
    #     """Utility to get all active cutss"""
    #     return {name: cut for name, cut in self.cuts.items() if cut["active"]}
    
    def combine_cuts(self, cut_names=None, active_only=True):
        """ Return a Boolean combined mask from specified cuts. Applies an AND operation across all cuts. 
        Args: 

        cut_names (list, optional): List of cut names to include (if None, use all cuts)
        active_only (bool, optional): Whether to only include active cuts
        """        
        if cut_names is None:
            # Then use all cuts in original order
            cut_names = list(self.cuts.keys())
        # Init mask
        combined = None
        # Loop through cuts        
        for name in cut_names:
            # Get info dict for this cut
            cut_info = self.cuts[name]
            # Active cuts
            if active_only and not cut_info["active"]:
                continue
            # If first cut, initialise 
            if combined is None:
                combined = cut_info["mask"]
            else:
                combined = combined & cut_info["mask"] 
        
        return combined

    ############################################
    # Generate and manage cut flows
    ############################################
    
    def _add_entry(self, name, events_passing, absolute_frac, relative_frac, description):
        return {
            "name": name,
            "events_passing": int(events_passing),
            "absolute_frac": round(float(absolute_frac), 2),
            "relative_frac": round(float(relative_frac), 2),
            "description": description
        }
            
    def create_cut_flow(self, data):
        """ Utility to calculate cut flow from array and cuts object
        
        Args:
            data (awkward.Array): Input data 
        """
        total_events = len(data)
        cut_flow = []
        
        # Base statistics (no cuts)
        cut_flow.append(
            self._add_entry(
                name = "No cuts",
                events_passing = total_events,
                absolute_frac =  100.00,
                relative_frac = 100.00,
                description = "No selection applied",
            )
        )
        
        # Get cuts, filter by active status
        cuts = [name for name in self.cuts.keys() if self.cuts[name]["active"]]

        # Initialise cumulative mask
        cumulative_mask = None
        
        for name in cuts:
            # Get info for this cut
            cut_info = self.cuts[name]
            # Get mask for this cut
            mask = cut_info["mask"]

            # Combine cuts progressively
            if cumulative_mask is None:
                # First cut
                current_mask = mask
            else:
                # Apply this cut on top of previous cuts
                current_mask = cumulative_mask & mask
                
            # Redefine cumulative mask    
            cumulative_mask = current_mask

            # Calculate event-level efficiency
            event_mask = ak.any(current_mask, axis=-1) # events that have ANY True combined mask
            events_passing = ak.sum(event_mask) # Count up these events
            absolute_frac = events_passing / total_events * 100
            relative_frac = (events_passing / cut_flow[-1]["events_passing"] * 100 
                           if cut_flow[-1]["events_passing"] > 0 else 0)

            # Append row
            cut_flow.append(
                self._add_entry(
                    name = name,
                    events_passing = events_passing,
                    absolute_frac = absolute_frac,
                    relative_frac = relative_frac,
                    description = cut_info["description"],
                )
            )

        return cut_flow

    def format_cut_flow(self, cut_flow):
        """Format cut flow as a DataFrame with more readable column names

            Args:
                cut_flow (dict): The cut flow to format
            Returns:
                df_cut_flow (pd.DataFrame)
        """
        df_cut_flow = pd.DataFrame(cut_flow)
        df_cut_flow = df_cut_flow.rename(columns={
            "name": "Cut",
            "events_passing": "Events Passing",
            "absolute_frac": "Absolute [%]", 
            "relative_frac": "Relative [%]",
            "description": "Description"
        })
        return df_cut_flow
        
    def combine_cut_flows(self, cut_flow_list, format_as_df=True):
        """Combine a list of cut flows after multiprocessing 
        
        Args:
            cut_flows: List of cut statistics lists from different files
            format_as_df (bool, optional): Format output as a pd.DataFrame. Defaults to True.
        
        Returns:
            list: Combined cut statistics
        """        
        # Return empty list if no input
        if not cut_flow_list:
            self.logger.log(f"No cut flows to combine", "error")
            return []
        
        # # Filter ALL stats lists based on active_only flag, not just the template??
        # if active_only:
        #     filtered_stats_list = []
        #     for stats in stats_list:
        #         filtered_stats = []
        #         for cut in stats:
        #             # Always include "No cuts" entry
        #             if cut["name"] == "No cuts":
        #                 filtered_stats.append(cut)
        #             # Include only active cuts
        #             elif cut.get("active", True):
        #                 filtered_stats.append(cut)
        #         filtered_stats_list.append(filtered_stats)
        #     stats_list = filtered_stats_list

        try:
            # Use the first (now filtered) list as template
            template = cut_flow_list[0]
            
            # Use the template to initialise combined stats
            combined_cut_flow = []
            for cut in template:
                # Create a copy (needed?)
                cut_copy = {k: v for k, v in cut.items()}
                # Reset the event count
                cut_copy["events_passing"] = 0
                combined_cut_flow.append(cut_copy)
            
            # Create a mapping of cut names to indices in combined_stats 
            cut_name_to_index = {cut["name"]: i for i, cut in enumerate(combined_cut_flow)}
            
            # Sum up events_passing for each cut across all files
            for cut_flow in cut_flow_list:
                for cut in cut_flow:
                    cut_name = cut["name"]
                    # Only process cuts that are in our combined_stats
                    if cut_name in cut_name_to_index:
                        idx = cut_name_to_index[cut_name]
                        combined_cut_flow[idx]["events_passing"] += cut["events_passing"]
            
            # Recalculate percentages
            if combined_cut_flow and combined_cut_flow[0]["events_passing"] > 0:
                total_events = combined_cut_flow[0]["events_passing"]
                
                for i, cut in enumerate(combined_cut_flow):
                    events = cut["events_passing"]
                    
                    # Absolute percentage
                    cut["absolute_frac"] = (events / total_events) * 100.0
                    
                    # Relative percentage
                    if i == 0:  # "No cuts"
                        cut["relative_frac"] = 100.0
                    else:
                        prev_events = combined_cut_flow[i-1]["events_passing"]
                        cut["relative_frac"] = (events / prev_events) * 100.0 if prev_events > 0 else 0.0
    
            if format_as_df:
                self.logger.log(f"Combined and formatted cut flows", "success")
                return self.format_cut_flow(combined_cut_flow)
            else:
                self.logger.log(f"Combined cut flows", "success")
                combined_cut_flow
        
        except Exception as e:
            self.logger.log(f"Exception when combining cut flows: {e}", "error")
            raise
            