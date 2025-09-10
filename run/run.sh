#!/bin/bash

# # Define analysis labels array
# ana_labels=(
#     # "alpha_CRY_offspill-LH_as"
#     # "alpha_CRY_onspill-LH_aq"
#     # "alpha_CRY_onspill-LH_au"
#     # "alpha_CRY_onspill-LH_aw"
#     # "alpha_signal_onspill-LH_aq"
#     # "alpha_signal_onspill-LH_au"
#     # # "alpha_signal_onspill-LH_aw"
#     # "SU2020a_CRY_offspill-LH_as"
#     # # "SU2020a_CRY_onspill-LH_aq"
#     # "SU2020a_CRY_onspill-LH_au"
#     # "SU2020a_CRY_onspill-LH_aw"
#     # # "SU2020a_signal_onspill-LH_aq"
#     # "SU2020a_signal_onspill-LH_au"
#     # "SU2020a_signal_onspill-LH_aw"
#     # "SU2020b_CRY_offspill-LH_as"
#     # # "SU2020b_CRY_onspill-LH_aq"
#     # "SU2020b_CRY_onspill-LH_au"
#     # "SU2020b_CRY_onspill-LH_aw"
#     # # "SU2020b_signal_onspill-LH_aq"
#     # "SU2020b_signal_onspill-LH_au"
#     # "SU2020b_signal_onspill-LH_aw"
#     # "SU2020c_CRY_offspill-LH_as"
#     # "SU2020c_CRY_onspill-LH_aq"
#     # "SU2020c_CRY_onspill-LH_au"
#     # "SU2020c_CRY_onspill-LH_aw"
#     # # "SU2020c_signal_onspill-LH_aq"
#     # "SU2020c_signal_onspill-LH_au"
#     # "SU2020c_signal_onspill-LH_aw"
#     # "preselection_CRY_offspill-LH_as"
#     # "preselection_CRY_onspill-LH_au"
#     # "preselection_CRY_onspill-LH_aw"
#     # # "preselection_signal_onspill-LH_aq"
#     # # "preselection_signal_onspill-LH_au"
#     # "preselection_signal_onspill-LH_aw"
#     # "SU2020d_CRY_offspill-LH_as"
#     # "SU2020d_CRY_onspill-LH_au"
#     # "SU2020d_CRY_onspill-LH_aw"
#     # "SU2020d_signal_onspill-LH_au"
#     # "SU2020d_signal_onspill-LH_aw"
#     # "SU2020e_CRY_offspill-LH_as"
#     # "SU2020e_CRY_onspill-LH_au"
#     "SU2020e_CRY_onspill-LH_aw"

#     # "SU2020a_CRY_offspill-LH_as"
#     # "SU2020d_signal_onspill-LH_au"
#     # "SU2020d_signal_onspill-LH_aw"
    
#     # "SU2020e_signal_onspill-LH_au"
#     "SU2020e_signal_onspill-LH_aw"
    
#     # "SU2020e_CRY_offspill-LH_as"
#     # "SU2020e_CRY_onspill-LH_au"
#     # "SU2020e_CRY_onspill-LH_aw"
#     # "SU2020e_signal_onspill-LH_au"
#     # "SU2020e_signal_onspill-LH_aw"
# )

# Define analysis labels array
# ana_labels=(
#     "SU2020a_CRY_onspill-LH_aw"
#     "SU2020a_signal_onspill-LH_aw"
#     "SU2020b_CRY_onspill-LH_aw"
#     "SU2020b_signal_onspill-LH_aw"
#     "SU2020c_CRY_onspill-LH_aw"
#     "SU2020c_signal_onspill-LH_aw"
#     "SU2020d_CRY_onspill-LH_aw"
#     "SU2020d_signal_onspill-LH_aw"
#     "SU2020e_CRY_onspill-LH_aw"
#     "SU2020e_signal_onspill-LH_aw"
# )

ana_labels=(
    "preselection_CRY_onspill-LH_aw_bug"
    "preselection_signal_onspill-LH_aw_bug"
)

# Process each analysis label
for ana_label in "${ana_labels[@]}"; do
    config_file="${ana_label}.yaml"
    python run.py -c $config_file
    # echo 'python run.py -c "${ana_label}.yaml"'
    # echo $config_file
    # Wait for all jobs to finish
    wait 
done