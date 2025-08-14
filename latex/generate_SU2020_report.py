from generate_report import generate_latex_report

ana_labels = [
    "SU2020a_CRY_offspill-LH_as",
    "SU2020a_CRY_onspill-LH_au", 
    "SU2020a_CRY_onspill-LH_aw",
    "SU2020a_signal_onspill-LH_au",
    "SU2020a_signal_onspill-LH_aw",
    "SU2020b_CRY_offspill-LH_as",
    "SU2020b_CRY_onspill-LH_au",
    "SU2020b_CRY_onspill-LH_aw", 
    "SU2020b_signal_onspill-LH_au",
    "SU2020b_signal_onspill-LH_aw",
    "SU2020c_CRY_offspill-LH_as",
    "SU2020c_CRY_onspill-LH_au",
    "SU2020c_CRY_onspill-LH_aw",
    "SU2020c_signal_onspill-LH_au",
    "SU2020c_signal_onspill-LH_aw"
]

# Generate the report
latex_file = generate_latex_report(ana_labels, "SU2020")
print(f"Report generated: {latex_file}")
