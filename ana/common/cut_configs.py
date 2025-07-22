cut_configs = [
    # Baseline 
    {"nominal": None},  
    # Single parameter OFF
    {"within_lhr_max": False}, 
    {"within_d0": False},  
    {"within_pitch_angle": False},  
    # Two parameters OFF
    {"within_lhr_max": False, "within_d0": False}, 
    {"within_lhr_max": False, "within_pitch_angle": False}, 
    {"within_d0": False, "within_pitch_angle": False},  
    # All three parameters OFF
    {"within_d0": False, "within_lhr_max": False, "within_pitch_angle": False}
]