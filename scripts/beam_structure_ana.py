# Mu2e Beam Structure Analysis 

def analyse_beam_mode(mode_name, supercycle_period, n_spills, spill_duration, 
                     interspill_gap, beam_off_period, initial_delay=0):
    """
    Analyse beam structure for a given mode
    
    Parameters:
    - mode_name: Name of the mode (e.g., "Single Batch", "Two Batch")
    - supercycle_period: Total supercycle time in ms
    - n_spills: Number of spills per supercycle
    - spill_duration: Duration of each spill in ms
    - interspill_gap: Gap between consecutive spills in ms
    - beam_off_period: Stated beam-off period in ms
    - initial_delay: Initial delay before first spill group in ms
    
    Returns:
    - Dictionary with calculated values
    """
    
    # Calculate timing components
    total_spill_time = n_spills * spill_duration
    total_interspill_time = (n_spills - 1) * interspill_gap
    calculated_beam_off = supercycle_period - total_spill_time - total_interspill_time
    
    # Total on-spill and off-spill times
    onspill_time = total_spill_time
    offspill_time = supercycle_period - onspill_time
    
    # Calculate fractions
    onspill_fraction = onspill_time / supercycle_period
    offspill_fraction = offspill_time / supercycle_period
    
    # Return results dictionary
    results = {
        'mode_name': mode_name,
        'supercycle_period': supercycle_period,
        'n_spills': n_spills,
        'spill_duration': spill_duration,
        'interspill_gap': interspill_gap,
        'beam_off_period': beam_off_period,
        'initial_delay': initial_delay,
        'total_spill_time': total_spill_time,
        'total_interspill_time': total_interspill_time,
        'calculated_beam_off': calculated_beam_off,
        'onspill_time': onspill_time,
        'offspill_time': offspill_time,
        'onspill_fraction': onspill_fraction,
        'offspill_fraction': offspill_fraction
    }
    
    return results


# Define parameters for both modes
single_batch_params = {
    'mode_name': 'Single Batch',
    'supercycle_period': 1333,  # ms
    'n_spills': 4,
    'spill_duration': 107.3,    # ms
    'interspill_gap': 5.0,      # ms
    'beam_off_period': 889,     # ms
}

two_batch_params = {
    'mode_name': 'Two Batch',
    'supercycle_period': 1400,  # ms
    'n_spills': 8,
    'spill_duration': 43.1,     # ms
    'interspill_gap': 5.0,      # ms
    'beam_off_period': 1020,    # ms
    'initial_delay': 0          # ms
}

# Analyse both modes
single_batch_results = analyse_beam_mode(**single_batch_params)
two_batch_results = analyse_beam_mode(**two_batch_params)

# Comparative analysis
print("="*60)
print(f"{'Parameter':<25} {'Single Batch':<15} {'Two Batch':<15}")
print("-" * 60)
print(f"{'Supercycle [ms]':<25} {single_batch_results['supercycle_period']:<15} {two_batch_results['supercycle_period']:<15}") 
print(f"{'Number of spills':<25} {single_batch_results['n_spills']:<15} {two_batch_results['n_spills']:<15}")
print(f"{'Spill duration [ms]':<25} {single_batch_results['spill_duration']:<15} {two_batch_results['spill_duration']:<15}")
print(f"{'Total spill time [ms]':<25} {single_batch_results['total_spill_time']:<15.1f} {two_batch_results['total_spill_time']:<15.1f}")
print("-" * 60)
print(f"{'Onspill fraction':<25} {single_batch_results['onspill_fraction']:.3f} ({single_batch_results['onspill_fraction']*100:.1f}%){'':<3} {two_batch_results['onspill_fraction']:.3f} ({two_batch_results['onspill_fraction']*100:.1f}%){'':<3}") 
print(f"{'Offspill fraction':<25} {single_batch_results['offspill_fraction']:.3f} ({single_batch_results['offspill_fraction']*100:.1f}%){'':<3} {two_batch_results['offspill_fraction']:.3f} ({two_batch_results['offspill_fraction']*100:.1f}%){'':<3}")
print("="*60)
print()