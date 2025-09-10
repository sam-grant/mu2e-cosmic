# Cutsets 

Cutsets are defined in `config/common/cuts.yaml`; cut definitions are defined in `src/core/analysis.py:Analyse().define_cuts`.

Analysis cutsets are based on the SU2020 analysis (https://arxiv.org/abs/2210.11380).

## Preselection 

All have common preselection cut group: 

* `has_trk_front`: Track intersects the tracker entrance
* `is_reco_electron`: Track fit uses electron hypothesis
* `one_reco_electron`: One electron fit per event
* `is_downstream`: Downstream tracks (p_z > 0 at tracker entrance)
* `is_truth_electron`: Truth PID (track parents are electrons)

## Signal-like background cuts 

All have common signal-like background cuts: 

* `unvetoed`: No veto: |dt| >= 150 ns
* `within_ext_win`: Extended window (100 < p < 110 MeV/c)
* `within_sig_win`: Signal window (103.6 < p < 104.9 MeV/c)

## Signal-like cuts 

Track-based signal-like cuts, using different combinations of trkqual and t0err. 

```
| trkqual \ t0err | < 0.9 ns |  > 0 (none) |
|-----------------|----------|-------------|
| > 0 (none)      | SU2020a  | SU2020d     |
| > 0.2           | SU2020b  | SU2020e     |
| > 0.8           | SU2020c  | SU2020f     |
```

For my own sanity I will stick primarily with SU2020b, which is the most comparable to baseline SU2020.

### SU2020a

`trkqual > 0 (none)` and `t0err < 0.9 ns`

* `good_trkqual`: Track quality result (trkqual > 0)
* `within_t0`: t0 at tracker entrance (640 < t_0 < 1650 ns)
* `within_t0err`: Track fit t0 uncertainty (t0err < 0.9 ns)
* `has_hits`: >20 active tracker hits
* `within_d0`: Distance of closest approach (d_0 < 100 mm)
* `within_pitch_angle_lo`: Extrapolated pitch angle (pz/pt > 0.5)
* `within_pitch_angle_hi`: Extrapolated pitch angle (pz/pt < 1.0)
* `within_lhr_max_hi`: Loop helix maximum radius (R_max < 680 mm)
  
### SU2020b

SU2020a, with `trkqual > 0.2`.

* `good_trkqual`: Track quality result (trkqual > 0.2)
* `within_t0`: t0 at tracker entrance (640 < t_0 < 1650 ns)
* `within_t0err`: Track fit t0 uncertainty (t0err < 0.9 ns)
* `has_hits`: >20 active tracker hits
* `within_d0`: Distance of closest approach (d_0 < 100 mm)
* `within_pitch_angle_lo`: Extrapolated pitch angle (pz/pt > 0.5)
* `within_pitch_angle_hi`: Extrapolated pitch angle (pz/pt < 1.0)
* `within_lhr_max_hi`: Loop helix maximum radius (R_max < 680 mm)
  
### SU2020c

SU2020a, with `trkqual > 0.8`.

* `good_trkqual`: Track quality result (trkqual > 0.8)
* `within_t0`: t0 at tracker entrance (640 < t_0 < 1650 ns)
* `within_t0err`: Track fit t0 uncertainty (t0err < 0.9 ns)
* `has_hits`: >20 active tracker hits
* `within_d0`: Distance of closest approach (d_0 < 100 mm)
* `within_pitch_angle_lo`: Extrapolated pitch angle (pz/pt > 0.5)
* `within_pitch_angle_hi`: Extrapolated pitch angle (pz/pt < 1.0)
* `within_lhr_max_hi`: Loop helix maximum radius (R_max < 680 mm)

### SU2020d

SU2020a, with `t0err > 0 (none)`.

* `good_trkqual`: Track quality result (trkqual > 0)
* `within_t0`: t0 at tracker entrance (640 < t_0 < 1650 ns)
* `within_t0err`: Track fit t0 uncertainty (t0err > 0 ns)
* `has_hits`: >20 active tracker hits
* `within_d0`: Distance of closest approach (d_0 < 100 mm)
* `within_pitch_angle_lo`: Extrapolated pitch angle (pz/pt > 0.5)
* `within_pitch_angle_hi`: Extrapolated pitch angle (pz/pt < 1.0)
* `within_lhr_max_hi`: Loop helix maximum radius (R_max < 680 mm)
  
### SU2020e

SU2020a, with `trkqual > 0.2` and `t0err < 0 (none)`.

* `good_trkqual`: Track quality result (trkqual > 0.2)
* `within_t0`: t0 at tracker entrance (640 < t_0 < 1650 ns)
* `within_t0err`: Track fit t0 uncertainty (t0err > 0 ns)
* `has_hits`: >20 active tracker hits
* `within_d0`: Distance of closest approach (d_0 < 100 mm)
* `within_pitch_angle_lo`: Extrapolated pitch angle (pz/pt > 0.5)
* `within_pitch_angle_hi`: Extrapolated pitch angle (pz/pt < 1.0)
* `within_lhr_max_hi`: Loop helix maximum radius (R_max < 680 mm)

### SU2020f

SU2020a, with trkqual > 0.8 and t0err > 0 (none).

* `good_trkqual`: Track quality result (trkqual > 0.8)
* `within_t0`: t0 at tracker entrance (640 < t_0 < 1650 ns)
* `within_t0err`: Track fit t0 uncertainty (t0err > 0 ns)
* `has_hits`: >20 active tracker hits
* `within_d0`: Distance of closest approach (d_0 < 100 mm)
* `within_pitch_angle_lo`: Extrapolated pitch angle (pz/pt > 0.5)
* `within_pitch_angle_hi`: Extrapolated pitch angle (pz/pt < 1.0)
* `within_lhr_max_hi`: Loop helix maximum radius (R_max < 680 mm)