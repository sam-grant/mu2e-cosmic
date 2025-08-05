# mu2e-cosmic

```
├── setup.sh
├── README.md
├── src
│   ├── core
│   │   ├── analyse.py
│   │   ├── cut_manager.py
│   │   ├── __init__.py
│   │   ├── io_manager.py
│   │   ├── postprocess.py
│   │   ├── process.py
│   ├── __init__.py
│   └── utils
│       ├── draw.py
│       ├── efficiency.py
│       ├── __init__.py
│       ├── mu2e.mplstyle
├── config
│   ├── ana
│   │   ├── alpha_CRY_offspill-LH_as.yaml
│   │   ├── alpha_CRY_onspill-LH_au.yaml
│   │   ├── alpha_CRY_onspill-LH_aw.yaml
│   │   ├── alpha_CRY_signal-LH_aq.yaml
│   │   └── test.yaml
│   └── common
│       └── cuts.yaml
├── run
│   ├── alpha_CRY_offspill-LH_as.log
│   ├── alpha_CRY_onspill-LH_au.log
│   ├── alpha_CRY_onspill-LH_aw.log
│   ├── alpha_CRY_signal-LH_aq.log
│   ├── run.py
│   ├── run.sh
│   └── run_test.py
├── notebooks
│   ├── ana
│   │   ├── ana_template.ipynb
│   │   ├── ana_test.ipynb
│   ├── comp
│   │   ├── count_tracks.ipynb
│   │   ├── tandip.ipynb
│   │   ├── test.csv
│   │   ├── test_lhrmax_indexing.ipynb
│   │   ├── track_cuts_1.ipynb
│   │   ├── track_cuts_2.ipynb
│   │   └── trkqual_and_t0.ipynb
│   └── models
│       ├── 1e-4efficiencyplot.ipynb
│       ├── ineff_uncertainty_wilson.png
│       ├── IneffUncvsWallTime.png
│       ├── ineff_vs_walltime.ipynb
│       ├── IneffvWalltime.ipynb
│       ├── walltime_offspill_as_alpha.ipynb
│       └── walltime_offspill_as_beta.ipynb
├── output
│   ├── images -> /exp/mu2e/data/users/sgrant/mu2e-cosmic/images
│   ├── logs -> /exp/mu2e/data/users/sgrant/mu2e-cosmic/logs
│   └── results -> /exp/mu2e/data/users/sgrant/mu2e-cosmic/results
```

<!-- ## Overview

mu2e-cosmic/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── core/
│   └── utils/              
│       ├── __init__.py
│       ├── plotting.py
│       ├── data.py
│       └── helpers.py
├── run/
├── notebooks/
└── config/



This repository contains code intended for **analyses of sensitivity to cosmic-ray-induced backgrounds at the Mu2e experiment and the performance of the Cosmic Ray Veto (CRV) system** using both on-spill and off-spill datasets. The analysis framework allows users to explore the cut strategies for signal-like track selection and cosmic ray induced background rejection. 

The tools in this repository were originally developed by **Sam Grant** and extended by **Victor Dorojan** as part of ongoing efforts to study sensitivity to cosmic-ray-induced backgrounds and optimise CRV efficiency measurements. The analysis aims to support detector studies by providing **reproducible workflows for cosmic background characterisation**.

---

## Contents

1. `offspill/` – Off-spill cosmic ray data analysis in `ana.ipynb`, also includes `run_cut_scan.py` and `ana_cut_scan.ipynb` for studying cut configurations
1. `onspill/` – On-spill cosmic ray data analysis in `ana.ipynb`
1. `signal/` – On-spill beam data analysis in `ana.ipynb`
1. `common/` – Core analysis utilities (`analyse.py`, `cut_manager.py`, `postprocess.py`)
1. `comp/` – Comparison between different datasets or configurations, particularly in comparing the impact of track cuts between signal and cosmic datasets
1. `models/` – Statisical modelling of measured CRV efficiency against wall time 

---

### Core components

- **`common/analyse.py`**  
    Defines `Analyse`, which houses the core analysis workflow, and `Utils`, which contains helper methods Processes particle tracking data and applies selection cuts to identify electron tracks using both truth-level and reconstructed information. It sets up logging, selection utilities, and prepares the data for further analysis or plotting.

- **`common/cut_manager.py`**  
  Defines the `CutManager` class used to manage, apply, and analyse cuts. It allows cuts to be added, toggled on/off, combined, and used to produce detailed statistics on event selection.

- **`common/postprocess.py`**  
  Defines a `PostProcess` class that consolidates filtered data, histograms, and cut statistics from multiple analysis result files. It merges awkward arrays, combines histograms, and aggregates cut statistics using `CutManager`.

---

## Users can

- Toggle cut parameters to suit their own analysis goals  
- Reproduce and extend results from existing studies (`offspill/` and `onspill/`)  
- Analyse background rates across datasets (`signal/`)  
- Compare the impact of track cuts between datasets (`comp/`)
- Model efficiency over time (`/models`)
  
---

This toolkit is designed to be **modular** and **user-friendly** for collaborators working on Mu2e cosmic-ray-induced backgrounds. -->
