# mu2e-cosmic

Cosmic-ray-induced background analyses of Monte Carlo data for the Mu2e experiment at Fermilab.

## Usage

- Run the analysis: [run](run/README.md)
- Display the results: [notebooks/ana](notebooks/ana/README.md)


<!-- >Note: Recently did some major restructuring, improved documentation is on the way.

This repository contains code intended for **analyses of sensitivity to cosmic-ray-induced backgrounds at the Mu2e experiment and the performance of the Cosmic Ray Veto (CRV) system** using both on-spill and off-spill datasets. The analysis framework allows users to explore the cut strategies for signal-like track selection and cosmic ray induced background rejection. 

The tools in this repository were originally developed by **Sam Grant** and extended by **Victor Dorojan** as part of ongoing efforts to study sensitivity to cosmic-ray-induced backgrounds and optimise CRV efficiency measurements. The analysis aims to support detector studies by providing **reproducible workflows for cosmic background characterisation**. -->


<!-- 
```
├── config
│   ├── ana
│   │   ├── alpha_CRY_offspill-LH_as.yaml
│   │   ├── alpha_CRY_onspill-LH_au.yaml
│   │   ├── alpha_CRY_onspill-LH_aw.yaml
│   │   ├── alpha_CRY_signal-LH_aq.yaml
│   │   └── test.yaml
│   └── common
│       └── cuts.yaml
├── docs
├── examples
│   ├── example_io.ipynb
│   ├── example_parquet.py
│   ├── input
│   │   └── ana_alpha_CRY_offspill-LH_as
│   │       ├── alpha_CRY_offspill-LH_as.log
│   │       ├── alpha_CRY_offspill-LH_as.yaml
│   │       ├── events.parquet
│   │       ├── hists.h5
│   │       ├── info.txt
│   │       ├── results.pkl
│   │       └── stats.csv
│   └── output
│       └── example_io
│           └── events.parquet
├── notebooks
│   ├── ana
│   │   ├── ana_template.ipynb
│   │   ├── ana_test.ipynb
│   │   └── attic
│   │       ├── offspill
│   │       │   ├── ana_as_alpha.ipynb
│   │       │   ├── ana_as_beta.ipynb
│   │       │   ├── ana_as_victor_report.ipynb
│   │       │   ├── ana-Copy1.ipynb
│   │       │   ├── ana_cut_scan.ipynb
│   │       │   ├── ana_single.ipynb
│   │       │   ├── attic
│   │       │   │   ├── ana_0.ipynb
│   │       │   │   ├── ana_1.ipynb
│   │       │   │   ├── ana.BK.06-11-25.ipynb
│   │       │   │   ├── ana.BK.ipynb
│   │       │   │   ├── ana-Copy1.ipynb
│   │       │   │   ├── ana_cut_eff.ipynb
│   │       │   │   ├── ana_cut_scan_OLD.ipynb
│   │       │   │   ├── ana_inactive_cuts.ipynb
│   │       │   │   ├── comp.ipynb
│   │       │   │   ├── debug.ipynb
│   │       │   │   ├── electron_cut_2.ipynb
│   │       │   │   ├── electron_cut.ipynb
│   │       │   │   ├── inspect_event.ipynb
│   │       │   │   ├── pileup_cut.ipynb
│   │       │   │   ├── reflected_cut.ipynb
│   │       │   │   ├── trkqual_cut-Copy1.ipynb
│   │       │   │   ├── trkqual_cut.ipynb
│   │       │   │   └── veto_cut.ipynb
│   │       │   ├── bar_chart.png
│   │       │   ├── count_coin.ipynb
│   │       │   ├── cuts
│   │       │   ├── image.png
│   │       │   ├── plotting.ipynb
│   │       │   ├── process_as.py
│   │       │   ├── README.md
│   │       │   ├── run_cut_scan.py
│   │       │   ├── scripts
│   │       │   └── test_read_pickle.ipynb
│   │       ├── onspill
│   │       │   ├── ana_06-30-25.ipynb
│   │       │   ├── ana_au.ipynb
│   │       │   └── ana_aw.ipynb
│   │       └── signal
│   │           ├── ana_06-30-25.ipynb
│   │           ├── ana_aq.ipynb
│   │           ├── ana_au_TODO.ipynb
│   │           ├── ana_aw_TODO.ipynb
│   │           ├── ana_cut_scan.ipynb
│   │           └── run_cut_scan.py
│   ├── comp
│   │   ├── count_tracks.ipynb
│   │   ├── tandip.ipynb
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
├── README.md
├── run
│   ├── alpha_CRY_offspill-LH_as.log
│   ├── alpha_CRY_onspill-LH_au.log
│   ├── alpha_CRY_onspill-LH_aw.log
│   ├── alpha_CRY_signal-LH_aq.log
│   ├── run.py
│   ├── run.sh
│   └── run_test.py
├── setup.sh
├── src
│   ├── attic
│   │   ├── cut_scan_configs.py
│   │   └── write.py
│   ├── core
│   │   ├── analyse.py
│   │   ├── cut_manager.py
│   │   ├── __init__.py
│   │   ├── io_manager.py
│   │   ├── postprocess.py
│   │   ├── process.py
│   │   └── __pycache__
│   │       ├── analyse.cpython-312.pyc
│   │       ├── cut_manager.cpython-312.pyc
│   │       ├── io_manager.cpython-310.pyc
│   │       ├── io_manager.cpython-312.pyc
│   │       ├── postprocess.cpython-312.pyc
│   │       └── process.cpython-312.pyc
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── analyse.cpython-312.pyc
│   │   └── cut_manager.cpython-312.pyc
│   └── utils
│       ├── draw.py
│       ├── efficiency.py
│       ├── __init__.py
│       ├── mu2e.mplstyle
│       ├── __pycache__
│       │   └── run_logger.cpython-312.pyc
│       └── run_logger.py
``` -->

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
