# mu2e-cosmic

This repository provides a comphrehensive framework for Monte Carlo-based cosmic-ray-induced background analyses, and performance evaluation studies, for the Cosmic Ray Veto (CRV) system at the Fermilab Mu2e experiment. It deploy [`Mu2e/pyutils`](https://github.com/Mu2e/pyutils) to explore cut strategies for signal-like track selection and cosmic ray background sensitivy studies through reproducible workflows.

## Structure

Please click the dropdown for detailed information. 

<details>
<summary>Repository structure</summary>
```bash
mu2e-cosmic/
├── config/             # Configuration files
│   ├── ana/            # YAML configs for analyses
│   └── common/         # YAML cut flow definitions
│       └── cuts.yaml   # Shared cut definitions
├── src/                # Core analysis code
│   ├── core/           # Main processing pipeline
│   │   ├── analyse.py      # Core analysis workflow (Analyse, Utils classes)
│   │   └── process.py      # Data processing 
│   │   ├── postprocess.py  # Result combination and aggregation
│   └── utils/          # Helper utilities
│       ├── cut_manager.py  # Cut management and statistics
│       ├── io_manager.py   # Input/output handling
│       ├── draw.py         # Plotting utilities
│       ├── efficiency.py   # CRV efficiency calculations
│       ├── run_logger.py   # Logging configuration
│       └── mu2e.mplstyle   # Matplotlib style for plots
├── run/                # Scripts to execute analyses
│   ├── run.py          # Main analysis runner
│   ├── run.sh          # Shell script wrapper
│   └── run_test.py     # Test runner
├── notebooks/          # Jupyter notebooks
│   ├── ana/            # Analysis result exploration
│   │   ├── generate_notebook.py  # Notebook generator script
│   │   ├── ana_template.ipynb    # Template for analysis notebooks
│   │   └── ana_test.ipynb        # Test analysis notebook
│   ├── comp/           # Dataset comparisons
│   └── models/         # Statistical modeling
├── examples/           # Example inputs and outputs
│   ├── example_io.ipynb        # I/O demonstration
│   ├── example_parquet.py      # Parquet file handling
│   ├── input/                  # Sample input data
│   └── output/                 # Sample output data
├── scripts/           # Utility scripts
├── docs/              # Additional documentation
├── latex/             # LaTeX documentation generation
└── setup.sh           # Environment setup script
```
</details>

## Quick start

Firstly, setup your Python environment on Mu2e VMs or Fermilab Elastic Analysis Facility (EAF):

```
mu2einit # source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh
pyenv ana
```

Clone the repository:

``` 
git clone https://github.com/sam-grant/mu2e-cosmic.git
cd mu2e-cosmic
```

Now, if running with remote files (using `xrootd`), source the setup script 

```
source setup.sh
```

and run the analysis

``` 
cd mu2e-cosmic/run 
python run.py -c example.yaml
```

this will produce some relatively versose output, which can be controlled via YAML (see below), ending in 

```
[run] ✅ Analysis complete, wrote results to ../output/results/example
```

>**Note**: Analyses are configured by YAML files in `config/ana` and `config/common`. Here we are using an example conifiguration that runs an example cut flow with a single local file.

Then, explore the results interactively by generating a notebook:

```bash
cd ../notebooks/ana
python generate_notebook.py example
```

which will output

```
(current) [sgrant@jupyter-sgrant ana]$ python generate_notebook.py example
[generate_notebook] ⭐️ Created/using directory: example
[generate_notebook] ✅ Successfully generated notebook: example/example.ipynb
```

## Analysis configuration template

Analyses are configured via YAML, which pass arguments directly to the `process.py` and `postprocess.py` modules. Many of the arguments are optional, but a template containing the full suite of options is below. 


```YAML
# ==================================
# ANALYSIS CONFIGURATION TEMPLATE
# ==================================
# This YAML file configures a cosmic-ray-induced background analysis
# It is used to steer run/run.py via 
# 
# python run.py -c my_config.yaml
#
# Naming: <cutset_name>_<generator>_<spill-reco>_<ds_type>.yaml
# 
# Example: "alpha_CRY_onspill-LH_au.yaml"
#


# Configure src/core/process.py
process:    
     defname (str, opt): Dataset definition name. Defaults to None.
     file_list_path (str, opt): Path to file list. Defaults to None.
     file_name (str, opt): Single file to process. Defaults to None.
     cutset_name (str, opt): Which cutset to use. Defaults to "alpha".
     branches (dict or list, opt): EventNtuple branches to extract.
     on_spill (bool, opt): Apply on-spill timing cuts. Defaults to True.
     cuts_to_toggle (dict, opt): Cuts to enable/disable. Defaults to None.
     groups_to_toggle (dict, opt): Cut groups to enable/disable. Defaults to None.
     use_remote (bool, opt): Use remote file access. Defaults to True.
     location (str, opt): File location ("disk", etc.). Defaults to "disk".
     max_workers (int, opt): Number of parallel workers. Defaults to 50.
     use_processes (bool, opt): Use processes rather than threads. Defaults to True.
     verbosity (int, opt): Logging verbosity level. Defaults to 2 (max).
     worker_verbosity (int, opt): Verbosity for worker processes (debug only!). Defaults to 0.

# Configure src/core/postprocess.py
postprocess:
    on_spill (bool): Whether we are using on spill cuts. Propagated from Process by run.py
    write_events (bool, opt): write filtered events. Defaults to False.
    write_event_info (bool, opt): write filtered event info. Defaults to False.
    generated_events (int, opt): Number of generated events. Defaults to 4.11e10.
    livetime (float, opt): The total livetime in seconds for this dataset. Defaults to 1e7.
    on_spill_frac (dict, opt): Fraction of walltime in single and two batch onspill. Defaults to 32.2% and 24.6%.
    veto (bool, opt): Whether running with CRV veto. Defaults to True.
    verbosity (int, opt): Printout level. Defaults to 1.

output:
    out_path: "../output/results/<name>" 
```

## Cut flows

Cut flows are also configured by YAML, in `config/common/cuts.yaml`. The structure follows a base cutset which is inherited by the others, where the use can modify thresholds and toggle cuts and cut groups on/off. 

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
