# Run analysis

`run.py` runs the core processing chain, configured by a YAML configuration file in `config/ana`. 

Example, from this directory run:

```bash
python run.py --config test.yaml
```

Once done, navigate to notebooks/ana and create a notebook to display your results:

```bash
cd ../notebooks/ana
python generate_notebook.c test 
```

Full example output for `run.py`: 

```
(ana_v2.2.0) [sgrant@jupyter-sgrant run]$ python run.py -c alpha_CRY_offspill-LH_as.yaml
[ConfigManager] ✅ Loading config from: /home/sgrant/mu2e-cosmic/src/core/../utils/../../config/ana/alpha_CRY_offspill-LH_as.yaml
[ConfigManager] ⭐️ Analysis configuration:
output:
  out_path: ../output/results/alpha_CRY_offspill-LH_as
postprocess:
  write_event_info: true
  write_events: true
process:
  defname: nts.mu2e.CosmicCRYSignalAllOffSpillTriggered-LH.MDC2020as_best_v1_3_v06_03_00.root
  max_workers: 50

[Skeleton] ⭐️ Skeleton init
[CosmicProcessor] ✅ Initialised with:
                defname         = nts.mu2e.CosmicCRYSignalAllOffSpillTriggered-LH.MDC2020as_best_v1_3_v06_03_00.root
                file_list_path  = None
                file_name       = None
                cutset_name     = alpha
                on_spill        = False
                cuts_to_toggle  = None
                branches        = ['evt', 'crv', 'trk', 'trkfit', 'trkmc']
                use_remote      = True
                location        = disk
                max_workers     = 50
                verbosity       = 1
                use_processes   = True
[CosmicProcessor] ⭐️ Starting analysis
[pyutils] ⭐️ Setting up...
[pyutils] ✅ Ready
[pyprocess] ⭐️ Initialised Processor:
        path = 'EventNtuple/ntuple'
        use_remote = True
        location = disk
        schema = root
        verbosity=1
[pyprocess] ✅ Successfully loaded file list
        SAM definition: nts.mu2e.CosmicCRYSignalAllOffSpillTriggered-LH.MDC2020as_best_v1_3_v06_03_00.root
        Count: 817 files
[pyprocess] ⭐️ Starting processing on 817 files with 50 processes
Processing: 100%|██████████████████████████████| 817/817 [02:16<00:00,  6.00file/s, successful=817, failed=0]
[pyprocess] ⭐️ Returning 817 results
[CosmicProcessor] ✅ Analysis complete
[Efficiency] ⭐️ Initialised
[PostProcess] ⭐️ Initialised
[CutManager] ✅ Combined and formatted cut flows
[PostProcess] ✅ Combined 4 histograms over 817 results
[Efficiency] ⭐️ Getting efficiency from histogams
[Efficiency] ✅ Returning efficiency information
[PostProcess] ✅ Calculated efficiency:
[PostProcess] ✅ Combined arrays, result contains 7 events
[PostProcess] ✅ Retrieved background event info
[PostProcess] ✅ Postprocessing complete:
        Returning dict of combined cut flows, histograms, filtered events, and filtered event info
[Write] ✅ Initialised with out_path=../output/results/alpha_CRY_offspill-LH_as
[Write] ⭐️ Saving all
[Write] ⭐️ Writing all results to pickle
[Write] ✅ Wrote results ../output/results/alpha_CRY_offspill-LH_as/results.pkl
[Write] ⭐️ Writing cut flow to csv
[Write] ✅ Wrote cut flow to ../output/results/alpha_CRY_offspill-LH_as/cut_flow.csv
[Write] ⭐️ Writing hists to h5
[Write] ✅ Wrote histograms to ../output/results/alpha_CRY_offspill-LH_as/hists.h5
[Write] ⭐️ Writing efficiency info to csv
[Write] ✅ Wrote efficiency to ../output/results/alpha_CRY_offspill-LH_as/efficiency.csv
[Write] ⭐️ Writing background events to parquet
[Write] ✅ Wrote events ../output/results/alpha_CRY_offspill-LH_as/events.parquet
[Write] ⭐️ Writing background info to txt
[Write] ✅ Wrote info to ../output/results/alpha_CRY_offspill-LH_as/info.txt
[run] ⭐️ Moved log file to ../output/results/alpha_CRY_offspill-LH_as/alpha_CRY_offspill-LH_as.log
[run] ✅ Analysis complete, wrote results to ../output/results/alpha_CRY_offspill-LH_as
```