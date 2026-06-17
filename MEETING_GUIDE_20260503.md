## Meeting Guide: 2026-05-03

This note is a quick orientation pass for Caren before reviewing the current repository state.

### What to look at first

- **Today's results & key checkpoint pointers:** [CURRENT_RESULTS.md](/data/projects/punim2637/nnliang/AVH-Align/CURRENT_RESULTS.md)
- Paper draft:
  - [paper/main.tex](/data/projects/punim2637/nnliang/AVH-Align/paper/main.tex)
- Main A6 / TriRoute code path:
  - [avh_sup/mlp_fcd_a6.py](/data/projects/punim2637/nnliang/AVH-Align/avh_sup/mlp_fcd_a6.py)
  - [avh_sup/train_test_fcd_a6_av1m_pseudodomain_v2.py](/data/projects/punim2637/nnliang/AVH-Align/avh_sup/train_test_fcd_a6_av1m_pseudodomain_v2.py)
- Recent unsupervised pilot path:
  - [avh_sup/train_eval_triroute_unsup_sync.py](/data/projects/punim2637/nnliang/AVH-Align/avh_sup/train_eval_triroute_unsup_sync.py)
  - [slurms/unsup_triroute/train_eval_triroute_unsup_sync_vox2_seeds.slurm](/data/projects/punim2637/nnliang/AVH-Align/slurms/unsup_triroute/train_eval_triroute_unsup_sync_vox2_seeds.slurm)

### Main themes of current work

- Finalizing the paper narrative around cross-domain routing, especially FV-RA.
- Continuing the A6 pseudodomain v2 supervised line for cleaner 50-epoch and 100-epoch comparisons.
- Running a small unsupervised VoxCeleb2 pilot:
  - `40k` subset
  - `20` epochs
  - later extended so it can print periodic cross-domain FAVC AUC during training

### Important repository areas

- `paper/`
  - Current manuscript and paper figures.
- `avh_sup/`
  - Main training, evaluation, probing, and figure-generation code.
- `slurms/`
  - All experiment entrypoints, organized by purpose (train/, eval/, retrain/, probe/, preprocess/, unsup_triroute/, vox2_chunks/, download/). See [slurms/README.md](/data/projects/punim2637/nnliang/AVH-Align/slurms/README.md).

### What is noise vs signal

Signal:

- `paper/main.tex`
- `avh_sup/*.py` related to A6, evaluation, unsup pilot, anomaly analysis
- selected `*.slurm` files for reproducible runs
- selected config YAMLs under `avh_sup/configs/`

Mostly noise / experiment products:

- `avh_sup/outputs_*`
- `avh_sup/anomaly_results_*`
- raw prediction CSVs and eval text dumps
- scratch-style helper binaries or download utilities

The `.gitignore` has been updated to hide a chunk of the new unsup output noise so that `git status` is easier to read during review.

### Unsup pilot status

- Full raw VoxCeleb2 archive exists locally, but the current pilot used only a processed `40k` subset.
- Current unsup training script now supports periodic cross-domain monitoring via:
  - `--cross_domain_eval_every N`
- The vox2 seed slurm currently uses:
  - `--cross_domain_eval_every 5`

### If reviewing results claims

The strongest current paper-facing evidence is still the supervised A6 / TriRoute cross-domain result set and routing analyses.  
The unsupervised Vox2 line should be read as pilot evidence, not yet as a fully scaled comparison against the stronger AVH-Align unsupervised baseline.
