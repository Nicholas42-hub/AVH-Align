# Submission Prep Assets

This folder collects the concrete artifacts needed for the next NeurIPS-facing cleanup:

1. `A2/A5/A6` 3-seed sweeps
2. explicit frozen-eval protocol unification
3. published FakeAVCeleb comparison table
4. a leakage-control A6 ablation
5. a concise NeurIPS submission plan

## 1. Build explicit matched FakeAVCeleb splits

The current repo mixes original FakeAVCeleb metadata with whatever NPZ features were actually extracted.  
Materialize the matched CSVs once:

```bash
cd /data/projects/punim2637/nnliang/AVH-Align/avh_sup
python build_favc_matched_splits.py
```

This creates:

- `avh_sup/csv_metadata/favc_npz_matched/train_split.csv`
- `avh_sup/csv_metadata/favc_npz_matched/val_split.csv`
- `avh_sup/csv_metadata/favc_npz_matched/test_split.csv`
- `avh_sup/csv_metadata/favc_npz_matched/summary.json`

Use this CSV root for all future frozen FakeAVCeleb evaluations.

## 2. Main 3-model seed sweep

Prepared Slurm files:

- [eval_main3_base_favc_matched.slurm](/data/gpfs/projects/punim2637/nnliang/AVH-Align/eval_main3_base_favc_matched.slurm)
- [train_main3_multiseed.slurm](/data/gpfs/projects/punim2637/nnliang/AVH-Align/train_main3_multiseed.slurm)
- [eval_main3_multiseed_favc_matched.slurm](/data/gpfs/projects/punim2637/nnliang/AVH-Align/eval_main3_multiseed_favc_matched.slurm)

These cover:

- base `A2 / A5 / A6` re-eval under the explicit matched protocol
- `A2 × {43,44,45}`
- `A5 × {43,44,45}`
- `A6 × {43,44,45}`

The A6 training entry-point now supports `--seed` and `--output_dir`, matching A2/A5.

## 3. Leakage-control A6 ablation

Prepared config + Slurm:

- [A6_leakage_trainreal.yaml](/data/gpfs/projects/punim2637/nnliang/AVH-Align/avh_sup/configs/A6_leakage_trainreal.yaml)
- [train_A6_leakage_trainreal.slurm](/data/gpfs/projects/punim2637/nnliang/AVH-Align/train_A6_leakage_trainreal.slurm)
- [eval_A6_leakage_trainreal_favc_matched.slurm](/data/gpfs/projects/punim2637/nnliang/AVH-Align/eval_A6_leakage_trainreal_favc_matched.slurm)

This keeps A6 fixed but switches the domain source from `FAVC val real-only` to `FAVC train real-only` using the explicit NPZ-matched CSVs.

## 4. Collect standardized results

Prepared scripts:

- [collect_main3_seed_results.py](/data/gpfs/projects/punim2637/nnliang/AVH-Align/submission_prep/collect_main3_seed_results.py)
- [collect_unified_frozen_protocol.py](/data/gpfs/projects/punim2637/nnliang/AVH-Align/submission_prep/collect_unified_frozen_protocol.py)

They write:

- `submission_prep/main3_seed_results_raw.csv`
- `submission_prep/main3_seed_results_summary.csv`
- `submission_prep/unified_frozen_protocol_main.csv`

## 5. Published FakeAVCeleb table

Prepared CSV:

- [published_fakeavceleb_sota.csv](/data/gpfs/projects/punim2637/nnliang/AVH-Align/submission_prep/published_fakeavceleb_sota.csv)

This file explicitly separates:

- `intra_dataset` results
- `cross_dataset` results

and includes protocol notes so we do not accidentally compare incompatible settings.

## 6. NeurIPS submission plan

- [NEURIPS_SUBMISSION_PLAN.md](/data/gpfs/projects/punim2637/nnliang/AVH-Align/submission_prep/NEURIPS_SUBMISSION_PLAN.md)

This file records:

- which benchmark should be the paper's main result
- which models belong in the main text vs appendix
- which tasks are still worth finishing before submission

## 7. Leading-silence correction

- [leading_silence_true_trim_results.csv](/data/gpfs/projects/punim2637/nnliang/AVH-Align/submission_prep/leading_silence_true_trim_results.csv)
- [LEADING_SILENCE_RESULTS.md](/data/gpfs/projects/punim2637/nnliang/AVH-Align/submission_prep/LEADING_SILENCE_RESULTS.md)

These files record the corrected paper-facing interpretation:

- the feature-prefix zeroing runs are diagnostic only
- the true-trimmed runs (`favc_features_trimmed`) are the final leading-silence result
- `FV-RA` stays stable after true trimming, especially for `A6`

## 8. Supervisor-facing briefing

- [CAREN_SUPERVISOR_UPDATE.md](/data/gpfs/projects/punim2637/nnliang/AVH-Align/submission_prep/CAREN_SUPERVISOR_UPDATE.md)

This file is a one-page briefing for presenting the current results to Caren:

- the stable paper-facing main result
- the minimum mechanistic evidence to show
- which recent results are promising but still higher-variance
- a claim-safe way to summarize the project in a meeting

## 9. Paper-story overview

- [PAPER_STORY_OVERVIEW.md](/data/gpfs/projects/punim2637/nnliang/AVH-Align/submission_prep/PAPER_STORY_OVERVIEW.md)

This file explains the end-to-end logic of the project:

- motivation
- hypothesis
- why `A2 -> A3 -> A5 -> A6` is the right model progression
- how each analysis experiment validates one step of the hypothesis
- which benchmarks are main vs supporting
