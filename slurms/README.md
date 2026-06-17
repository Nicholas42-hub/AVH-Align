# Slurm scripts — by purpose

All SBATCH scripts have been categorized here. To submit, run from the project root:

```bash
cd /data/gpfs/projects/punim2637/nnliang/AVH-Align
sbatch slurms/<category>/<script>.slurm
```

(The internal `cd` paths in every script are relative to where `sbatch` is invoked, not where the .slurm file lives, so submission still works the same as before.)

## Layout

| Folder | Count | Purpose |
|--------|-------|---------|
| `train/`            | 21 | New-model training (A2, A3, A5, A6 base, CORAL, DANN, main3 baselines, AV1M-pseudodomain variants) |
| `unsup_triroute/`   |  3 | Unsupervised TriRoute sync experiments (paper / AV1M / Vox2 variants) — main current work |
| `retrain/`          | 11 | Continuations & re-trains of A6 pseudodomain runs (50ep, to100, clean50ep, etc.) |
| `eval/`             | 39 | All evaluations: FAVC 70/30, AV1M cross-domain, anomaly scoring, silence robustness, perturbation, head ablation eval, etc. |
| `probe/`            |  9 | Diagnostic probes: subspace, domain, label, head ablation training, routing map |
| `preprocess/`       |  5 | Feature extraction + preprocessing for FAVC / AVLips |
| `vox2_chunks/`      |  3 | VoxCeleb2 chunk preprocessing + feature extraction |
| `download/`         |  2 | Vox2 raw download (BaiduPCS / official) |

## Most-recent / actively-used scripts (May 3, 2026)

- `unsup_triroute/train_eval_triroute_unsup_sync_vox2_seeds.slurm` — current Vox2 unsup training (produced the 0.918 ckpt)
- `eval/eval_anomaly_three_ways_seed43.slurm` — Caren's three post-hoc methods
- `probe/label_probe_seed45_best.slurm` — label predictability probe (per-FAVC-subset)
- `retrain/retrain_A6_pdv2_clean50ep_seed{43,44,45}.slurm` — clean 50-epoch supervised baselines
- `eval/eval_swa_seed{43,44}.slurm` — SWA evaluation
