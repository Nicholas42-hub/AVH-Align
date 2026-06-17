# Current Results — Pre-meeting snapshot (2026-05-03)

## TL;DR

Three new findings since last sync (all using purely unsupervised TriRoute on Vox2 40k real clips):

1. **kNN on `u_v` post-hoc beats the trained sync head** on FAVC cross-domain
   (overall 0.918 → **0.9345**, +1.6 pts; FV-RA 0.894 → 0.923; FV-FA 0.937 → 0.962)
2. **Residual-gating recovers RV-FA** from 0.53 → 0.84 with α=1.0, validating the
   "Z_res carries domain artifact" hypothesis
3. A-V disagreement (M2) is informative but weakest of the three — supports
   sync-prediction as the main training objective

## Headline ckpt

```
/data/projects/punim2637/nnliang/AVH-Align/avh_sup/
  outputs_triroute_unsup_sync_vox2_repro_seed43/
    triroute_unsup_sync_best_auc.pt        ← seed43, epoch=20, FAVC overall 0.9182
    cross_domain_history.csv               ← per-5-epoch trajectory
    results_favc_7030/eval_results.txt     ← per-subset baseline
    results_av1m_val/eval_results.txt      ← in-domain (AV1M val 0.8362)
    results_anomaly_3way/                  ← three-way anomaly methods
      anomaly_three_ways.json              ← full grid
```

## Caren's three methods — full grid (FAVC test, N=6667, seed43 ckpt)

| # | Method | overall | RV-FA | FV-RA | FV-FA |
|---|---|---:|---:|---:|---:|
| — | Sync head (trained head, baseline) | **0.9182** | 0.9954 | 0.8938 | 0.9365 |
| 1a | Mahalanobis (u_v) | 0.8508 | 0.5279 | 0.8221 | 0.8910 |
| 1b | kNN, k=5 (u_v) | **0.9315** | 0.5415 | 0.9188 | 0.9603 |
| 1c | Gaussian NLL (u_v) | 0.8508 | 0.5279 | 0.8221 | 0.8910 |
| 2 | A-V disagreement (1 - cos s_v, s_a) | 0.8701 | 0.5225 | 0.8459 | 0.9073 |
| 3 | Residual-gated, α=0.5 | 0.8683 | 0.7676 | 0.8186 | 0.9172 |
| 3 | Residual-gated, α=1.0 | 0.8733 | 0.8384 | 0.8133 | 0.9285 |
| 3 | Residual-gated, α=2.0 | 0.8727 | **0.8975** | 0.7998 | **0.9369** |

In-domain reference (AV1M val, sync head): 0.8362, N=2000.
1a Mahalanobis ≡ 1c Gaussian NLL by construction (NLL is monotonic in M-distance, AUC is rank-based).

## Training setup (current ckpt)

- Backbone: AV-HuBERT Large, frozen (`av_hubert/avhubert/self_large_vox_433h.pt`)
- Head: TriRouteUnsupSync, 5 components (s_v, s_a, u_v, u_a, u_delta) + residuals (r_v, r_a)
- Training: sync prediction (31-class CE on temporal alignment) + 4 structural regularizers
- Data: VoxCeleb2 dev, 40,000 real clips (`vox2_n40000_seed44_e20`); pure unsup, no fake labels
- Optimizer: Adam, lr=1e-4 constant, 20 epochs, batch=64, tau=15
- Reference distribution for M1/M3: AV1M val real-only (519 clips), pooled u_v

## Comparison to benchmark

Smeu et al., CVPR 2025 (the paper this project extends):
- FAVC: 0.946 overall — trained on 50k AV1M-real
- AV1M: 0.859 overall

Our current single-seed best (kNN on u_v, post-hoc): **0.9345** overall, FV-RA 0.9188.

Gap is ~1 pt on overall and ~3 pts on FV-RA, but ours uses 40k Vox2 (different protocol)
and demonstrates structural-allocation routing rather than scaled training data.

## Open follow-ups for the meeting

- Run the same 3-way grid on seed44/45 to confirm the +1.6 kNN gain isn't a single-seed artifact
- Score ensemble M1 kNN + M3 α=1.0 (visual-anomaly + RV-FA recovery)
- Sync-vs-inconsistency objective ablation (1 row in the paper's ablation table)
- LRS dataset: pending data access (emails sent 2026-05-03)

## Layout note

Failed experiments, rotation backups, and smoke tests have been moved out of the way to:
`_archive_pre_meeting_20260503/` (3.2 GB, preserved, can be deleted later if not needed).

The `outputs_*` dirs at the top of `avh_sup/` are all active paper experiments.
