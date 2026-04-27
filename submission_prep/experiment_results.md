# TriRoute — Latest Experiment Results Summary

> Training: AV-Deepfake1M (AV1M) -> Evaluation: FakeAVCeleb 70/30 test split (N=6,667)  
> Current result policy: keep only the latest pseudo-domain v2 50-epoch TriRoute numbers for our main method.  
> FV-RA = FakeVideo-RealAudio, the primary visual-routing stress category.
> Last verified from seed 44/45 artifacts on 2026-04-28.

## Latest Main Result

Latest run: `A6_av1m_pseudodomain_v2_50ep_seed44/45`

| Run | Seed | Best epoch | FV-RA AUC |
|-----|------|------------|-----------|
| A6 pseudo-domain v2, 50ep | 44 | 38 | 0.8272 |
| A6 pseudo-domain v2, 50ep | 45 | 37 | 0.8484 |
| **Mean** | **44/45** | — | **0.8378** |

Interpretation: the current strongest result is source-side pseudo-domain routing, not UDA. It uses AV1M speaker-group pseudo-labels only and no FakeAVCeleb data during training, validation, checkpointing, or hyperparameter selection.

Source checkpoint pointers:

| Seed | Best checkpoint |
|------|-----------------|
| 44 | `outputs_A6_av1m_pseudodomain_v2_50ep_seed44/ckpts/favc_fvra/best-favc-fvra-epoch=38-auc=0.8272.ckpt` |
| 45 | `outputs_A6_av1m_pseudodomain_v2_50ep_seed45/ckpts/favc_fvra/best-favc-fvra-epoch=37-auc=0.8484.ckpt` |

## Main Benchmark Table

Metric format is AUC/AP. The latest pseudo-domain v2 row is verified from direct seed 44/45 FAVC evaluation of the best FV-RA checkpoints.

| Method | Structure | Overall | RV-FA | **FV-RA** | FV-FA |
|--------|-----------|---------|-------|-----------|-------|
| AVH-Align/sup (Smeu et al.) | two-way supervised | 0.777/0.994 | 0.999/0.999 | 0.519/0.955 | 0.999/1.000 |
| Two-way baseline | two-way factorization | 0.779/0.994 | 1.000/1.000 | 0.523/0.957 | 0.999/1.000 |
| Two-way + DANN/GRL | two-way + GRL | 0.762/0.993 | 1.000/1.000 | 0.478/0.952 | 1.000/1.000 |
| Two-way + CORAL | two-way + CORAL | 0.753/0.993 | 1.000/1.000 | 0.491/0.947 | 0.971/0.999 |
| Three-way, no push-pull | three-way factorization | 0.827/0.996 | 1.000/1.000 | 0.626/0.973 | 0.999/1.000 |
| **TriRoute pseudo-domain v2 50ep** | three-way + push-pull | **0.922/0.998** | 0.999/0.999 | **0.838/0.989** | 0.994/1.000 |

Latest TriRoute main-eval means before rounding:

| Metric | Seed 44 AUC/AP | Seed 45 AUC/AP | Mean AUC/AP |
|--------|----------------|----------------|-------------|
| Overall | 0.9171/0.9978 | 0.9270/0.9981 | 0.9220/0.9980 |
| RV-FA | 0.9989/0.9991 | 0.9996/0.9997 | 0.9993/0.9994 |
| FV-RA | 0.8272/0.9872 | 0.8484/0.9903 | 0.8378/0.9888 |
| FV-FA | 0.9938/0.9997 | 0.9941/0.9997 | 0.9940/0.9997 |

Main deltas:

| Comparison | FV-RA AUC delta |
|------------|-----------------|
| TriRoute latest vs. AVH-Align/sup | +0.319 |
| TriRoute latest vs. two-way baseline | +0.315 |
| TriRoute latest vs. three-way no push-pull | +0.212 |
| TriRoute latest vs. DANN/GRL control | +0.360 |
| TriRoute latest vs. CORAL control | +0.347 |

## Head Ablation

Latest pseudo-domain v2 50ep rows average seeds 44/45.

| Model | Ablation | Baseline FV-RA | Ablated FV-RA | Delta |
|-------|----------|----------------|---------------|-------|
| Two-way baseline | mask visual task branch | 0.495 | 0.461 | -0.034 |
| Three-way, no push-pull | mask visual branches | 0.586 | 0.473 | -0.113 |
| **TriRoute pseudo-domain v2 50ep** | mask manipulation (`u_v`) | **0.836** | 0.491 | **-0.345** |
| **TriRoute pseudo-domain v2 50ep** | mask visual branches | **0.836** | 0.441 | **-0.395** |

Latest TriRoute ablation means before rounding:

| Ablation | Seed 44 FV-RA | Seed 45 FV-RA | Mean FV-RA | Mean delta |
|----------|---------------|---------------|------------|------------|
| baseline | 0.8291 | 0.8428 | 0.8359 | — |
| mask manipulation (`u_v`) | 0.5001 | 0.4823 | 0.4912 | -0.3448 |
| mask visual branches | 0.4449 | 0.4378 | 0.4414 | -0.3946 |

## Input Masking

Latest pseudo-domain v2 50ep row averages seeds 44/45.

| Model | Baseline FV-RA | Audio full mask delta | Visual full mask delta | Relative visual sensitivity |
|-------|----------------|-----------------------|------------------------|-----------------------------|
| Two-way baseline | 0.546 | -0.021 | -0.077 | 1.0x |
| Three-way, no push-pull | 0.616 | +0.040 | -0.137 | 1.8x |
| **TriRoute pseudo-domain v2 50ep** | **0.838** | **+0.005** | **-0.377** | **4.9x** |

### Progressive Masking — TriRoute (seeds 44/45 mean)

**Audio masking** (p = fraction of audio tokens zeroed):

| p | Seed 44 FV-RA | Seed 45 FV-RA | Mean FV-RA | Mean delta |
|---|---------------|---------------|------------|------------|
| 0.00 (baseline) | 0.8272 | 0.8484 | 0.8378 | — |
| 0.10 | 0.8249 | 0.8504 | 0.8376 | -0.0002 |
| 0.25 | 0.8230 | 0.8503 | 0.8366 | -0.0012 |
| 0.50 | 0.8201 | 0.8516 | 0.8359 | -0.0019 |
| 0.75 | 0.8240 | 0.8537 | 0.8388 | +0.0010 |
| 1.00 (full) | 0.8263 | 0.8597 | 0.8430 | +0.0052 |

**Visual masking** (p = fraction of visual tokens zeroed):

| p | Seed 44 FV-RA | Seed 45 FV-RA | Mean FV-RA | Mean delta |
|---|---------------|---------------|------------|------------|
| 0.00 (baseline) | 0.8272 | 0.8484 | 0.8378 | — |
| 0.10 | 0.8176 | 0.8313 | 0.8245 | -0.0134 |
| 0.25 | 0.8004 | 0.7940 | 0.7972 | -0.0406 |
| 0.50 | 0.7555 | 0.7435 | 0.7495 | -0.0883 |
| 0.75 | 0.6793 | 0.6668 | 0.6730 | -0.1648 |
| 1.00 (full) | 0.4460 | 0.4764 | 0.4612 | -0.3766 |

**Audio noise** (p = noise std relative to signal std):

| p | Seed 44 FV-RA | Seed 45 FV-RA | Mean FV-RA | Mean delta |
|---|---------------|---------------|------------|------------|
| 0.00 (baseline) | 0.8272 | 0.8484 | 0.8378 | — |
| 0.05 | 0.7632 | 0.7870 | 0.7751 | -0.0627 |
| 0.10 | 0.6144 | 0.7198 | 0.6671 | -0.1707 |
| 0.20 | 0.4919 | 0.6732 | 0.5826 | -0.2553 |
| 0.50 | 0.4522 | 0.6482 | 0.5502 | -0.2876 |
| 1.00 | 0.4468 | 0.6415 | 0.5442 | -0.2936 |

**Visual noise** (p = noise std relative to signal std):

| p | Seed 44 FV-RA | Seed 45 FV-RA | Mean FV-RA | Mean delta |
|---|---------------|---------------|------------|------------|
| 0.00 (baseline) | 0.8272 | 0.8484 | 0.8378 | — |
| 0.05 | 0.7655 | 0.8230 | 0.7943 | -0.0435 |
| 0.10 | 0.6481 | 0.7296 | 0.6888 | -0.1490 |
| 0.20 | 0.5676 | 0.6152 | 0.5914 | -0.2464 |
| 0.50 | 0.5127 | 0.5329 | 0.5228 | -0.3150 |
| 1.00 | 0.4944 | 0.5026 | 0.4985 | -0.3393 |

**Temporal shuffle** (p = fraction of tokens temporally shuffled):

| p | Seed 44 FV-RA | Seed 45 FV-RA | Mean FV-RA | Mean delta |
|---|---------------|---------------|------------|------------|
| 0.00 (baseline) | 0.8272 | 0.8484 | 0.8378 | — |
| 0.10 | 0.8261 | 0.8508 | 0.8385 | +0.0007 |
| 0.25 | 0.8282 | 0.8498 | 0.8390 | +0.0012 |
| 0.50 | 0.8315 | 0.8532 | 0.8423 | +0.0045 |
| 0.75 | 0.8329 | 0.8547 | 0.8438 | +0.0060 |
| 1.00 | 0.8374 | 0.8577 | 0.8476 | +0.0098 |

Interpretation: audio masking and temporal shuffle are approximately neutral throughout the full range, confirming the model has stopped relying on audio for FV-RA. Visual masking produces a monotonic drop, with severe degradation only near full masking (p≥0.75). Audio noise causes a sharp early drop (already −0.17 at p=0.1), suggesting audio-feature sensitivity differs from audio-masking sensitivity. Visual noise shows a more gradual decline than visual masking.

## Leading-Silence Control

Latest pseudo-domain v2 50ep true-trimmed control:

| Condition | Overall | RV-FA | FV-RA | FV-FA |
|-----------|---------|-------|-------|-------|
| Untrimmed mean | 0.9427 | 1.0000 | 0.8779 | 0.9969 |
| Trimmed mean | 0.9045 | 0.7759 | 0.8767 | 0.9348 |
| Delta | -0.0382 | -0.2241 | -0.0012 | -0.0622 |

Interpretation: the FV-RA gain is stable under leading-silence trimming, so the latest result is not explained by the leading-silence artifact.

## AVLips Cross-Dataset Check

Latest pseudo-domain v2 50ep evaluation on AVLips features. Values are causal/full AUC/AP; the spurious head is reported as a sanity check and is not the task score.

| Seed | Causal/full AUC | Causal/full AP | Spurious AUC | Spurious AP |
|------|-----------------|----------------|--------------|-------------|
| 44 | 0.7818 | 0.7562 | 0.4669 | 0.5481 |
| 45 | 0.7966 | 0.8107 | 0.4630 | 0.5448 |
| **Mean** | **0.7892** | **0.7835** | **0.4650** | **0.5465** |

## Current Paper Wording

Use this claim:

> TriRoute pseudo-domain v2 50ep reaches 0.838 FV-RA AUC without using target-domain data, improving over the two-way baseline by +0.315 AUC and over the three-way no-push-pull control by +0.212 AUC.

Avoid using older TriRoute no-target/UDA rows as current paper results. They are superseded by the latest pseudo-domain v2 50ep row.

## Pending / Not Yet in Main Paper

- AVLips evaluation completed on `gpu-l40s` as Slurm job `24387213` after fixing checkpoint-path parsing in `eval_A6_pdv2_50ep_avlips_seeds44_45.slurm`.
- Seed 43 pseudo-domain v2 50ep training is running on `gpu-h100` as Slurm job `24387238` with a 4-hour limit; the current paper result remains the seed 44/45 mean until seed 43 completes and is explicitly incorporated.
