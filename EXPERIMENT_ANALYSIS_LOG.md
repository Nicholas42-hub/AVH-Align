# AVH-Align Experiment Analysis Log
Generated: 2026-04-02; updated: 2026-04-06

---

## 1. Dataset Statistics

### AV-Deepfake1M (AV1M) — Primary Training/Eval Dataset
| Split | N clips | Fake | Real | Note |
|-------|---------|------|------|------|
| train | 8,000 | 6,024 (75%) | 1,976 (25%) | Heavily imbalanced |
| val | 2,000 | 1,481 (74%) | 519 (26%) | Used for validation metric |
| test | 10,000 | 7,505 (75%) | 2,495 (25%) | In-domain test |

### FakeAVCeleb (FAVC) — Cross-Domain Evaluation Dataset
| Split | N clips | A (RV-RA real) | B (RV-FA) | C (FV-RA) | D (FV-FA) |
|-------|---------|----------|----------|----------|----------|
| test | 6,667 | 151 | 151 | 3,007 | 3,358 |
| val | 1,497 | 35 | 35 | 679 | 748 |

**Used in evaluation:** N=1,114 balanced subset (52 RV-FA + 522 FV-RA + 592 FV-FA, all fake vs real)
**Used in B1 favc_eval:** N=4,089 (inconsistent — reconciliation needed)

Category key:
- A: RealVideo-RealAudio (genuine)
- B: RealVideo-FakeAudio (audio synthesis only)
- C: FakeVideo-RealAudio (video synthesis only — hardest case)
- D: FakeVideo-FakeAudio (both synthetic)

### Feature Extraction Pipeline
- Encoder: AV-HuBERT `self_large_vox_433h.pt` (325M parameters)
- Visual input: mouth ROI frames (grayscale, 88×88 px), 25fps
- Audio input: log-filterbank features (104-d, stacked)
- Output: per-frame visual [T×1024] + audio [T×1024] features
- Normalization: L2 normalization applied (`apply_l2: true` in all configs)
- Variable-length clips: batch_size=1 for frozen experiments

---

## 2. Model Architecture Families

### Family 1: Binary Causal/Spurious (A0–A4, A42–A44)
```
Input [B,T,1024] × 2 (video, audio)
  ↓ Linear projections
  v_c, v_s [B,T,512],  a_c, a_s [B,T,512]
  Z_c = cat(v_c, a_c)  [B,T,1024]  ← causal/invariant
  Z_s = cat(v_s, a_s)  [B,T,1024]  ← spurious/variable
  ↓ LogSumExp pooling over T → clip-level score
  causal_head(Z_c) → 1     [primary in A2+]
  spurious_head(Z_s.detach()) → 1  [probe only, never trains encoder]
  adv_head(GRL(Z_s)) → 1   [label adversarial on Z_s]
```

Shared MLP: Linear(1024,512) → LN → ReLU → Linear(512,256) → LN → ReLU → Linear(256,128) → LN → ReLU → Linear(128,1)

### Family 2: FCD Three-Way (A5–A7)
```
Input [B,T,1024] × 2
  ↓ CCD_v, CCD_a (per-modality MLP branches)
  s_v, h_v [B,T,256]   s_a, h_a [B,T,256]
  ↓ URD_v, URD_a (unique-redundant decomposition with SVD regularization)
  u_v=fc2(ReLU(LN(fc1(h_v)))),  r_v = h_v - u_v
  u_a, r_a  (same)
  Z_c = cat(s_v,u_v,s_a,u_a) [B,T,1024]  ← causal (syn+unique)
  Z_s = cat(r_v, r_a)         [B,T,512]   ← spurious (redundant)
```
SVD regularization: every 50 steps, clamp near-zero singular values of URD.fc1 to ≥ 1e-5

**A5 additional components:**
- L_SDA: Sinkhorn divergence between sda_proj(mean s_v) and sda_proj(mean s_a) — ε=0.05, n_iter=50
- L_MI: CE proxy on unique_head_v(u_v) + unique_head_a(u_a)
- L_Dis: BCE(modality_head(cat(r_v_pool, r_a_pool)), {0,1}) — keeps r modality-discriminative
- L_orth: ||cos(u,r)||² per modality

**A6 adds domain push-pull to A5:**
- L_dadv: BCE(domain_head_c(GRL(Z_c_pool)), domain) — expels domain from Z_c
- L_ddis: BCE(domain_head_s(Z_s_pool), domain) — concentrates domain in Z_s
- Domain data: FAVC val-real clips (domain=1) + AV1M (domain=0)

**A7 adds label adversarial on Z_s to A6:**
- L_ladv: BCE(label_head_s(GRL(Z_s_pool)), label) — expels label info from Z_s
- Motivation: A6 had Z_s AUC > Z_c AUC in-domain (label info inverted in Z_s)
- lambda_dadv increased 1.0→3.0, lambda_ladv=2.0

### Family 3: SAD Synchrony-Appearance (A8–A9)
```
Input: X_v [B,T,1024], X_a [B,T,1024]
  Z_sync = MLP(cat(V, A, V⊙A, V-A)) [B,T,512]   ← cross-modal interaction
  Z_app_v = MLP(V) [B,T,256],  Z_app_a = MLP(A) [B,T,256]
  Z_app = cat(Z_app_v, Z_app_a) [B,T,512]        ← unimodal appearance
```
Causal = Z_sync (AV interaction encodes coherence breaks)
Spurious = Z_app (identity, recording conditions, speaker style)

**A8:** Domain push-pull (L_dadv on Z_sync, L_ddis on Z_app, L_ladv on Z_app)
**A9:** Replace L_dadv with multi-kernel RBF MMD² between Z_sync domain distributions. batch_size=16 required for meaningful MMD.

### Family 4: Modal Balance Extensions (B1–B3, built on A2)
```
A2 base (adv + stop-grad) +
  B1:   modal_head_visual(v_c) + modal_head_audio(a_c) → CE(labels) each
  B1b:  B1 + balance_penalty = (CE_audio - CE_visual)²
  B2:   cross_modal_loss = mean(1-cosine(v_c,a_c)) on real samples only
  B3:   modal_disc(GRL(cat(v_c_tokens, a_c_tokens))) → BCE(modality)
```

---

## 3. Frozen Experiments — Full Results

### 3.1 FakeAVCeleb Cross-Dataset AUC (N=1114 test, unless noted)

| Model | Overall AUC | RV-FA | FV-RA | FV-FA | Spurious AUC | Best Epoch |
|-------|:-----------:|:-----:|:-----:|:-----:|:------------:|:----------:|
| A0 | 0.7703 | — | — | — | 0.7764 ❌ | 10 |
| A1 | 0.7827 | — | — | — | 0.2866 | 3 |
| A2 | 0.7928 | — | — | — | 0.3167 | 9 |
| A3 | 0.7490 | 1.000 | 0.4496 | 0.9999 | 0.3778 | 10 |
| A4 | — | — | — | — | — | 0 (failed) |
| A5 | 0.8186 | 0.9956 | 0.6156 | 0.9883 | 0.3163 | 17 |
| **A6** | **0.8885** | 1.000 | **0.7568** | 0.9987 | 0.3667 | 15 |
| A7 | 0.8251 | 1.000 | 0.6168 | 0.9997 | **0.2674** | 5 |
| A8 | 0.7729 | 0.9985 | 0.5154 | 0.9882 | 0.3823 | 4 |
| A9 | — | — | — | — | — | 84 |
| A42 | 0.7991 | 1.000 | 0.5594 | 1.000 | 0.3054 | 5 |
| A43 | 0.7840 | 1.000 | 0.5264 | 0.9998 | 0.3135 | 5 |
| A44 | 0.7667 | 1.000 | 0.4887 | 0.9997 | 0.3002 | 2 |
| B1 | 0.7538* | 1.000 | 0.4705 | 0.9998 | 0.3515 | 13 |
| B1b | 0.8430 | 1.000 | 0.6560 | 0.9996 | 0.4240 | 58 |
| B2 | 0.7891† | — | — | — | — | — |
| B3 | 0.8079† | — | — | — | — | — |

*N=4089 (full FAVC) — different test set from others
†perturbation base AUC used as proxy; no eval_results.txt has been run for B2/B3

### 3.2 In-Domain Performance (AV1M val, from training metrics)

| Model | Best val_auc_causal | spurious peak val |
|-------|:-------------------:|:-----------------:|
| A0 | 0.9978 (full_head) | 0.5141 |
| A1 | 0.9981 | 0.5043 |
| A2 | 0.9984 | 0.5633 |
| A3 | 0.9980 | 0.5187 |
| A4 | 0.9856 | 0.5045 |
| A5 | 0.9983 | 0.5148 |
| A6 | 0.9982 | 0.5341 |
| A7 | 0.9988 | 0.5157 |
| A8 | 0.9973 | 0.5183 |
| A9 | 0.9976 | 0.5252 |
| A42 | 0.9978 | 0.5113 |
| A43 | 0.9974 | 0.5016 |
| A44 | 0.9978 | 0.5113 |

Note: A0 seed variants (42/43/44) show best_val_causal ≈ 0.50 — correct, because in A0 (use_sg=False) the causal_head is never the primary trained head. The checkpoint is selected on val_auc_causal which tracks the unused probe head.

### 3.3 Seed Variant In-Domain Val AUC (proxy for variance)

| Variant | Best val_auc_causal |
|---------|:-------------------:|
| A0_seed42 | 0.5098 (causal head untrained in A0) |
| A0_seed43 | 0.5093 |
| A0_seed44 | 0.5029 |
| A1_seed42 | 0.9983 |
| A1_seed43 | 0.9977 |
| A1_seed44 | 0.9985 |
| A2_seed42 | 0.9978 |
| A2_seed43 | 0.9984 |
| A2_seed44 | 0.9977 |
| A2_mine | 0.9980 |

In-domain variance is small (~0.001) across seeds for A1/A2. Cross-domain variance unknown (seed sweep not yet run on frozen experiments).

### 3.4 Perturbation Robustness — Full Table

All values = causal AUC at specified perturbation level. Base (0.0) = clean FAVC test.

#### Audio Noise (Gaussian noise added to audio features)
| Model | 0.0 | 0.05 | 0.1 | 0.2 | 0.5 | 1.0 | Δ (0→1.0) |
|-------|-----|------|-----|-----|-----|-----|-----------|
| A0 | 0.7703 | 0.7859 | 0.6619 | 0.5447 | 0.4864 | 0.4779 | **−0.292** |
| A1 | 0.7827 | 0.8056 | 0.7619 | 0.6362 | 0.5322 | 0.4883 | −0.294 |
| A2 | 0.7928 | 0.7776 | 0.7068 | 0.5547 | 0.4514 | 0.4460 | **−0.347** |
| A3 | 0.7490 | 0.7963 | 0.5503 | 0.4895 | 0.3921 | 0.3948 | −0.354 |
| A5 | 0.8186 | 0.8035 | 0.6945 | 0.5912 | 0.5247 | 0.5011 | −0.318 |
| **A6** | **0.8885** | 0.8768 | 0.7416 | 0.5947 | 0.5237 | 0.5173 | **−0.371** |
| A7 | 0.8251 | — | — | — | — | 0.5568 | −0.268 |
| A8 | 0.7729 | — | — | — | — | 0.2140 | **−0.559 ❌** |

#### Audio Masking (zeroing out fraction of audio frames)
| Model | 0.0 | 0.1 | 0.25 | 0.5 | 0.75 | 1.0 | Δ (0→1.0) |
|-------|-----|-----|------|-----|------|-----|-----------|
| A0 | 0.7703 | 0.7567 | 0.7479 | 0.6787 | 0.5860 | 0.3448 | **−0.426** |
| A1 | 0.7827 | 0.7670 | 0.7642 | 0.6947 | 0.5935 | 0.4600 | −0.323 |
| A2 | 0.7928 | 0.7757 | 0.7518 | 0.6917 | 0.5865 | 0.4622 | −0.330 |
| A5 | 0.8186 | 0.8077 | 0.8161 | 0.7647 | 0.6877 | 0.6077 | −0.211 |
| **A6** | **0.8885** | 0.8808 | 0.8675 | 0.8379 | 0.8071 | **0.7691** | **−0.119** |
| A7 | 0.8251 | — | — | — | — | 0.5490 | −0.276 |
| A8 | 0.7729 | — | — | — | — | 0.1510 | **−0.622 ❌** |
| B1 | 0.7679 | — | — | — | — | — | — |
| B1b | 0.8430 | — | — | — | — | 0.6085 | −0.234 |
| B2 | 0.7891 | — | — | — | — | — | — |
| B3 | 0.8079 | — | — | — | — | 0.6200 | −0.188 |

#### Visual Masking (zeroing out fraction of visual frames)
| Model | 0.0 | 0.1 | 0.25 | 0.5 | 0.75 | 1.0 | Δ (0→1.0) |
|-------|-----|-----|------|-----|------|-----|-----------|
| A0 | 0.7703 | 0.7645 | 0.7748 | 0.7733 | 0.7585 | 0.7468 | −0.024 |
| A1 | 0.7827 | 0.7833 | 0.8009 | 0.7886 | 0.7759 | 0.7545 | −0.028 |
| A2 | 0.7928 | 0.7902 | 0.7820 | 0.7721 | 0.7558 | 0.7580 | −0.035 |
| A5 | 0.8186 | 0.8199 | 0.8410 | 0.8239 | 0.7994 | 0.7593 | −0.059 |
| **A6** | **0.8885** | 0.8952 | 0.8909 | 0.8733 | 0.8296 | 0.7678 | **−0.121** |
| A7 | 0.8251 | — | — | — | — | 0.7898 | −0.035 |
| A8 | 0.7729 | — | — | — | — | 0.7591 | −0.014 |
| B1b | 0.8430 | — | — | — | — | 0.7735 | −0.070 |
| B3 | 0.8079 | — | — | — | — | 0.7265 | −0.081 |

#### Temporal Shuffle (randomly permuting frame order)
| Model | 0.0 | 0.1 | 0.25 | 0.5 | 0.75 | 1.0 | Δ (0→1.0) |
|-------|-----|-----|------|-----|------|-----|-----------|
| A0 | 0.7703 | 0.7654 | 0.7712 | 0.7630 | 0.7647 | 0.7588 | −0.012 |
| A1 | 0.7827 | 0.7807 | 0.7948 | 0.7855 | 0.7718 | 0.7723 | −0.010 |
| A2 | 0.7928 | 0.7892 | 0.7863 | 0.7594 | 0.7925 | 0.7647 | −0.028 |
| A5 | 0.8186 | 0.8075 | 0.8196 | 0.8500 | 0.8010 | 0.8326 | **+0.014** |
| **A6** | **0.8885** | 0.8871 | 0.8894 | 0.8954 | 0.8898 | **0.8937** | **+0.005** |
| A8 | 0.7729 | — | — | — | — | 0.6811 | −0.092 |

Key observation: A5/A6 are essentially invariant to temporal shuffling (Δ≈0), while A8 degrades (−0.092). This means A5/A6 rely on per-frame features, not temporal synchrony — the "temporal coherence" as the causal signal is NOT what FCD is learning.

#### Visual Noise (Gaussian noise on visual features)
| Model | 0.0 | 0.05 | 0.1 | 0.2 | 0.5 | 1.0 | Δ (0→1.0) |
|-------|-----|------|-----|-----|-----|-----|-----------|
| A0 | 0.7703 | 0.7684 | 0.7643 | 0.7611 | 0.7623 | 0.7631 | −0.007 |
| A1 | 0.7827 | 0.7774 | 0.7789 | 0.7795 | 0.7709 | 0.7663 | −0.016 |
| A2 | 0.7928 | 0.7793 | 0.7682 | 0.7648 | 0.7641 | 0.7640 | −0.029 |
| A5 | 0.8186 | 0.8165 | 0.8042 | 0.7857 | 0.7739 | 0.7652 | −0.053 |
| **A6** | **0.8885** | **0.9052** | 0.8449 | 0.7922 | 0.7609 | 0.7515 | −0.137 |

A6 at visual_noise=0.05 actually *improves* to 0.9052 (slight noise as regularization). Larger visual noise degrades similarly to others.

### 3.5 Label Probe Results

**⚠️ Methodological correction (2026-04-06)**: The original `eval_label_probe.py` pre-mean-pooled features *before* passing them into `_encode()`, producing a fake T=1 input `[B,1,D]`. For A2 (per-frame linear projections), this is equivalent to the correct T>1 input. For FCD models (A5/A6), T=1 is out-of-distribution (models trained on T=25–150 frames) and produces severely degraded representations. The A5/A6 values below marked ❌ are artifacts. The corrected sequence-based values are in § 3.5.1.

Probe design: linear classifier trained on frozen representations, 5-fold CV on AV1M val.

| Model | Raw AUC | Z_c AUC | Z_s AUC | OOD Z_c (FAVC) | Domain probe Z_c |
|-------|---------|---------|---------|----------------|:----------------:|
| A0 | 0.9654 | 0.9398 | 0.9334 | 0.7011 | 0.9976 |
| A1 | 0.9654 | 0.9475 | 0.9588 ⚠️ | 0.7077 | 0.9972 |
| A2 | 0.9654 | 0.9421 | 0.9187 | 0.7056 | 0.9979 |
| A5 | 0.9654 | 0.8191 ❌ | 0.8134 ❌ | 0.7200 ❌ | 0.9943 ❌ |
| **A6** | 0.9654 | 0.7704 ❌ | 0.8316 ❌ | 0.7226 ❌ | 0.9908 ❌ |

### 3.5.1 Label Probe — Sequence-Based (Corrected, 2026-04-06)

Protocol: full [T,D] sequences → `model._encode(video [B,T,D], audio [B,T,D])` → mean-pool Z_c/Z_s over T.
This is the correct way to query FCD models (same as training).

| Model | Z_c AUC (in-domain) | Z_s AUC (in-domain) | OOD Z_c | OOD Z_s | Domain Z_c | Domain Z_s |
|-------|:-------------------:|:-------------------:|:-------:|:-------:|:----------:|:----------:|
| A2 | 0.9420 ± 0.0083 | 0.9190 ± 0.0080 | 0.7052 | 0.7396 ⚠️ | 0.9975 | 0.9962 |
| A5 | **0.9970** ± 0.0026 | **0.9967** ± 0.0021 | **0.7744** | 0.7078 | 0.9966 | 0.9956 |
| **A6** | 0.9944 ± 0.0033 | 0.9975 ± 0.0025 ⚠️ | 0.7480 | 0.7127 | **0.9971** | 0.9974 |

**Key revised findings (sequence-based is the truth):**

1. **A5/A6 in-domain Z_c probe result overturned**: old probe showed A5 Z_c=0.8191, new shows **0.9970**. The T=1 artifact was causing FCD models to look like they had less task information in Z_c — the opposite is true. FCD both branches carry near-full in-domain task signal.

2. **A2 is unchanged** (0.9421 → 0.9420): binary per-frame projection is commutative with mean-pooling, confirming T=1 probe was only broken for FCD models.

3. **OOD Z_c ranking is preserved but gaps change**: A5 wins (0.7744) > A6 (0.7480) > A2 (0.7052). Previously A5 appeared equal to A2 (0.7200 ≈ 0.7056) — the true gap is +6.9pp for A5.

4. **A2 OOD Z_s (0.7396) > Z_c (0.7052)** ⚠️: binary split spurious branch has *better* cross-domain transfer than causal. This is surprising — Z_s for A2 apparently encodes something globally useful (likely speaker acoustics that are domain-invariant).

5. **A6 in-domain Z_s (0.9975) > Z_c (0.9944)** ⚠️: the Z_s > Z_c inversion persists in the corrected probe. Domain adversarial pressure in A6 redirects label-predictive information toward Z_s even in the correct setting.

6. **Domain probes (Z_c ≈ 0.997)**: all three models show near-identical domain predictability from Z_c under sequence encoding. A6's domain push-pull does not reduce domain content in Z_c as measured sequentially.

### 3.6 Domain Predictability Results (binary: AV1M=0 vs FAVC=1)

| Model | Raw AUC | Causal AUC | Spurious AUC |
|-------|---------|-----------|-------------|
| A0 | 0.9990 | 0.9976 | 0.9969 |
| A1 | 0.9990 | 0.9972 | 0.9975 |
| A2 | 0.9990 | 0.9979 | 0.9959 |
| A3 | 0.9990 | 0.9976 | 0.9985 |
| A5 | 0.9990 | **0.9943** | **0.9927** |
| **A6** | 0.9990 | **0.9923** | 0.9937 |

Critical finding: **No model achieves meaningful domain invariance.** All Z_c domain AUCs ≥ 0.992. The causal features remain highly domain-predictable despite all adversarial objectives. A6's domain push-pull yields only marginal improvement (0.9979→0.9923 Z_c). This is a fundamental weakness for the paper's domain invariance claims — to be acknowledged as "domain feature compression improves generalization without requiring full domain invariance" per Karen's framing.

### 3.7 Branch Probe Under Perturbation

Probe design: label probe AUC (fake/real on AV1M val, 5-fold CV) after applying perturbation to input features before `_encode`. Measures how much label-predictive information survives in Z_c and Z_s when one modality is degraded.

Note: AV1M is primarily an **audio deepfake** dataset (voice dubbed onto real video). Visual perturbation barely hurts any model — there is negligible visual-manipulation signal in AV1M. The real test is audio perturbation.

**Updated 2026-04-06**: Added `audio_noise` (Gaussian noise σ=[0.05,0.10,0.20,0.50,1.00]) and `visual_noise` perturbation types (jobs 23656789–91).

#### Audio Masking (zeroing out fraction of audio frames before encoding)
| Model | Branch | 0% | 10% | 25% | 50% | 75% | **100%** | Δ (0→100%) |
|-------|--------|-----|-----|-----|-----|-----|--------|------------|
| A2 | Z_c | 0.9421 | 0.9405 | 0.9357 | 0.9226 | 0.8983 | **0.5332** | **−0.409** |
| A2 | Z_s | 0.9189 | 0.9179 | 0.9154 | 0.9052 | 0.8890 | 0.5287 | −0.390 |
| A5 | Z_c | 0.9970 | 0.9838 | 0.9615 | 0.9360 | 0.8843 | **0.6990** | −0.298 |
| A5 | Z_s | 0.9968 | 0.9822 | 0.9619 | 0.9398 | 0.8919 | 0.6882 | −0.309 |
| A6 | Z_c | 0.9948 | 0.9785 | 0.9595 | 0.9266 | 0.8718 | **0.6624** | **−0.332** |
| A6 | Z_s | 0.9975 | 0.9811 | 0.9626 | 0.9349 | 0.8835 | 0.6888 | −0.309 |

#### Visual Masking (zeroing out fraction of visual frames before encoding)
| Model | Branch | 0% | 10% | 25% | 50% | 75% | **100%** | Δ (0→100%) |
|-------|--------|-----|-----|-----|-----|-----|--------|------------|
| A2 | Z_c | 0.9421 | 0.9413 | 0.9417 | 0.9361 | 0.9338 | **0.9057** | −0.036 |
| A2 | Z_s | 0.9189 | 0.9175 | 0.9170 | 0.9145 | 0.9107 | 0.8471 | −0.072 |
| A5 | Z_c | 0.9970 | 0.9970 | 0.9971 | 0.9964 | 0.9967 | **0.9958** | **−0.001** |
| A5 | Z_s | 0.9968 | 0.9968 | 0.9965 | 0.9965 | 0.9969 | 0.9941 | −0.003 |
| A6 | Z_c | 0.9948 | 0.9939 | 0.9954 | 0.9965 | 0.9954 | **0.9960** | **+0.001** |
| A6 | Z_s | 0.9975 | 0.9973 | 0.9970 | 0.9967 | 0.9978 | 0.9961 | −0.001 |

#### Audio Noise (Gaussian σ added to audio features)
| Model | Branch | σ=0 | σ=0.05 | σ=0.10 | σ=0.20 | σ=0.50 | **σ=1.00** | Δ (0→max) |
|-------|--------|-----|--------|--------|--------|--------|----------|------------|
| A2 | Z_c | 0.9421 | 0.9105 | 0.8523 | 0.7430 | 0.6111 | **0.5664** | **−0.376** |
| A2 | Z_s | 0.9189 | 0.8963 | 0.8509 | 0.7548 | 0.6087 | 0.5555 | −0.363 |
| A5 | Z_c | 0.9970 | 0.9867 | 0.9002 | 0.7589 | 0.7045 | **0.6972** | −0.300 |
| A5 | Z_s | 0.9968 | 0.9842 | 0.9166 | 0.7943 | 0.7261 | 0.7120 | −0.285 |
| A6 | Z_c | 0.9948 | 0.9850 | 0.8635 | 0.7393 | 0.6850 | **0.6741** | **−0.321** |
| A6 | Z_s | 0.9975 | 0.9856 | 0.8889 | 0.7715 | 0.7236 | 0.7115 | −0.286 |

#### Visual Noise (Gaussian σ added to visual features)
| Model | Branch | σ=0 | σ=0.05 | σ=0.10 | σ=0.20 | σ=0.50 | **σ=1.00** | Δ (0→max) |
|-------|--------|-----|--------|--------|--------|--------|----------|------------|
| A2 | Z_c | 0.9421 | 0.9365 | 0.9283 | 0.9162 | 0.9032 | **0.8995** | −0.043 |
| A2 | Z_s | 0.9189 | 0.9188 | 0.9159 | 0.9068 | 0.8872 | 0.8784 | −0.041 |
| A5 | Z_c | 0.9970 | 0.9949 | 0.9934 | 0.9932 | 0.9937 | **0.9940** | **−0.003** |
| A5 | Z_s | 0.9968 | 0.9937 | 0.9924 | 0.9916 | 0.9919 | 0.9922 | −0.005 |
| A6 | Z_c | 0.9948 | 0.9945 | 0.9943 | 0.9950 | 0.9950 | **0.9939** | **−0.001** |
| A6 | Z_s | 0.9975 | 0.9974 | 0.9969 | 0.9959 | 0.9954 | 0.9953 | −0.002 |

**Key findings (updated):**

1. **Audio masking ≈ audio noise in effect**: A2 Z_c drops to 0.5332 (mask) vs 0.5664 (noise). A5: 0.6990 vs 0.6972. A6: 0.6624 vs 0.6741. Both perturbation types converge to the same saturation level — the residual signal at max perturbation reflects the model's non-audio capacity.

2. **A2 Z_c collapses under audio perturbation**: ~0.54–0.57 at max. Near-random. Binary Z_c depends entirely on audio signal on AV1M. **Proves audio dominance in binary split.**

3. **A5/A6 Z_c survives better**: saturates at 0.67–0.70. FCD's u_v channel contributes visual residual even with audio destroyed. Gap over A2: +0.13–0.16pp at max audio perturbation. **Confirms FCD forces partial visual specialization in Z_c.**

4. **Visual perturbation is negligible for A5/A6** (Δ≈−0.001 to −0.003): AV1M has no visual manipulation signal. A2 visual noise is slightly worse (Δ=−0.043) because binary Z_c leaks some visual acoustic-correlation signal. FCD's shared block s_v/s_a cleanly separates modalities so visual noise doesn't contaminate audio signal.

5. **Z_s behaves slightly more robustly than Z_c under audio noise** for A5/A6 (e.g., A5 Z_s@σ=1.00 = 0.7120 > Z_c = 0.6972; A6 Z_s = 0.7115 > Z_c = 0.6741). This is the Z_s > Z_c inversion from the label probe (§ 3.5.1) — Z_s carries at least as much label signal as Z_c even under audio corruption.

6. **The critical test is FAVC FV-RA clips specifically**: AV1M branch probe is limited by audio-only-fake data. FAVC branch probe (§ 3.11, jobs 23656792-94 running) will provide the visual-fake counterpart.

### 3.7.1 A2 Per-Category Perturbation Robustness (gap fill — job 23656795, ✅ done 2026-04-06)

This fills the gap in Section 3.4: A2's per-category files previously only had `audio_noise`+`visual_noise`; now includes `audio_masking`, `visual_masking`, `temporal_shuffle`. Comparing A2 to A6 (which already had all 5 types).

Key metric: `causal_auc` (A2's Z_c-equivalent branch detection on held-out test clips, no probe training).

**A2 FakeVideo-RealAudio (n=522 fakes, 100 reals) — baseline causal_auc = 0.5462 (near chance ⚠️)**

| Perturbation | @100%/max causal_auc | Δ |
|-------------|---------------------|---|
| audio_masking | 0.5253 | −0.021 (barely changes — audio is real, irrelevant) |
| visual_masking | 0.4694 | −0.077 (removes the fake video signal) |
| audio_noise | 0.3198 | −0.226 (audio noise tanks below chance!) |
| visual_noise | 0.4714 | −0.075 |
| temporal_shuffle | 0.4903 | −0.056 |

**A2 RealVideo-FakeAudio (n=100 fakes, 100 reals) — baseline causal_auc = 1.0000 (perfect ✅)**

| Perturbation | @100%/max causal_auc | Δ |
|-------------|---------------------|---|
| audio_masking | 0.5340 | −0.466 (collapses — audio mask removes the only fake signal) |
| visual_masking | 1.0000 | 0.000 (no effect — video is real, masking it changes nothing) |
| audio_noise | 0.4822 | −0.518 (collapses below chance!) |
| visual_noise | 1.0000 | 0.000 |
| temporal_shuffle | 1.0000 | 0.000 |

**Critical interpretation for paper**: A2's causal branch is functionally an **audio-only detector**.
- FV-RA (video fake, real audio): baseline ≈ chance (0.55). A2's Z_c cannot detect video-only fakes.
- RV-FA (audio fake, real video): baseline = perfect (1.00), collapses under audio perturbation.
- Audio noise on FV-RA tanks to **0.32** (below chance) — the model gets actively misled when audio is corrupted, because audio is its only relevant feature and corrupted audio produces misleading confidence.
- This strongly motivates FCD: the binary causal/spurious split does not produce a "causal+visual" channel — it produces an "audio" channel and a "correlated audio+visual" channel. FCD's u_v unique-visual channel is required to detect FV-RA fakes.

### 3.11 FAVC Branch Probe Under Perturbation (jobs 23656792–794, ✅ done 2026-04-06)

Script: `avh_sup/eval_favc_branch_probe_perturb.py` — linear probe on frozen Z_c/Z_s, 5-fold CV on FAVC test clips, per fake category vs RV-RA (real) controls.

Categories: FV-RA (fake video, real audio, n=1854), RV-FA (real video, fake audio, n=100), FV-FA (both fake, n=2035). All vs RV-RA (n=100) controls.

Perturbation levels: masking [0, 10, 25, 50, 75, 100%]; noise σ [0, 0.05, 0.10, 0.20, 0.50, 1.00].

#### Z_c AUC at max perturbation — FakeVideo-RealAudio (video fake, real audio)

| Model | baseline | audio_mask@100% | visual_mask@100% | audio_noise@σ=1 | visual_noise@σ=1 |
|-------|:--------:|:---------------:|:----------------:|:---------------:|:----------------:|
| A2 | 0.9927 | **0.9882** (+0.0) ✅ | **0.3565** ❌ | 0.9789 | 0.6519 |
| A5 | 0.9898 | **0.9924** (+0.0) ✅ | **0.3400** ❌ | 0.9856 | 0.5251 |
| **A6** | 0.9892 | **0.9807** (+0.0) ✅ | **0.3248** ❌ | 0.9767 | **0.4964** ⚠️ |

#### Z_c AUC at max perturbation — RealVideo-FakeAudio (real video, fake audio)

| Model | baseline | audio_mask@100% | visual_mask@100% | audio_noise@σ=1 | visual_noise@σ=1 |
|-------|:--------:|:---------------:|:----------------:|:---------------:|:----------------:|
| A2 | 0.9910 | **0.4240** ❌ | 0.9910 ✅ | 0.7710 | 0.9890 |
| A5 | 1.0000 | **0.3700** ❌ | 1.0000 ✅ | 0.5930 | 1.0000 |
| **A6** | 1.0000 | **0.3505** ❌ | 1.0000 ✅ | 0.7105 | 1.0000 |

#### Z_c AUC at max perturbation — FakeVideo-FakeAudio (both fake)

| Model | baseline | audio_mask@100% | visual_mask@100% | audio_noise@σ=1 | visual_noise@σ=1 |
|-------|:--------:|:---------------:|:----------------:|:---------------:|:----------------:|
| A2 | 1.0000 | 1.0000 | 0.9970 | 1.0000 | 0.9990 |
| A5 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **A6** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

#### Z_s AUC at max perturbation — key categories for comparison

| Model | FV-RA audio_mask | FV-RA visual_mask | RV-FA audio_mask | RV-FA visual_mask |
|-------|:----------------:|:-----------------:|:----------------:|:-----------------:|
| A2 | 0.9814 | 0.3714 | 0.4800 | 0.9830 |
| A5 | 0.9866 | 0.3255 | 0.3360 | 1.0000 |
| **A6** | 0.9916 | 0.3439 | 0.3365 | 1.0000 |

**Key findings:**

1. **Correct modality-specific collapse pattern** (all models): FV-RA Z_c collapses under visual masking (~0.33–0.36) but survives audio masking (~0.98). RV-FA Z_c collapses under audio masking (~0.35–0.42) but survives visual masking (~0.99–1.00). This confirms all three models correctly specialize their Z_c to the relevant modality for each fake type.

2. **FV-RA audio masking survival** — A2 Z_c also survives (0.9882): This contradicts Section 3.7.1's finding that A2's own detection head gives only 0.55 AUC on FV-RA. The reconciliation: A2 Z_c DOES encode video-fake features (v_c = `cat(v_c, ...)`), but A2's *trained classification head* was optimized on AV1M (audio-only fakes) so it ignores the visual fake signal. A linear probe over Z_c can extract this latent visual signal. So the binary split does capture cross-modal features — the bottleneck is downstream head specialization, not representation quality.

3. **Visual noise gradient (FV-RA) A2 > A5 > A6** ⚠️: Under visual noise σ=1.0 on FV-RA, A2 Z_c = 0.6519, A5 = 0.5251, A6 = 0.4964 (near chance). FCD Z_c is *more sensitive* to visual feature corruption on video-fake clips. This is because FCD's u_v channel explicitly encodes visual-unique features — specialized representations are more precisely query-matched to visual quality, so noise hurts more. A2's Z_c is a mixed audio+visual hash that is inherently more redundant.

4. **RV-FA audio noise** — A2 Z_c more robust (0.771) than A5 (0.593): binary Z_c retains audio detection capability better under noise. FCD's factored audio representation (s_a, u_a split) is more tightly specialized, so Gaussian noise on audio features is more disruptive.

5. **Z_s ≈ Z_c** in most rows — same pattern holds for spurious branch. For A5/A6, both branches collapse identically under visual mask (FV-RA: Z_c=0.34 ≈ Z_s=0.33) and audio mask (RV-FA: Z_c=0.37 ≈ Z_s=0.34). The causal/spurious distinction does not produce strongly different per-category behaviors — both branches see the same factored modality space. The distinction is in degree, not direction.

6. **FV-FA is near-perfect everywhere**: both manipulation signals are present — removing one modality still leaves the other. All models, both branches. This is the easy case and provides a sanity check.

**Reconciliation with Section 3.7.1**:
- Section 3.7.1 used A2's own classification head on FAVC → causal_auc FV-RA baseline = 0.55 (near chance). The head can't detect video fakes because it was trained on audio fakes.
- Section 3.11 uses fresh linear probe on frozen Z_c → baseline = 0.99. The representation CAN encode video fakes; the trained head doesn't use it.
- This is an important nuance for the paper: **FCD advantage is not representation quality but loss landscape forcing**. Both binary and FCD representations encode visual fake features. However, FCD's training objective (via u_v uniqueness pressure + OOD adversarial pressure) causes the final model weights to actually *use* those visual features, whereas binary model training on AV1M collapses to pure audio detection.

---

### 3.12 Subspace Probe — Individual FCD Components (jobs 23658622–624, ✅ done 2026-04-06)

Script: `avh_sup/eval_subspace_probe.py` — linear probe on each individual FCD subspace vector (dim=256 each), probing for label (fake/real) and domain (AV1M/FAVC) information.

Subspace dimensions: s_v [0:256], u_v [256:512], s_a [512:768], u_a [768:1024] within Z_c (dim=1024); r_v [0:256], r_a [256:512] within Z_s (dim=512). Aggregates: vis_all=cat(s_v,u_v), aud_all=cat(s_a,u_a), syn_all=cat(s_v,s_a), uniq_all=cat(u_v,u_a), Z_c=full, Z_s=full.

For A2: only Z_c and Z_s reported (no subspace decomposition).

#### A2 baseline (reference)
| Subspace | dim | in-dom label | OOD label | domain |
|----------|:---:|:------------:|:---------:|:------:|
| Z_c | 1024 | 0.9420 ± 0.008 | 0.7052 | 0.9975 |
| Z_s | 1024 | 0.9190 ± 0.008 | 0.7396 | 0.9962 |

#### A5 subspace probe
| Subspace | dim | in-dom label | OOD label | domain | Notes |
|----------|:---:|:------------:|:---------:|:------:|-------|
| **s_v** | 256 | 0.5672 ± 0.017 | **0.8138** | 0.9921 | ⚠️ Low in-dom, high OOD |
| **u_v** | 256 | 0.7879 ± 0.010 | **0.8248** | 0.9668 | Visual-unique: best OOD of all subspaces |
| s_a | 256 | 0.9906 ± 0.006 | 0.7496 | 0.9820 | Strong in-dom |
| u_a | 256 | 0.9950 ± 0.005 | 0.7570 | 0.9377 | Lowest domain AUC → less domain-correlated |
| r_v | 256 | 0.7354 ± 0.017 | 0.7890 | 0.9899 | Spurious visual |
| r_a | 256 | 0.9949 ± 0.004 | 0.7575 | 0.9836 | Spurious audio |
| *vis_all* | 512 | 0.7369 | 0.8220 | 0.9924 | |
| *aud_all* | 512 | 0.9958 | 0.7559 | 0.9836 | |
| *syn_all* | 512 | 0.9866 | 0.7686 | 0.9966 | |
| *uniq_all* | 512 | 0.9964 | 0.6909 | 0.9878 | Uniq subspc lower OOD than syn |
| **Z_c** | 1024 | **0.9970** | **0.7744** | 0.9966 | |
| Z_s | 512 | 0.9967 | 0.7078 | 0.9956 | |

#### A6 subspace probe
| Subspace | dim | in-dom label | OOD label | domain | Notes |
|----------|:---:|:------------:|:---------:|:------:|-------|
| **s_v** | 256 | 0.5297 ± 0.012 | 0.7258 | 0.9890 | Even lower in-dom than A5 (domain push-pull effect) |
| **u_v** | 256 | 0.7818 ± 0.029 | **0.9238** ⭐ | 0.9763 | **Best OOD of ALL subspaces across all models** |
| s_a | 256 | 0.9678 ± 0.009 | 0.7391 | 0.9835 | Slightly lower in-dom than A5 s_a |
| u_a | 256 | 0.9959 ± 0.004 | 0.7573 | **0.9422** | Lowest domain AUC — most domain-invariant audio |
| r_v | 256 | 0.7384 ± 0.021 | **0.8350** | 0.9922 | A6 r_v has better OOD than A5 (0.789) |
| r_a | 256 | 0.9971 ± 0.002 | 0.7619 | 0.9817 | |
| *vis_all* | 512 | 0.7042 | 0.8150 | 0.9909 | |
| *aud_all* | 512 | 0.9956 | 0.7639 | 0.9873 | |
| *syn_all* | 512 | 0.9532 | 0.6809 | 0.9961 | ⚠️ Lower in-dom than A5 syn_all (0.987) |
| *uniq_all* | 512 | 0.9963 | **0.7736** | 0.9898 | A6 uniq_all better OOD than A5 (0.691) |
| **Z_c** | 1024 | 0.9944 | 0.7480 | 0.9971 | |
| Z_s | 512 | 0.9975 | 0.7127 | 0.9974 | |

**Key findings (subspace):**

1. **u_v is the OOD champion** ⭐: A6 u_v OOD = **0.9238** — the single best any 256-dim subspace achieves. A5 u_v OOD = 0.8248. Both far above everything else. u_v is the visual-unique subspace: features that are specific to the visual modality and not shared with audio. This is exactly what the three-way FCD factorization is designed to create. **This is the strongest evidence for the paper: the u_v subspace alone generalizes to FAVC far better than the entire audio subspace.**

2. **s_v in-domain ≈ chance** (A5: 0.567, A6: 0.530): the visual-shared subspace carries almost no in-domain fake label information. This confirms s_v is capturing the *acoustic correlation* of visual movement (the AV-HuBERT shared manifold) which doesn't discriminate real/fake on AV1M. Yet s_v OOD is surprisingly high (A5: 0.814, A6: 0.726) — this means s_v does contain cross-domain discriminative information even when in-domain it's near random.

3. **Audio subspaces (s_a, u_a, r_a) dominate in-domain**: 0.968–0.997 in-domain AUC. AV1M is entirely about audio fakes, so audio subspaces encode all the in-domain signal.

4. **A6 domain push-pull effects on subspaces**: comparing A5 vs A6 —
   - u_v: A5 domain=0.967, A6 domain=0.976 (slightly higher, unexpected)
   - u_a: A5 domain=0.938, A6 domain=**0.942** (A6 u_a is the least domain-correlated audio component — adversarial pressure is working on audio uniqueness)
   - syn_all in-dom: A5=0.987 → A6=0.953 (adversarial pressure squeezes label info out of shared branch)
   - uniq_all OOD: A5=0.691 → A6=**0.774** (adversarial pressure improves OOD for unique branch)

5. **vis_all OOD > aud_all OOD** consistently: visual components have better cross-domain transfer (A5: 0.822 vs 0.756; A6: 0.815 vs 0.764). Audio features are more domain-specific (AV1M audio codec/style ≠ FAVC audio).

6. **r_v OOD is surprisingly strong** (A5: 0.789, A6: 0.835): the spurious *visual* subspace also generalizes well OOD. This corroborates Section 3.11 finding that Z_s and Z_c are nearly symmetric in cross-modal behavior — the causal/spurious split doesn't strongly partition visual vs audio, rather it partitions by training signal strength.

---

### 3.13 Head Ablation — Trained Detector's Actual Feature Usage (jobs 23658625–627, ✅ done 2026-04-06)

Script: `avh_sup/eval_head_ablation.py` — zero out specific dimensions of Z_c at inference time, measure AUC drop. This directly answers "which subspace does the trained head actually use?"

Notation: ΔAUC = ablated − baseline (negative = the ablated subspace was being used).

#### A2 head ablation

Baseline: AV1M=0.9975, FV-RA=0.4949 (near chance!), RV-FA=1.000, FV-FA=0.999, overall=0.765

| Ablation | ΔAV1M | ΔFV-RA | ΔRV-FA | ΔFV-FA | Δoverall |
|----------|:-----:|:------:|:------:|:------:|:--------:|
| mask_vc (zero v_c half) | −0.001 | **−0.034** | 0.000 | +0.000 | −0.016 |
| **mask_ac (zero a_c half)** | **−0.470** | −0.024 | **−0.463** | **−0.616** | **−0.337** |

**A2 head practically ignores visual causal features** (v_c ablation: Δ AV1M = −0.001). All task performance lives in a_c. Zeroing a_c destroys AV1M detection (−0.470) and RV-FA (−0.463). FV-RA was already near chance (0.49) and stays near chance — the head never learned to use visual features for FV-RA detection. This directly proves the "A2 is an audio-only detector" claim from Section 3.7.1.

#### A5 head ablation

Baseline: AV1M=0.9984, FV-RA=0.5860 (still low!), RV-FA=0.9994, FV-FA=0.9870, overall=0.8634 (n.b. higher than A2's 0.765 due to multimodal training)

| Ablation | ΔAV1M | ΔFV-RA | ΔRV-FA | ΔFV-FA | Δoverall |
|----------|:-----:|:------:|:------:|:------:|:--------:|
| mask_sv | −0.000 | **−0.040** | +0.001 | +0.002 | **−0.018** |
| **mask_uv** | −0.001 | **−0.060** | +0.000 | +0.000 | **−0.028** |
| mask_sa | −0.002 | **+0.193** ⬆️ | −0.001 | −0.008 | **+0.086** |
| mask_ua | **−0.056** | +0.077 | −0.036 | −0.039 | +0.015 |
| mask_vis (s_v+u_v) | −0.003 | **−0.113** | +0.001 | +0.002 | **−0.052** |
| **mask_aud (s_a+u_a)** | **−0.241** | **+0.207** ⬆️ | **−0.447** | **−0.181** | −0.007 |
| mask_syn (s_v+s_a) | −0.008 | **+0.218** ⬆️ | −0.002 | −0.010 | **+0.096** |
| mask_uniq (u_v+u_a) | **−0.027** | −0.060 | −0.003 | −0.012 | **−0.034** |

**A5 head still primarily uses audio** (mask_aud Δ AV1M = −0.241). But crucially:
- mask_vis (zero all visual): FV-RA drops by −0.113 → A5 head IS using visual features for video-fake detection, unlike A2
- mask_sa FV-RA = **+0.193** (removing shared audio HELPS FV-RA detection): the shared audio block was providing misleading audio-correlation signal that confused the head on FV-RA cases; removing it frees the model to use visual channels
- mask_aud FV-RA = **+0.207** (removing all audio helps FV-RA detection) — same mechanism at scale

#### A6 head ablation

Baseline: AV1M=0.9984 (same as A5), FV-RA=**0.7207** (much better than A2/A5 ✅), RV-FA=1.000, FV-FA=0.9959, overall=**0.9284** (best of all models)

| Ablation | ΔAV1M | ΔFV-RA | ΔRV-FA | ΔFV-FA | Δoverall |
|----------|:-----:|:------:|:------:|:------:|:--------:|
| mask_sv | +0.000 | **+0.060** ⬆️ | 0.000 | +0.001 | **+0.028** |
| **mask_uv** | −0.002 | **−0.354** ⭐ | 0.000 | +0.000 | **−0.165** |
| mask_sa | −0.001 | **+0.068** ⬆️ | 0.000 | 0.000 | **+0.031** |
| **mask_ua** | **−0.155** | +0.008 | **−0.254** | **−0.257** | **−0.134** |
| mask_vis | −0.002 | **−0.237** | 0.000 | +0.001 | **−0.110** |
| **mask_aud** | **−0.234** | **+0.067** ⬆️ | **−0.450** | **−0.220** | −0.092 |
| mask_syn | −0.002 | **+0.141** ⬆️ | 0.000 | +0.001 | **+0.066** |
| **mask_uniq** | **−0.285** ⭐ | **−0.379** ⭐ | **−0.131** | **−0.652** ⭐ | **−0.512** ⭐ |

**A6 head usage is qualitatively different from A2 and A5:**

1. **u_v is the critical FV-RA detector** ⭐: mask_uv FV-RA = −0.354. Zeroing only u_v (the visual-unique 256-dim subspace) drops FV-RA detection by 35pp. Nothing like this exists in A2 (Δ=−0.034) or A5 (Δ=−0.060). A6's domain adversarial pressure has caused the trained head to *actually utilize* the u_v visual-unique channel.

2. **mask_uniq is catastrophic across the board** ⭐: Δ AV1M = −0.285, FV-RA = −0.379, FV-FA = **−0.652**, overall = −0.512. Zeroing both unique channels (u_v + u_a) essentially destroys the A6 head. This is unlike A5 (mask_uniq overall = −0.034). A6's head has strongly concentrated task performance in the unique subspaces.

3. **mask_sv and mask_sa have POSITIVE FV-RA deltas** for A6 (+0.060 and +0.068): exactly the same mechanism as A5's mask_sa (+0.193) but weaker — shared subspaces provide conflicting audio-correlation signal, and their removal helps the model focus on u_v for video-fake detection.

4. **mask_ua dominant for RV-FA and FV-FA** (Δ −0.254 and −0.257): the audio-unique channel (u_a) is the primary detector for audio fakes. NOT u_a+s_a together — specifically the u_a unique component.

5. **A6 baseline FV-RA = 0.7207** vs A2 = 0.4949 and A5 = 0.5860: this gap **directly traces to u_v usage**. Both A2 and A5 have u_v in their subspace probe (OOD label ≈ 0.92) but their heads don't use it (mask_uv Δ FV-RA: A2=−0.034, A5=−0.060). A6's adversarial training forces the head to route FV-RA detection through u_v (mask_uv FV-RA: −0.354).

**Summary table — where each model's head routes FV-RA detection:**

| Model | FV-RA baseline | Primary FV-RA pathway | Δ when pathway removed |
|-------|:--------------:|----------------------|:---------------------:|
| A2 | 0.4949 (chance) | a_c (audio) — but audio is real, so random | −0.024 |
| A5 | 0.5860 (low) | vis block (s_v+u_v) partial | −0.113 |
| **A6** | **0.7207** ✅ | **u_v exclusively** | **−0.354** ⭐ |

**This is the strongest evidence for the paper**: the FCD three-way factorization + domain adversarial pressure caused the trained model to develop a dedicated visual-unique fake detection pathway (u_v) that activates specifically for video fakes. Binary causal/spurious decomposition cannot produce this because there is no mechanism to force visual-unique features into a separate subspace that the head learns to query.

### 3.9 Modality-Origin Probe Results (FV-RA vs RV-FA — which modality was faked?)

Probe design: linear classifier trained on frozen Z_c or Z_s, predicting FV-RA (video fake, label=0) vs RV-FA (audio fake, label=1) from FAVC test clips.
Note: class imbalance — 1854 FV-RA vs 100 RV-FA.

| Model | n_FV-RA | n_RV-FA | Z_c AUC | Z_s AUC |
|-------|---------|---------|---------|------|
| A2 | 1854 | 100 | 0.9988 ± 0.0025 | 0.9992 ± 0.0015 |
| **A6** | 1854 | 100 | **1.0000 ± 0.000** | 1.0000 ± 0.000 |

Note: Near-perfect AUC for both models, both branches. This is consistent with Section 3.7 (manipulation probe) — audio synthesis vs no-audio-synthesis is trivially detectable in all representations. The severe class imbalance (1854:100) means the probe is essentially identifying the rare RV-FA class by audio-manipulation signal, which all branches encode strongly. This probe does NOT distinguish FCD from binary split — the modality-of-origin signal is too strong in both. **Not useful as a differentiating experiment for the paper.**

### 3.10 Manipulation Probe Results (in-domain, FakeVideo-RealAudio vs FakeVideo-FakeAudio)

Probe design: can the representation distinguish FV-RA from FV-FA? (i.e., does it detect whether audio was also manipulated given that video is fake)

| Model | Raw AUC | Z_c AUC | Z_s AUC |
|-------|---------|---------|---------|
| A0 | 1.0000 | 0.9999 | 0.9999 |
| A1 | 1.0000 | 0.9999 | 1.0000 |
| A2 | 1.0000 | 0.9999 | 0.9999 |
| A3 | 1.0000 | 0.9999 | 1.0000 |
| **A6** | 1.0000 | 0.9995 | 0.9999 |

Near-perfect for all, including A6: every representation can distinguish audio manipulation from non-manipulation when video manipulation is held constant. This tells us audio manipulation signal is trivially present in all representations — confirming audio dominance. The manipulation probe is therefore useful mainly as a sanity check, not as a differentiating experiment.

---

## 4. End-to-End (E2E) Experiments

### 4.1 Architecture Shared Across E2E Variants
All e2e variants use AVHubertWrapper (AV-HuBERT, 325M params) + classification head.
- n_unfreeze_layers: -1 (all layers) in most variants
- batch_size: 8, accumulate_grad_batches: 4 → effective batch = 32
- max_frames: 150 (cap at 6s @ 25fps)
- lr_encoder: 5e-6 (very small), lr_head: 1e-4 or 1e-3
- Domain data: FAVC val-real clips oversampled (favc_oversample: 3–14×) for domain signal

### 4.2 E2E Experiments by Domain Alignment Method

#### Phase 1: SAD + Distribution Alignment (A9–A18)
All achieve in-domain val_auc_causal ≈ 1.0 but cross-domain eval not run for most.

| Exp | Domain Method | Application Point | Augmentation | Best val |
|-----|-------------|-----------------|-------------|----------|
| A9_e2e | MMD | Z_sync (512-d) | None | 1.0000 |
| A10_e2e | CORAL | Z_sync (512-d) | Strong (flip, noise, brightness) | 0.9999 |
| A11_e2e | CORAL | Encoder output (2048-d) | Strong | 0.9999 |
| A12_e2e | CORAL variant | Encoder output | Strong | 0.9997 |
| A13_e2e | CORAL variant | Encoder output | Strong | 0.9956 |
| A14_e2e | CORAL variant | Encoder output | Strong | 0.9998 |
| A15_e2e | RFF-MMD | Encoder output | Strong | 1.0000 |
| A16_e2e | LayerNorm+CORAL | Encoder output | Strong | 0.9995 |
| A17_e2e | DANN (GRL) | Encoder output | Strong | 0.9995 |
| A18_e2e | Dual DANN | Encoder + Z_sync | Strong | 0.9929 |

#### Phase 2: Strong Adversarial — FAILED ZONE (A19–A25, A30)

**Common failure mode: lambda ≥ 10-15 DANN/MMD on 325M encoder kills task signal.**

| Exp | Domain Method | Lambda | val_causal | Status |
|-----|-------------|--------|-----------|--------|
| A19_e2e | LayerNorm + Dual DANN | 15 | ~0.4452 | ❌ FAILED |
| A20_e2e | Dual DANN, no λ_ddis | 10–15 | ~0.5000 | ❌ FAILED |
| A21_e2e | BatchNorm + Dual DANN | 15 | ~0.5000 | ❌ FAILED |
| A22_e2e | BN on enc+Z_sync + DANN | 15 | ~0.5000 | ❌ FAILED |
| A23_e2e | RFF-MMD on Z_sync + enc DANN | λ_zsync=25, λ_enc=15 | ~0.5000 | ❌ FAILED |
| A24_e2e | CDAN + enc DANN | 15 | ~0.5188 | ❌ FAILED |
| A25_e2e | Mean centering + Dual DANN | 15 | ~0.5031 | ❌ FAILED |
| A30_e2e | Whitened MMD (Cohen's d²) | 1.0 (effective large) | ~0.5000 | ❌ FAILED |

Root cause: 325M encoder vs ~260K classifier. At lambda ≥ 10-15, adversarial gradient from domain classifier overwhelms the task signal in the encoder. The encoder converges to domain-agnostic but task-useless features.

Fix discovered: lambda_grl = 0.1–0.2 only on mean-pooled Z_c (not on raw encoder layers).

#### Phase 3: Recovery — Moderate Domain Alignment (A26–A41)

| Exp | Architecture | Cross-Domain AUC | In-Domain AUC | In-Dom Spu |
|-----|-------------|:----------------:|:-------------:|:----------:|
| **A6_e2e** | FCD + domain adv/disc | **0.8667** | 1.0000 | 0.6209 |
| A34_e2e | IPW + light GRL (λ=0.2) | 0.8115 | 1.0000 | 0.5369 |
| A26_e2e | SAD + mean align + DANN | 0.7914 | 0.9999 | **0.1748** |
| A2_e2e_dadv | A2_e2e + domain adv | 0.7836 | 1.0000 | 0.5384 |
| A1_e2e_dadv | A1_e2e + domain adv | 0.7630 | 1.0000 | 0.5321 |
| A34_v2_e2e | IPW + gentler GRL + balanced sampler | 0.7466 | 1.0000 | 0.4004 |
| A33_e2e | IPW backdoor adjustment | 0.7438 | 1.0000 | 0.4309 |
| A27_e2e | — | 0.7280 | 1.0000 | 0.4969 |
| A28_e2e | — | 0.7177 | 1.0000 | 0.0576 |
| A2_e2e | A2 end-to-end baseline | 0.7079 | 0.9990 | 0.3198 |
| A39_e2e | — | 0.6814 | 1.0000 | 0.5843 |
| A41_e2e | — | 0.6618 | 1.0000 | 0.4776 |
| A29_e2e | — | 0.6553 | 1.0000 | 0.3120 |

Notable:
- A6_e2e (0.8667) is far ahead of everything else in e2e
- A26_e2e achieves lowest spurious AUC (0.1748) = best disentanglement in e2e
- A2_e2e baseline (0.7079) is significantly below A6_e2e (0.8667): +15.9pp from FCD+domain

### 4.3 E2E Domain Alignment Methods Compared

| Method | Used in | Principle | Gradient Target | Works? |
|--------|---------|-----------|----------------|--------|
| MMD² (RBF, per-batch) | A9, A23 | OT-free distribution matching | Z_sync or enc | Partial |
| CORAL | A10–A14, A16 | Covariance alignment | Z_sync or enc | Partial |
| RFF-MMD | A15 | Random Fourier Features MMD | enc | Partial |
| DANN (GRL) | A17–A25 | Adversarial domain classifier | enc / Z_sync | ❌ at λ≥10 |
| CDAN | A24 | Conditional DANN | Z_sync × task | ❌ at λ≥10 |
| Mean centering | A25 | Structural mean alignment | Z_sync | ❌ at λ=15 |
| Whitened MMD (Cohen's d²) | A30 | Scale-invariant mean alignment | Z_sync | ❌ |
| Mean align (MSE, λ=10) | A26 | Direct MSE(mu_0, mu_1) | Z_sync | ✓ (partial) |
| IPW backdoor adjust | A33 | Reweight task loss by domain pred | Classifier only | ✓ |
| IPW + light GRL (λ=0.2) | A34 | IPW + gentle domain pressure | Z_c (pooled) | ✓ |
| FCD + domain push-pull | A6_e2e | Structured decomp + push-pull | Z_c/Z_s (pooled) | ✓ (best) |

---

## 5. Individual Experiment Notes

### A0 — Control (Orth loss only)
- use_adv=False, use_sg=False: full_head(Z_c+Z_s) only
- Z_s AUC = 0.7764 (label leaks into spurious — complete failure of disentanglement)
- Audio masking at 100% → causal=0.3448 (worst: model barely above random without audio)
- Domain predictability: Z_c=0.9976, Z_s=0.9969 (both totally domain-predictable)
- Seed variants: causal head val_auc ≈ 0.5 (causal_head not trained in A0 — expected)

### A1 — Adversarial Only
- GRL on Z_s, but full_head(Z_c+Z_s) for classification
- Spurious AUC = 0.2866 (below 0.5 — see note in section 6)
- Z_s label probe AUC = 0.9588 > Z_c = 0.9475: spurious branch retains more label signal than causal
- Best epoch: 3 (early convergence, may indicate instability)

### A2 — Adversarial + Stop-Gradient (Core Baseline)
- causal_head(Z_c) only for classification; Z_s gradient only from GRL
- Best frozen binary model: 0.7928 overall
- Z_s label probe = 0.9187 (−0.047 vs raw) — meaningful but incomplete disentanglement

### A3 — A2 + Domain Adversarial
- Mixes AV1M (domain=0) + FAVC real clips (domain=1) during training
- Domain GRL on Z_c (expels domain) + domain disc on Z_s (concentrates domain)
- **Hurts performance: 0.7490 vs A2's 0.7928**
- FV-RA = 0.4496 (worse than random for visual-only fakes)
- Spurious AUC rises to 0.3778 (more uniform) but task AUC falls
- Root cause: domain GRL on Z_c conflicts with task loss in binary framework

### A4 — A2 + β-VAE
- Z_s projected to 32-d bottleneck (vs 512 in A2) — geometric information bottleneck
- lambda_adv=0.0, grl_alpha=0.0 (GRL disabled — KL pressure alone separates Z_s)
- beta_c=0.5, beta_s=2.0; beta_warmup_epochs=10
- Best epoch: 0 (stopped at first checkpoint — convergence failure or dataset issue)
- In-domain val_auc = 0.9856 (lowest of all models)
- No cross-dataset eval run

### A5 — FCD (Three-Way Decomposition)
- Best model before domain-aware training
- FV-RA: 0.6156 (vs A3's 0.4496 — +16.6pp from three-way decomposition)
- Temporal shuffle: 0.8186→0.8326 (+0.014) — improves with shuffle → not using temporal structure
- Audio masking 100%: 0.6077 (vs A2's 0.4622 — +14.5pp)
- Domain pred Z_c: 0.9943 (best of all, still very high)
- SVD regularization on URD every 50 steps
- Sinkhorn n_iter reduced from 50 to 5 in A6 (speed optimization)

### A6 — FCD + Domain Push-Pull (BEST FROZEN MODEL)
- Overall FAVC AUC: 0.8885 (+6.99pp vs A5, +9.57pp vs A2)
- FV-RA: 0.7568 (+15.1pp vs A5)
- Audio masking 100%: 0.7691 (+16.1pp vs A5) — genuinely bimodal
- Audio vs visual mask degradation: −0.119 vs −0.121 (balanced, unlike all others)
- Temporal shuffle: +0.005 (robust)
- Why it works vs A3: FCD structure provides cleaner gradient separation. Domain adversarial applied to pooled Z_c (not per-frame) with separate domain head reduces conflict with task loss.
- Sequence-based probes now complete: Z_c=0.9944 / Z_s=0.9975 in-domain, OOD Z_c=0.7480, manipulation probe Z_c=0.9995
- Remaining major gap: seed sweep

### A7 — A6 + Label Adversarial on Z_s
- Adds GRL(Z_s_pool) → label_head_s → BCE(fake/real) to expel label from Z_s
- Motivation: A6 had in-domain label probe showing Z_s > Z_c AUC (inverted label info in Z_s)
- Achieves lowest spurious AUC: 0.2674
- But overall AUC drops: 0.8251 vs A6's 0.8885 (−6.3pp)
- FV-RA: 0.6168 (vs A6's 0.7568 — −14pp)
- Trade-off: better disentanglement but worse detection
- Best epoch: 5 (fast convergence with combined objectives)

### A8 — SAD Synchrony-Appearance Decomposition
- Architecturally different: Z_sync = interaction(V,A) as causal
- Audio masking 100%: 0.1510 ❌ — catastrophic failure
- Root cause: Z_sync = cat(V, A, V⊙A, V−A) — V⊙A and V−A collapse to zero without audio, destroying all signal
- Temporal shuffle: −0.092 (moderate degradation — more temporal-dependent than FCD)
- Spurious AUC = 0.3823 (highest among supervised models — less label-suppression)
- Conclusion: "AV interaction as causal signal" is structurally wrong for this task

### A9 — SAD + MMD
- MMD² with multi-kernel RBF to replace GRL for domain alignment
- batch_size=16 required (vs batch_size=1 in others) for per-batch MMD
- Best epoch: 84 (slow convergence, likely MMD struggling with batch variance)
- No cross-dataset eval run

### A42 — A2 + A6-Style Domain Adv/Disc (Binary Split)
- Key comparison: A6's domain push-pull applied to binary Z_c/Z_s (without FCD)
- Overall: 0.7991 vs A6's 0.8885 → confirms FCD structure is critical, not just domain objective
- FV-RA: 0.5594 (vs A6's 0.7568 — −19.7pp) — binary split still audio-dominant

### A43 — A42 with Stronger Domain Pressure
- lambda_dadv: 1→5, grl_domain_alpha: 1→3, lambda_ddis: 1→2
- epochs: 10 (converged by epoch 5 in A42)
- Result: 0.7840 < A42's 0.7991 — stronger domain pressure slightly hurts
- FV-RA: 0.5264 < A42's 0.5594 — worse

### A44 — A42 + Sinkhorn Cross-Modal Alignment
- Adds SDA loss (Sinkhorn between v_c and a_c projections) to A42
- Result: 0.7667 < A42's 0.7991 — SDA hurts binary split ❌
- Explanation: forcing v_c ≈ a_c without modality-specific unique channels forces visual projector toward easier audio solution (amplifies audio dominance)
- This is the opposite of A5's SDA effect (which has unique channels to anchor each modality)
- Best epoch: 2 (extremely fast — model collapsed quickly)

### B1 — A2 + Per-Modality Classification
- Adds modal_head_visual(v_c) and modal_head_audio(a_c) supervised on labels
- FAVC eval N=4089 (full set, different from other N=1114 evaluations)
- FV-RA: 0.4705 (marginal improvement over A2's assumed ~0.45)
- Evaluated on 4089 clips — not directly comparable to other experiments

### B1b — A2 + Per-Modality + Balance Penalty
- Adds (CE_audio - CE_visual)² to equalize gradient magnitude across modalities
- Cross-dataset eval now run on the standard N=1114 subset: overall 0.8430, RV-FA 1.000, FV-RA 0.6560, FV-FA 0.9996
- Audio masking 100%: 0.6085 (comparable to A5)
- Spurious head remains weak/inverted: 0.4240 overall spurious AUC

### B2 — A2 + Cross-Modal Alignment (Real Samples Only)
- Cosine alignment between v_c and a_c, restricted to label=0 (real) clips
- Base: 0.7891 (slight improvement over A2's 0.7928 baseline)
- Rationale: fake clips may have unilateral manipulation, so unconditional alignment would destroy fake signal

### B3 — A2 + Modal Discriminator Adversarial
- GRL between projectors and binary modality discriminator
- Forces v_c and a_c to become modality-agnostic (indistinguishable)
- Base: 0.8079, audio masking 100%: 0.6200

---

## 6. Identified Weaknesses and Issues

### 6.1 Sub-0.5 Spurious AUC (Inverted Label Signal)
Multiple models have spurious head AUC well below 0.5:
- A7: 0.2674, A44: 0.3002, A43: 0.3135, A42: 0.3054, A5: 0.3163

Values below 0.5 mean the spurious head predicts *the wrong class* — anti-predicting fakes. This means Z_s still contains label-relevant information, just with inverted polarity. The probe head (trained from scratch) learns to invert the signal. This contradicts the disentanglement claim and must be either explained (score sign convention in the spurious head) or fixed (add explicit label adversarial on Z_s like A7 does).

### 6.2 Domain Invariance Not Achieved (All Models)
Domain predictability remains extremely high for all models (roughly 0.992–0.998 AUC). Even A6 only reduces Z_c domain AUC from A2's 0.9979 to 0.9923. The causal features remain highly domain-predictable, so the paper should avoid claiming true domain invariance.

### 6.3 Inconsistent Test Set Sizes
- Most frozen experiments: N=1,114 (balanced subset of FAVC test)
- B1b eval: N=1,114 (now reconciled with the standard frozen setting)
- B1 favc_eval: N=4,089 (full FAVC test)
- E2E eval_e2e_auc: N=500 (balanced 500 AV1M + 500 FAVC)
All three are different. Need to reconcile before paper submission.

### 6.4 A3 Worse Than A2 (Domain Adversarial Backfires in Binary Framework)
Adding domain adversarial to the binary Z_c/Z_s split (A3) decreases performance from 0.7928 to 0.7490. Same objective applied to FCD (A6) improves from 0.8186 to 0.8885. The structural incompatibility between domain adversarial and binary causal/spurious decomposition is a key theoretical finding.

### 6.5 A8 SAD Architecture Failure on Audio Masking
The "AV interaction = causal" assumption leads to catastrophic failure when audio is removed (AUC 0.1510). The V⊙A term in Z_sync collapses without audio. This architectural hypothesis is falsified.

### 6.6 A44 SDA Hurts Binary Split
Adding Sinkhorn cross-modal alignment to binary Z_c/Z_s (A44) degrades performance (0.7991→0.7667). The same SDA alignment helps FCD (A5 vs A2 without SDA would be worse). The cross-modal alignment objective requires modality-specific unique channels to be effective.

### 6.7 Temporal Shuffle Robustness Paradox
A5 and A6 are temporally robust (shuffle Δ≈0 or even slight improvement). This reveals a potential theoretical issue: if the models don't use temporal structure, they cannot be detecting "AV synchrony breaks" which require temporal alignment. They are likely detecting per-frame synthesis artifacts instead.

---

## 7. Architecture Details: Loss Functions and Hyperparameters

### A6 Loss Weights (Best Config)
```
L_total = L_task + 1.0·L_MI + 0.5·L_SDA + 1.0·L_Dis + 0.1·L_orth
        + 1.0·L_dadv + 1.0·L_ddis
grl_alpha = 1.0 (for domain adversarial)
sda_eps = 0.05, sda_n_iter = 5
lr = 1e-3
```

### A2 Loss Weights (Core Baseline)
```
L_total = L_cls + 10.0·L_adv + 0.5·L_orth
grl_alpha = 5.0
lr = 1e-3
```

### Shared Components
- LogSumExp temporal pooling: `logsumexp(head(repr)[...,0], dim=-1)` — clip-level score
- CE loss: `F.cross_entropy(stack([-score, score], dim=1), labels)` — binary via 2-logit formulation
- Orth loss: `mean(cosine(Z_c_frame, Z_s_frame)²)` — per-frame mean-squared cosine

---

## 8. Paper Framing Analysis

### Core Narrative
1. Binary causal/spurious decomposition fails due to audio dominance
2. Audio dominance is mathematically unavoidable when I(audio;Y) >> I(visual;Y) in binary framework
3. FCD three-way decomposition (synergistic/unique/redundant) resolves (1) via modality-specific unique channels
4. Domain adversarial applied to FCD's pooled representation resolves domain generalization without gradient conflict
5. A3 failure (same objective, wrong structure) confirms (4) is architecture-dependent

### Evidence Chain
- A0 audio mask Δ=-0.426, visual mask Δ=-0.024 → audio dominance in baseline
- A2 audio mask Δ=-0.330, visual mask Δ=-0.035 → binary adversarial doesn't fix audio dominance
- A5 audio mask Δ=-0.211, visual mask Δ=-0.059 → FCD fixes audio dominance partially
- A6 audio mask Δ=-0.119, visual mask Δ=-0.121 → balanced (solved)
- A3 (0.7490) < A2 (0.7928): domain adversarial on binary split HURTS
- A6 (0.8885) > A5 (0.8186): domain adversarial on FCD HELPS
- A42 (0.7991) << A6 (0.8885): same domain objective on binary split is insufficient

### NeurIPS Weaknesses
1. "Causal" framing narrative only, no formal SCM/identification
2. A6 = FCD (cited NeurIPS 2025) + domain adv — needs clear novelty articulation
3. Domain invariance claim quantitatively unsupported (domain pred AUC still ≈0.992+)
4. Sub-0.5 spurious AUC unexplained
5. No SOTA comparison on standard benchmarks
6. Single seed results; no variance

---

## 9. Missing Experiments (Priority Order)

### Critical for paper:
1. A6 seed sweep (A6_seed42, A6_seed43, A6_seed44) — at least 3 seeds for all key models
2. Consistent test set: re-run B1 eval on N=1114 subset
3. Perturbation robustness for A42/A43/A44
4. Domain predictability for A7, A8, A42, A43, A44

### Important for paper:
5. A4 full evaluation (β-VAE — currently only epoch=0 checkpoint)
6. A9 evaluation (SAD+MMD — 84 epochs, presumably converged)
7. Published SOTA comparison on FakeAVCeleb

### For NeurIPS upgrade:
8. OOD domain ablation: train A6 with a different domain (not FAVC val-real) and still eval on FAVC test — rules out test-distribution leakage critique
9. MINE or HSIC estimates of I(Z_c; D) to empirically connect theory to experiments

---

## 10. Information-Theoretic Proof Strategy

### Goal
Prove that FCD decomposition bounds target-domain generalization error, and binary decomposition cannot achieve the same bound due to audio dominance.

### Proposition 1: Domain Generalization Bound (Tractable)
From Ben-David et al. (2010) + Pinsker's inequality:
```
R_T ≤ R_S + 2√(I(Z_c; D)) + λ*
```
Minimizing I(Z_c; D) via L_dadv directly bounds target risk. Proof: straightforward.

### Proposition 2: Binary Decomp Audio Dominance Lower Bound (Novel, Medium Difficulty)
For binary Z_c with single task loss:
```
I(Z_c^binary; M) ≥ |I(X_a; Y) - I(X_v; Y)| - ε(n)
```
Audio dominance in Z_c is mathematically unavoidable when audio task signal exceeds visual.
Proof sketch: data processing inequality + gradient dynamics under cross-entropy.
Corollary: I(Z_c^binary; D) ≥ lower bound > I(Z_c^FCD; D) when domain correlates with modality preferences.

### Proposition 3: PID Alignment (Hard, Needs Gaussian Assumption)
Under Gaussian approximation, FCD decomposition implements Partial Information Decomposition:
- Z_c captures Π_syn + Π_uniq_V + Π_uniq_A (synergistic + unique task info)
- Z_s captures Π_red (redundant = domain-specific shortcut info)
Therefore: I(Z_c^FCD; D) < I(Z_c^binary; D) strictly.
Key reference: Bertschinger et al. (2014) "Quantifying Unique Information" (I-BROJA measure).

### Main Theorem
FCD achieves strictly tighter generalization bound than binary decomposition.

### Key References
- Ben-David et al. (2010) — domain adaptation bound
- Williams & Beer (2010) — PID (discrete)
- Bertschinger et al. (2014) — I-BROJA for continuous PID
- Federici et al. (2020) — multi-view information bottleneck (closest existing work)
- Zhao et al. (2019) — invariant representation + generalization bounds

### Empirical Connection
Domain predictability AUC = proxy for I(Z_c; D):
- A2: 0.9979, A5: 0.9943, A6: 0.9923
For tighter connection: run MINE (Belghazi 2018) or HSIC to estimate I(Z_c; D) numerically.

---

## 11. Recommended Next Steps

### Week 1–2:
1. Run A6 with seeds 42, 43, 44
2. Re-run B1 eval on the N=1114 subset for apples-to-apples comparison
3. Run perturbation robustness for A42/A43/A44
4. Run domain predictability for A7, A8, A42, A43, A44

### Week 3–4:
5. Locate published SOTA numbers on FakeAVCeleb (FACTOR, AVoiD-GAN, etc.)
6. Design and run OOD-domain ablation for A6 (use different OOD domain during training)
7. A4 full evaluation
8. A9 full evaluation

### Month 2 (for NeurIPS theory):
9. Implement Prop 1 (Ben-David bound): ~2 days
10. Implement Prop 2 (audio dominance lower bound): ~1 week
11. Gaussian PID for Prop 3: ~2–3 weeks
12. MINE estimation for empirical I(Z_c;D): ~3 days

---

## 12. Meeting Notes — 2026-04-06 (Karen)

### Strategic Pivot: Framing

**Avoid**: "disentanglement" framing (too vague, invites domain-AUC attack)
**Use instead**: "factorization of multimodal representation space" — more concrete and mechanistic

Karen's proposed three-contribution structure for NeurIPS:

1. **Identify fundamental limitation of binary AV decomposition**: the binary split conflates modality-specific useful signals with domain-specific (spurious) signals, making disentanglement by loss design alone insufficient. (Supported by A3 < A2, A42 << A6, A44 hurts.)

2. **Introduce three-way factorization**: decompose multimodal representation into shared (synergistic) + modality-specific (unique) + residual (redundant) subspace — provides a more expressive and controllable representation bottleneck. (FCD = the proposed mechanism.)

3. **Yield substantial gains**: the approach achieves strong cross-domain generalization and robustness under perturbations. Name TBD (can name the full approach). (Supported by A6 0.8885 vs A2 0.7928 frozen; A6_e2e 0.8667 vs A2_e2e 0.7079.)

### Key Story

How the three-way factorization changes *what* each branch actually captures, compared to binary decomposition. This is what the analysis experiments (label predictability, domain predictability, perturbation per-branch) need to demonstrate.

### Action Items (from meeting)

**Immediate priority — Analysis experiments:**
- Per-branch label predictability for binary (A2) vs. FCD (A5/A6): show that Z_c^FCD captures less audio-only label signal than Z_c^binary
- Per-branch domain predictability for A2 vs. A5/A6: show that Z_c^FCD is less domain-predictable (or that Z_s^FCD concentrates domain better)
- Determine which perturbations best demonstrate that FCD forces branch specialization:
  - Audio masking / noise: already shows A6 Z_c survives, A2 Z_c collapses (done)
  - Per-branch response to audio masking (if Z_s^FCD = redundant/audio-dominant, masking should kill Z_s but not Z_c)
  - Visual masking per-branch counterpart
  - Per-branch probing under perturbation (new: run label probe after corrupting one modality)

**Share perturbation testing plan with Karen ASAP** (she requested a written plan like the one shared the morning of 2026-04-06).

### De-prioritized (Karen's direction)
- Chasing domain AUC reduction (Z_c domain pred ≥ 0.994) — acknowledged but not the primary claim
- Disentanglement narrative framing
- These can be addressed briefly: "domain feature compression improves generalization without requiring full domain invariance"

### Plan to Share with Karen (draft)

**Perturbation × Branch Analysis Grid:**

| Perturbation | Metric | Branch | Purpose |
|---|---|---|---|
| Audio masking 0–100% | Label probe AUC | Z_c, Z_s for A2 vs A6 | Show Z_c^FCD retains label signal when audio removed; Z_c^binary collapses |
| Visual masking 0–100% | Label probe AUC | Z_c, Z_s for A2 vs A6 | Show symmetric behavior in FCD |
| Audio noise 0–1.0 | Direct AUC | Z_c for A2, A5, A6 | Already have data; extend with per-branch probe |
| Clean (no perturbation) | Domain probe AUC | Z_c, Z_s for A2 vs A5/A6 | Quantify how much domain is concentrated in Z_s^FCD vs binary |
| Clean | Modality-of-origin probe | Z_c for A2 vs A6 | Can we predict if a clip has fakeV or fakeA from Z_c alone? |

### Submitted Jobs (2026-04-06)

| Job ID | Experiment | Status | Key Results |
|--------|-----------|--------|-------------|
| 23656617 | Label probe A6 | ✅ Done | Z_c=0.7704, Z_s=0.8316, domain_Z_c=0.9908, OOD_Zc=0.7226 |
| 23656618 | Domain predictability A6 | ✅ Done | causal=0.9923, spurious=0.9937 |
| 23656619 | Branch probe × perturbation A2 | ✅ Done | audio_mask@100%: Zc=0.5332 (collapse); visual_mask@100%: Zc=0.9057 |
| 23656620 | Branch probe × perturbation A5 | ✅ Done | audio_mask@100%: Zc=0.6990; visual_mask@100%: Zc=0.9958 |
| 23656621 | Branch probe × perturbation A6 | ✅ Done | audio_mask@100%: Zc=0.6624; visual_mask@100%: Zc=0.9960 |
| 23656622 | Modality-origin probe A2 | ✅ Done | Z_c=0.9988, Z_s=0.9992 (trivial — see §3.7) |
| 23656623 | Modality-origin probe A6 | ✅ Done | Z_c=1.000, Z_s=1.000 (trivial — see §3.7) |

**New scripts created:**
- `avh_sup/eval_branch_probe_perturb.py` — per-branch label probe at each perturbation level (A2/A5/A6)
- `avh_sup/eval_modality_origin_probe.py` — FV-RA vs RV-FA probe from Z_c (A2/A6)

**Expected results to fill in table (Section 3.5 extension):**

Branch probe under perturbation — expected pattern:
- A2 Z_c probe AUC: high at 0% audio mask, collapses toward 0.5 as mask increases (audio-dominated)
- A6 Z_c probe AUC: stays high even at 100% audio mask (u_v visual channel survives)
- A6 Z_s probe AUC: collapses under audio mask (Z_s = redundant, audio-correlated)
- A6 visual masking: symmetric degradation (u_a audio channel provides backup signal)

Modality-origin probe — expected pattern:
- A6 Z_c: AUC > 0.5, balanced (u_v activates for FV-RA, u_a for RV-FA)
- A2 Z_c: potentially audio-biased (high AUC but driven by RV-FA being easy, FV-RA hard)

### Follow-up Experiments Submitted (2026-04-06, batch 2)

**Motivation:** Four gaps identified after first-batch results:
1. Label/domain probe used pre-mean-pooled features (fake T=1 input) — needs full-sequence re-run.
2. Branch probe covered only masking; missing noise perturbations at branch level.
3. AV1M branch probe limited (audio-only fakes) — FAVC per-category probe with FV-RA is the key test.
4. A2 per-category perturbation missing masking + temporal_shuffle types (A6 had all 5).

**New scripts created (2026-04-06 batch 2):**
- `avh_sup/eval_label_probe_seq.py` — sequence-based label + domain probe (correct T>1 input to _encode)
- `avh_sup/eval_branch_probe_perturb.py` — **updated**: added `audio_noise` + `visual_noise` to PERTURB_CONFIG
- `avh_sup/eval_favc_branch_probe_perturb.py` — FAVC per-category (FV-RA/RV-FA/FV-FA) branch probe under all 4 perturbation types

| Job ID | Experiment | Status | Expected key result |
|--------|-----------|--------|---------------------|
| 23656786 | Label probe (seq) A2 | ✅ Done | Z_c=0.9420, Z_s=0.9190, OOD_Zc=0.7052, Domain_Zc=0.9975 |
| 23656787 | Label probe (seq) A5 | ✅ Done | Z_c=0.9970, Z_s=0.9967, OOD_Zc=0.7744, Domain_Zc=0.9966 |
| 23656788 | Label probe (seq) A6 | ✅ Done | Z_c=0.9944, Z_s=0.9975⚠️, OOD_Zc=0.7480, Domain_Zc=0.9971 |
| 23656789 | Branch probe v2 A2 | ✅ Done | audio_noise: Z_c 0.9421→0.5664; visual_noise: →0.8995 |
| 23656790 | Branch probe v2 A5 | ✅ Done | audio_noise: Z_c 0.9970→0.6972; visual_noise: →0.9940 |
| 23656791 | Branch probe v2 A6 | ✅ Done | audio_noise: Z_c 0.9948→0.6741; visual_noise: →0.9939 |
| 23656792 | FAVC branch probe A2 | ✅ Done | FV-RA: vmask@100%=0.3565❌, audio_mask=0.9882✅; RV-FA: amask=0.4240❌ |
| 23656793 | FAVC branch probe A5 | ✅ Done | FV-RA: vmask@100%=0.3400❌, audio_mask=0.9924✅; RV-FA: amask=0.3700❌ |
| 23656794 | FAVC branch probe A6 | ✅ Done | FV-RA: vmask@100%=0.3248❌, vnoise=0.4964⚠️; RV-FA: amask=0.3505❌ |
| 23656795 | A2 per-cat perturbation (gap fill) | ✅ Done | FV-RA causal_auc@baseline=0.55 (chance!); RV-FA=1.00 |

#### Batch 3 — Subspace Probe + Head Ablation (submitted 2026-04-06)

| Job ID | Experiment | Status | Key result |
|--------|-----------|--------|------------|
| 23658622 | Subspace probe A2 | ✅ Done | Z_c ind=0.9420, OOD=0.7052 |
| 23658623 | Subspace probe A5 | ✅ Done | u_v OOD=**0.8248**⭐; s_v ind=0.567 (near chance) |
| 23658624 | Subspace probe A6 | ✅ Done | u_v OOD=**0.9238**⭐ (best any subspace); uniq_all OOD=0.774 |
| 23658625 | Head ablation A2 | ✅ Done | mask_ac Δ=−0.470 (head ignores visual); mask_vc Δ=−0.001 |
| 23658626 | Head ablation A5 | ✅ Done | mask_uv FV-RA=−0.060; mask_vis FV-RA=−0.113; mask_sa FV-RA=+0.193 |
| 23658627 | Head ablation A6 | ✅ Done | **mask_uv FV-RA=−0.354**⭐; mask_uniq overall=−0.512⭐ |

### Consolidated Summary Table — Meeting Follow-Up Experiments

This table reorganizes the post-meeting experiments by **question asked**, **what was run**, **key protocol/parameters**, **main result**, and **paper conclusion**. It is intended to be the shortest path from Karen's action items to the paper story.

| Analysis question | Experiment / script / jobs | Key protocol / parameters | Main result | Conclusion for paper |
|---|---|---|---|---|
| **Does three-way factorization matter, or do extra domain losses alone explain the gain?** | Core structural ablations from frozen + e2e comparisons: A2/A3/A5/A6/A42/A43/A44; A2_e2e vs A6_e2e | Frozen eval on standard FAVC subset `N=1114`; compare binary vs FCD; domain push-pull on/off; stronger binary domain pressure in A43; Sinkhorn added to binary in A44. E2E compare balanced 500+500 eval. | Frozen overall AUC: A2=0.7928, A3=0.7490, A5=0.8186, **A6=0.8885**, A42=0.7991, A43=0.7840, A44=0.7667. E2E: A2_e2e=0.7079, **A6_e2e=0.8667**. | The gain is **architecture-dependent**, not just "more losses". The same domain objective hurts or barely helps the binary split, but works strongly with FCD. |
| **Are the probe results methodologically valid for FCD models?** | Probe protocol correction: old `eval_label_probe.py` / `eval_domain_predictability.py` vs corrected `avh_sup/eval_label_probe_seq.py`; jobs 23656786-88 supersede first-batch probe interpretation | Old protocol mean-pooled **before** `_encode()` and fed fake `T=1` input. Corrected protocol feeds full `[T,D]` sequences into `_encode()` and mean-pools **after** encoding; 5-fold CV. | A5 `Z_c` in-domain probe changes **0.8191 -> 0.9970**; A6 `Z_c` changes **0.7704 -> 0.9944**; A2 stays unchanged (0.9421 -> 0.9420). | All FCD probe interpretation must use the **sequence-based protocol**. Old A5/A6 probe values are artifacts and should not be used as main evidence. |
| **Does FCD reduce label/domain leakage in the branches?** | Sequence-based label + domain probe: `avh_sup/eval_label_probe_seq.py`; jobs 23656786-88 | Models A2/A5/A6. Label probe trained on AV1M val with 5-fold CV. OOD eval on FAVC. Domain probe uses AV1M=0 vs FAVC=1. | OOD `Z_c`: A2=0.7052, **A5=0.7744**, A6=0.7480. Domain `Z_c` stays very high for all: A2=0.9975, A5=0.9966, A6=0.9971. A6 still has `Z_s > Z_c` in-domain (0.9975 vs 0.9944). | A5 has the best **representation-level** OOD transfer. A6 does **not** achieve domain invariance. Therefore A6's gain is not from making `Z_c` domain-free. |
| **Do branch representations survive modality corruption?** | AV1M branch probe under perturbation: `avh_sup/eval_branch_probe_perturb.py`; jobs 23656619-21 and 23656789-91 | Linear probe on frozen `Z_c/Z_s`. Audio/visual masking at {0,10,25,50,75,100%}. Audio/visual noise at `sigma={0,0.05,0.10,0.20,0.50,1.00}`. Models A2/A5/A6. | `Z_c` under max audio corruption: A2 `audio_mask@100%=0.5332`, `audio_noise@1.0=0.5664`; A5 `0.6990 / 0.6972`; A6 `0.6624 / 0.6741`. Visual perturbation is nearly negligible on A5/A6. | Binary A2 is clearly **audio-dominant**. FCD retains more non-audio residual signal, but `Z_s` remains nearly as predictive as `Z_c`, so this is **not** clean causal/spurious separation. |
| **What does the trained binary detector actually do on each fake type?** | A2 per-category perturbation using the trained causal head; gap-fill job 23656795 | Uses **A2's own detection head**, not a fresh probe. FAVC per-category eval on FV-RA and RV-FA with audio_mask, visual_mask, audio_noise, visual_noise, temporal_shuffle. | FV-RA baseline = **0.5462** (near chance), RV-FA baseline = **1.0000**. FV-RA `audio_noise@1.0=0.3198`; RV-FA `audio_mask@100%=0.5340`. | The trained A2 detector is functionally an **audio-only detector** and can be actively misled when audio is corrupted. |
| **Is A2 missing visual fake information, or just failing to use it?** | FAVC branch probe under perturbation: `avh_sup/eval_favc_branch_probe_perturb.py`; jobs 23656792-94 | Fresh linear probe on frozen `Z_c/Z_s`. Categories `FV-RA`, `RV-FA`, `FV-FA` vs `RV-RA` controls. Masking levels {0,10,25,50,75,100%}; noise `sigma={0,0.05,0.10,0.20,0.50,1.00}`. | For FV-RA, A2 `Z_c` stays **0.9882** under `audio_mask@100%` but collapses to **0.3565** under `visual_mask@100%`. Similar modality-specific collapse pattern holds for A5/A6. | Even binary `Z_c` already contains **separable visual fake evidence**. The main binary failure is **downstream head usage**, not lack of representation quality. |
| **Which FCD subspace is actually carrying transferable signal?** | Subspace probe: `avh_sup/eval_subspace_probe.py`; jobs 23658622-24 | Probe individual 256-dim subspaces `s_v, u_v, s_a, u_a, r_v, r_a` plus aggregates `vis_all, aud_all, syn_all, uniq_all`; label and domain probes. | A6 `u_v` achieves **OOD AUC = 0.9238** (best single subspace overall). A5 `u_v` reaches 0.8248. `s_v` is near chance in-domain (~0.53-0.57) but still useful OOD. Visual aggregates transfer better OOD than audio aggregates. | The strongest mechanistic evidence is that FCD creates a high-value **visual-unique subspace**. `u_v` is the main OOD carrier. |
| **What features does the trained detector actually route through at test time?** | Head ablation: `avh_sup/eval_head_ablation.py`; jobs 23658625-27 | Zero selected dimensions of `Z_c` at inference. A2 uses `mask_vc` / `mask_ac`; A5/A6 use `mask_sv, mask_uv, mask_sa, mask_ua, mask_vis, mask_aud, mask_syn, mask_uniq`. | A2: `mask_vc` almost no effect, `mask_ac` gives `delta_AV1M=-0.470`, `delta_overall=-0.337`. A6: `mask_uv` gives `delta_FV-RA=-0.354`; `mask_ua` gives `delta_RV-FA=-0.254`, `delta_FV-FA=-0.257`; `mask_uniq` gives `delta_overall=-0.512`. | This is the **strongest paper evidence**: A6 reroutes video-only detection through **`u_v`** and audio-fake detection through **`u_a`**. Binary A2 never learns this routing. |
| **Do modality-origin or manipulation probes actually differentiate models?** | Modality-origin probe `avh_sup/eval_modality_origin_probe.py`; manipulation probe runs; jobs 23656622-23 plus manipulation probe outputs | Linear probes on frozen `Z_c/Z_s`. Tasks: FV-RA vs RV-FA, and FV-RA vs FV-FA. | Both are nearly perfect for all models (e.g., A2/A6 modality-origin AUC approx. 1.0). | These are useful as **sanity checks only**. They are too easy to support the main claim and should not be central in the paper. |

**Most defensible paper-level conclusion after these experiments:**

1. Binary decomposition does **not** fail because the representation lacks visual evidence; it fails because training causes the final detector to ignore that visual evidence and collapse to audio-dominant routing.
2. Three-way factorization creates meaningful modality-specific subspaces, especially the visual-unique channel `u_v`, which is the strongest cross-domain carrier.
3. A6's advantage is best framed as **factorization changes feature usage and downstream routing**, not as clean disentanglement and not as true domain invariance.

**Do not use as main evidence:** the first-batch A6 label/domain probe jobs (23656617-18) used the obsolete fake-`T=1` protocol and should be kept only for provenance.
