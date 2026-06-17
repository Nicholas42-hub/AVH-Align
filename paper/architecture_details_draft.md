# TriRoute Architecture Details Draft

Prepared for appendix/supplementary material, but not added to `main.tex` yet.

## Inputs

- Frozen AV-HuBERT features per frame:
  - visual feature `z_v`: 1024 dimensions
  - audio feature `z_a`: 1024 dimensions
- Temporal resolution: 25 fps, variable-length clips.

## Factorization Modules

Per modality, `CCD` maps 1024-d AV-HuBERT features into:

- shared/content factor `s`: `Linear(1024, 256) -> LayerNorm(256) -> ReLU`
- modality-specific hidden factor `h`: `Linear(1024, 256) -> LayerNorm(256) -> ReLU`

Per modality, `URD` decomposes `h` into unique and residual factors:

- `u = Linear(256, 256) -> LayerNorm(256) -> ReLU -> Linear(256, 256)`
- `r = h - u`

TriRoute also includes a cross-modal inconsistency factor:

- `MultiheadAttention(1024, num_heads=4)` for visual-to-audio attention
- `MultiheadAttention(1024, num_heads=4)` for audio-to-visual attention
- asymmetric difference projected by `Linear(1024, 256) -> LayerNorm(256) -> ReLU`

The task representation is:

`cat(s_v, u_v, u_delta, s_a, u_a)`, dimension 1280.

The residual/domain representation is:

`cat(r_v, r_a)`, dimension 512.

## Heads

Task head:

- `Linear(1280, 512) -> LayerNorm(512) -> ReLU`
- `Linear(512, 256) -> LayerNorm(256) -> ReLU`
- `Linear(256, 128) -> LayerNorm(128) -> ReLU`
- `Linear(128, 1)`

Residual probe head:

- `Linear(512, 512) -> LayerNorm(512) -> ReLU`
- `Linear(512, 256) -> LayerNorm(256) -> ReLU`
- `Linear(256, 128) -> LayerNorm(128) -> ReLU`
- `Linear(128, 1)`

Domain heads:

- task-branch adversarial head: `Linear(1280, 256) -> LayerNorm(256) -> ReLU -> Linear(256, 64) -> ReLU -> Linear(64, 1)`
- residual/domain head: `Linear(512, 256) -> LayerNorm(256) -> ReLU -> Linear(256, 64) -> ReLU -> Linear(64, 1)`

Auxiliary heads:

- visual unique head: `Linear(256, 128) -> LayerNorm(128) -> ReLU -> Linear(128, 1)`
- audio unique head: `Linear(256, 128) -> LayerNorm(128) -> ReLU -> Linear(128, 1)`
- inconsistency head: `Linear(256, 128) -> LayerNorm(128) -> ReLU -> Linear(128, 1)`
- modality head: `Linear(256, 64) -> ReLU -> Linear(64, 1)`
- SDA projection: `Linear(256, 128)`

## Parameter Count

Computed from `avh_sup/mlp_fcd_a6.py` with `feat_dim=1024`, `syn_dim=256`, `spec_dim=256`, `sda_dim=128`, `mi_dim=128`, and `incon_dim=256`.

| Component | Parameters |
|---|---:|
| `ccd_v` | 525,824 |
| `ccd_a` | 525,824 |
| `urd_v` | 132,096 |
| `urd_a` | 132,096 |
| `sda` | 32,896 |
| `cross_incon` | 8,659,712 |
| `unique_head_v` | 33,281 |
| `unique_head_a` | 33,281 |
| `incon_head` | 33,281 |
| `modality_head` | 16,513 |
| `causal_head` | 822,017 |
| `redundant_head` | 428,801 |
| `domain_head_c` | 344,961 |
| `domain_head_s` | 148,353 |
| **Total trainable detector parameters** | **11,868,936** |

The AV-HuBERT backbone is frozen and is not included in the trainable detector parameter count.
