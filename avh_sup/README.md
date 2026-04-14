# AVH-Align/sup

For data preprocessing, AV-HuBERT setup, installation, and feature extraction,
see the main `README.md` at the repo root.

---

## Overview

This directory contains the supervised deepfake detection experiments.
The active architecture is an **end-to-end causal disentanglement model**
built on top of a fine-tuned AV-HuBERT encoder.

### Core idea

The model separates representations into two branches:
- **Z_c (causal)** — captures audio-visual synchrony, which is informative for
  deepfake detection and should generalise across domains.
- **Z_s (spurious)** — absorbs dataset-specific artefacts; should be label-
  uninformative when probed out-of-domain.

Domain invariance is enforced via IPW reweighting, a light GRL on Z_c, and a
label-aware domain pull on Z_s (A38/A39).

---

## Key source files

| File | Purpose |
|------|---------|
| `train_e2e_a38.py` | **Active training script** — `E2EModelA38` with IPW + GRL + label-aware domain pull |
| `mlp_causal_ablation.py` | Shared causal/spurious projection heads used by all e2e models |
| `avhubert_wrapper.py` | Wraps the fairseq AV-HuBERT model; supports partial/full unfreezing |
| `datasets_e2e.py` | Raw-video datasets (`AV1M_E2E_FullPathDataset`, `FakeAVCeleb_E2E_Dataset`) |
| `eval_e2e_auc.py` | Direct-head AUC eval: in-domain (AV1M val) + cross-domain (FakeAVCeleb) |
| `eval_e2e_domain.py` | Domain + label probe: fits logistic regression on Z_c / Z_s |
| `eval_favc.py` | Cross-dataset AUC on FakeAVCeleb NPZ features (frozen-encoder experiments) |
| `eval_label_probe.py` | Label + domain probe for frozen-encoder experiments |

Experiment configs live in `configs/`. The current active configs are
`configs/A38_e2e.yaml` (with IPW + GRL) and `configs/A39_e2e.yaml` (task loss
+ label-aware Z_s pull only, no IPW/GRL).

---

## Training (end-to-end experiments)

```bash
python train_e2e_a38.py --config_path configs/A38_e2e.yaml
```

Key config fields in `configs/A38_e2e.yaml`:

| Field | Description |
|-------|-------------|
| `data_paths.e2e_train_csv` | Absolute-path CSV for AV1M training clips |
| `data_paths.favc_raw_root` | Root of `favc_preprocessed/` (raw video/audio) |
| `model_hparams.avhubert_ckpt` | Path to `self_large_vox_433h.pt` |
| `model_hparams.n_unfreeze_layers` | `-1` = fully unfrozen; `0` = frozen baseline |
| `model_hparams.lambda_ipw` | Weight on IPW domain predictor loss |
| `model_hparams.lambda_grl` | Weight on GRL adversarial loss on Z_c |
| `model_hparams.lambda_ladv_ds` | Weight on label-aware domain pull on Z_s |

---

## Evaluation (end-to-end experiments)

**Direct-head AUC** (in-domain + cross-domain):
```bash
python eval_auc_a39_e2e.py \
    --ckpt    outputs_A39_e2e/ckpts/model-best.ckpt \
    --config  configs/A39_e2e.yaml \
    --av1m_raw   /scratch/.../avd1m_preprocessed/val \
    --favc_raw   /scratch/.../favc_preprocessed \
    --av1m_csv   csv_metadata/av1m_e2e \
    --output_dir outputs_A39_e2e/results
```

**Domain + label probe** (logistic regression on Z_c / Z_s):
```bash
python eval_domain_a39_e2e.py \
    --ckpt    outputs_A39_e2e/ckpts/model-best.ckpt \
    --config  configs/A39_e2e.yaml \
    --av1m_raw   /scratch/.../avd1m_preprocessed/val \
    --favc_raw   /scratch/.../favc_preprocessed \
    --av1m_csv   csv_metadata/av1m_e2e \
    --output_dir outputs_A39_e2e/results
```

The `eval_auc_a39_e2e.py` / `eval_domain_a39_e2e.py` scripts are thin
wrappers: they set `--model_module` / `--model_class` / `--use_fullpath_csv`
then call `eval_e2e_auc.py` / `eval_e2e_domain.py` directly. All other
per-experiment eval scripts (A26–A39) follow the same pattern.

---

## Experiment history

Earlier experiments (A0–A9) used **frozen AV-HuBERT features** (pre-extracted
`.npz` files). From A10 onwards the encoder is fine-tuned end-to-end.
The key progression leading to A38/A39:

| Exp | Key addition |
|-----|--------------|
| A2  | Stop-gradient on Z_s; Z_c-only classification |
| A26 | IPW reweighting on task loss |
| A34 | Light GRL on Z_c for domain invariance |
| A38 | Label-aware domain pull on Z_s (`lambda_ladv_ds`) |
| A39 | A38 without IPW/GRL (ablation: task loss + Z_s pull only) |

## License

<p xmlns:cc="http://creativecommons.org/ns#">The code is licensed under <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">CC BY-NC-SA 4.0 <img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/nc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/sa.svg?ref=chooser-v1" alt=""></a></p>

## Citation

If you find this work useful in your research, please cite it.

```
@InProceedings{AVH-Align,
    author    = {Smeu, Stefan and Boldisor, Dragos-Alexandru and Oneata, Dan and Oneata, Elisabeta},
    title     = {Circumventing shortcuts in audio-visual deepfake detection datasets with unsupervised learning localization},
    booktitle = {Proceedings of The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2025}
}
```