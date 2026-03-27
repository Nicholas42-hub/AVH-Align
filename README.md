# AVH-Align

[![arXiv](https://img.shields.io/badge/-arXiv-B31B1B.svg?style=for-the-badge)](https://arxiv.org/abs/2412.00175)

**Official PyTorch Implementation of the Paper:**

> **Ștefan Smeu, Dragoș-Alexandru Boldisor, Dan Oneață and Elisabeta Oneață**  
> [Circumventing shortcuts in audio-visual deepfake detection datasets with unsupervised learning](https://arxiv.org/abs/2412.00175)  
> *CVPR, 2025*

## Data

To set up your data, follow these steps:

**Download the datasets:**
   - **AV-Deepfake1M(AV1M) Dataset:** Follow instructions from [AV-Deepfake1M](https://github.com/ControlNet/AV-Deepfake1M)
   - **FakeAVCeleb Dataset:** Follow instructions from [FakeAVCeleb GitHub repo](https://github.com/DASH-Lab/FakeAVCeleb)
   - **AVLips Dataset:** Follow instructions from [LipFD GitHub repo](https://github.com/AaronComo/LipFD)

## Set-up AV-Hubert
```bash 
# clone/install AV-Hubert
git clone https://github.com/facebookresearch/av_hubert.git
cd av_hubert/avhubert
git submodule init
git submodule update
cd ../fairseq
pip install --editable ./
cd ../avhubert
# install additional files for AV-Hubert
mkdir -p content/data/misc/
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 -O content/data/misc/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d content/data/misc/shape_predictor_68_face_landmarks.dat.bz2
wget --content-disposition https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/raw/master/preprocessing/20words_mean_face.npy -O content/data/misc/20words_mean_face.npy
cd ../../
# moving our feature extraction files into avhubert space
cp deepfake_feature_extraction.py av_hubert/avhubert/deepfake_feature_extraction.py 
cp deepfake_preprocess.py av_hubert/avhubert/deepfake_preprocess.py

# download avhubert checkpoint
wget https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/vsr/self_large_vox_433h.pt
mv self_large_vox_433h.pt av_hubert/avhubert/self_large_vox_433h.pt
```

This repository also integrates code from the following repositories:
- [FACTOR](https://github.com/talreiss/FACTOR)
- [AV-Hubert](https://github.com/facebookresearch/av_hubert)

## Installation

Main prerequisites:

* `Python 3.10.14`
* `pytorch=2.2.0` (older version for compability with AVHubert)
* `pytorch-cuda=12.4`
* `lightning=2.4.0`
* `torchvision>=0.17`
* `scikit-learn>=1.3.2`
* `pandas>=2.1.1`
* `numpy>=1.26.4`
* `pillow>=10.0.1`
* `librosa>=0.9.1`
* `dlib>=19.24.9`
* `skvideo>=1.1.10`
* `ffmpeg>=4.3`

## Feature extraction

1. **Preprocess video files**
Run deepfake_preprocess.py from av_hubert/avhubert. Example for AV-Deepfake1M
```bash
python deepfake_preprocess.py \
    --dataset AV1M \
    --split train \
    --metadata /av1m_metadata/train_metadata.csv \
    --data_path /path/to/AV1M_root \
    --save_path /path/to/save/output_videos_and_audio
```
and FakeAVCeleb
```bash
python deepfake_preprocess.py \
    --dataset FakeAVCeleb \
    --metadata /path/to/FakeAVCeleb_metadata.csv \
    --data_path /path/to/FakeAVCeleb_root \
    --save_path /path/to/save/output_videos_and_audio
    --category all \
```

2. **Extract features**
Run deepfake_feature_extraction.py from av_hubert/avhubert. Example for AV-Deepfake1M

```bash
python deepfake_feature_extraction.py \
    --dataset AV1M \
    --split train \
    --metadata /av1m_metadata/train_metadata.csv \
    --ckpt_path self_large_vox_433h.pt \
    --data_path /path/to/preprocessed/data \
    --save_path /path/to/save/features
```
and FakeAVCeleb
```bash
python deepfake_feature_extraction.py \
    --dataset FakeAVCeleb \
    --metadata /path/to/FakeAVCeleb_metadata.csv \
    --ckpt_path self_large_vox_433h.pt \
    --data_path /path/to/preprocessed/data \
    --save_path /path/to/save/features \
    --category all
```

add ```--trimmed``` for the trimmed version of features

## Train

 To train the models mentioned in the article, follow:

 **Set up training and validation data paths** in `config.py` or specify them as arguments when running the training routine as the example below:

 ```bash
 python train.py --name=<experiment_name> --data_root_path=<path_to_the_features_data> --metadata_root_path=<path_to_the_folder_containing_the_dataset_metadata_files>
 ```
 The model weights will be available at `<save_path>/<name>.pt`

## Pretrained Models
We provide weights for our AVH-Align model trained on 45000 real videos from AV-Deepfake1M in `checkpoints/AVH-Align_AV1M.pt`.

## Evaluation

To evaluate a model, use/modify the following example:

```bash 
python eval.py \ 
    --checkpoint_path checkpoints/AVH-Align_AV1M.pt \ 
    --features_path /path/to/saved/features \ 
    --metadata /av1m_metadata/test_metadata.csv \ 
    --dataset AV1M 
```

## Ablation Study Variants

All ablations share a common backbone: AV-HuBERT features (1024-d per modality) are projected independently via two parallel MLP heads into a **causal subspace Z_c** and a **spurious subspace Z_s** (each 1024-d, 512-d per modality). An orthogonality constraint $\mathcal{L}_\text{orth} = \lambda_\text{orth}\|\mathbf{Z}_c^\top \mathbf{Z}_s\|_F^2$ is applied in every variant to prevent Z_c and Z_s from encoding the same information.

### Series A — Causal-Spurious Disentanglement

Series A ablations progressively add components to isolate what is needed to push spurious label signal out of Z_s.

| Variant | Adversarial on Z_s | Stop-Gradient | Domain-Aware | β-VAE |
|---------|:-----------------:|:-------------:|:------------:|:-----:|
| A0      | ✗ | ✗ | ✗ | ✗ |
| A1      | ✓ | ✗ | ✗ | ✗ |
| A2      | ✓ | ✓ | ✗ | ✗ |
| A3      | ✓ | ✓ | ✓ | ✗ |
| A4      | ✓ | ✓ | ✗ | ✓ |

**A0 — Orthogonal Loss Only (control)**

Classification uses `full_head(Z_c + Z_s)` with gradient flowing to both subspaces. Additionally, `causal_head(Z_c)` and `spurious_head(Z_s)` are trained independently so their AUCs are meaningful.  
Total loss: $\mathcal{L} = \mathcal{L}_\text{CE}^\text{full} + \mathcal{L}_\text{CE}^\text{causal} + \mathcal{L}_\text{CE}^\text{spurious} + \lambda_\text{orth}\mathcal{L}_\text{orth}$  
Without adversarial pressure, Z_s can still freely retain label information despite the orthogonality constraint. This is the control condition establishing that orthogonality alone is insufficient.

**A1 — Adversarial Loss Only**

Adds a GRL-based adversarial head on Z_s: `adv_head(GRL(Z_s))` is trained to predict labels; reversed gradients push Z_s toward being label-non-informative. Classification still uses `full_head(Z_c + Z_s)`, so Z_s can still contribute to the main classifier via this path.  
Total loss: $\mathcal{L} = \mathcal{L}_\text{CE}^\text{full} + \mathcal{L}_\text{CE}^\text{causal} + \lambda_\text{adv}\mathcal{L}_\text{adv} + \lambda_\text{orth}\mathcal{L}_\text{orth}$  
Expected: causal AUC < full AUC (Z_s still helps via full_head), spurious AUC remains > 0.5.

**A2 — Adversarial + Stop-Gradient**

Adds a stop-gradient: classification is **restricted** to `causal_head(Z_c)` only. Z_s receives gradient solely from the GRL adversarial path, with no route into the classification loss.  
Total loss: $\mathcal{L} = \mathcal{L}_\text{CE}^\text{causal} + \lambda_\text{adv}\mathcal{L}_\text{adv} + \lambda_\text{orth}\mathcal{L}_\text{orth}$  
Expected: causal AUC ≈ full AUC; spurious AUC ≈ 0.5 in-domain.

**A3 — Adversarial + Stop-Gradient + Domain-Aware**

Extends A2 by adding a second GRL on Z_s for domain prediction (AV1M=0 vs FAVC=1). FAVC samples are mixed into training as unlabelled domain-contrast data. Reversed gradients force Z_s to be domain-invariant, preventing it from encoding dataset-specific shortcuts.  
Total loss: $\mathcal{L} = \mathcal{L}_\text{CE}^\text{causal} + \lambda_\text{adv}\mathcal{L}_\text{adv} + \lambda_\text{dom}\mathcal{L}_\text{domain} + \lambda_\text{orth}\mathcal{L}_\text{orth}$

**A4 — Adversarial + Stop-Gradient + β-VAE**

Extends A2 by replacing deterministic projections with variational (μ, log σ²) heads. Separate KL penalties $\beta_c \cdot \mathcal{L}_\text{KL}^c$ and $\beta_s \cdot \mathcal{L}_\text{KL}^s$ (with $\beta_s \gg \beta_c$) compress Z_s more aggressively toward $\mathcal{N}(0, I)$, geometrically limiting the capacity available for spurious shortcuts. A β warmup ramps both from 0 to target over the first 10 epochs.  
Total loss: $\mathcal{L} = \mathcal{L}_\text{CE}^\text{causal} + \lambda_\text{adv}\mathcal{L}_\text{adv} + \beta_c\mathcal{L}_\text{KL}^c + \beta_s\mathcal{L}_\text{KL}^s + \lambda_\text{orth}\mathcal{L}_\text{orth}$

---

### Series B — Modal Balance

All B variants build on A2 (adversarial + stop-gradient) and address the problem of **audio dominance** in the causal branch: without additional supervision, the model tends to rely almost entirely on audio when classifying, leaving the visual causal projector undertrained.

| Variant | Per-Modality Cls | Balance Penalty | Cross-Modal Align | Modal Discriminator |
|---------|:----------------:|:---------------:|:-----------------:|:-------------------:|
| B1      | ✓ | ✗ | ✗ | ✗ |
| B1b     | ✓ | ✓ | ✗ | ✗ |
| B2      | ✗ | ✗ | ✓ | ✗ |
| B3      | ✗ | ✗ | ✗ | ✓ |

**B1 — Per-Modality Classification**

Adds independent classification heads for each modality: `modal_head_visual(v_c)` and `modal_head_audio(a_c)` each predict labels on their own. This applies direct positive gradient to the visual causal projector, counteracting audio dominance.  
Additional loss: $\mathcal{L}_\text{modal} = \lambda_\text{modal}(\mathcal{L}_\text{CE}^\text{visual} + \mathcal{L}_\text{CE}^\text{audio})$

**B1b — Per-Modality Classification + Balance Penalty**

Extends B1 with an equilibrium constraint $\mathcal{L}_\text{balance} = (\mathcal{L}_\text{CE}^\text{audio} - \mathcal{L}_\text{CE}^\text{visual})^2$. If audio classification loss drops faster than visual (audio dominates), the squared gap grows and redirects gradient to the visual branch, enforcing equal per-modality contribution throughout training.  
Additional loss: $\mathcal{L}_\text{modal} + \lambda_\text{balance}(\mathcal{L}_\text{CE}^\text{audio} - \mathcal{L}_\text{CE}^\text{visual})^2$

**B2 — Cross-Modal Alignment**

Adds a cosine alignment loss between per-frame visual and audio causal features: $\mathcal{L}_\text{cross} = \text{mean}(1 - \cos(v_c, a_c))$, applied only on real (RealVideo-RealAudio) samples. The hypothesis is that genuine AV content shares the same causal signal across modalities, so aligning v_c and a_c in feature space steers both projectors toward the same semantic representation and discourages modality-specific over-fitting.  
Additional loss: $\lambda_\text{cross}\mathcal{L}_\text{cross}$

**B3 — Modal Discriminator Adversarial**

Adds a GRL-based modal discriminator trained to predict whether a token came from the visual causal projector (label=0) or the audio causal projector (label=1). Reversed gradients force both projectors to produce modality-agnostic features, preventing either branch from specialising in modality-specific artifacts.  
Additional loss: $\lambda_\text{disc}\mathcal{L}_\text{disc}$ (via GRL, so projectors receive reversed gradient)

---

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
