# Related Paper Protocol Matrix

Working question from Caren: do existing audio-visual deepfake papers use the same environment/protocol as ours, and if not, how should we justify our cross-dataset setting?

## Our Current Protocol

| Item | Setting |
|---|---|
| Main claim | Cross-dataset transfer, not intra-dataset leaderboard |
| Train | AV-Deepfake1M / AV1M-trained detectors |
| Main test | Matched AV1M -> FakeAVCeleb subset, `N=1114` |
| Public benchmark add-on | AV1M -> FakeAVCeleb 70/30 public test split, `N=6667` |
| Seeds | Public 70/30 table currently uses 43/44/45 |
| Metrics | Video-level AUC, with category AUC for RV-FA, FV-RA, FV-FA |
| Core wording | "We focus on cross-dataset transfer under shortcut-prone audio-visual datasets, rather than intra-FakeAVCeleb training/testing." |

## Protocol Matrix

| Paper | Venue / year | Train data / training protocol | Test data / split | Metric(s) | Same as ours? | What to say in our paper |
|---|---|---|---|---|---|---|
| **Smeu et al., "Circumventing Shortcuts in Audio-visual Deepfake Detection Datasets with Unsupervised Learning" / AVH-Align** | CVPR 2025 | Unsupervised AVH-Align trains only on real videos. For AV1M, they use 50k real samples from AV1M train split: 45k train / 5k validation. They also train supervised AVH-Align/sup variants with real+fake labels. | FakeAVCeleb: 70% train+val / 30% test. AV-Deepfake1M: evaluates on 10k original validation samples and, for best models, official withheld test. They also evaluate original vs trimmed data for leading-silence shortcut. | AUC and AP, video-level. | **Closest benchmark reference, but not identical.** They emphasize real-only/unsupervised training and shortcut robustness. Our public 70/30 FAVC test is aligned in spirit, but our train side is AV1M-trained supervised/factorized variants. | Cite as the main protocol/shortcut motivation. State that we add full FAVC 70/30 public test as a protocol bridge, while keeping our main setting as AV1M -> FAVC cross-dataset transfer. Also mention leading-silence trimmed control. |
| **Cai et al., "AV-Deepfake1M"** | ACM MM 2024 | Official AV-Deepfake1M train/val/test splits. Benchmarks include localization and video-level detection; label access levels include frame-level, segment-level, and video-level. | AV-Deepfake1M official validation/test; official test has unseen speakers and is evaluated through competition/Codabench. They also compare/transfer with LAV-DF in additional experiments. | Detection: video-level AUC and accuracy. Localization: AP/AR variants. | **No.** It defines our source dataset and an official AV1M benchmark, but does not use AV1M -> FakeAVCeleb as the main setting. | Use to justify AV1M as large, difficult source dataset. If reviewers ask for AV1M official test, explain it is a different in-dataset/official benchmark from our cross-dataset transfer question. |
| **Khalid et al., "FakeAVCeleb"** | NeurIPS Datasets & Benchmarks 2021 | Dataset paper. Baseline experiments mostly evaluate existing visual/audio detectors, not a unified AV1M -> FAVC cross-dataset protocol. | FakeAVCeleb v1.2. The paper reports frame-level AUC for baseline detectors; later papers commonly impose their own train/test splits, including 70/30. | Frame-level AUC in the original benchmark section. | **No.** It introduces the dataset but does not define our cross-dataset protocol. | Cite for dataset construction and categories. Do not imply that our 70/30 table is the original FakeAVCeleb leaderboard unless we train on FakeAVCeleb train. |
| **Feng et al., "Self-Supervised Video Forensics by Audio-Visual Anomaly Detection" / AVAD** | CVPR 2023 | Self-supervised anomaly detector trained on real, unlabeled speech videos from LRS2/LRS3, not on FakeAVCeleb fakes. Supervised baselines are retrained on FakeAVCeleb. | FakeAVCeleb: paper samples 2400 videos as train/val and 600 as test; their method does not use FAVC train/val. Also evaluates KoDF for cross-dataset/language generalization. | AP and AUC. | **Related but not same.** It is real-only cross-dataset/anomaly detection, with a much smaller FAVC split than our full 70/30 `N=6667`. | Use in related work as a self-supervised / real-only AV consistency baseline. It supports the argument that real-only and cross-dataset settings are accepted, but its split is not directly comparable to ours. |
| **Liang et al., "SpeechForensics"** | NeurIPS 2024 | Learns audio-visual speech representations on real videos via self-supervised masked prediction; no fake video participation in representation learning. For cross-dataset tables, supervised comparators are trained on FF++. | Cross-manipulation on FF++; cross-dataset testing on FakeAVCeleb and KoDF, with category-level FAVC results. | Video-level AUC; robustness AUC under perturbations. | **Related but not same.** It is a real-only/self-supervised representation approach; not AV1M -> FAVC and not the same full 70/30 FAVC protocol. | Cite as recent NeurIPS evidence that cross-dataset generalization and real-only training are central in this area. Distinguish our contribution as factorized routing on AV1M -> FAVC. |
| **Zhou and Lim, "Joint Audio-Visual Deepfake Detection"** | ICCV 2021 | Curates AV deepfake data from FF++ and DFDC by modifying audio streams; follows original train/val/test splits from those datasets. Also runs unseen-category evaluations. | FF++ and DFDC, not FakeAVCeleb or AV1M. | Accuracy and AUC. | **No.** It is important historically for joint audio-visual detection, but not our datasets or protocol. | Related work only. Do not benchmark against directly unless reproducing on their FF++/DFDC synthetic-audio setup. |
| **Koutlis and Papadopoulos, "DiMoDif"** | arXiv 2024/2025 | Trains/evaluates on FakeAVCeleb, LAV-DF, and AV-Deepfake1M for deepfake detection and localization. For FakeAVCeleb, they use 70% train / 30% test and 5% of train for validation; for AV1M, they use official train/val/test and Codabench for test. Also reports cross-dataset generalization. | FakeAVCeleb 70/30, LAV-DF official splits, AV-Deepfake1M official splits/Codabench. | DFD: ACC/AP/AUC. TFL: AP@p and AR@n. | **Partially comparable.** Their FAVC 70/30 protocol is close, but they do in-dataset FAVC training/evaluation and localization, not only AV1M -> FAVC transfer. | Useful to show that 70/30 FAVC and AV1M official splits are common. Our distinction: we evaluate AV1M-trained models on FAVC, not train on FAVC. |
| **Como et al., "Lips Are Lying" / AVLips** | NeurIPS Datasets & Benchmarks 2024 | Trains LipFD on Wav2Lip-modified LRS3 / AVLips subset. Baselines are adapted/fine-tuned for lip-sync detection. | Cross-dataset validation on AVLips, FF++, and DFDC; robustness under perturbations. Not FakeAVCeleb/AV1M as the main protocol. | ACC, AP, FPR, FNR; AUC for perturbation robustness. | **No.** Different task emphasis: lip-sync forgery detection. | Use as external-direction related work and optional external evaluation. Also useful because Smeu et al. report no strong leading-silence bias in AVLips. |

## Bottom Line for Caren

1. Most related papers are **in-domain** on their proposed dataset or use their own evaluation split.
2. The closest public benchmark bridge is **Smeu et al. / AVH-Align**, because they explicitly study FakeAVCeleb and AV-Deepfake1M shortcuts and use a FakeAVCeleb 70/30 split.
3. Our current setup is defensible if phrased as:

   > We do not aim to report an intra-FakeAVCeleb leaderboard. Instead, we study AV1M -> FakeAVCeleb cross-dataset transfer under shortcut-prone audio-visual datasets. To connect to the public protocol used in prior shortcut analysis, we additionally evaluate on the full FakeAVCeleb 70/30 public test split.

4. The paper should include:
   - Main matched AV1M -> FAVC result (`N=1114`).
   - Public FAVC 70/30 result (`N=6667`).
   - Leading-silence / true-trimmed robustness control.
   - A sentence saying that in-domain FakeAVCeleb training/testing is a different leaderboard-style protocol and not the focus here.

## Sources Checked

- Smeu et al., CVPR 2025, AVH-Align shortcut paper: https://openaccess.thecvf.com/content/CVPR2025/html/Smeu_Circumventing_Shortcuts_in_Audio-visual_Deepfake_Detection_Datasets_with_Unsupervised_Learning_CVPR_2025_paper.html
- Cai et al., ACM MM 2024, AV-Deepfake1M: https://arxiv.org/abs/2311.15308
- Khalid et al., NeurIPS Datasets 2021, FakeAVCeleb: https://arxiv.org/abs/2108.05080
- Feng et al., CVPR 2023, AVAD: https://openaccess.thecvf.com/content/CVPR2023/html/Feng_Self-Supervised_Video_Forensics_by_Audio-Visual_Anomaly_Detection_CVPR_2023_paper.html
- Liang et al., NeurIPS 2024, SpeechForensics: https://arxiv.org/abs/2508.09913
- Zhou and Lim, ICCV 2021, Joint Audio-Visual Deepfake Detection: https://openaccess.thecvf.com/content/ICCV2021/html/Zhou_Joint_Audio-Visual_Deepfake_Detection_ICCV_2021_paper.html
- Koutlis and Papadopoulos, DiMoDif: https://arxiv.org/abs/2411.10193
- Como et al., NeurIPS 2024, AVLips / Lips Are Lying: https://proceedings.neurips.cc/paper_files/paper/2024/file/a5a5b0ff87c59172a13342d428b1e033-Paper-Conference.pdf
