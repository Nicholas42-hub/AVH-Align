#!/bin/bash
#SBATCH --job-name=eval_full
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_full_comprehensive_%j.out
#SBATCH --error=logs/eval_full_comprehensive_%j.err

# ============================================================
#  Comprehensive Evaluation on ALL Available Data
#  1. AV-Deepfake1M Val (10,000 samples) - In-domain
#  2. FakeAVCeleb (2,702 samples) - Cross-domain
# ============================================================

echo "============================================================"
echo "  Comprehensive Model Evaluation"
echo "  Job ID : $SLURM_JOB_ID"
echo "  Node   : $SLURM_NODELIST"
echo "  Start  : $(date)"
echo "============================================================"

cd /data/projects/punim2637/nnliang/AVH-Align/avh_sup

module purge
module load GCCcore/11.3.0
module load CUDA/12.2.0

# Use conda environment
source /data/projects/punim2637/nnliang/avh_env/etc/profile.d/conda.sh
conda activate /data/projects/punim2637/nnliang/avh_env

echo "Python: $(which python)"
echo ""

# Set environment
export PATH=/data/projects/punim2637/nnliang/avh_env/bin:$PATH
export PYTHONPATH=/data/projects/punim2637/nnliang/avh_env/lib/python3.9/site-packages:$PYTHONPATH
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# ============================================================
# PART 1: AV-Deepfake1M Val Set (10,000 samples - In-domain)
# ============================================================

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  PART 1: AV-Deepfake1M Validation Set (In-domain)                 ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

python << 'EOF'
import torch
import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader
from datasets import AV1M_trainval_dataset
from mlp import AVH_Sup
from mlp_causal import AVH_Causal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

# Configuration for AV1M validation set
config = {
    "name": "AV1M_NPZ",
    "root_path": "../data/avh_features",
    "csv_root_path": "../av1m_metadata",
    "apply_l2": True,
}

# Load full validation dataset (10,000 samples)
val_dataset = AV1M_trainval_dataset(config, split="val")
val_loader = DataLoader(val_dataset, shuffle=False, batch_size=1, num_workers=8)

print("="*70)
print(f"AV-Deepfake1M Validation Set: {len(val_dataset)} samples")
print("="*70)
print()

# ─── Evaluate Baseline Model ───────────────────────────────────────────
print("┌─ Baseline Model ──────────────────────────────────────────────────┐")
baseline_model = AVH_Sup.load_from_checkpoint("outputs/ckpts/model-epoch=05.ckpt")
baseline_model.to(device)
baseline_model.eval()

all_scores, all_labels = [], []
with torch.no_grad():
    for video, audio, labels in tqdm.tqdm(val_loader, desc="Baseline"):
        video = video.to(device)
        audio = audio.to(device)
        scores = baseline_model.predict_scores(video, audio)
        all_scores.extend(scores.cpu().numpy().tolist())
        all_labels.extend(labels.numpy().tolist())

n_real = all_labels.count(0)
n_fake = all_labels.count(1)
auc = roc_auc_score(all_labels, all_scores)
ap = average_precision_score(all_labels, all_scores)

print(f"│ Samples: {len(all_labels):,} (real={n_real:,}, fake={n_fake:,})")
print(f"│ AUC: {auc:.4f}")
print(f"│ AP:  {ap:.4f}")
print("└────────────────────────────────────────────────────────────────────┘")
print()

# ─── Evaluate Causal Model ─────────────────────────────────────────────
print("┌─ Causal Model ────────────────────────────────────────────────────┐")
causal_model = AVH_Causal.load_from_checkpoint("outputs_causal/ckpts/model-epoch=09.ckpt")
causal_model.to(device)
causal_model.eval()

all_full, all_causal, all_spurious, all_labels = [], [], [], []
with torch.no_grad():
    for video, audio, labels in tqdm.tqdm(val_loader, desc="Causal "):
        video = video.to(device)
        audio = audio.to(device)
        full_s = causal_model.predict_scores(video, audio, mode="full")
        causal_s = causal_model.predict_scores(video, audio, mode="causal")
        spur_s = causal_model.predict_scores(video, audio, mode="spurious")
        all_full.extend(full_s.cpu().numpy().tolist())
        all_causal.extend(causal_s.cpu().numpy().tolist())
        all_spurious.extend(spur_s.cpu().numpy().tolist())
        all_labels.extend(labels.numpy().tolist())

n_real = all_labels.count(0)
n_fake = all_labels.count(1)

print(f"│ Samples: {len(all_labels):,} (real={n_real:,}, fake={n_fake:,})")
print(f"│")
for head_name, scores in [("Full", all_full), ("Causal", all_causal), ("Spurious", all_spurious)]:
    auc = roc_auc_score(all_labels, scores)
    ap = average_precision_score(all_labels, scores)
    print(f"│ {head_name:10s} Head:  AUC={auc:.4f}  AP={ap:.4f}")
print("└────────────────────────────────────────────────────────────────────┘")
print()

EOF

# ============================================================
# PART 2: FakeAVCeleb (2,702 samples - Cross-domain)
# ============================================================

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  PART 2: FakeAVCeleb (Cross-domain)                                ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

python << 'EOF'
import torch
import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader
from datasets import FakeAVCeleb_NPZ_Dataset
from mlp import AVH_Sup
from mlp_causal import AVH_Causal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load FakeAVCeleb dataset (all features, no split filtering)
config = {"root_path": "../data/favc_features", "apply_l2": True}
dataset = FakeAVCeleb_NPZ_Dataset(config, split=None)
loader = DataLoader(dataset, shuffle=False, batch_size=1, num_workers=8)

print("="*70)
print(f"FakeAVCeleb: {len(dataset)} samples")
print("="*70)
print()

# ─── Evaluate Baseline Model ───────────────────────────────────────────
print("┌─ Baseline Model ──────────────────────────────────────────────────┐")
baseline_model = AVH_Sup.load_from_checkpoint("outputs/ckpts/model-epoch=05.ckpt")
baseline_model.to(device)
baseline_model.eval()

all_scores, all_labels = [], []
with torch.no_grad():
    for video, audio, labels, paths in tqdm.tqdm(loader, desc="Baseline"):
        video = video.to(device)
        audio = audio.to(device)
        scores = baseline_model.predict_scores(video, audio)
        all_scores.extend(scores.cpu().numpy().tolist())
        all_labels.extend(labels.numpy().tolist())

n_real = all_labels.count(0)
n_fake = all_labels.count(1)
auc = roc_auc_score(all_labels, all_scores)
ap = average_precision_score(all_labels, all_scores)

print(f"│ Samples: {len(all_labels):,} (real={n_real:,}, fake={n_fake:,})")
print(f"│ AUC: {auc:.4f}")
print(f"│ AP:  {ap:.4f}")
print("└────────────────────────────────────────────────────────────────────┘")
print()

# ─── Evaluate Causal Model ─────────────────────────────────────────────
print("┌─ Causal Model ────────────────────────────────────────────────────┐")
causal_model = AVH_Causal.load_from_checkpoint("outputs_causal/ckpts/model-epoch=09.ckpt")
causal_model.to(device)
causal_model.eval()

all_full, all_causal, all_spurious, all_labels = [], [], [], []
with torch.no_grad():
    for video, audio, labels, paths in tqdm.tqdm(loader, desc="Causal "):
        video = video.to(device)
        audio = audio.to(device)
        full_s = causal_model.predict_scores(video, audio, mode="full")
        causal_s = causal_model.predict_scores(video, audio, mode="causal")
        spur_s = causal_model.predict_scores(video, audio, mode="spurious")
        all_full.extend(full_s.cpu().numpy().tolist())
        all_causal.extend(causal_s.cpu().numpy().tolist())
        all_spurious.extend(spur_s.cpu().numpy().tolist())
        all_labels.extend(labels.numpy().tolist())

n_real = all_labels.count(0)
n_fake = all_labels.count(1)

print(f"│ Samples: {len(all_labels):,} (real={n_real:,}, fake={n_fake:,})")
print(f"│")
for head_name, scores in [("Full", all_full), ("Causal", all_causal), ("Spurious", all_spurious)]:
    auc = roc_auc_score(all_labels, scores)
    ap = average_precision_score(all_labels, scores)
    print(f"│ {head_name:10s} Head:  AUC={auc:.4f}  AP={ap:.4f}")
print("└────────────────────────────────────────────────────────────────────┘")
print()

EOF

# ============================================================
# Summary
# ============================================================

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  Evaluation Summary                                                ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "✓ AV-Deepfake1M Val (10,000 samples) - In-domain performance"
echo "✓ FakeAVCeleb (2,702 samples) - Cross-domain generalization"
echo ""
echo "Check the results above to compare:"
echo "  • Baseline vs Causal model"
echo "  • Causal vs Spurious heads"
echo "  • In-domain vs Cross-domain performance"
echo ""
echo "============================================================"
echo "  Evaluation completed: $(date)"
echo "============================================================"
