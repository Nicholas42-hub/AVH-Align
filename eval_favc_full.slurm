#!/bin/bash
#SBATCH --job-name=eval_favc_full
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/eval_favc_full_%j.out
#SBATCH --error=logs/eval_favc_full_%j.err

# ============================================================
#  Evaluate on FULL FakeAVCeleb Dataset (~21k samples)
#  
#  Prerequisite: favc_full_features.slurm completed
# ============================================================

echo "============================================================"
echo "  FakeAVCeleb FULL Dataset Evaluation"
echo "  Job ID : $SLURM_JOB_ID"
echo "  Node   : $SLURM_NODELIST"
echo "  Start  : $(date)"
echo "============================================================"

cd /data/projects/punim2637/nnliang/AVH-Align/avh_sup

module purge
module load Miniconda3/23.10.0-1
module load GCC/11.3.0
module load CUDA/12.2.0

# Use conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/projects/punim2637/nnliang/avh_env

echo "Python: $(which python) — $(python --version)"
echo ""

# Set environment (avoid user site packages to prevent conflicts)
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Count available features
FEAT_COUNT=$(find ../data/favc_features -name "*.npz" 2>/dev/null | wc -l)
echo "Available features: $FEAT_COUNT"
echo ""

# ============================================================
# Run Evaluation on ALL Available Features
# ============================================================

python << 'EOF'
import torch
import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader
from datasets import FakeAVCeleb_NPZ_Dataset
from mlp import AVH_Sup
from mlp_causal import AVH_Causal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

# Load ALL FakeAVCeleb features (no split filtering)
config = {"root_path": "../data/favc_features", "apply_l2": True}
dataset = FakeAVCeleb_NPZ_Dataset(config, split=None)
loader = DataLoader(dataset, shuffle=False, batch_size=1, num_workers=8)

print("="*75)
print(f"FakeAVCeleb FULL Dataset Evaluation: {len(dataset):,} samples")
print("="*75)
print()

# ─── Evaluate Baseline Model ─────────────────────────────────────────────
print("╔═══════════════════════════════════════════════════════════════════════╗")
print("║  BASELINE MODEL (AVH-Sup)                                            ║")
print("╚═══════════════════════════════════════════════════════════════════════╝")
print()

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

print(f"\n Results:")
print(f"  • Total samples: {len(all_labels):,}")
print(f"  • Real: {n_real:,} ({n_real/len(all_labels)*100:.1f}%)")
print(f"  • Fake: {n_fake:,} ({n_fake/len(all_labels)*100:.1f}%)")
print(f"  • AUC:  {auc:.4f}")
print(f"  • AP:   {ap:.4f}")
print()

# ─── Evaluate Causal Model ───────────────────────────────────────────────
print("╔═══════════════════════════════════════════════════════════════════════╗")
print("║  CAUSAL MODEL (AVH-Causal)                                           ║")
print("╚═══════════════════════════════════════════════════════════════════════╝")
print()

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

print(f"\n Results:")
print(f"  • Total samples: {len(all_labels):,}")
print(f"  • Real: {n_real:,} ({n_real/len(all_labels)*100:.1f}%)")
print(f"  • Fake: {n_fake:,} ({n_fake/len(all_labels)*100:.1f}%)")
print()

for head_name, scores in [("Full", all_full), ("Causal", all_causal), ("Spurious", all_spurious)]:
    auc = roc_auc_score(all_labels, scores)
    ap = average_precision_score(all_labels, scores)
    print(f"  {head_name:10s} HEAD:  AUC={auc:.4f}  AP={ap:.4f}")

print()
print("="*75)

# Calculate causal hypothesis metrics
delta_auc = all_causal_auc - all_spurious_auc if 'all_causal_auc' in locals() else 0
all_causal_auc = roc_auc_score(all_labels, all_causal)
all_spurious_auc = roc_auc_score(all_labels, all_spurious)
delta_auc = all_causal_auc - all_spurious_auc

print("\n Causal Hypothesis Analysis:")
print(f"  Δ_AUC (Causal - Spurious) = {delta_auc:+.4f}")
if delta_auc > 0.15:
    print("  ✓ Strong evidence for causal hypothesis (Δ > 0.15)")
elif delta_auc > 0.05:
    print("  ⚠ Moderate evidence for causal hypothesis (0.05 < Δ < 0.15)")
else:
    print("  ✗ Weak evidence for causal hypothesis (Δ < 0.05)")
print()

EOF

echo ""
echo "============================================================"
echo "  Evaluation completed: $(date)"
echo "============================================================"
