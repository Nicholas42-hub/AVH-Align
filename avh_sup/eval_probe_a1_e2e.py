"""
Domain & Label Probe for A1_e2e_dadv checkpoint
=================================================
Thin wrapper around eval_e2e_domain.py that substitutes E2EModelA1DADV for
E2EModel (the only difference vs the generic eval).

Usage:
  python eval_probe_a1_e2e.py \\
      --ckpt  outputs_A1_e2e_dadv/ckpts/model-epoch=06-val_auc_causal=1.0000.ckpt \\
      --config configs/A1_e2e_dadv.yaml \\
      --av1m_raw  /scratch/punim2637/nnliang/avd1m_preprocessed/val \\
      --av1m_csv  csv_metadata/av1m_e2e \\
      --favc_raw  /scratch/punim2637/nnliang/favc_preprocessed \\
      --output_dir outputs_A1_e2e_dadv/results \\
      --max_clips 500 --batch_size 4
"""

import sys
import types
import os

# ── Inject E2EModelA1DADV as "E2EModel" in a fake train_e2e module so that
#    eval_e2e_domain.py's  "from train_e2e import E2EModel" resolves correctly.
sys.path.insert(0, os.path.dirname(__file__))
from train_e2e_a1_dadv import E2EModelA1DADV
from datasets_e2e import AV1M_E2E_FullPathDataset, FakeAVCeleb_E2E_Dataset, e2e_collate_fn

# Build a minimal fake module so csv-based dataset is available too
_fake_train_e2e = types.ModuleType("train_e2e")
_fake_train_e2e.E2EModel = E2EModelA1DADV
sys.modules["train_e2e"] = _fake_train_e2e

# Also need AV1M_E2E_Dataset shim: eval_e2e_domain uses AV1M_E2E_Dataset(root, csv, split, max_frames)
# but our data is best addressed via AV1M_E2E_FullPathDataset(csv_path, max_frames).
# We shim it so av1m_csv / "val_e2e_full.csv" is used automatically.
class _AV1M_E2E_Dataset_Shim:
    """Shim: ignores root/split, reads absolute-path CSV directly.
    eval_e2e_domain calls:  AV1M_E2E_Dataset(av1m_raw, av1m_csv, split, max_frames)
    where av1m_csv is our csv_metadata/av1m_e2e directory.
    We always resolve to  <av1m_csv>/val_e2e_full.csv  regardless of split.
    """
    def __new__(cls, root, csv_dir_or_path, split="val", max_frames=150):
        # Strip any trailing filename component added by eval_e2e_domain
        # (it does os.path.join(args.av1m_csv, "val_labels.csv") internally)
        candidate = csv_dir_or_path
        if candidate.endswith("val_labels.csv"):
            candidate = os.path.dirname(candidate)
        if os.path.isfile(candidate):
            csv_path = candidate
        else:
            csv_path = os.path.join(candidate, "val_e2e_full.csv")
        print(f"[_AV1M_E2E_Dataset_Shim] resolved csv: {csv_path}", flush=True)
        return AV1M_E2E_FullPathDataset(csv_path, max_frames=max_frames)

_fake_datasets_e2e = types.ModuleType("datasets_e2e")
_fake_datasets_e2e.AV1M_E2E_Dataset      = _AV1M_E2E_Dataset_Shim
_fake_datasets_e2e.FakeAVCeleb_E2E_Dataset = FakeAVCeleb_E2E_Dataset
_fake_datasets_e2e.e2e_collate_fn        = e2e_collate_fn
sys.modules["datasets_e2e"] = _fake_datasets_e2e

# ── Now run eval_e2e_domain as __main__ ───────────────────────────────────────
import importlib, importlib.util
spec = importlib.util.spec_from_file_location(
    "__main__",
    os.path.join(os.path.dirname(__file__), "eval_e2e_domain.py"),
)
mod = importlib.util.module_from_spec(spec)
# Forward all sys.argv
spec.loader.exec_module(mod)
