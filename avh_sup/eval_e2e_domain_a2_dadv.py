"""
E2E Domain Probe — A2_e2e_dadv shim
=====================================
Wraps eval_e2e_domain.py with E2EModelA1DADV substituted for E2EModel,
and AV1M_E2E_FullPathDataset substituted for AV1M_E2E_Dataset.

Probes:
  1. Domain AUC: can a logistic regression on Z_c / Z_s predict AV1M vs FAVC?
     → Z_c domain AUC should be low if GRL worked; Z_s should soak domain
  2. Label AUC: can Z_c / Z_s predict real vs fake in-domain and OOD?

Usage:
  python eval_e2e_domain_a2_dadv.py \\
      --ckpt      outputs_A2_e2e_dadv/ckpts/model-epoch=08-val_auc_causal=0.9999.ckpt \\
      --config    configs/A2_e2e_dadv.yaml \\
      --av1m_raw  /scratch/punim2637/nnliang/avd1m_preprocessed/val \\
      --av1m_csv  csv_metadata/av1m_e2e \\
      --favc_raw  /scratch/punim2637/nnliang/favc_preprocessed \\
      --output_dir outputs_A2_e2e_dadv/results \\
      --max_clips 500 --batch_size 4
"""

import sys
import types
import os

sys.path.insert(0, os.path.dirname(__file__))
from train_e2e_a1_dadv import E2EModelA1DADV
from datasets_e2e import AV1M_E2E_FullPathDataset, FakeAVCeleb_E2E_Dataset, e2e_collate_fn

# Inject E2EModelA1DADV as "E2EModel"
_fake_train_e2e = types.ModuleType("train_e2e")
_fake_train_e2e.E2EModel = E2EModelA1DADV
sys.modules["train_e2e"] = _fake_train_e2e

# Shim AV1M_E2E_Dataset → AV1M_E2E_FullPathDataset (absolute-path CSV)
class _AV1M_E2E_Dataset_Shim:
    def __new__(cls, root, csv_dir_or_path, split="val", max_frames=150):
        candidate = csv_dir_or_path
        if candidate.endswith("val_labels.csv"):
            candidate = os.path.dirname(candidate)
        csv_path = candidate if os.path.isfile(candidate) else os.path.join(candidate, "val_e2e_full.csv")
        print(f"[_AV1M_E2E_Dataset_Shim] resolved csv: {csv_path}", flush=True)
        return AV1M_E2E_FullPathDataset(csv_path, max_frames=max_frames)

_fake_datasets_e2e = types.ModuleType("datasets_e2e")
_fake_datasets_e2e.AV1M_E2E_Dataset        = _AV1M_E2E_Dataset_Shim
_fake_datasets_e2e.FakeAVCeleb_E2E_Dataset = FakeAVCeleb_E2E_Dataset
_fake_datasets_e2e.e2e_collate_fn          = e2e_collate_fn
sys.modules["datasets_e2e"] = _fake_datasets_e2e

# Run eval_e2e_domain.py as __main__
import importlib.util
spec = importlib.util.spec_from_file_location(
    "__main__",
    os.path.join(os.path.dirname(__file__), "eval_e2e_domain.py"),
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
