"""Direct Model AUC Eval — A39_e2e shim (uses E2EModelA38 codepath)."""
import sys, types, os
sys.path.insert(0, os.path.dirname(__file__))
from train_e2e_a38 import E2EModelA38
from datasets_e2e import AV1M_E2E_FullPathDataset, FakeAVCeleb_E2E_Dataset, e2e_collate_fn

_fake_train_e2e = types.ModuleType("train_e2e")
_fake_train_e2e.E2EModel = E2EModelA38
sys.modules["train_e2e"] = _fake_train_e2e


class _AV1M_E2E_Dataset_Shim:
    def __new__(cls, root, csv_dir_or_path, split="val", max_frames=150):
        candidate = csv_dir_or_path
        if candidate.endswith("val_labels.csv"):
            candidate = os.path.dirname(candidate)
        csv_path = candidate if os.path.isfile(candidate) else os.path.join(candidate, "val_e2e_full.csv")
        print(f"[Shim] resolved csv: {csv_path}", flush=True)
        return AV1M_E2E_FullPathDataset(csv_path, max_frames=max_frames)


_fake_datasets_e2e = types.ModuleType("datasets_e2e")
_fake_datasets_e2e.AV1M_E2E_Dataset        = _AV1M_E2E_Dataset_Shim
_fake_datasets_e2e.FakeAVCeleb_E2E_Dataset = FakeAVCeleb_E2E_Dataset
_fake_datasets_e2e.e2e_collate_fn          = e2e_collate_fn
sys.modules["datasets_e2e"] = _fake_datasets_e2e

import importlib.util
spec = importlib.util.spec_from_file_location("__main__", os.path.join(os.path.dirname(__file__), "eval_e2e_auc.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
