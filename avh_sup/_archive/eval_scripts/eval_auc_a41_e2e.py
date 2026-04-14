"""Direct Model AUC Eval — A41_e2e (uses E2EModelA41)."""
import sys
sys.argv += ["--model_module", "train_e2e_a41", "--model_class", "E2EModelA41",
             "--use_fullpath_csv"]
from eval_e2e_auc import main
main()
