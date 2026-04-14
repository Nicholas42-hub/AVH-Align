"""Direct Model AUC Eval — A2_e2e (E2EModelA1DADV codepath)."""
import sys
sys.argv += ["--model_module", "train_e2e_a1_dadv", "--model_class", "E2EModelA1DADV",
             "--use_fullpath_csv"]
from eval_e2e_auc import main
main()
