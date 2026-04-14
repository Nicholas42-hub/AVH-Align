"""Direct Model AUC Eval — A6_e2e."""
import sys
sys.argv += ["--model_module", "train_e2e_a6", "--model_class", "E2EModelA6",
             "--use_fullpath_csv"]
from eval_e2e_auc import main
main()
