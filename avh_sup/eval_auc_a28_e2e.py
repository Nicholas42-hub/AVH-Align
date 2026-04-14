"""Direct Model AUC Eval — A28_e2e."""
import sys
sys.argv += ["--model_module", "train_e2e_a28", "--model_class", "E2EModelA28",
             "--use_fullpath_csv"]
from eval_e2e_auc import main
main()
