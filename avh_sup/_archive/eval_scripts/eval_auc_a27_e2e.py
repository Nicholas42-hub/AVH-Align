"""Direct Model AUC Eval — A27_e2e."""
import sys
sys.argv += ["--model_module", "train_e2e_a27", "--model_class", "E2EModelA27",
             "--use_fullpath_csv"]
from eval_e2e_auc import main
main()
