"""Direct Model AUC Eval — A26_e2e."""
import sys
sys.argv += ["--model_module", "train_e2e_a26", "--model_class", "E2EModelA26",
             "--use_fullpath_csv"]
from eval_e2e_auc import main
main()
