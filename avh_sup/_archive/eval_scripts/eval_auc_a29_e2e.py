"""Direct Model AUC Eval — A29_e2e."""
import sys
sys.argv += ["--model_module", "train_e2e_a29", "--model_class", "E2EModelA29",
             "--use_fullpath_csv"]
from eval_e2e_auc import main
main()
