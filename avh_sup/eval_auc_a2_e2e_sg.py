"""Direct Model AUC Eval — A2_e2e stop-gradient (default E2EModel)."""
import sys
sys.argv += ["--use_fullpath_csv"]
from eval_e2e_auc import main
main()
