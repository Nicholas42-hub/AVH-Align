"""Domain & Label Probe Eval — A35_e2e."""
import sys
sys.argv += ["--model_module", "train_e2e_a35", "--model_class", "E2EModelA35",
             "--use_fullpath_csv"]
from eval_e2e_domain import main
main()
