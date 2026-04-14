"""Domain & Label Probe Eval — A34_e2e."""
import sys
sys.argv += ["--model_module", "train_e2e_a34", "--model_class", "E2EModelA34",
             "--use_fullpath_csv"]
from eval_e2e_domain import main
main()
