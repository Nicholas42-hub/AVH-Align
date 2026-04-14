"""Domain & Label Probe Eval — A33_e2e."""
import sys
sys.argv += ["--model_module", "train_e2e_a33", "--model_class", "E2EModelA33",
             "--use_fullpath_csv"]
from eval_e2e_domain import main
main()
