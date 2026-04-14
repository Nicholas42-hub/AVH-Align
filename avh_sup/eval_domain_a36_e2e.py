"""Domain & Label Probe Eval — A36_e2e."""
import sys
sys.argv += ["--model_module", "train_e2e_a36", "--model_class", "E2EModelA36",
             "--use_fullpath_csv"]
from eval_e2e_domain import main
main()
