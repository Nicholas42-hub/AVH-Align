"""Domain & Label Probe Eval — A39_e2e (uses E2EModelA38)."""
import sys
sys.argv += ["--model_module", "train_e2e_a38", "--model_class", "E2EModelA38",
             "--use_fullpath_csv"]
from eval_e2e_domain import main
main()
