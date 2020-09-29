import os
from pathlib import Path

SRC_ROOT = Path(os.path.dirname(os.path.realpath(__file__)))
PRO_ROOT = SRC_ROOT.parent

CHAOSNLI_SNLI = PRO_ROOT / "data/chaosNLI_v1.0/chaosNLI_snli.jsonl"
CHAOSNLI_MNLI = PRO_ROOT / "data/chaosNLI_v1.0/chaosNLI_mnli_m.jsonl"
CHAOSNLI_ALPHANLI = PRO_ROOT / "data/chaosNLI_v1.0/chaosNLI_alphanli.jsonl"

MODEL_PRED_ABDNLI = PRO_ROOT / "data/model_predictions/model_predictions_for_abdnli.json"
MODEL_PRED_NLI = PRO_ROOT / "data/model_predictions/model_predictions_for_snli_mnli.json"


if __name__ == '__main__':
    print(SRC_ROOT)
    print(PRO_ROOT)