import argparse
from pathlib import Path
from dotenv import load_dotenv

from clinical_assistant.eval.summarization.run_eval import run_eval

load_dotenv()

EVAL_DIR = Path("outputs/summarization/eval")
DEFAULT_REFERENCE = Path("outputs/summarization/datasets/claude-sonnet-4-6-test.jsonl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Model directory name")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE, help="Path to reference JSONL")
    args = parser.parse_args()

    if args.model:
        paths = [EVAL_DIR / args.model / "preds.jsonl"]
    else:
        paths = sorted(EVAL_DIR.glob("*/preds.jsonl"))

    for predicted_path in paths:
        print(f"\nEvaluating {predicted_path.parent.name}...")
        run_eval(predicted_path, args.reference)