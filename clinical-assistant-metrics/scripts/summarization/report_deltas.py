import argparse
import time
from collections import defaultdict
from pathlib import Path
from dotenv import load_dotenv
from rich.progress import Progress, SpinnerColumn, BarColumn, MofNCompleteColumn, TextColumn, TimeElapsedColumn

from clinical_assistant.summarization.patient import PatientSummary
from clinical_assistant.summarization.encounter import EncounterMeta
from clinical_assistant.eval.summarization.run_eval import compute_all_metrics, save_json, save_jsonl, compute_summary
from clinical_assistant.eval.summarization.aggregate import compute_aggregate_scores
from clinical_assistant.eval.summarization.filter import delta_to_patient_summary
from clinical_assistant.eval.summarization.loaders import load_predicted_encounters, load_reference_encounters

load_dotenv()

DEFAULT_REFERENCE = Path("outputs/summarization/datasets/claude-sonnet-4-6-test.jsonl")

def compute_by_encounter_index(
    all_scores_by_index: dict[int, list[dict]],
    digits: int = 6,
) -> dict[str, dict]:
    result = {}
    for n in sorted(all_scores_by_index.keys()):
        scores = all_scores_by_index[n]
        summary = compute_summary(scores, llm_name="", digits=digits)
        summary["n_patients"] = len(scores)
        result[str(n)] = summary
    return result


def zero_scores(encounter_index: int) -> dict:
    return {"patient_score": 0.0, "sections": {}, "encounter_index": encounter_index}


def run_delta_eval(predicted_path: Path, reference_path: Path) -> None:
    llm_name = predicted_path.parent.name
    llm_dir = predicted_path.parent / "report_deltas"
    llm_dir.mkdir(parents=True, exist_ok=True)

    ref_encounters = load_reference_encounters(reference_path)
    pred_encounters = load_predicted_encounters(predicted_path)

    all_scores = []
    all_scores_by_index: dict[int, list[dict]] = defaultdict(list)
    encounter_count_distribution: dict[int, int] = defaultdict(int)

    # Only evaluate patients present in both reference and predicted
    patient_ids = [pid for pid in pred_encounters if pid in ref_encounters]
    total = sum(len(ref_encounters[pid]) for pid in patient_ids)
    eval_times = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("{task.fields[status]}"),
    ) as progress:
        task = progress.add_task("Evaluating deltas", total=total, status="")

        i = 0
        for patient_id in patient_ids:
            ref_list = ref_encounters[patient_id]
            pred_list = pred_encounters[patient_id]
            n_expected = len(ref_list)
            n_actual = len(pred_list)
            encounter_count_distribution[n_expected] += 1

            # Build a map of doc_id -> pred encounter for matching
            pred_map = {
                enc["document"]["meta"]["id"]: enc
                for enc in pred_list
            }

            patient_scores = []
            for n, ref_enc in enumerate(ref_list):
                t0 = time.time()
                doc_id = ref_enc["document"]["meta"]["id"]

                if doc_id in pred_map:
                    encounter_data = pred_map[doc_id]
                    prior = PatientSummary.model_validate(encounter_data["summary"])
                    meta = EncounterMeta.model_validate(encounter_data["document"]["meta"])

                    ref_delta_ps = delta_to_patient_summary(prior, encounter_data["target_summary_delta"], meta)
                    pred_delta_ps = delta_to_patient_summary(prior, encounter_data["pred_summary_delta"], meta)

                    metrics = compute_all_metrics(ref_delta_ps, pred_delta_ps)
                    scores = compute_aggregate_scores(metrics)
                    scores["encounter_index"] = n
                    save_json(metrics, llm_dir / "patients" / f"{patient_id}_enc{n}_metrics.json")
                else:
                    # Missing encounter — penalize with score 0
                    progress.console.log(f"[yellow]Missing encounter {doc_id} for patient {patient_id} → score 0[/yellow]")
                    scores = zero_scores(n)

                scores["patient_id"] = patient_id
                scores["doc_id"] = doc_id

                all_scores.append(scores)
                all_scores_by_index[n].append(scores)
                patient_scores.append(scores)

                eval_times.append(time.time() - t0)
                i += 1
                avg = sum(eval_times) / len(eval_times)
                remaining = avg * (total - i)
                mins, secs = divmod(int(remaining), 60)
                progress.update(task, status=f"[cyan]~{mins}m{secs:02d}s left ({avg:.1f}s/ex)")
                progress.advance(task)

            save_json(
                {
                    "patient_id": patient_id,
                    "n_encounters_expected": n_expected,
                    "n_encounters_found": n_actual,
                    **compute_summary(patient_scores, llm_name=llm_name),
                },
                llm_dir / "patients" / f"{patient_id}_report.json"
            )

    save_jsonl(all_scores, llm_dir / "cumulative_scores.jsonl")

    report = {
        "llm": llm_name,
        "encounter_distribution": {
            str(k): v for k, v in sorted(encounter_count_distribution.items())
        },
        "by_encounter_index": compute_by_encounter_index(all_scores_by_index),
        "global": compute_summary(all_scores, llm_name),
    }
    save_json(report, llm_dir / "delta_report.json")

    print(f"\nSaved to {llm_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Model directory name")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE, help="Path to reference JSONL")
    args = parser.parse_args()

    eval_path = Path("outputs/summarization/eval")

    if args.model:
        paths = [eval_path / args.model / "preds.jsonl"]
    else:
        paths = sorted(eval_path.glob("*/preds.jsonl"))

    for predicted_path in paths:
        print(f"\nEvaluating {predicted_path.parent.name}...")
        run_delta_eval(predicted_path, args.reference)