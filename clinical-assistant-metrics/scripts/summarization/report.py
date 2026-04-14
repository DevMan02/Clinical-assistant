import json
import math
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

from clinical_assistant.summarization.patient import PatientSummary
from clinical_assistant.eval.summarization.medications import compute_medication_metrics
from clinical_assistant.eval.summarization.allergies import compute_allergy_metrics
from clinical_assistant.eval.summarization.substances import compute_substance_metrics
from clinical_assistant.eval.summarization.clinical_problems import compute_clinical_problem_metrics
from clinical_assistant.eval.summarization.measurements import compute_measurement_metrics
from clinical_assistant.eval.summarization.family_history import compute_family_history_metrics
from clinical_assistant.eval.summarization.procedures import compute_procedure_metrics
from clinical_assistant.eval.summarization.aggregate import compute_aggregate_scores

load_dotenv()

BASE_DIR = Path("data")
OUTPUT_DIR = Path("output") / "summaries_reports"


def load_jsonl(path: Path) -> list[PatientSummary]:
    summaries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if "id" in data and "patient_id" not in data:
                data["patient_id"] = data.pop("id")
            if "document_ids" not in data:
                data["document_ids"] = []
            summaries.append(PatientSummary.model_validate(data))
    return summaries


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record, default=str) + "\n")


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def age_score(pred_age: int, ref_age: int, half_life: float = 5.0) -> float:
    sigma = half_life / math.sqrt(2 * math.log(2))
    return math.exp(-0.5 * ((pred_age - ref_age) / sigma) ** 2)


def compute_patient_info_metrics(pred: PatientSummary, ref: PatientSummary) -> dict:
    return {
        "id": ref.patient_id,
        "sex": {
            "reference": ref.sex,
            "predicted": pred.sex,
            "correct": pred.sex == ref.sex,
            "score": 1.0 if pred.sex == ref.sex else 0.0,
        },
        "age": {
            "reference": ref.age,
            "predicted": pred.age,
            "correct": pred.age == ref.age,
            "difference": abs(pred.age - ref.age),
            "score": age_score(pred.age, ref.age),
        },
    }


def compute_all_metrics(ref: PatientSummary, pred: PatientSummary) -> dict:
    metrics = {"patient_info": compute_patient_info_metrics(pred, ref)}

    if ref.medications or pred.medications:
        metrics["medications"] = compute_medication_metrics(pred.medications, ref.medications)
    if ref.allergies or pred.allergies:
        metrics["allergies"] = compute_allergy_metrics(pred.allergies, ref.allergies)
    if ref.substances or pred.substances:
        metrics["substances"] = compute_substance_metrics(pred.substances, ref.substances)
    if ref.clinical_problems or pred.clinical_problems:
        metrics["clinical_problems"] = compute_clinical_problem_metrics(pred.clinical_problems, ref.clinical_problems)
    if ref.measurements or pred.measurements:
        metrics["measurements"] = compute_measurement_metrics(pred.measurements, ref.measurements)
    if ref.family_history or pred.family_history:
        metrics["family_history"] = compute_family_history_metrics(pred.family_history, ref.family_history)
    if ref.procedures or pred.procedures:
        metrics["procedures"] = compute_procedure_metrics(pred.procedures, ref.procedures)

    return metrics


def compute_summary(all_scores: list[dict], llm_name: str, digits: int = 6) -> dict:
    """Average all section scores and patient scores across patients."""
    if not all_scores:
        return {"llm": llm_name, "n_patients": 0}

    # collect section scores
    section_keys = set()
    for s in all_scores:
        section_keys.update(s.get("sections", {}).keys())

    section_averages = {}
    for section in section_keys:
        metric_keys = set()
        for s in all_scores:
            if section in s.get("sections", {}):
                metric_keys.update(s["sections"][section].keys())
        metric_keys.discard("section_score")

        section_averages[section] = {}
        for metric in metric_keys:
            values = [
                s["sections"][section][metric]
                for s in all_scores
                if section in s.get("sections", {}) and metric in s["sections"][section]
            ]
            section_averages[section][metric] = (
                round(float(np.mean(values)), digits) if values else 0.0
            )

        section_scores = [
            s["sections"][section]["section_score"]
            for s in all_scores
            if section in s.get("sections", {})
        ]
        section_averages[section]["section_score"] = (
            round(float(np.mean(section_scores)), digits) if section_scores else 0.0
        )

    patient_scores = [s["patient_score"] for s in all_scores if s.get("patient_score") is not None]

    return {
        "llm": llm_name,
        "n_patients": len(all_scores),
        "patient_score": (
            round(float(np.mean(patient_scores)), digits) if patient_scores else 0.0
        ),
        "sections": section_averages,
    }


def run_eval(reference_path: Path, predicted_path: Path, output_dir: Path) -> None:
    llm_name = predicted_path.stem
    llm_dir = output_dir / llm_name
    llm_dir.mkdir(parents=True, exist_ok=True)

    ref_summaries = {s.patient_id: s for s in load_jsonl(reference_path)}
    pred_summaries = load_jsonl(predicted_path)

    all_scores = []

    for pred in pred_summaries:
        if pred.patient_id not in ref_summaries:
            print(f"  No reference for {pred.patient_id}, skipping")
            continue

        ref = ref_summaries[pred.patient_id]
        print(f"  Evaluating {pred.patient_id}...")

        metrics = compute_all_metrics(ref, pred)
        scores = compute_aggregate_scores(metrics)

        # detailed metrics
        save_json(metrics, llm_dir / "patients" / f"{pred.patient_id}_metrics.json")

        all_scores.append(scores)

    # aggregated scores
    save_jsonl(all_scores, output_dir / llm_name / "cumulative_scores.jsonl")

    # summary
    save_json(compute_summary(all_scores, llm_name), output_dir / llm_name/ "report.json")

    print(f"\nSaved to {llm_dir}")


if __name__ == "__main__":
    summaries_path = BASE_DIR / "summaries"
    ref_path = BASE_DIR / "reference" / "reference.jsonl"

    for predicted_path in sorted(summaries_path.glob("*.jsonl")):
        print(f"\nEvaluating {predicted_path.name}...")
        run_eval(ref_path, predicted_path, OUTPUT_DIR)