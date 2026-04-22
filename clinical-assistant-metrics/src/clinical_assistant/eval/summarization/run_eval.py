import json
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from rich.progress import Progress, SpinnerColumn, BarColumn, MofNCompleteColumn, TextColumn, TimeElapsedColumn

from clinical_assistant.summarization.patient import PatientSummary
from clinical_assistant.eval.summarization.loaders import load_final_summaries
from clinical_assistant.eval.summarization.medications import compute_medication_metrics
from clinical_assistant.eval.summarization.allergies import compute_allergy_metrics
from clinical_assistant.eval.summarization.substances import compute_substance_metrics
from clinical_assistant.eval.summarization.clinical_problems import compute_clinical_problem_metrics
from clinical_assistant.eval.summarization.measurements import compute_measurement_metrics
from clinical_assistant.eval.summarization.family_history import compute_family_history_metrics
from clinical_assistant.eval.summarization.procedures import compute_procedure_metrics

load_dotenv()


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
    import math
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
    if not all_scores:
        return {"llm": llm_name, "n_patients": 0}

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
                if section in s.get("sections", {})
                and metric in s["sections"][section]
                and s["sections"][section][metric] is not None
            ]
            section_averages[section][metric] = round(
                float(np.mean(values)), digits
            ) if values else None

        section_scores = [
            s["sections"][section]["section_score"]
            for s in all_scores
            if section in s.get("sections", {})
            and s["sections"][section].get("section_score") is not None
        ]
        section_averages[section]["section_score"] = round(
            float(np.mean(section_scores)), digits
        ) if section_scores else None

    patient_scores = [
        s["patient_score"] for s in all_scores
        if s.get("patient_score") is not None
    ]

    return {
        "llm": llm_name,
        "n_patients": len(all_scores),
        "patient_score": round(float(np.mean(patient_scores)), digits) if patient_scores else None,
        "sections": section_averages,
    }


def run_eval(predicted_path: Path, reference_path: Path) -> None:
    from clinical_assistant.eval.summarization.aggregate import compute_aggregate_scores

    llm_name = predicted_path.parent.name
    llm_dir = predicted_path.parent / "report"
    llm_dir.mkdir(parents=True, exist_ok=True)

    summaries = load_final_summaries(predicted_path, reference_path)
    all_scores = []
    eval_times = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("{task.fields[status]}"),
    ) as progress:
        task = progress.add_task("Evaluating summaries", total=len(summaries), status="")

        for i, (patient_id, (ref, pred)) in enumerate(summaries.items()):
            t0 = time.time()
            metrics = compute_all_metrics(ref, pred)
            scores = compute_aggregate_scores(metrics)
            save_json(metrics, llm_dir / "patients" / f"{patient_id}_metrics.json")
            all_scores.append(scores)

            eval_times.append(time.time() - t0)
            avg = sum(eval_times) / len(eval_times)
            remaining = avg * (len(summaries) - i - 1)
            mins, secs = divmod(int(remaining), 60)
            progress.update(task, status=f"[cyan]~{mins}m{secs:02d}s left ({avg:.1f}s/ex)")
            progress.advance(task)

    save_jsonl(all_scores, llm_dir / "cumulative_scores.jsonl")
    save_json(compute_summary(all_scores, llm_name), llm_dir / "report.json")
    print(f"\nSaved to {llm_dir}")