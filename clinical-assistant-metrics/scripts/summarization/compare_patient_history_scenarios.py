import argparse
import json
import math
from pathlib import Path

from clinical_assistant.eval.summarization.aggregate import compute_aggregate_scores
from clinical_assistant.eval.summarization.allergies import compute_allergy_metrics
from clinical_assistant.eval.summarization.clinical_problems import compute_clinical_problem_metrics
from clinical_assistant.eval.summarization.family_history import compute_family_history_metrics
from clinical_assistant.eval.summarization.measurements import compute_measurement_metrics
from clinical_assistant.eval.summarization.medications import compute_medication_metrics
from clinical_assistant.eval.summarization.procedures import compute_procedure_metrics
from clinical_assistant.eval.summarization.substances import compute_substance_metrics
from clinical_assistant.summarization.patient import PatientSummary, PatientSummaryDelta, PatientSummaryExtractor


class _NoopClient:
    async def structured_output(self, prompt, output_format):
        return None


def _age_score(pred_age: int, ref_age: int, half_life: float = 5.0) -> float:
    sigma = half_life / math.sqrt(2 * math.log(2))
    return math.exp(-0.5 * ((pred_age - ref_age) / sigma) ** 2)


def _compute_patient_info_metrics(pred: PatientSummary, ref: PatientSummary) -> dict:
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
            "score": _age_score(pred.age, ref.age),
        },
    }


def _compute_all_metrics(ref: PatientSummary, pred: PatientSummary) -> dict:
    metrics = {"patient_info": _compute_patient_info_metrics(pred, ref)}

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


def _final_score(reference: PatientSummary, predicted: PatientSummary) -> float:
    metrics = _compute_all_metrics(reference, predicted)
    scores = compute_aggregate_scores(metrics)
    return float(scores.get("patient_score", 0.0))


def _load_rows_by_patient(path: Path, patient_id: str) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("summary", {}).get("patient_id") == patient_id:
                rows.append(row)
    return rows


def _sort_key(row: dict) -> tuple[str, str, str]:
    meta = row.get("document", {}).get("meta", {})
    return (
        str(meta.get("start_date") or ""),
        str(meta.get("end_date") or ""),
        str(meta.get("id") or ""),
    )


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate scenario A/B comparison JSONs for one patient.")
    parser.add_argument("--gold-path", default="data/claude-sonnet-4-6.jsonl")
    parser.add_argument("--self-path", default="data/Qwen-Qwen3-4B-Instruct-2507.jsonl")
    parser.add_argument("--patient-id", default="10000032")
    parser.add_argument("--output-dir", default="outputs/summarization/scenario_comparison")
    args = parser.parse_args()

    gold_path = Path(args.gold_path)
    self_path = Path(args.self_path)
    patient_id = args.patient_id
    output_dir = Path(args.output_dir) / patient_id

    gold_rows = sorted(_load_rows_by_patient(gold_path, patient_id), key=_sort_key)
    self_rows = _load_rows_by_patient(self_path, patient_id)
    self_by_doc_id = {str(r["document"]["meta"]["id"]): r for r in self_rows}

    if not gold_rows:
        raise ValueError(f"No rows found for patient {patient_id} in {gold_path}")

    if len(gold_rows) != 4:
        print(f"Warning: expected 4 rows for patient {patient_id}, found {len(gold_rows)}")

    extractor = PatientSummaryExtractor(_NoopClient())

    for idx, gold_row in enumerate(gold_rows, start=1):
        doc_id = str(gold_row["document"]["meta"]["id"])
        if doc_id not in self_by_doc_id:
            raise ValueError(f"Missing document {doc_id} in self-history file {self_path}")

        self_row = self_by_doc_id[doc_id]

        target_summary_before = PatientSummary.model_validate(gold_row["summary"])
        target_delta = PatientSummaryDelta.model_validate(gold_row["summary_delta"])
        target_summary_after = extractor.update(target_summary_before, target_delta)

        scenario_a_summary_after = target_summary_after

        scenario_b_summary_before = PatientSummary.model_validate(self_row["summary"])
        scenario_b_delta = PatientSummaryDelta.model_validate(self_row["summary_delta"])
        scenario_b_summary_after = extractor.update(scenario_b_summary_before, scenario_b_delta)

        score_a = round(_final_score(target_summary_after, scenario_a_summary_after), 6)
        score_b = round(_final_score(target_summary_after, scenario_b_summary_after), 6)

        base_payload = {
            "patient_id": patient_id,
            "document_id": doc_id,
            "document_index": idx,
            "target": gold_row["summary_delta"],
        }

        scenario_a_payload = {
            **base_payload,
            "scenario": "A",
            "history_source": "gold_claude",
            "model_source": "claude-sonnet-4-6",
            "score_final": score_a,
            "prediction": gold_row["summary_delta"],
        }
        scenario_b_payload = {
            **base_payload,
            "scenario": "B",
            "history_source": "self_predicted",
            "model_source": "Qwen-Qwen3-4B-Instruct-2507",
            "score_final": score_b,
            "prediction": self_row["summary_delta"],
        }

        _save_json(output_dir / f"doc_{idx:02d}_scenario_a.json", scenario_a_payload)
        _save_json(output_dir / f"doc_{idx:02d}_scenario_b.json", scenario_b_payload)

    print(f"Saved {len(gold_rows) * 2} comparison JSON files to {output_dir}")


if __name__ == "__main__":
    main()
