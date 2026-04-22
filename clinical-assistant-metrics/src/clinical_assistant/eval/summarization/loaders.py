import json
from collections import defaultdict
from pathlib import Path

from clinical_assistant.summarization.patient import PatientSummary
from clinical_assistant.summarization.encounter import EncounterMeta
from clinical_assistant.eval.summarization.filter import (
    merge_summary,
    delta_to_patient_summary,
)


def load_reference_encounters(path: Path) -> dict[str, list[dict]]:
    """
    Load reference dataset grouped by patient_id.
    Format: {summary, document, summary_delta} per line.
    """
    patients: dict[str, list[dict]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            patient_id = data["summary"]["patient_id"]
            patients[patient_id].append(data)
    return dict(patients)


def load_predicted_encounters(path: Path) -> dict[str, list[dict]]:
    """
    Loads eval.py output format (preds.jsonl).
    Each line has: summary, document, pred_summary_delta, target_summary_delta.
    Returns patient_id -> list of encounter dicts in order.
    """
    patients: dict[str, list[dict]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            patient_id = data["summary"]["patient_id"]
            patients[patient_id].append(data)
    return dict(patients)


def load_final_summaries(
    predicted_path: Path,
    reference_path: Path,
) -> dict[str, tuple[PatientSummary, PatientSummary]]:
    """
    Reconstructs final cumulative summaries for reference and predicted.

    Reference is built from the reference dataset (all encounters, using summary_delta).
    Predicted is built from preds.jsonl (only successful encounters, using pred_summary_delta).

    Returns patient_id -> (ref_summary, pred_summary).
    Only includes patients present in both files.
    """
    ref_encounters = load_reference_encounters(reference_path)
    pred_encounters = load_predicted_encounters(predicted_path)

    summaries = {}

    for patient_id, ref_list in ref_encounters.items():
        if patient_id not in pred_encounters:
            continue

        pred_list = pred_encounters[patient_id]
        pred_map = {
            enc["document"]["meta"]["id"]: enc
            for enc in pred_list
        }

        # Build reference cumulative summary from all encounters
        ref_summary = PatientSummary.model_validate(ref_list[0]["summary"])
        for ref_enc in ref_list:
            meta = EncounterMeta.model_validate(ref_enc["document"]["meta"])
            delta_ps = delta_to_patient_summary(ref_summary, ref_enc["summary_delta"], meta)
            ref_summary = merge_summary(ref_summary, delta_ps)

        # Build predicted cumulative summary from successful encounters only
        pred_summary = PatientSummary.model_validate(ref_list[0]["summary"])
        for ref_enc in ref_list:
            doc_id = ref_enc["document"]["meta"]["id"]
            if doc_id not in pred_map:
                continue
            pred_enc = pred_map[doc_id]
            meta = EncounterMeta.model_validate(pred_enc["document"]["meta"])
            delta_ps = delta_to_patient_summary(pred_summary, pred_enc["pred_summary_delta"], meta)
            pred_summary = merge_summary(pred_summary, delta_ps)

        summaries[patient_id] = (ref_summary, pred_summary)

    return summaries