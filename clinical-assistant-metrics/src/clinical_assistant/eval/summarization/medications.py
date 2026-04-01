import numpy as np
from dataclasses import dataclass
from rapidfuzz.fuzz import ratio, partial_ratio, token_sort_ratio, token_set_ratio, partial_token_set_ratio, WRatio
from rapidfuzz import process

from clinical_assistant.summarization.medications import MedicationHistory, MedicationEvent
from clinical_assistant.eval.text_similarity import compare_dosage, compare_optional_text, match_names
from clinical_assistant.eval.sequence_alignment import compare_event_sequences_by_id
from clinical_assistant.eval.summarization.base import (
    compute_detection_metrics, compute_categorical_metrics, display_peritem_metrics
)


scorers = {
    "ratio": ratio,
    "partial_ratio": partial_ratio,
    "token_sort_ratio": token_sort_ratio,
    "token_set_ratio": token_set_ratio,
    "partial_token_set_ratio": partial_token_set_ratio,
    "WRatio": WRatio,
}


def debug_match(pred_name: str, ref_names: set[str]) -> None:
    for scorer_name, scorer in scorers.items():
        # noinspection PyTypeChecker
        result = process.extractOne(pred_name, list(ref_names), scorer=scorer)
        if result:
            print(f"{scorer_name:30s} → '{result[0]}' score={result[1]:.1f}")


@dataclass
@dataclass
class EventWeights:
    status: float = 0.4
    dosage: float = 0.3
    reason: float = 0.2
    detail: float = 0.05
    category: float = 0.05


def event_similarity(
    pred: MedicationEvent,
    ref: MedicationEvent,
    weights: EventWeights = EventWeights(),
) -> float:
    status_score = 1.0 if pred.status == ref.status else 0.0
    category_score = 1.0 if pred.category == ref.category else 0.0
    dosage_score = compare_dosage(pred.dosage, ref.dosage)
    reason_score = compare_optional_text(pred.reason, ref.reason)
    detail_score = compare_optional_text(pred.detail, ref.detail)
    return (
        weights.status * status_score
        + weights.category * category_score
        + weights.dosage * dosage_score
        + weights.reason * reason_score
        + weights.detail * detail_score
    )


def compute_medication_metrics(
    predicted: list[MedicationHistory],
    reference: list[MedicationHistory],
) -> dict:
    metrics = {}

    pred_names = {m.name.lower() for m in predicted}
    ref_names = {m.name.lower() for m in reference}

    fuzzy_mapping = match_names(pred_names, ref_names)
    detection, name_mapping, score_mapping, common, common_weights = compute_detection_metrics(
        pred_names, ref_names, fuzzy_mapping
    )
    metrics["detection"] = detection

    pred_map = {name_mapping.get(m.name.lower(), m.name.lower()): m for m in predicted}
    ref_map = {m.name.lower(): m for m in reference}

    if common:
        # --- Therapeutic category ---
        metrics["therapeutic_category"] = compute_categorical_metrics(
            ref_vals=[ref_map[n].events[-1][1].category for n in common],
            pred_vals=[pred_map[n].events[-1][1].category for n in common],
            common_weights=common_weights,
        )

        # --- Medication status ---
        metrics["medication_status"] = compute_categorical_metrics(
            ref_vals=[ref_map[n].events[-1][1].status for n in common],
            pred_vals=[pred_map[n].events[-1][1].status for n in common],
            common_weights=common_weights,
        )

        # --- Dosage ---
        dosage_scores = [
            compare_dosage(pred_map[n].events[-1][1].dosage, ref_map[n].events[-1][1].dosage)
            for n in common
        ]
        metrics["dosage"] = {
            **display_peritem_metrics(dosage_scores, common, weights=common_weights),
            "exact_match_rate": float(sum(1 for s in dosage_scores if s == 1.0) / len(dosage_scores)),
        }

        # --- Reason ---
        reason_scores = [
            compare_optional_text(pred_map[n].events[-1][1].reason, ref_map[n].events[-1][1].reason)
            for n in common
        ]
        metrics["reason"] = display_peritem_metrics(reason_scores, common, weights=common_weights)

        # --- Detail ---
        detail_scores = [
            compare_optional_text(pred_map[n].events[-1][1].detail, ref_map[n].events[-1][1].detail)
            for n in common
        ]
        metrics["detail"] = display_peritem_metrics(detail_scores, common, weights=common_weights)

        # --- Event history ---
        history_results = [
            compare_event_sequences_by_id(
                pred_map[n].events,
                ref_map[n].events,
                similarity_fn=lambda p, r: event_similarity(p, r),
            )
            for n in common
        ]
        metrics["event_history"] = display_peritem_metrics(history_results, common, weights=common_weights)

    else:
        metrics["therapeutic_category"] = {}
        metrics["medication_status"] = {}
        metrics["dosage"] = {}
        metrics["reason"] = {}
        metrics["detail"] = {}
        metrics["event_history"] = {}

    return metrics