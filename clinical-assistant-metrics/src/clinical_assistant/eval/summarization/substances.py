import numpy as np
from dataclasses import dataclass

from clinical_assistant.summarization.substances import SubstanceUseHistory, SubstanceUseEvent
from clinical_assistant.eval.text_similarity import compare_optional_text, match_names
from clinical_assistant.eval.sequence_alignment import compare_event_sequences_by_id
from clinical_assistant.eval.summarization.base import (
    compute_detection_metrics, compute_categorical_metrics, display_peritem_metrics
)


@dataclass
class SubstanceEventWeights:
    status: float = 0.5
    quantity: float = 0.3
    detail: float = 0.2


def substance_event_similarity(
    pred: SubstanceUseEvent,
    ref: SubstanceUseEvent,
    weights: SubstanceEventWeights = SubstanceEventWeights(),
) -> float:
    status_score = 1.0 if pred.status == ref.status else 0.0
    quantity_score = compare_optional_text(pred.quantity, ref.quantity)
    detail_score = compare_optional_text(pred.detail, ref.detail)
    return (
        weights.status * status_score
        + weights.quantity * quantity_score
        + weights.detail * detail_score
    )


def compute_substance_metrics(
    predicted: list[SubstanceUseHistory],
    reference: list[SubstanceUseHistory],
) -> dict:
    metrics = {}

    pred_names = {s.name.lower() for s in predicted}
    ref_names = {s.name.lower() for s in reference}

    fuzzy_mapping = match_names(pred_names, ref_names)
    detection, name_mapping, score_mapping, common, common_weights = compute_detection_metrics(
        pred_names, ref_names, fuzzy_mapping
    )
    metrics["detection"] = detection

    pred_map = {name_mapping.get(s.name.lower(), s.name.lower()): s for s in predicted}
    ref_map = {s.name.lower(): s for s in reference}

    if common:
        # --- Status ---
        metrics["substance_status"] = compute_categorical_metrics(
            ref_vals=[ref_map[n].events[-1][1].status for n in common],
            pred_vals=[pred_map[n].events[-1][1].status for n in common],
            common_weights=common_weights,
        )

        # --- Quantity ---
        quantity_scores = [
            compare_optional_text(pred_map[n].events[-1][1].quantity, ref_map[n].events[-1][1].quantity)
            for n in common
        ]
        metrics["quantity"] = display_peritem_metrics(quantity_scores, common, weights=common_weights)

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
                similarity_fn=lambda p, r: substance_event_similarity(p, r),
            )
            for n in common
        ]
        metrics["event_history"] = display_peritem_metrics(history_results, common, weights=common_weights)

    else:
        metrics["substance_status"] = {}
        metrics["quantity"] = {}
        metrics["detail"] = {}
        metrics["event_history"] = {}

    return metrics