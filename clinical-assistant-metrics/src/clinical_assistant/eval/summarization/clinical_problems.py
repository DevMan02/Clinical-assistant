import numpy as np
from dataclasses import dataclass

from clinical_assistant.summarization.clinical_problems import ClinicalProblemEvent, ClinicalProblem
from clinical_assistant.eval.text_similarity import compare_optional_text, match_names
from clinical_assistant.eval.sequence_alignment import compare_event_sequences_by_id
from clinical_assistant.eval.summarization.base import (
    compute_detection_metrics, compute_categorical_metrics, display_peritem_metrics
)


@dataclass
class ClinicalProblemEventWeights:
    status: float = 0.5
    detail: float = 0.4
    new_info: float = 0.1


def event_similarity(
    pred: ClinicalProblemEvent,
    ref: ClinicalProblemEvent,
    weights: ClinicalProblemEventWeights = ClinicalProblemEventWeights(),
) -> float:
    status_score = 1.0 if pred.status == ref.status else 0.0
    detail_score = compare_optional_text(pred.detail, ref.detail)
    new_info_score = 1.0 if pred.new_info == ref.new_info else 0.0
    return (
        weights.status * status_score
        + weights.detail * detail_score
        + weights.new_info * new_info_score
    )


def compute_clinical_problem_metrics(
    predicted: list[ClinicalProblem],
    reference: list[ClinicalProblem],
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
        # --- Category ---
        metrics["category"] = compute_categorical_metrics(
            ref_vals=[ref_map[n].events[-1][1].category for n in common],
            pred_vals=[pred_map[n].events[-1][1].category for n in common],
            common_weights=common_weights,
        )

        # --- Status ---
        metrics["status"] = compute_categorical_metrics(
            ref_vals=[ref_map[n].events[-1][1].status for n in common],
            pred_vals=[pred_map[n].events[-1][1].status for n in common],
            common_weights=common_weights,
        )

        # --- Detail ---
        detail_scores = [
            compare_optional_text(pred_map[n].events[-1][1].detail, ref_map[n].events[-1][1].detail)
            for n in common
        ]
        metrics["detail"] = display_peritem_metrics(detail_scores, common, weights=common_weights)

        # --- New Info ---
        new_info_scores = [
            1.0 if pred_map[n].events[-1][1].new_info == ref_map[n].events[-1][1].new_info else 0.0
            for n in common
        ]
        metrics["new_info"] = display_peritem_metrics(new_info_scores, common, weights=common_weights)

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
        metrics["category"] = {}
        metrics["status"] = {}
        metrics["detail"] = {}
        metrics["new_info"] = {}
        metrics["event_history"] = {}

    return metrics