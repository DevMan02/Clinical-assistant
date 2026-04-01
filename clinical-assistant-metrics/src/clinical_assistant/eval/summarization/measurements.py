import numpy as np
from dataclasses import dataclass

from clinical_assistant.summarization.measurements import Measurement, MeasurementEvent
from clinical_assistant.eval.text_similarity import match_names
from clinical_assistant.eval.sequence_alignment import compare_event_sequences_by_id
from clinical_assistant.eval.summarization.base import (
    compute_detection_metrics, compute_categorical_metrics, display_peritem_metrics
)


@dataclass
class MeasurementEventWeights:
    value: float = 0.6
    unit: float = 0.2
    flag: float = 0.1
    category: float = 0.1


def compare_value(pred: float, ref: float) -> float:
    if ref == 0:
        return 1.0 if pred == 0 else 0.0
    relative_error = abs(pred - ref) / abs(ref)
    return max(0.0, 1.0 - relative_error)


def compare_unit(pred: str, ref: str) -> float:
    return 1.0 if pred.lower().strip() == ref.lower().strip() else 0.0


def event_similarity(
    pred: MeasurementEvent,
    ref: MeasurementEvent,
    weights: MeasurementEventWeights = MeasurementEventWeights(),
) -> float:
    category_score = 1.0 if pred.category == ref.category else 0.0
    value_score = compare_value(pred.value, ref.value)
    unit_score = compare_unit(pred.unit, ref.unit)
    flag_score = 1.0 if pred.flag == ref.flag else 0.0
    return (
        weights.category * category_score
        + weights.value * value_score
        + weights.unit * unit_score
        + weights.flag * flag_score
    )


def compute_measurement_metrics(
    predicted: list[Measurement],
    reference: list[Measurement],
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
            ref_vals=[ref_map[n].data_points[-1][1].category for n in common],
            pred_vals=[pred_map[n].data_points[-1][1].category for n in common],
            common_weights=common_weights,
        )

        # --- Flag ---
        flag_common = [n for n in common if ref_map[n].data_points[-1][1].flag is not None
                       or pred_map[n].data_points[-1][1].flag is not None]
        flag_weights = [common_weights[i] for i, n in enumerate(common) if n in flag_common]

        if flag_common:
            metrics["flag"] = compute_categorical_metrics(
                ref_vals=[ref_map[n].data_points[-1][1].flag or "none" for n in flag_common],
                pred_vals=[pred_map[n].data_points[-1][1].flag or "none" for n in flag_common],
                common_weights=flag_weights,
            )
        else:
            metrics["flag"] = {}

        # --- Value ---
        value_scores = [
            compare_value(pred_map[n].data_points[-1][1].value, ref_map[n].data_points[-1][1].value)
            for n in common
        ]
        metrics["value"] = display_peritem_metrics(value_scores, common, weights=common_weights)

        # --- Unit ---
        unit_scores = [
            compare_unit(pred_map[n].data_points[-1][1].unit, ref_map[n].data_points[-1][1].unit)
            for n in common
        ]
        metrics["unit"] = display_peritem_metrics(unit_scores, common, weights=common_weights)

        # --- Event history ---
        history_results = [
            compare_event_sequences_by_id(
                pred_map[n].data_points,
                ref_map[n].data_points,
                similarity_fn=lambda p, r: event_similarity(p, r),
            )
            for n in common
        ]
        metrics["event_history"] = display_peritem_metrics(history_results, common, weights=common_weights)

    else:
        metrics["category"] = {}
        metrics["flag"] = {}
        metrics["value"] = {}
        metrics["unit"] = {}
        metrics["event_history"] = {}

    return metrics