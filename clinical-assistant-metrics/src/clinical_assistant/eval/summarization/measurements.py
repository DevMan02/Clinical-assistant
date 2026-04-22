import numpy as np
from dataclasses import dataclass

from clinical_assistant.summarization.measurements import Measurement, MeasurementEvent
from clinical_assistant.summarization.encounter import EncounterMeta
from clinical_assistant.eval.text_similarity import match_names
from clinical_assistant.eval.sequence_alignment import compare_event_sequences_by_id, match_by_assignment
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


def compare_intra_encounter(
    pred_points: list[MeasurementEvent],
    ref_points: list[MeasurementEvent],
) -> float:
    """
    Compare two sets of data_points from the same encounter using Hungarian matching.
    Returns a score in [0, 1].
    """
    if not ref_points and not pred_points:
        return 1.0
    if not ref_points or not pred_points:
        return 0.0

    if len(ref_points) == 1 and len(pred_points) == 1:
        return event_similarity(pred_points[0], ref_points[0])

    matched_pairs, match_scores = match_by_assignment(
        pred_items=pred_points,
        ref_items=ref_points,
        similarity_fn=event_similarity,
        threshold=0.0,
    )

    tp = len(matched_pairs)
    fp = len(pred_points) - tp
    fn = len(ref_points) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    coverage_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    similarity_score = float(np.mean(match_scores)) if match_scores else 0.0

    return min(1.0, similarity_score * coverage_f1)


def compare_measurement_history(
    pred_data_points: list[tuple[EncounterMeta, MeasurementEvent]],
    ref_data_points: list[tuple[EncounterMeta, MeasurementEvent]],
) -> dict:
    """
    Combines intra-encounter and inter-encounter history comparison.

    Inter-encounter: group data_points by encounter ID, take last per encounter,
    then use compare_event_sequences_by_id.

    Intra-encounter: for encounters present in both, compare all data_points
    within that encounter using Hungarian matching.

    Final score: mean of inter and intra scores.
    """
    from collections import defaultdict

    # Group by encounter ID
    def group_by_enc(data_points):
        groups = defaultdict(list)
        metas = {}
        for meta, e in data_points:
            groups[meta.id].append(e)
            metas[meta.id] = meta
        return groups, metas

    pred_groups, pred_metas = group_by_enc(pred_data_points)
    ref_groups, ref_metas = group_by_enc(ref_data_points)

    # --- Inter-encounter: use last data_point per encounter ---
    pred_inter = [(pred_metas[enc_id], events[-1]) for enc_id, events in pred_groups.items()]
    ref_inter = [(ref_metas[enc_id], events[-1]) for enc_id, events in ref_groups.items()]

    inter_result = compare_event_sequences_by_id(
        pred_inter,
        ref_inter,
        similarity_fn=event_similarity,
    )
    inter_score = inter_result["score"]

    # --- Intra-encounter: compare all data_points within common encounters ---
    common_enc_ids = set(pred_groups.keys()) & set(ref_groups.keys())

    if common_enc_ids:
        intra_scores = [
            compare_intra_encounter(pred_groups[enc_id], ref_groups[enc_id])
            for enc_id in common_enc_ids
        ]
        intra_score = float(np.mean(intra_scores))
    else:
        intra_score = 0.0

    return {
        "inter_encounter": inter_score,
        "intra_encounter": intra_score,
    }


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
        # --- Category (last data_point) ---
        metrics["category"] = compute_categorical_metrics(
            ref_vals=[ref_map[n].data_points[-1][1].category for n in common],
            pred_vals=[pred_map[n].data_points[-1][1].category for n in common],
            common_weights=common_weights,
        )

        # --- Flag (last data_point) ---
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

        # --- Value (last data_point) ---
        value_scores = [
            compare_value(pred_map[n].data_points[-1][1].value, ref_map[n].data_points[-1][1].value)
            for n in common
        ]
        metrics["value"] = display_peritem_metrics(value_scores, common, weights=common_weights)

        # --- Unit (last data_point) ---
        unit_scores = [
            compare_unit(pred_map[n].data_points[-1][1].unit, ref_map[n].data_points[-1][1].unit)
            for n in common
        ]
        metrics["unit"] = display_peritem_metrics(unit_scores, common, weights=common_weights)

        # --- Event history (intra + inter encounter) ---
        history_results = [
            compare_measurement_history(
                pred_map[n].data_points,
                ref_map[n].data_points,
            )
            for n in common
        ]
        inter_scores = [r["inter_encounter"] for r in history_results]
        intra_scores = [r["intra_encounter"] for r in history_results]
        metrics["event_history_inter"] = display_peritem_metrics(inter_scores, common, weights=common_weights)
        metrics["event_history_intra"] = display_peritem_metrics(intra_scores, common, weights=common_weights)

    else:
        metrics["category"] = {}
        metrics["flag"] = {}
        metrics["value"] = {}
        metrics["unit"] = {}
        metrics["event_history_inter"] = {}
        metrics["event_history_intra"] = {}

    return metrics