import numpy as np
from dataclasses import dataclass

from clinical_assistant.summarization.family_history import FamilyHistoryEvent
from clinical_assistant.eval.text_similarity import compare_optional_text
from clinical_assistant.eval.embeddings import embed
from clinical_assistant.eval.sequence_alignment import match_by_assignment
from clinical_assistant.eval.summarization.base import (
    compute_categorical_metrics, display_peritem_metrics
)


@dataclass
class FamilyHistoryWeights:
    relationship: float = 0.35
    condition: float = 0.5
    status: float = 0.1
    detail: float = 0.05


def compute_family_history_metrics(
    predicted: list[FamilyHistoryEvent],
    reference: list[FamilyHistoryEvent],
    weights: FamilyHistoryWeights = FamilyHistoryWeights(),
    threshold: float = 0.5,
) -> dict:
    metrics = {}

    if not reference and not predicted:
        return metrics

    if not reference or not predicted:
        metrics["detection"] = {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "missing": [f"{e.relationship}|{e.condition}" for e in reference],
            "hallucinated": [f"{e.relationship}|{e.condition}" for e in predicted],
        }
        metrics["status"] = {}
        metrics["detail"] = {}
        return metrics

    # --- Precompute all condition embeddings at once ---
    all_conditions = [e.condition for e in reference] + [e.condition for e in predicted]
    all_embeddings = embed(all_conditions)
    ref_embeddings = all_embeddings[:len(reference)]
    pred_embeddings = all_embeddings[len(reference):]

    def similarity_fn(pred_idx: int, ref_idx: int) -> float:
        ref_entry = reference[ref_idx]
        pred_entry = predicted[pred_idx]

        relationship_score = 1.0 if pred_entry.relationship == ref_entry.relationship else 0.0
        condition_score = float(np.dot(ref_embeddings[ref_idx], pred_embeddings[pred_idx]))
        status_score = 1.0 if pred_entry.status == ref_entry.status else 0.0
        detail_score = compare_optional_text(pred_entry.detail, ref_entry.detail)

        return (
            weights.relationship * relationship_score
            + weights.condition * condition_score
            + weights.status * status_score
            + weights.detail * detail_score
        )

    matched_pairs, match_scores = match_by_assignment(
        pred_items=list(range(len(predicted))),
        ref_items=list(range(len(reference))),
        similarity_fn=similarity_fn,
        threshold=threshold,
    )

    matched_ref = {i for i, _ in matched_pairs}
    matched_pred = {j for _, j in matched_pairs}

    missing = [
        f"{reference[i].relationship}|{reference[i].condition}"
        for i in range(len(reference)) if i not in matched_ref
    ]
    hallucinated = [
        f"{predicted[j].relationship}|{predicted[j].condition}"
        for j in range(len(predicted)) if j not in matched_pred
    ]

    tp = len(matched_pairs)
    fp = len(hallucinated)
    fn = len(missing)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics["detection"] = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "missing": missing,
        "hallucinated": hallucinated,
        "matched": [
            {
                "ref": f"{reference[i].relationship}|{reference[i].condition}",
                "pred": f"{predicted[j].relationship}|{predicted[j].condition}",
                "score": match_scores[k],
            }
            for k, (i, j) in enumerate(matched_pairs)
        ],
    }

    similarity_score = float(np.mean(match_scores)) if match_scores else 0.0
    metrics["detection"]["similarity_score"] = similarity_score
    metrics["detection"]["score"] = similarity_score * f1

    if matched_pairs:
        # --- Status ---
        metrics["status"] = compute_categorical_metrics(
            ref_vals=[reference[i].status for i, _ in matched_pairs],
            pred_vals=[predicted[j].status for _, j in matched_pairs],
            common_weights=match_scores,
        )

        # --- Detail ---
        detail_scores = [
            compare_optional_text(predicted[j].detail, reference[i].detail)
            for i, j in matched_pairs
        ]
        matched_keys = [
            f"{reference[i].relationship}|{reference[i].condition}"
            for i, _ in matched_pairs
        ]
        metrics["detail"] = display_peritem_metrics(detail_scores, matched_keys, weights=match_scores)

    else:
        metrics["status"] = {}
        metrics["detail"] = {}

    return metrics