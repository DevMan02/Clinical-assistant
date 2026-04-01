import numpy as np
from typing import TypeVar, Protocol
from sklearn.metrics import precision_recall_fscore_support


class HasName(Protocol):
    name: str


class HasEvents(Protocol):
    events: list


T = TypeVar("T")


def compute_detection_metrics(
    pred_names: set[str],
    ref_names: set[str],
    fuzzy_mapping: dict[str, tuple[str, float]],
) -> tuple[dict, dict, dict, list, list]:
    """Returns detection metrics, name_mapping, score_mapping, common, common_weights."""
    name_mapping = {pred: ref_name for pred, (ref_name, _) in fuzzy_mapping.items()}
    score_mapping = {pred: score for pred, (_, score) in fuzzy_mapping.items()}

    normalized_pred_names = set(name_mapping.values())
    matched_ref_names = normalized_pred_names & ref_names
    unmatched_pred_names = {name_mapping.get(n, n) for n in pred_names} - ref_names

    weighted_tp = sum(score_mapping[pred] for pred in name_mapping if name_mapping[pred] in ref_names)
    fp = len(unmatched_pred_names)
    fn = len(ref_names - matched_ref_names)
    precision = weighted_tp / (weighted_tp + fp) if (weighted_tp + fp) > 0 else 0.0
    recall = weighted_tp / (weighted_tp + fn) if (weighted_tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    detection = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "missing": sorted(ref_names - matched_ref_names),
        "hallucinated": sorted(unmatched_pred_names),
        "fuzzy_matches": {
            pred: {"matched_to": ref_name, "score": score}
            for pred, (ref_name, score) in fuzzy_mapping.items()
            if ref_name in ref_names and score < 1.0
        },
    }

    common = sorted(matched_ref_names)
    common_weights = [score_mapping.get(
        next((p for p, r in name_mapping.items() if r == n), n), 1.0
    ) for n in common]

    return detection, name_mapping, score_mapping, common, common_weights


def compute_categorical_metrics(
    ref_vals: list,
    pred_vals: list,
    common_weights: list,
) -> dict:
    """Compute per-class + macro + weighted P/R/F1."""
    labels = sorted(set(ref_vals))
    p, r, f1, _ = precision_recall_fscore_support(
        ref_vals, pred_vals, average=None, labels=labels,
        sample_weight=common_weights, zero_division=0
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        ref_vals, pred_vals, average="macro", labels=labels,
        sample_weight=common_weights, zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        ref_vals, pred_vals, average="weighted", labels=labels,
        sample_weight=common_weights, zero_division=0
    )
    result = {
        cls: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i])}
        for i, cls in enumerate(labels)
    }
    result["macro"] = {"precision": float(p_macro), "recall": float(r_macro), "f1": float(f1_macro)}
    result["weighted"] = {"precision": float(p_weighted), "recall": float(r_weighted), "f1": float(f1_weighted)}
    return result


def display_peritem_metrics(
    scores: list,
    keys: list[str],
    per_key: str = "per_item",
    weights: list[float] | None = None,
) -> dict:
    """Compute mean score + per-item breakdown. Scores can be float or dict."""
    if not scores:
        return {"mean_score": 0.0, per_key: {}}

    if isinstance(scores[0], dict):
        float_scores = [min(1.0, max(0.0, s["score"])) for s in scores]
        clipped_scores = [
            {**s, "score": min(1.0, max(0.0, s["score"])),
                  "similarity_score": min(1.0, max(0.0, s["similarity_score"]))}
            for s in scores
        ]
        result = {
            "mean_score": float(np.mean(float_scores)),
            "mean_coverage_f1": float(np.mean([s["coverage_f1"] for s in scores])),
            "mean_similarity_score": float(np.mean([min(1.0, max(0.0, s["similarity_score"])) for s in scores])),
            per_key: {k: clipped_scores[i] for i, k in enumerate(keys)},
        }
        if weights:
            result["weighted_mean_score"] = float(np.average(float_scores, weights=weights))
    else:
        clipped = [min(1.0, max(0.0, float(s))) for s in scores]
        result = {
            "mean_score": float(np.mean(clipped)),
            per_key: {k: clipped[i] for i, k in enumerate(keys)},
        }
        if weights:
            result["weighted_mean_score"] = float(np.average(clipped, weights=weights))

    return result