import numpy as np
from dataclasses import dataclass

from clinical_assistant.summarization.procedures import Procedure, EncounterProcedures
from clinical_assistant.eval.text_similarity import compare_optional_text
from clinical_assistant.eval.embeddings import embed
from clinical_assistant.eval.sequence_alignment import match_by_assignment
from clinical_assistant.eval.summarization.base import (
    compute_categorical_metrics, display_peritem_metrics
)


@dataclass
class ProcedureWeights:
    name: float = 0.5
    category: float = 0.2
    significance: float = 0.2
    detail: float = 0.1


def procedure_similarity(
    pred: Procedure,
    ref: Procedure,
    pred_name_emb: np.ndarray,
    ref_name_emb: np.ndarray,
    weights: ProcedureWeights = ProcedureWeights(),
) -> float:
    name_score = float(np.dot(ref_name_emb, pred_name_emb))
    category_score = 1.0 if pred.category == ref.category else 0.0
    significance_score = 1.0 if pred.significance == ref.significance else 0.0
    detail_score = compare_optional_text(pred.detail, ref.detail)
    return (
        weights.name * name_score
        + weights.category * category_score
        + weights.significance * significance_score
        + weights.detail * detail_score
    )


def match_procedures(
    predicted: list[Procedure],
    reference: list[Procedure],
    weights: ProcedureWeights = ProcedureWeights(),
    threshold: float = 0.5,
) -> tuple[list[tuple[int, int]], list[float]]:
    """Match procedures within a single encounter using Hungarian algorithm."""
    if not predicted or not reference:
        return [], []

    all_names = [p.name for p in reference] + [p.name for p in predicted]
    all_embeddings = embed(all_names)
    ref_embeddings = all_embeddings[:len(reference)]
    pred_embeddings = all_embeddings[len(reference):]

    def similarity_fn(pred_idx: int, ref_idx: int) -> float:
        return procedure_similarity(
            pred=predicted[pred_idx],
            ref=reference[ref_idx],
            pred_name_emb=pred_embeddings[pred_idx],
            ref_name_emb=ref_embeddings[ref_idx],
            weights=weights,
        )

    return match_by_assignment(
        pred_items=list(range(len(predicted))),
        ref_items=list(range(len(reference))),
        similarity_fn=similarity_fn,
        threshold=threshold,
    )


def compute_procedure_metrics(
    predicted: list[EncounterProcedures],
    reference: list[EncounterProcedures],
    weights: ProcedureWeights = ProcedureWeights(),
    threshold: float = 0.5,
) -> dict:
    metrics = {}

    if not reference and not predicted:
        return metrics

    # --- Match encounters by id ---
    ref_enc_map = {ep.encounter.id: ep for ep in reference}
    pred_enc_map = {ep.encounter.id: ep for ep in predicted}

    common_enc_ids = set(ref_enc_map.keys()) & set(pred_enc_map.keys())
    missing_enc_ids = set(ref_enc_map.keys()) - common_enc_ids
    extra_enc_ids = set(pred_enc_map.keys()) - common_enc_ids

    # --- Encounter coverage ---
    tp_enc = len(common_enc_ids)
    fp_enc = len(extra_enc_ids)
    fn_enc = len(missing_enc_ids)
    enc_precision = tp_enc / (tp_enc + fp_enc) if (tp_enc + fp_enc) > 0 else 0.0
    enc_recall = tp_enc / (tp_enc + fn_enc) if (tp_enc + fn_enc) > 0 else 0.0
    enc_f1 = 2 * enc_precision * enc_recall / (enc_precision + enc_recall) if (enc_precision + enc_recall) > 0 else 0.0

    metrics["encounter_coverage"] = {
        "precision": enc_precision,
        "recall": enc_recall,
        "f1": enc_f1,
        "missing_encounters": sorted(missing_enc_ids),
        "extra_encounters": sorted(extra_enc_ids),
    }

    if not common_enc_ids:
        metrics["detection"] = {}
        metrics["category"] = {}
        metrics["significance"] = {}
        metrics["detail"] = {}
        return metrics

    # --- Per-encounter procedure matching ---
    all_matched_pairs_data = []
    all_missing = []
    all_hallucinated = []
    all_match_scores = []

    for enc_id in sorted(common_enc_ids):
        ref_procs = ref_enc_map[enc_id].procedures
        pred_procs = pred_enc_map[enc_id].procedures

        matched_pairs, match_scores = match_procedures(pred_procs, ref_procs, weights, threshold)

        matched_ref = {i for i, _ in matched_pairs}
        matched_pred = {j for _, j in matched_pairs}

        for k, (i, j) in enumerate(matched_pairs):
            all_matched_pairs_data.append({
                "enc_id": enc_id,
                "ref_proc": ref_procs[i],
                "pred_proc": pred_procs[j],
                "score": match_scores[k],
            })
            all_match_scores.append(match_scores[k])

        all_missing.extend([
            f"{enc_id}|{ref_procs[i].name}"
            for i in range(len(ref_procs)) if i not in matched_ref
        ])
        all_hallucinated.extend([
            f"{enc_id}|{pred_procs[j].name}"
            for j in range(len(pred_procs)) if j not in matched_pred
        ])

    # --- Procedure detection ---
    tp = len(all_matched_pairs_data)
    fp = len(all_hallucinated)
    fn = len(all_missing)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics["detection"] = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "missing": all_missing,
        "hallucinated": all_hallucinated,
        "matched": [
            {
                "encounter": d["enc_id"],
                "ref": d["ref_proc"].name,
                "pred": d["pred_proc"].name,
                "score": d["score"],
            }
            for d in all_matched_pairs_data
        ],
    }

    similarity_score = float(np.mean(all_match_scores)) if all_match_scores else 0.0
    metrics["detection"]["similarity_score"] = similarity_score
    metrics["detection"]["score"] = similarity_score * f1

    if all_matched_pairs_data:
        # --- Category ---
        metrics["category"] = compute_categorical_metrics(
            ref_vals=[d["ref_proc"].category for d in all_matched_pairs_data],
            pred_vals=[d["pred_proc"].category for d in all_matched_pairs_data],
            common_weights=all_match_scores,
        )

        # --- Significance ---
        metrics["significance"] = compute_categorical_metrics(
            ref_vals=[d["ref_proc"].significance for d in all_matched_pairs_data],
            pred_vals=[d["pred_proc"].significance for d in all_matched_pairs_data],
            common_weights=all_match_scores,
        )

        # --- Detail ---
        detail_scores = [
            compare_optional_text(d["pred_proc"].detail, d["ref_proc"].detail)
            for d in all_matched_pairs_data
        ]
        matched_keys = [
            f"{d['enc_id']}|{d['ref_proc'].name}"
            for d in all_matched_pairs_data
        ]
        metrics["detail"] = display_peritem_metrics(detail_scores, matched_keys, weights=all_match_scores)

    else:
        metrics["category"] = {}
        metrics["significance"] = {}
        metrics["detail"] = {}

    return metrics