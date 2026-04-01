from typing import TypeVar, Callable
import numpy as np
from scipy.optimize import linear_sum_assignment

T = TypeVar("T")


# def needleman_wunsch(
#     pred_seq: list[T],
#     ref_seq: list[T],
#     similarity_fn: Callable[[T, T], float],
#     gap_penalty: float = -0.5,
# ) -> float:
#     """
#     Generic global sequence alignment using Needleman-Wunsch.
#     Returns a normalized score in [0, 1].
#     """
#     n = len(ref_seq)
#     m = len(pred_seq)
#
#     if n == 0 and m == 0:
#         return 1.0
#     if n == 0 or m == 0:
#         return 0.0
#
#     dp = [[0.0] * (m + 1) for _ in range(n + 1)]
#     for i in range(n + 1):
#         dp[i][0] = i * gap_penalty
#     for j in range(m + 1):
#         dp[0][j] = j * gap_penalty
#
#     for i in range(1, n + 1):
#         for j in range(1, m + 1):
#             match = dp[i - 1][j - 1] + similarity_fn(pred_seq[j - 1], ref_seq[i - 1])
#             delete = dp[i - 1][j] + gap_penalty
#             insert = dp[i][j - 1] + gap_penalty
#             dp[i][j] = max(match, delete, insert)
#
#     raw_score = dp[n][m]
#     best_score = min(n, m) * 1.0
#     worst_score = max(n, m) * gap_penalty
#
#     if best_score == worst_score:
#         return 1.0
#
#     return max(0.0, (raw_score - worst_score) / (best_score - worst_score))


def compare_event_sequences_by_id(
    pred_events: list[tuple],
    ref_events: list[tuple],
    similarity_fn: Callable[[T, T], float],
) -> dict:
    """
    Match events by encounter id, compute similarity on matched pairs,
    and coverage F1 for missing/extra encounters.
    Returns a dict with score, coverage_f1, similarity_score, missing, extra.
    """
    ref_map = {meta.id: e for meta, e in ref_events}
    pred_map = {meta.id: e for meta, e in pred_events}

    common_ids = set(ref_map.keys()) & set(pred_map.keys())
    missing_ids = set(ref_map.keys()) - common_ids
    extra_ids = set(pred_map.keys()) - common_ids

    # coverage F1
    tp = len(common_ids)
    fp = len(extra_ids)
    fn = len(missing_ids)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    coverage_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # similarity on matched events
    if common_ids:
        similarity_score = min(1.0, max(0.0, float(np.mean([
            similarity_fn(pred_map[id_], ref_map[id_])
            for id_ in common_ids
        ]))))
    else:
        similarity_score = 0.0

    return {
        "score": min(1.0, similarity_score * coverage_f1),
        "coverage_f1": coverage_f1,
        "similarity_score": similarity_score,
        "missing_encounters": sorted(missing_ids),
        "extra_encounters": sorted(extra_ids),
    }

def match_by_assignment(
    pred_items: list,
    ref_items: list,
    similarity_fn: Callable[[T, T], float],
    threshold: float = 0.5,
) -> tuple[list[tuple[int, int]], list[float]]:
    """
    Find optimal matching between pred and ref items using Hungarian algorithm.
    Returns matched pairs (ref_idx, pred_idx) and their similarity scores,
    only for pairs above threshold.
    """
    if not pred_items or not ref_items:
        return [], []

    n, m = len(ref_items), len(pred_items)
    sim_matrix = np.clip(np.array([
        [similarity_fn(pred_items[j], ref_items[i]) for j in range(m)]
        for i in range(n)
    ]), 0.0, 1.0)

    row_ind, col_ind = linear_sum_assignment(-sim_matrix)

    matched_pairs = [
        (i, j) for i, j in zip(row_ind, col_ind)
        if sim_matrix[i, j] >= threshold
    ]
    match_scores = [float(sim_matrix[i, j]) for i, j in matched_pairs]

    return matched_pairs, match_scores