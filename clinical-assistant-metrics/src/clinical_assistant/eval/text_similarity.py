import re
from rapidfuzz import process
from rapidfuzz.fuzz import partial_token_set_ratio
from clinical_assistant.eval.embeddings import pairwise_similarity


def normalize_dosage(dosage: str) -> str:
    return re.sub(r'\s+', ' ', dosage.lower().strip())


def extract_dose_numbers(dosage: str) -> set[str]:
    return set(re.findall(r'\d+\.?\d*', dosage))


def compare_dosage(pred: str | None, ref: str | None) -> float:
    if pred is None and ref is None:
        return 1.0
    if pred is None or ref is None:
        return 0.0

    pred_norm = normalize_dosage(pred)
    ref_norm = normalize_dosage(ref)

    if pred_norm == ref_norm:
        return 1.0

    pred_numbers = extract_dose_numbers(pred_norm)
    ref_numbers = extract_dose_numbers(ref_norm)
    if pred_numbers and pred_numbers == ref_numbers:
        return 0.8

    return float(pairwise_similarity([pred_norm], [ref_norm])[0])


def match_names(
        pred_names: set[str],
        ref_names: set[str],
        threshold: float = 60.0,
) -> dict[str, tuple[str, float]]:
    """Returns a mapping from pred_name -> (ref_name, normalized_score).

    First tries exact match, then falls back to fuzzy matching.
    Ensures each ref_name is matched at most once.
    """
    mapping = {}
    remaining_pred = set()
    remaining_ref = set(ref_names)

    # --- Exact match first ---
    for pred_name in pred_names:
        if pred_name in remaining_ref:
            mapping[pred_name] = (pred_name, 1.0)
            remaining_ref.discard(pred_name)
        else:
            remaining_pred.add(pred_name)

    # --- Fuzzy match on unmatched names ---
    for pred_name in remaining_pred:
        if not remaining_ref:
            break
        # noinspection PyTypeChecker
        result = process.extractOne(pred_name, list(remaining_ref), scorer=partial_token_set_ratio)
        if result and result[1] >= threshold:
            mapping[pred_name] = (result[0], result[1] / 100)
            remaining_ref.discard(result[0])

    return mapping


def compare_optional_text(pred: str | None, ref: str | None) -> float:
    if pred is None and ref is None:
        return 1.0
    if pred is None or ref is None:
        return 0.0
    return float(pairwise_similarity([pred], [ref])[0])