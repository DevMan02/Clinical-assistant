from clinical_assistant.summarization.allergies import Allergy
from clinical_assistant.eval.text_similarity import compare_optional_text, match_names
from clinical_assistant.eval.summarization.base import (
    compute_detection_metrics, compute_categorical_metrics, display_peritem_metrics
)


def compute_allergy_metrics(
    predicted: list[Allergy],
    reference: list[Allergy],
) -> dict:
    metrics = {}

    pred_names = {a.allergen.lower() for a in predicted}
    ref_names = {a.allergen.lower() for a in reference}

    fuzzy_mapping = match_names(pred_names, ref_names)
    detection, name_mapping, score_mapping, common, common_weights = compute_detection_metrics(
        pred_names, ref_names, fuzzy_mapping
    )
    metrics["detection"] = detection

    pred_map = {name_mapping.get(a.allergen.lower(), a.allergen.lower()): a for a in predicted}
    ref_map = {a.allergen.lower(): a for a in reference}

    if common:
        # --- Category ---
        metrics["allergy_category"] = compute_categorical_metrics(
            ref_vals=[ref_map[n].category for n in common],
            pred_vals=[pred_map[n].category for n in common],
            common_weights=common_weights,
        )

        # --- Severity ---
        severity_common = [n for n in common if ref_map[n].severity is not None or pred_map[n].severity is not None]
        severity_weights = [common_weights[i] for i, n in enumerate(common) if n in severity_common]

        if severity_common:
            metrics["severity"] = compute_categorical_metrics(
                ref_vals=[ref_map[n].severity or "unknown" for n in severity_common],
                pred_vals=[pred_map[n].severity or "unknown" for n in severity_common],
                common_weights=severity_weights,
            )
        else:
            metrics["severity"] = {}

        # --- Reaction ---
        reaction_scores = [
            compare_optional_text(pred_map[n].reaction, ref_map[n].reaction)
            for n in common
        ]
        metrics["reaction"] = display_peritem_metrics(reaction_scores, common, weights=common_weights)

    else:
        metrics["allergy_category"] = {}
        metrics["severity"] = {}
        metrics["reaction"] = {}

    return metrics