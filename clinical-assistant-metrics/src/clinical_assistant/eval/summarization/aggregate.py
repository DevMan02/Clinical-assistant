import numpy as np


def _safe_get(d: dict, *keys, default=None) -> float | None:
    current = d
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    if current == {} or current is None:
        return default
    return float(current)


def _section_score(scores: dict) -> float | None:
    values = [v for k, v in scores.items() if v is not None and k != "section_score"]
    return float(np.mean(values)) if values else None


def compute_aggregate_scores(metrics: dict) -> dict:
    scores = {"patient_id": metrics.get("patient_info", {}).get("id", "unknown"),
              "sex_score": _safe_get(metrics, "patient_info", "sex", "score"),
              "age_score": _safe_get(metrics, "patient_info", "age", "score")}

    # --- Patient info ---

    sections = {}

    # --- Allergies ---
    if "allergies" in metrics and metrics["allergies"]:
        m = metrics["allergies"]
        s = {
            "detection_f1": _safe_get(m, "detection", "f1"),
            "category_f1": _safe_get(m, "allergy_category", "weighted", "f1"),
            "reaction": _safe_get(m, "reaction", "weighted_mean_score"),
        }
        s["section_score"] = _section_score(s)
        if s["section_score"] is not None:
            sections["allergies"] = s

    # --- Substances ---
    if "substances" in metrics and metrics["substances"]:
        m = metrics["substances"]
        s = {
            "detection_f1": _safe_get(m, "detection", "f1"),
            "status_f1": _safe_get(m, "substance_status", "weighted", "f1"),
            "quantity": _safe_get(m, "quantity", "weighted_mean_score"),
            "detail": _safe_get(m, "detail", "weighted_mean_score"),
            "event_history": _safe_get(m, "event_history", "mean_score"),
        }
        s["section_score"] = _section_score(s)
        if s["section_score"] is not None:
            sections["substances"] = s

    # --- Family history ---
    if "family_history" in metrics and metrics["family_history"]:
        m = metrics["family_history"]
        s = {
            "detection_f1": _safe_get(m, "detection", "f1"),
            "detection_score": _safe_get(m, "detection", "score"),
            "status_f1": _safe_get(m, "status", "weighted", "f1"),
            "detail": _safe_get(m, "detail", "weighted_mean_score"),
        }
        s["section_score"] = _section_score(s)
        if s["section_score"] is not None:
            sections["family_history"] = s

    # --- Clinical problems ---
    if "clinical_problems" in metrics and metrics["clinical_problems"]:
        m = metrics["clinical_problems"]
        s = {
            "detection_f1": _safe_get(m, "detection", "f1"),
            "category_f1": _safe_get(m, "category", "weighted", "f1"),
            "status_f1": _safe_get(m, "status", "weighted", "f1"),
            "detail": _safe_get(m, "detail", "weighted_mean_score"),
            "event_history": _safe_get(m, "event_history", "mean_score"),
        }
        s["section_score"] = _section_score(s)
        if s["section_score"] is not None:
            sections["clinical_problems"] = s

    # --- Procedures ---
    if "procedures" in metrics and metrics["procedures"]:
        m = metrics["procedures"]
        s = {
            "encounter_f1": _safe_get(m, "encounter_coverage", "f1"),
            "detection_f1": _safe_get(m, "detection", "f1"),
            "detection_score": _safe_get(m, "detection", "score"),
            "category_f1": _safe_get(m, "category", "weighted", "f1"),
            "significance_f1": _safe_get(m, "significance", "weighted", "f1"),
            "detail": _safe_get(m, "detail", "weighted_mean_score"),
        }
        s["section_score"] = _section_score(s)
        if s["section_score"] is not None:
            sections["procedures"] = s

    # --- Medications ---
    if "medications" in metrics and metrics["medications"]:
        m = metrics["medications"]
        s = {
            "detection_f1": _safe_get(m, "detection", "f1"),
            "category_f1": _safe_get(m, "therapeutic_category", "weighted", "f1"),
            "status_f1": _safe_get(m, "medication_status", "weighted", "f1"),
            "dosage": _safe_get(m, "dosage", "weighted_mean_score"),
            "reason": _safe_get(m, "reason", "weighted_mean_score"),
            "detail": _safe_get(m, "detail", "weighted_mean_score"),
            "event_history": _safe_get(m, "event_history", "mean_score"),
        }
        s["section_score"] = _section_score(s)
        if s["section_score"] is not None:
            sections["medications"] = s

    # --- Measurements ---
    if "measurements" in metrics and metrics["measurements"]:
        m = metrics["measurements"]
        s = {
            "detection_f1": _safe_get(m, "detection", "f1"),
            "category_f1": _safe_get(m, "category", "weighted", "f1"),
            "value": _safe_get(m, "value", "weighted_mean_score"),
            "unit": _safe_get(m, "unit", "weighted_mean_score"),
            "event_history_intra": _safe_get(m, "event_history_intra", "mean_score"),
            "event_history_inter": _safe_get(m, "event_history_inter", "mean_score"),
        }
        s["section_score"] = _section_score(s)
        if s["section_score"] is not None:
            sections["measurements"] = s

    scores["sections"] = sections
    scores["patient_score"] = float(np.mean([s["section_score"] for s in sections.values()])) if sections else 0.0

    return scores