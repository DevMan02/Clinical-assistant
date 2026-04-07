"""
plot_section_scores.py

Confronto dei section_score tra Scenario A (storia gold) e Scenario B (storia Qwen).

Risponde a: quale sezione clinica beneficia di più di avere la storia Claude come contesto?

Due visualizzazioni:
  1. Bar chart per documento: section_score A vs B per ogni sezione
  2. Gain summary: media (A - B) per sezione su tutti i documenti

Modalità score:
  --score final  : usa accumulated_prediction vs accumulated_target
  --score delta  : usa predicted_delta vs target_delta (estrazione pura, senza rumore storia)

Esecuzione:
  cd clinical-assistant-metrics
  PYTHONPATH=src uv run python scripts/summarization/plot_section_scores.py
  PYTHONPATH=src uv run python scripts/summarization/plot_section_scores.py --score delta
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from clinical_assistant.eval.summarization.aggregate import compute_aggregate_scores
from clinical_assistant.eval.summarization.allergies import compute_allergy_metrics
from clinical_assistant.eval.summarization.clinical_problems import compute_clinical_problem_metrics
from clinical_assistant.eval.summarization.family_history import compute_family_history_metrics
from clinical_assistant.eval.summarization.measurements import compute_measurement_metrics
from clinical_assistant.eval.summarization.medications import compute_medication_metrics
from clinical_assistant.eval.summarization.procedures import compute_procedure_metrics
from clinical_assistant.eval.summarization.substances import compute_substance_metrics
from clinical_assistant.summarization.patient import PatientSummary, PatientSummaryDelta, PatientSummaryExtractor
from clinical_assistant.summarization.structured_output import AsyncClient


class _NoopClient(AsyncClient):
    async def structured_output(self, prompt, output_format):
        return None


_noop = PatientSummaryExtractor(_NoopClient())

SECTIONS = ["medications", "allergies", "substances", "clinical_problems",
            "measurements", "family_history", "procedures"]

SECTION_LABELS = {
    "medications": "Medications",
    "allergies": "Allergies",
    "substances": "Substances",
    "clinical_problems": "Clin. Problems",
    "measurements": "Measurements",
    "family_history": "Family Hist.",
    "procedures": "Procedures",
}


def _metrics_from_summaries(pred: PatientSummary, ref: PatientSummary) -> dict:
    metrics = {}
    if ref.medications or pred.medications:
        metrics["medications"] = compute_medication_metrics(pred.medications, ref.medications)
    if ref.allergies or pred.allergies:
        metrics["allergies"] = compute_allergy_metrics(pred.allergies, ref.allergies)
    if ref.substances or pred.substances:
        metrics["substances"] = compute_substance_metrics(pred.substances, ref.substances)
    if ref.clinical_problems or pred.clinical_problems:
        metrics["clinical_problems"] = compute_clinical_problem_metrics(pred.clinical_problems, ref.clinical_problems)
    if ref.measurements or pred.measurements:
        metrics["measurements"] = compute_measurement_metrics(pred.measurements, ref.measurements)
    if ref.family_history or pred.family_history:
        metrics["family_history"] = compute_family_history_metrics(pred.family_history, ref.family_history)
    if ref.procedures or pred.procedures:
        metrics["procedures"] = compute_procedure_metrics(pred.procedures, ref.procedures)
    return metrics


def _section_scores_from_json(row: dict, mode: str) -> dict[str, float]:
    """Ritorna {section: score} per un dato JSON di scenario."""
    if mode == "final":
        pred = PatientSummary.model_validate(row["accumulated_prediction"])
        ref = PatientSummary.model_validate(row["accumulated_target"])
    else:
        patient_id = row["patient_id"]
        ref_sum = PatientSummary.model_validate(row["accumulated_target"])
        empty = PatientSummary(
            patient_id=patient_id, document_ids=[],
            sex=ref_sum.sex, age=ref_sum.age,
        )
        pred_delta = PatientSummaryDelta.model_validate(row["predicted_delta"])
        ref_delta = PatientSummaryDelta.model_validate(row["target_delta"])
        pred = _noop.update(empty, pred_delta)
        ref = _noop.update(empty, ref_delta)

    metrics = _metrics_from_summaries(pred, ref)
    agg = compute_aggregate_scores(metrics)
    return {s: agg["sections"][s]["section_score"] for s in SECTIONS if s in agg.get("sections", {})}


def _load_rows(scenario_dir: Path) -> tuple[list[dict], list[dict]]:
    rows_a = sorted(scenario_dir.glob("doc_*_scenario_a.json"), key=lambda p: p.name)
    rows_b = sorted(scenario_dir.glob("doc_*_scenario_b.json"), key=lambda p: p.name)
    return (
        [json.loads(p.read_text(encoding="utf-8")) for p in rows_a],
        [json.loads(p.read_text(encoding="utf-8")) for p in rows_b],
    )


def plot(patient_id: str, output_dir: Path, mode: str = "final", save_path: Path | None = None) -> None:
    scenario_dir = output_dir / patient_id
    rows_a, rows_b = _load_rows(scenario_dir)
    if not rows_a:
        raise FileNotFoundError(f"Nessun file JSON in {scenario_dir}")

    n_docs = len(rows_a)
    scores_a = [_section_scores_from_json(r, mode) for r in rows_a]
    scores_b = [_section_scores_from_json(r, mode) for r in rows_b]

    # sezioni presenti in almeno un documento
    present_sections = [s for s in SECTIONS if any(s in sc for sc in scores_a + scores_b)]
    n_sec = len(present_sections)
    x_labels = [r["document_date"]["start"] for r in rows_a]

    COLOR_A = "#2563eb"
    COLOR_B = "#f97316"
    bar_w = 0.35

    fig, axes = plt.subplots(1, n_docs + 1, figsize=(4 * (n_docs + 1), 5), sharey=True)
    fig.suptitle(
        f"Paziente {patient_id} — Section scores ({mode})\nScenario A (gold history) vs B (self history)",
        fontsize=12, fontweight="bold",
    )

    x = np.arange(n_sec)
    sec_labels = [SECTION_LABELS[s] for s in present_sections]

    for doc_i, (sa, sb, ax) in enumerate(zip(scores_a, scores_b, axes[:-1])):
        vals_a = [sa.get(s, 0.0) for s in present_sections]
        vals_b = [sb.get(s, 0.0) for s in present_sections]

        bars_a = ax.bar(x - bar_w / 2, vals_a, bar_w, color=COLOR_A, alpha=0.85, label="A")
        bars_b = ax.bar(x + bar_w / 2, vals_b, bar_w, color=COLOR_B, alpha=0.85, label="B")

        for bar in bars_a:
            h = bar.get_height()
            if h > 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=7, color=COLOR_A)
        for bar in bars_b:
            h = bar.get_height()
            if h > 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=7, color=COLOR_B)

        ax.set_title(f"doc_{doc_i + 1:02d}\n{x_labels[doc_i]}", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(sec_labels, rotation=45, ha="right", fontsize=7.5)
        ax.set_ylim(0, 1.15)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        if doc_i == 0:
            ax.set_ylabel("Section score [0–1]", fontsize=9)
        if doc_i == 0:
            ax.legend(fontsize=8)

    # Gain summary (ultimo subplot)
    ax_gain = axes[-1]
    gains = []
    for s in present_sections:
        g = np.mean([sa.get(s, 0.0) - sb.get(s, 0.0) for sa, sb in zip(scores_a, scores_b)])
        gains.append(g)

    colors = ["#16a34a" if g >= 0 else "#dc2626" for g in gains]
    ax_gain.barh(np.arange(n_sec), gains, color=colors, alpha=0.85)
    for i, g in enumerate(gains):
        ax_gain.text(g + (0.005 if g >= 0 else -0.005), i,
                     f"{g:+.3f}", va="center", ha="left" if g >= 0 else "right", fontsize=8)
    ax_gain.axvline(0, color="black", linewidth=0.8)
    ax_gain.set_yticks(np.arange(n_sec))
    ax_gain.set_yticklabels(sec_labels, fontsize=8)
    ax_gain.set_xlabel("Gain medio A − B", fontsize=9)
    ax_gain.set_title("Gain\n(A − B, media docs)", fontsize=9)
    ax_gain.grid(axis="x", linestyle=":", alpha=0.4)

    plt.tight_layout()

    if save_path is None:
        save_path = scenario_dir / f"section_scores_{mode}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Grafico salvato: {save_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Section scores A vs B per documento.")
    parser.add_argument("--patient-id", default="10000032")
    parser.add_argument("--output-dir", default="outputs/summarization/scenario_comparison")
    parser.add_argument("--score", default="final", choices=["final", "delta"],
                        help="'final'=accumulated score, 'delta'=estrazione pura")
    parser.add_argument("--save-path", default=None)
    args = parser.parse_args()

    plot(
        patient_id=args.patient_id,
        output_dir=Path(args.output_dir),
        mode=args.score,
        save_path=Path(args.save_path) if args.save_path else None,
    )
