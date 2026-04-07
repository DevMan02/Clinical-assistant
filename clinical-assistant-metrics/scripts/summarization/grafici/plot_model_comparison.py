"""
plot_model_comparison.py

Confronto pulito tra i tre scenari con due letture separate:

  C vs B = effetto modello puro
           FP8 self-history  vs  2507 self-history
           stessa struttura di contesto, modelli diversi

  A vs C = effetto contesto puro
           FP8 gold-history  vs  FP8 self-history
           stesso modello, contesto diverso

Visualizzazioni:
  1. score_delta per documento: A, B, C su un unico grafico
  2. Gain C−B (effetto modello) e A−C (effetto contesto) per documento
  3. Item overlap: gold items trovati da A, B, C (stacked bar per documento)

Esecuzione:
  cd clinical-assistant-metrics
  PYTHONPATH=src uv run python scripts/summarization/grafici/plot_model_comparison.py
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

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

SECTION_ITEM_FIELDS: dict[str, tuple[str, str]] = {
    "medications":       ("events",         "medication_name"),
    "measurements":      ("measurements",   "name"),
    "clinical_problems": ("problems",       "name"),
    "procedures":        ("procedures",     "name"),
    "allergies":         ("allergies",      "allergen"),
    "substances":        ("substances",     "name"),
    "family_history":    ("family_history", "condition"),
}


def _get_item_set(delta_dict: dict, section: str) -> set[str]:
    list_key, name_field = SECTION_ITEM_FIELDS[section]
    items = delta_dict.get(section, {}).get(list_key, [])
    return {str(item.get(name_field, "")).lower().strip()
            for item in items if item.get(name_field)}


def _section_scores_delta(row: dict) -> dict[str, float]:
    ref_sum = PatientSummary.model_validate(row["accumulated_target"])
    empty = PatientSummary(
        patient_id=row["patient_id"], document_ids=[],
        sex=ref_sum.sex, age=ref_sum.age,
    )
    pred = _noop.update(empty, PatientSummaryDelta.model_validate(row["predicted_delta"]))
    ref  = _noop.update(empty, PatientSummaryDelta.model_validate(row["target_delta"]))

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

    agg = compute_aggregate_scores(metrics)
    return {s: agg["sections"][s]["section_score"] for s in SECTIONS if s in agg.get("sections", {})}


def _item_overlap(ra: dict, rb: dict, rc: dict) -> dict:
    counts = {"only_a": 0, "only_b": 0, "only_c": 0,
              "ab": 0, "ac": 0, "bc": 0, "abc": 0, "none": 0, "total": 0}
    for section in SECTION_ITEM_FIELDS:
        gold   = _get_item_set(ra["target_delta"], section)
        pred_a = _get_item_set(ra["predicted_delta"], section)
        pred_b = _get_item_set(rb["predicted_delta"], section)
        pred_c = _get_item_set(rc["predicted_delta"], section)
        for item in gold:
            in_a = any(item in p or p in item for p in pred_a)
            in_b = any(item in p or p in item for p in pred_b)
            in_c = any(item in p or p in item for p in pred_c)
            if in_a and in_b and in_c:   counts["abc"] += 1
            elif in_a and in_b:          counts["ab"] += 1
            elif in_a and in_c:          counts["ac"] += 1
            elif in_b and in_c:          counts["bc"] += 1
            elif in_a:                   counts["only_a"] += 1
            elif in_b:                   counts["only_b"] += 1
            elif in_c:                   counts["only_c"] += 1
            else:                        counts["none"] += 1
            counts["total"] += 1
    return counts


def plot(patient_id: str, output_dir: Path, save_path: Path | None = None) -> None:
    scenario_dir = output_dir / patient_id

    def _load(pat):
        rows = [json.loads(p.read_text(encoding="utf-8"))
                for p in sorted(scenario_dir.glob(pat))]
        return sorted(rows, key=lambda r: r["document_index"])

    rows_a = _load("doc_*_scenario_a.json")
    rows_b = _load("doc_*_scenario_b.json")
    rows_c = _load("doc_*_scenario_c.json")

    if not rows_c:
        raise FileNotFoundError(
            f"Nessun file scenario_c in {scenario_dir}. "
            "Riesegui generate_scenario_jsons.py per generare il Scenario C."
        )

    n_docs = len(rows_a)
    x = np.arange(n_docs)
    x_labels = [f"doc_{i+1:02d}\n{r['document_date']['start']}" for i, r in enumerate(rows_a)]

    COLOR_A = "#2563eb"
    COLOR_B = "#f97316"
    COLOR_C = "#16a34a"

    delta_a = [r["score_delta"] for r in rows_a]
    delta_b = [r["score_delta"] for r in rows_b]
    delta_c = [r["score_delta"] for r in rows_c]

    overlap = [_item_overlap(ra, rb, rc) for ra, rb, rc in zip(rows_a, rows_b, rows_c)]

    fig, axes = plt.subplots(3, 1, figsize=(11, 14))
    fig.suptitle(
        f"Paziente {patient_id} — Model comparison\n"
        "C vs B = effetto modello puro  |  A vs C = effetto contesto puro",
        fontsize=13, fontweight="bold",
    )

    # ── subplot 1: score_delta A / B / C ──────────────────────────────────
    ax1 = axes[0]
    ax1.plot(x, delta_a, color=COLOR_A, marker="o", linewidth=2.2, markersize=7,
             label="A — FP8 + gold history")
    ax1.plot(x, delta_b, color=COLOR_B, marker="o", linewidth=2.2, markersize=7,
             label="B — 2507 + self history", linestyle="--")
    ax1.plot(x, delta_c, color=COLOR_C, marker="o", linewidth=2.2, markersize=7,
             label="C — FP8 + self history", linestyle=":")

    for xi, va, vb, vc in zip(x, delta_a, delta_b, delta_c):
        ax1.annotate(f"{va:.2f}", (xi, va), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8.5, color=COLOR_A)
        ax1.annotate(f"{vb:.2f}", (xi, vb), textcoords="offset points",
                     xytext=(0, -14), ha="center", fontsize=8.5, color=COLOR_B)
        ax1.annotate(f"{vc:.2f}", (xi, vc), textcoords="offset points",
                     xytext=(0, -26), ha="center", fontsize=8.5, color=COLOR_C)

    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, fontsize=9)
    ax1.set_ylabel("score_delta [0–1]", fontsize=10)
    ax1.set_ylim(0, 1.12)
    ax1.set_title("score_delta per documento", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", linestyle=":", alpha=0.5)

    # ── subplot 2: gain C−B e A−C ─────────────────────────────────────────
    ax2 = axes[1]
    bar_w = 0.35
    gain_model   = [vc - vb for vc, vb in zip(delta_c, delta_b)]
    gain_context = [va - vc for va, vc in zip(delta_a, delta_c)]

    ax2.bar(x - bar_w / 2, gain_model,   bar_w,
            color=[COLOR_C if g >= 0 else "#dc2626" for g in gain_model],
            alpha=0.85, label="C−B  effetto modello (FP8 vs 2507)")
    ax2.bar(x + bar_w / 2, gain_context, bar_w,
            color=[COLOR_A if g >= 0 else "#dc2626" for g in gain_context],
            alpha=0.85, label="A−C  effetto contesto (gold vs self)")

    for xi, gm, gc in zip(x, gain_model, gain_context):
        ax2.text(xi - bar_w / 2, gm + (0.005 if gm >= 0 else -0.02),
                 f"{gm:+.2f}", ha="center", va="bottom" if gm >= 0 else "top", fontsize=8.5)
        ax2.text(xi + bar_w / 2, gc + (0.005 if gc >= 0 else -0.02),
                 f"{gc:+.2f}", ha="center", va="bottom" if gc >= 0 else "top", fontsize=8.5)

    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels, fontsize=9)
    ax2.set_ylabel("Gain [0–1]", fontsize=10)
    ax2.set_title("Gain score_delta\nverde = miglioramento, rosso = peggioramento", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", linestyle=":", alpha=0.4)

    # ── subplot 3: item overlap stacked bar ───────────────────────────────
    ax3 = axes[2]
    norm = [max(d["total"], 1) for d in overlap]

    def _pct(key):
        return [d[key] / n for d, n in zip(overlap, norm)]

    bot = np.zeros(n_docs)
    for values, color, label, alpha in [
        (_pct("abc"),    "#7c3aed", "A∩B∩C (tutti)",              0.90),
        (_pct("ac"),     COLOR_A,   "A∩C (FP8, entrambi contesti)", 0.70),
        (_pct("ab"),     "#60a5fa", "A∩B (gold+2507)",             0.70),
        (_pct("bc"),     "#fdba74", "B∩C (self history)",          0.70),
        (_pct("only_a"), COLOR_A,   "Solo A",                      0.45),
        (_pct("only_b"), COLOR_B,   "Solo B",                      0.45),
        (_pct("only_c"), COLOR_C,   "Solo C",                      0.45),
        (_pct("none"),   "#9ca3af", "Nessuno (miss)",              0.50),
    ]:
        ax3.bar(x, values, bottom=bot, color=color, alpha=alpha, label=label)
        bot = bot + np.array(values)

    for xi, d in enumerate(overlap):
        ax3.text(xi, 1.02, f"n={d['total']}", ha="center", fontsize=8.5)

    ax3.set_xticks(x)
    ax3.set_xticklabels(x_labels, fontsize=9)
    ax3.set_ylabel("Frazione item gold", fontsize=10)
    ax3.set_ylim(0, 1.12)
    ax3.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax3.set_title("Overlap item estratti rispetto al gold (A, B, C)", fontsize=10)
    ax3.legend(fontsize=8, loc="lower right", ncol=2)
    ax3.grid(axis="y", linestyle=":", alpha=0.4)

    plt.tight_layout()

    if save_path is None:
        save_path = scenario_dir / "model_comparison.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Grafico salvato: {save_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Confronto modello FP8 vs 2507 con Scenario C.")
    parser.add_argument("--patient-id", default="10000032")
    parser.add_argument("--output-dir", default="outputs/summarization/scenario_comparison")
    parser.add_argument("--save-path", default=None)
    args = parser.parse_args()

    plot(
        patient_id=args.patient_id,
        output_dir=Path(args.output_dir),
        save_path=Path(args.save_path) if args.save_path else None,
    )
