"""
plot_fallback_rates.py

Fallback rate per sezione e per modello.

Legge i file JSONL prodotti da run_batch_scenarios.py:
  outputs/summarization/scores/{model_slug}.jsonl

Per ogni modello calcola, su tutti i documenti di tutti i pazienti:
  fallback_rate(sezione, scenario) = documenti con fallback su quella sezione
                                     ─────────────────────────────────────────
                                     documenti totali × 100

Produce un grafico a barre raggruppate per sezione, una barra per ogni
combinazione modello × scenario (A/B).

Esecuzione (singolo modello):
  cd clinical-assistant-metrics
  PYTHONPATH=src uv run python scripts/summarization/grafici/plot_fallback_rates.py \\
    --models Qwen-Qwen3-0-6B

Esecuzione (confronto multi-modello):
  PYTHONPATH=src uv run python scripts/summarization/grafici/plot_fallback_rates.py \\
    --models Qwen-Qwen3-0-6B Qwen-Qwen3-1-7B Qwen-Qwen3-4B-Instruct-2507-FP8
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

SECTIONS = [
    "medications", "allergies", "substances",
    "clinical_problems", "measurements", "family_history", "procedures",
]

SECTION_LABELS = {
    "medications": "Medications",
    "allergies": "Allergies",
    "substances": "Substances",
    "clinical_problems": "Clin. Problems",
    "measurements": "Measurements",
    "family_history": "Family Hist.",
    "procedures": "Procedures",
}

MODEL_COLORS = [
    "#2563eb",  # blu
    "#dc2626",  # rosso
    "#16a34a",  # verde
    "#9333ea",  # viola
    "#d97706",  # ambra
    "#0891b2",  # ciano
]


def _load_scores(scores_path: Path) -> list[dict]:
    records = []
    with scores_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _compute_fallback_rates(records: list[dict]) -> dict[str, dict[str, float]]:
    """
    Ritorna {scenario: {section: fallback_rate_percent}}
    per scenario in ["a", "b"].
    Solo record che hanno il campo fallbacks_* vengono conteggiati
    (retrocompatibile con JSONL prodotti prima dell'aggiunta del tracking).
    """
    rates: dict[str, dict[str, float]] = {}

    for scenario in ("a", "b"):
        key = f"fallbacks_{scenario}"
        eligible = [r for r in records if key in r]
        total = len(eligible)
        if total == 0:
            rates[scenario] = {}
            continue

        counts = {s: 0 for s in SECTIONS}
        for r in eligible:
            for s in r[key]:
                if s in counts:
                    counts[s] += 1

        rates[scenario] = {s: counts[s] / total * 100 for s in SECTIONS}

    return rates


def plot_fallback_rates(
    models_records: dict[str, list[dict]],
    output_path: Path,
) -> None:
    n_models = len(models_records)
    n_scenarios = 2  # A e B
    n_bars_per_section = n_models * n_scenarios

    x = np.arange(len(SECTIONS))
    bar_width = 0.8 / n_bars_per_section

    fig, ax = plt.subplots(figsize=(max(10, 1.5 * len(SECTIONS)), 6))
    ax.set_title(
        "Fallback rate per sezione e per modello\n"
        "(% di documenti in cui la sezione ha restituito lista vuota)",
        fontsize=12, fontweight="bold",
    )

    for i, (model_slug, records) in enumerate(models_records.items()):
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        short = model_slug.split("-")[-1] if "-" in model_slug else model_slug
        rates = _compute_fallback_rates(records)

        for j, (scenario, linestyle, hatch) in enumerate([("a", "-", ""), ("b", "--", "///")]):
            scenario_rates = rates.get(scenario, {})
            if not scenario_rates:
                continue

            values = [scenario_rates.get(s, 0.0) for s in SECTIONS]
            offset = (i * n_scenarios + j - n_bars_per_section / 2 + 0.5) * bar_width
            bars = ax.bar(
                x + offset, values,
                width=bar_width * 0.9,
                color=color,
                alpha=0.85 if scenario == "a" else 0.5,
                hatch=hatch,
                label=f"{short} — {'A (gold hist.)' if scenario == 'a' else 'B (self hist.)'}",
                edgecolor="white",
            )

            # Etichetta valore sopra ogni barra (solo se > 0)
            for bar, val in zip(bars, values):
                if val > 1:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 1.5,
                        f"{val:.0f}%",
                        ha="center", va="bottom",
                        fontsize=7, color=color,
                    )

    total_docs = {slug: len(recs) for slug, recs in models_records.items()}
    subtitle = "  |  ".join(f"{slug.split('-')[-1]}: {n} doc" for slug, n in total_docs.items())
    ax.set_xlabel(subtitle, fontsize=9, color="gray")
    ax.set_ylabel("Fallback rate (%)", fontsize=10)
    ax.set_ylim(0, 115)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.set_xticks(x)
    ax.set_xticklabels([SECTION_LABELS[s] for s in SECTIONS], fontsize=9)
    ax.legend(fontsize=8, loc="upper right", ncol=max(1, n_bars_per_section // 3))
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.axhline(y=100, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Grafico salvato: {output_path}")
    plt.show()


def print_fallback_table(models_records: dict[str, list[dict]]) -> None:
    """Stampa una tabella testuale del fallback rate per tesi/report."""
    col_w = 16
    header = f"{'Sezione':<18}" + "".join(
        f"{slug.split('-')[-1]+' A':>{col_w}}{slug.split('-')[-1]+' B':>{col_w}}"
        for slug in models_records
    )
    print("\n" + header)
    print("-" * len(header))

    for section in SECTIONS:
        row = f"{SECTION_LABELS[section]:<18}"
        for slug, records in models_records.items():
            rates = _compute_fallback_rates(records)
            rate_a = rates.get("a", {}).get(section, float("nan"))
            rate_b = rates.get("b", {}).get(section, float("nan"))
            row += f"{rate_a:>{col_w}.1f}%{rate_b:>{col_w}.1f}%"
        print(row)

    print()
    for slug, records in models_records.items():
        eligible = [r for r in records if "fallbacks_a" in r]
        print(f"  {slug}: {len(eligible)} documenti con tracking fallback "
              f"({len(records)} totali nel JSONL)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fallback rate per sezione e modello."
    )
    parser.add_argument(
        "--models", nargs="+", required=True,
        help="Lista di model_slug (es. Qwen-Qwen3-0-6B). "
             "Corrisponde al nome file in outputs/summarization/scores/.",
    )
    parser.add_argument("--scores-dir", default="outputs/summarization/scores")
    parser.add_argument("--output-dir", default="outputs/summarization/grafici_aggregati")
    args = parser.parse_args()

    scores_dir = Path(args.scores_dir)
    output_dir = Path(args.output_dir)

    models_records = {}
    for slug in args.models:
        path = scores_dir / f"{slug}.jsonl"
        if not path.exists():
            print(f"[WARN] File non trovato: {path} — modello skippato")
            continue
        records = _load_scores(path)
        n_with_tracking = sum(1 for r in records if "fallbacks_a" in r)
        print(f"Caricato {slug}: {len(records)} record "
              f"({len({r['patient_id'] for r in records})} pazienti, "
              f"{n_with_tracking} con fallback tracking)")
        models_records[slug] = records

    if not models_records:
        print("Nessun file trovato. Controlla --scores-dir e --models.")
        raise SystemExit(1)

    print_fallback_table(models_records)

    suffix = "_".join(args.models)
    plot_fallback_rates(models_records, output_dir / f"fallback_rates_{suffix}.png")