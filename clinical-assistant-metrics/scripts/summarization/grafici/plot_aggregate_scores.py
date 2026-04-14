"""
plot_aggregate_scores.py

Grafici aggregati multi-paziente per confronto scenari A/B tra modelli diversi.

Legge i file JSONL compatti prodotti da run_batch_scenarios.py:
  outputs/summarization/scores/{model_slug}.jsonl

Produce due grafici:
  1. score_delta aggregato: media ± std per posizione documento, una linea per
     ogni combinazione modello+scenario (A e B).
  2. Section scores delta: una linea per sezione, aggregata su tutti i pazienti,
     per ogni modello (scenario A e B sovrapposti).

Esecuzione (singolo modello):
  cd clinical-assistant-metrics
  PYTHONPATH=src uv run python scripts/summarization/grafici/plot_aggregate_scores.py \\
    --models Qwen-Qwen3-0-6B

Esecuzione (confronto multi-modello):
  PYTHONPATH=src uv run python scripts/summarization/grafici/plot_aggregate_scores.py \\
    --models Qwen-Qwen3-0-6B Qwen-Qwen3-4B-Instruct-2507
"""

import argparse
import json
from collections import defaultdict
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

# Palette colori: un colore per modello, linestyle diverso per scenario A/B
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


def _aggregate_by_doc_index(
    records: list[dict],
    score_key: str,
) -> dict[int, dict]:
    """
    Raggruppa i valori per document_index.
    Ritorna {doc_index: {"values": [...], "mean": float, "std": float, "n": int}}
    I None vengono esclusi.
    """
    by_index = defaultdict(list)
    for r in records:
        v = r.get(score_key)
        if v is not None:
            by_index[r["document_index"]].append(float(v))

    result = {}
    for idx, vals in sorted(by_index.items()):
        result[idx] = {
            "values": vals,
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "n": len(vals),
        }
    return result


def _aggregate_sections_by_doc_index(
    records: list[dict],
    scenario: str,  # "a" o "b"
) -> dict[str, dict[int, dict]]:
    """
    Per ogni sezione, aggrega i section_score per document_index.
    Ritorna {section: {doc_index: {"mean": float, "std": float, "n": int}}}
    """
    key = f"sections_delta_{scenario}"
    by_section = defaultdict(lambda: defaultdict(list))

    for r in records:
        sections = r.get(key, {})
        doc_idx = r["document_index"]
        for section, score in sections.items():
            if score is not None:
                by_section[section][doc_idx].append(float(score))

    result = {}
    for section, by_idx in by_section.items():
        result[section] = {}
        for idx, vals in sorted(by_idx.items()):
            result[section][idx] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "n": len(vals),
            }
    return result


def _plot_overall(
    ax: plt.Axes,
    model_data: list[tuple[str, list[dict]]],
    score_key: str,
    scenario: str,
    color: str,
    linestyle: str,
    label: str,
) -> None:
    agg = _aggregate_by_doc_index(model_data, score_key)
    if not agg:
        return

    x = sorted(agg.keys())
    means = [agg[i]["mean"] for i in x]
    stds = [agg[i]["std"] for i in x]
    ns = [agg[i]["n"] for i in x]

    ax.plot(x, means, color=color, linestyle=linestyle, marker="o",
            linewidth=2.2, markersize=7, label=label)
    ax.fill_between(x,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    color=color, alpha=0.12)

    for xi, m, n in zip(x, means, ns):
        ax.annotate(f"{m:.2f}\nn={n}", (xi, m),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=7.5, color=color)


def plot_overall(
    models_records: dict[str, list[dict]],
    output_path: Path,
) -> None:
    """Grafico 1: score_delta aggregato per posizione documento, A e B per ogni modello."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Score delta aggregato per posizione documento\n(media ± std su tutti i pazienti)",
                 fontsize=12, fontweight="bold")

    for i, (model_slug, records) in enumerate(models_records.items()):
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        short = model_slug.split("-")[-1] if "-" in model_slug else model_slug

        _plot_overall(ax, records, "score_delta_a", "a", color, "-",
                      f"{short} — A (gold history)")
        _plot_overall(ax, records, "score_delta_b", "b", color, "--",
                      f"{short} — B (self history)")

    # x ticks = tutti gli indici documento presenti
    all_indices = sorted({r["document_index"] for recs in models_records.values() for r in recs})
    ax.set_xticks(all_indices)
    ax.set_xticklabels([f"doc {i}" for i in all_indices], fontsize=9)
    ax.set_xlabel("Posizione documento", fontsize=10)
    ax.set_ylabel("Score delta [0–1]", fontsize=10)
    ax.set_ylim(0, 1.2)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Grafico salvato: {output_path}")
    plt.show()


def plot_sections(
    models_records: dict[str, list[dict]],
    output_path: Path,
) -> None:
    """Grafico 2: section scores delta per posizione documento, una sezione per subplot."""
    # Raccogli sezioni presenti
    present_sections = []
    for sec in SECTIONS:
        for records in models_records.values():
            if any(sec in r.get("sections_delta_a", {}) or sec in r.get("sections_delta_b", {})
                   for r in records):
                if sec not in present_sections:
                    present_sections.append(sec)

    if not present_sections:
        print("Nessuna section score disponibile.")
        return

    n_sec = len(present_sections)
    fig, axes = plt.subplots(1, n_sec, figsize=(4.5 * n_sec, 5), sharey=True)
    if n_sec == 1:
        axes = [axes]

    fig.suptitle("Section scores delta per posizione documento\n(media ± std su tutti i pazienti)",
                 fontsize=12, fontweight="bold")

    all_indices = sorted({r["document_index"] for recs in models_records.values() for r in recs})

    for ax, section in zip(axes, present_sections):
        ax.set_title(SECTION_LABELS.get(section, section), fontsize=10)

        for i, (model_slug, records) in enumerate(models_records.items()):
            color = MODEL_COLORS[i % len(MODEL_COLORS)]
            short = model_slug.split("-")[-1] if "-" in model_slug else model_slug

            for scenario, linestyle, suffix in [("a", "-", "A"), ("b", "--", "B")]:
                agg = _aggregate_sections_by_doc_index(records, scenario)
                sec_data = agg.get(section, {})
                if not sec_data:
                    continue

                x = sorted(sec_data.keys())
                means = [sec_data[xi]["mean"] for xi in x]
                stds = [sec_data[xi]["std"] for xi in x]
                ns = [sec_data[xi]["n"] for xi in x]

                ax.plot(x, means, color=color, linestyle=linestyle, marker="o",
                        linewidth=2, markersize=6,
                        label=f"{short} {suffix}")
                ax.fill_between(x,
                                [m - s for m, s in zip(means, stds)],
                                [m + s for m, s in zip(means, stds)],
                                color=color, alpha=0.1)

                for xi, m, n in zip(x, means, ns):
                    ax.annotate(f"{m:.2f}", (xi, m),
                                textcoords="offset points", xytext=(0, 8),
                                ha="center", fontsize=7, color=color)

        ax.set_xticks(all_indices)
        ax.set_xticklabels([f"doc {i}" for i in all_indices], fontsize=8.5)
        ax.set_ylim(0, 1.2)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        if ax == axes[0]:
            ax.set_ylabel("Section score [0–1]", fontsize=9)
            ax.legend(fontsize=7.5, loc="lower right")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Grafico salvato: {output_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Grafici aggregati multi-paziente per confronto A/B tra modelli."
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
        print(f"Caricato {slug}: {len(records)} record ({len({r['patient_id'] for r in records})} pazienti)")
        models_records[slug] = records

    if not models_records:
        print("Nessun file trovato. Controlla --scores-dir e --models.")
        raise SystemExit(1)

    suffix = "_".join(args.models)
    plot_overall(models_records, output_dir / f"aggregate_score_delta_{suffix}.png")
    plot_sections(models_records, output_dir / f"aggregate_section_scores_{suffix}.png")
