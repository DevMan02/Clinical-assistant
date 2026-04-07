"""
plot_scenario_scores.py

Genera i grafici di confronto Scenario A / B / C per un paziente.

Due subplot:
  1. score_final  — similarità tra accumulated_prediction e accumulated_target
                    (include il traino della storia accumulata)
  2. score_delta  — similarità tra predicted_delta e target_delta
                    (estrazione pura del singolo documento, senza rumore della storia)

Scenario A = FP8 + gold history (Claude)
Scenario B = Qwen 2507 + self history (dal JSONL, zero inferenza)
Scenario C = FP8 + self history (inferenza live iterativa)

Confronti leggibili dal grafico:
  A vs B = effetto combinato modello + contesto
  A vs C = effetto contesto puro  (stesso modello FP8, storia diversa)
  C vs B = effetto modello puro   (stessa self-history, modelli diversi)

Esecuzione:
  cd clinical-assistant-metrics
  uv run python scripts/summarization/grafici/plot_scenario_scores.py
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def _load_scenarios(scenario_dir: Path) -> tuple[list[dict], list[dict], list[dict]]:
    def _load(pattern):
        rows = [json.loads(p.read_text(encoding="utf-8"))
                for p in sorted(scenario_dir.glob(pattern))]
        return sorted(rows, key=lambda r: r["document_index"])

    return _load("doc_*_scenario_a.json"), _load("doc_*_scenario_b.json"), _load("doc_*_scenario_c.json")


def plot(patient_id: str, output_dir: Path, save_path: Path | None = None) -> None:
    scenario_dir = output_dir / patient_id
    rows_a, rows_b, rows_c = _load_scenarios(scenario_dir)

    if not rows_a or not rows_b:
        raise FileNotFoundError(f"Nessun file JSON trovato in {scenario_dir}")

    has_c = len(rows_c) == len(rows_a)
    has_delta = "score_delta" in rows_a[0]

    x_labels = [r["document_date"]["start"] for r in rows_a]
    x = list(range(1, len(rows_a) + 1))

    COLOR_A = "#2563eb"   # blu       — FP8 + gold history
    COLOR_B = "#f97316"   # arancio   — 2507 + self history
    COLOR_C = "#16a34a"   # verde     — FP8 + self history
    MARKER = "o"
    LW = 2.2
    MS = 7

    n_plots = 2 if has_delta else 1
    fig, axes = plt.subplots(n_plots, 1, figsize=(9, 4.5 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    fig.suptitle(
        f"Paziente {patient_id} — Confronto Scenario A / B / C",
        fontsize=13, fontweight="bold", y=1.01,
    )

    def _plot_lines(ax, score_key, title):
        sa = [r[score_key] for r in rows_a]
        sb = [r[score_key] for r in rows_b]

        ax.plot(x, sa, color=COLOR_A, marker=MARKER, linewidth=LW, markersize=MS,
                label="A — FP8 + gold history")
        ax.plot(x, sb, color=COLOR_B, marker=MARKER, linewidth=LW, markersize=MS,
                label="B — 2507 + self history", linestyle="--")

        offsets = {"a": 8, "b": -14, "c": -28}

        for xi, ya, yb in zip(x, sa, sb):
            ax.annotate(f"{ya:.2f}", (xi, ya), textcoords="offset points",
                        xytext=(0, offsets["a"]), ha="center", fontsize=8.5, color=COLOR_A)
            ax.annotate(f"{yb:.2f}", (xi, yb), textcoords="offset points",
                        xytext=(0, offsets["b"]), ha="center", fontsize=8.5, color=COLOR_B)

        if has_c:
            sc = [r[score_key] for r in rows_c]
            ax.plot(x, sc, color=COLOR_C, marker=MARKER, linewidth=LW, markersize=MS,
                    label="C — FP8 + self history", linestyle=":")
            for xi, yc in zip(x, sc):
                ax.annotate(f"{yc:.2f}", (xi, yc), textcoords="offset points",
                            xytext=(0, offsets["c"]), ha="center", fontsize=8.5, color=COLOR_C)

        ax.set_ylabel("Score [0–1]", fontsize=10)
        ax.set_ylim(0, 1.12)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.5)

    _plot_lines(axes[0], "score_final",
                "score_final  (accumulated_prediction vs accumulated_target)")

    if has_delta:
        _plot_lines(axes[1], "score_delta",
                    "score_delta  (predicted_delta vs target_delta — estrazione pura)")

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(
        [f"doc_{i}\n{lbl}" for i, lbl in zip(x, x_labels)], fontsize=8.5,
    )
    axes[-1].set_xlabel("Documento (data inizio ricovero)", fontsize=10)

    plt.tight_layout()

    if save_path is None:
        save_path = scenario_dir / "scores_plot.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Grafico salvato in: {save_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plotta score_final e score_delta per Scenario A/B/C.")
    parser.add_argument("--patient-id", default="10000032")
    parser.add_argument("--output-dir", default="outputs/summarization/scenario_comparison")
    parser.add_argument("--save-path", default=None)
    args = parser.parse_args()

    plot(
        patient_id=args.patient_id,
        output_dir=Path(args.output_dir),
        save_path=Path(args.save_path) if args.save_path else None,
    )
