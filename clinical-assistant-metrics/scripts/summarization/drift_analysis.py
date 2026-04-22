"""
drift_analysis.py

Fit lineare del patient_score vs encounter_index per scenario A e B.
La slope negativa in B quantifica il drift da auto-storia; A dovrebbe essere
piatto (slope ≈ 0).

Input:
  outputs/summarization/eval/{model_slug}/scenario_{a,b}/report_deltas/cumulative_scores.jsonl

Output:
  outputs/summarization/drift_analysis/drift_table.csv
  outputs/summarization/drift_analysis/drift_table.md
  outputs/summarization/drift_analysis/drift_plot_{suffix}.png

Esecuzione:
  cd clinical-assistant-metrics
  PYTHONPATH=src uv run python scripts/summarization/drift_analysis.py \\
    --models Qwen-Qwen3-0-6B Qwen-Qwen3-1-7B Qwen-Qwen3-4B-Instruct-2507
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

DEFAULT_EVAL_DIR = Path("outputs/summarization/eval")
DEFAULT_OUTPUT_DIR = Path("outputs/summarization/drift_analysis")

MODEL_DISPLAY = {
    "Qwen-Qwen3-0-6B": "0.6B",
    "Qwen-Qwen3-1-7B": "1.7B",
    "Qwen-Qwen3-4B-Instruct-2507": "4B",
    "Qwen-Qwen3-4B-Instruct-2507-FP8": "4B-FP8",
}

MODEL_COLORS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#d97706", "#0891b2"]

SCENARIO_STYLE = {
    "a": {"linestyle": "-",  "marker": "o", "label": "A (gold history)"},
    "b": {"linestyle": "--", "marker": "s", "label": "B (self history)"},
}


def _display(slug: str) -> str:
    return MODEL_DISPLAY.get(slug, slug)


def _load_cumulative(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _fit(records: list[dict]) -> dict:
    pairs = [
        (int(r["encounter_index"]), float(r["patient_score"]))
        for r in records
        if r.get("patient_score") is not None
    ]
    if len(pairs) < 2:
        return {"n": len(pairs), "slope": None, "intercept": None,
                "r_squared": None, "p_value": None, "x": [], "y": []}

    x = np.array([p[0] for p in pairs])
    y = np.array([p[1] for p in pairs])
    res = linregress(x, y)
    return {
        "n": len(pairs),
        "slope": float(res.slope),
        "intercept": float(res.intercept),
        "r_squared": float(res.rvalue ** 2),
        "p_value": float(res.pvalue),
        "x": x.tolist(),
        "y": y.tolist(),
    }


def _collect(models: list[str], eval_dir: Path) -> list[dict]:
    rows = []
    for slug in models:
        for scenario in ("a", "b"):
            path = eval_dir / slug / f"scenario_{scenario}" / "report_deltas" / "cumulative_scores.jsonl"
            if not path.exists():
                print(f"[WARN] Mancante: {path}")
                continue
            records = _load_cumulative(path)
            fit = _fit(records)
            fit["model"] = slug
            fit["scenario"] = scenario
            rows.append(fit)
    return rows


def _save_table(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "drift_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "scenario", "n", "slope", "intercept", "r_squared", "p_value"])
        for r in rows:
            writer.writerow([
                r["model"], r["scenario"].upper(), r["n"],
                f"{r['slope']:.5f}" if r["slope"] is not None else "",
                f"{r['intercept']:.5f}" if r["intercept"] is not None else "",
                f"{r['r_squared']:.4f}" if r["r_squared"] is not None else "",
                f"{r['p_value']:.4g}" if r["p_value"] is not None else "",
            ])
    print(f"CSV → {csv_path}")

    md_path = output_dir / "drift_table.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("| Modello | Scenario | n | Slope | Intercept | R² | p |\n")
        f.write("|---------|----------|---|-------|-----------|-----|---|\n")
        for r in rows:
            f.write(f"| {_display(r['model'])} | {r['scenario'].upper()} | {r['n']} | "
                    f"{r['slope']:+.4f} | {r['intercept']:.3f} | {r['r_squared']:.3f} | {r['p_value']:.3g} |\n")
    print(f"Markdown → {md_path}")


def _plot(rows: list[dict], models: list[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_title("Drift analysis: patient_score vs encounter index\n"
                 "(regressione lineare su tutti i pazienti)",
                 fontsize=12, fontweight="bold")

    for i, slug in enumerate(models):
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        for scenario_key in ("a", "b"):
            row = next((r for r in rows if r["model"] == slug and r["scenario"] == scenario_key), None)
            if row is None or row["slope"] is None:
                continue
            style = SCENARIO_STYLE[scenario_key]

            ax.scatter(row["x"], row["y"], color=color, alpha=0.15,
                       marker=style["marker"], s=25)

            x_line = np.array([min(row["x"]), max(row["x"])])
            y_line = row["intercept"] + row["slope"] * x_line
            label = (f"{_display(slug)} — {style['label']}  "
                     f"(slope={row['slope']:+.4f}, R²={row['r_squared']:.2f})")
            ax.plot(x_line, y_line, color=color, linestyle=style["linestyle"],
                    linewidth=2.2, label=label)

    ax.set_xlabel("Encounter index (0 = primo documento)", fontsize=10)
    ax.set_ylabel("Patient score (delta)", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(fontsize=8.5, loc="lower left", framealpha=0.9, edgecolor="#cccccc")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Grafico → {output_path}")


def _print_summary(rows: list[dict]) -> None:
    print("\n=== Drift summary ===")
    for r in rows:
        if r["slope"] is None:
            continue
        tag = "PIATTO  " if abs(r["slope"]) < 0.005 else ("DRIFT-  " if r["slope"] < 0 else "DRIFT+  ")
        print(f"  {tag}{_display(r['model']):<10}  scenario {r['scenario'].upper()}  "
              f"slope={r['slope']:+.4f}  R²={r['r_squared']:.3f}  "
              f"p={r['p_value']:.3g}  n={r['n']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True,
                        help="Lista di model_slug (es. Qwen-Qwen3-0-6B).")
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    rows = _collect(args.models, args.eval_dir)
    if not rows:
        raise SystemExit("Nessun report trovato. Controlla --eval-dir e --models.")

    _save_table(rows, args.output_dir)
    suffix = "_".join(args.models)
    _plot(rows, args.models, args.output_dir / f"drift_plot_{suffix}.png")
    _print_summary(rows)
