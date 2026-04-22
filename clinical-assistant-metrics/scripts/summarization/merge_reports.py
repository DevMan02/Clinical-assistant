"""
merge_reports.py

Combina i report di scenario A e B prodotti da report_deltas.py in un unico
JSONL compatto consumato da grafici/plot_aggregate_scores.py.

Input:
  outputs/summarization/eval/{model_slug}/scenario_a/report_deltas/cumulative_scores.jsonl
  outputs/summarization/eval/{model_slug}/scenario_b/report_deltas/cumulative_scores.jsonl

Output:
  outputs/summarization/scores/{model_slug}.jsonl

Ogni riga di output:
  {
    "patient_id":       str,
    "document_index":   int,            # 1-based, da encounter_index + 1
    "score_delta_a":    float | None,
    "score_delta_b":    float | None,
    "sections_delta_a": {section: float},
    "sections_delta_b": {section: float},
    "fallbacks_a":      [section, ...], # solo se trovati i debug JSON
    "fallbacks_b":      [section, ...],
  }

I `fallbacks_*` vengono letti dai file di debug
  outputs/summarization/scenario_comparison/{model_slug}/{patient_id}/doc_{nn}_scenario_{a,b}.json
e sono opzionali (se la cartella manca, vengono semplicemente omessi).

Esecuzione:
  PYTHONPATH=src uv run python scripts/summarization/merge_reports.py --model Qwen-Qwen3-1-7B
"""

import argparse
import json
from pathlib import Path

DEFAULT_EVAL_DIR = Path("outputs/summarization/eval")
DEFAULT_OUTPUT_DIR = Path("outputs/summarization/scores")
DEFAULT_SCENARIO_DIR = Path("outputs/summarization/scenario_comparison")


def _load_cumulative(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _index_by_key(records: list[dict]) -> dict[tuple[str, str], dict]:
    return {(r["patient_id"], r["doc_id"]): r for r in records}


def _section_scores(record: dict) -> dict[str, float]:
    sections = record.get("sections") or {}
    return {
        name: data["section_score"]
        for name, data in sections.items()
        if isinstance(data, dict) and data.get("section_score") is not None
    }


def _load_fallbacks(scenario_dir: Path, model_slug: str) -> dict[tuple[str, str, str], list[str]]:
    """
    Indicizza i fallback dai debug JSON per (patient_id, doc_id, scenario).
    Ritorna {} se la cartella non esiste.
    """
    root = scenario_dir / model_slug
    if not root.exists():
        return {}

    fallbacks: dict[tuple[str, str, str], list[str]] = {}
    for path in root.glob("*/doc_*_scenario_*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        pid = data.get("patient_id")
        doc_id = str(data.get("document_id", ""))
        scenario = (data.get("scenario") or "").lower()
        if pid and doc_id and scenario in ("a", "b"):
            fallbacks[(pid, doc_id, scenario)] = list(data.get("fallbacks") or [])
    return fallbacks


def merge_model(
    model_slug: str,
    eval_dir: Path,
    output_dir: Path,
    scenario_dir: Path,
) -> None:
    path_a = eval_dir / model_slug / "scenario_a" / "report_deltas" / "cumulative_scores.jsonl"
    path_b = eval_dir / model_slug / "scenario_b" / "report_deltas" / "cumulative_scores.jsonl"

    if not path_a.exists():
        raise FileNotFoundError(f"Manca scenario A: {path_a}")
    if not path_b.exists():
        raise FileNotFoundError(f"Manca scenario B: {path_b}")

    records_a = _load_cumulative(path_a)
    records_b = _load_cumulative(path_b)
    index_b = _index_by_key(records_b)
    fallbacks = _load_fallbacks(scenario_dir, model_slug)
    has_fallbacks = bool(fallbacks)

    merged = []
    for rec_a in records_a:
        key = (rec_a["patient_id"], rec_a["doc_id"])
        rec_b = index_b.get(key)
        if rec_b is None:
            print(f"[WARN] Nessun match B per {key} — salto")
            continue

        record = {
            "patient_id": rec_a["patient_id"],
            "document_index": int(rec_a["encounter_index"]) + 1,
            "score_delta_a": rec_a.get("patient_score"),
            "score_delta_b": rec_b.get("patient_score"),
            "sections_delta_a": _section_scores(rec_a),
            "sections_delta_b": _section_scores(rec_b),
        }
        if has_fallbacks:
            record["fallbacks_a"] = fallbacks.get((*key, "a"), [])
            record["fallbacks_b"] = fallbacks.get((*key, "b"), [])
        merged.append(record)

    output_path = output_dir / f"{model_slug}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for rec in merged:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Scritto {output_path}: {len(merged)} record "
          f"({len({r['patient_id'] for r in merged})} pazienti)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model slug (es. Qwen-Qwen3-1-7B).")
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scenario-dir", type=Path, default=DEFAULT_SCENARIO_DIR,
                        help="Directory dei debug JSON (per recuperare i fallback).")
    args = parser.parse_args()

    merge_model(args.model, args.eval_dir, args.output_dir, args.scenario_dir)
