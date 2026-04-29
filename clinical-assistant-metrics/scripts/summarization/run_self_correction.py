"""
run_self_correction.py

Secondo passaggio "verify-and-rewrite" sui preds prodotti da run_batch_scenarios.py.

Per ogni sezione del pred_summary_delta:
  1. Costruisce un prompt che mostra il prompt originale (prior + document)
     + l'estrazione v1 + istruzioni di self-review.
  2. Chiama l'LLM con la stessa OutputFormat → v2.
  3. Se v2 non parsa o è None, mantiene v1 come fallback.

Scopo della tesi:
  Misurare se un secondo passaggio critico migliora la qualità di estrazione
  di modelli piccoli (Qwen 0.6B / 1.7B / 4B) su documenti clinici in italiano.
  Confronto baseline vs self-correct, su scenario A (gold history) e B (self
  history). In B il miglioramento può ridurre il drift cumulativo.

Input:
  outputs/summarization/eval/{model_slug}/scenario_{a,b}/preds.jsonl

Output:
  outputs/summarization/eval/{model_slug}/scenario_{a,b}_selfcorrect/preds.jsonl

Esecuzione (proof-of-concept):
  cd clinical-assistant-metrics
  PYTHONPATH=src uv run python scripts/summarization/run_self_correction.py \\
    --model Qwen/Qwen3-1.7B \\
    --scenarios a \\
    --max-patients 3 --max-encounters 2 \\
    --sections medications clinical_problems

Esecuzione (full run):
  PYTHONPATH=src uv run python scripts/summarization/run_self_correction.py \\
    --model Qwen/Qwen3-1.7B \\
    --scenarios a b

Valutazione (riusa pipeline esistente):
  PYTHONPATH=src uv run python scripts/summarization/report_deltas.py \\
    --model Qwen-Qwen3-1-7B/scenario_a_selfcorrect
"""

import argparse
import asyncio
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from clinical_assistant.summarization.allergies import (
    ALLERGIES_EXTRACTION_PROMPT_TEMPLATE,
    AllergyOutputFormat,
)
from clinical_assistant.summarization.clinical_problems import (
    CLINICAL_PROBLEM_HISTORY_PROMPT_TEMPLATE,
    ClinicalProblemOutputFormat,
)
from clinical_assistant.summarization.encounter import EncounterDocument
from clinical_assistant.summarization.family_history import (
    FAMILY_HISTORY_EXTRACTION_PROMPT_TEMPLATE,
    FamilyHistoryOutputFormat,
)
from clinical_assistant.summarization.measurements import (
    MEASUREMENT_EXTRACTION_PROMPT_TEMPLATE,
    MeasurementEventOutputFormat,
)
from clinical_assistant.summarization.medications import (
    MEDICATION_EVENT_PROMPT_TEMPLATE,
    MedicationHistoryOutputFormat,
)
from clinical_assistant.summarization.patient import PatientSummary
from clinical_assistant.summarization.procedures import (
    PROCEDURE_EXTRACTION_PROMPT_TEMPLATE,
    ProcedureOutputFormat,
)
from clinical_assistant.summarization.structured_output import OpenAIClient
from clinical_assistant.summarization.substances import (
    SUBSTANCE_USE_EXTRACTION_PROMPT_TEMPLATE,
    SubstanceUseOutputFormat,
)

logging.basicConfig(level=logging.WARNING, format="%(name)s — %(levelname)s — %(message)s")
console = Console()


SELF_REVIEW_BLOCK = """

---

## Self-review

A previous attempt at the extraction above produced this output:

<previous_attempt>
{v1_json}
</previous_attempt>

Critically review the previous attempt against the encounter document:
1. Verify each entity is grounded in the document (not hallucinated).
2. Check for omissions: any relevant entity in the document missing from the previous attempt?
3. Verify required fields are correctly populated and not invented.

If the previous attempt is correct and complete, return it unchanged.
Otherwise, return the corrected version in the same JSON schema.
"""


# -------------------------------------------------------------------------
# Section registry
# -------------------------------------------------------------------------
# Ogni voce descrive come costruire il prompt verify per la sezione:
#   - output_format: pydantic class della risposta
#   - prompt_template: template originale (con {document} e opzionalmente {prior_kwarg})
#   - prior_attr: attributo di PatientSummary che contiene la prior list (None per procedures)
#   - prior_kwarg: nome del placeholder nel template (es. "medications")
#   - document_as_content: True se il template si aspetta document.content (testo)
#                         invece dell'EncounterDocument (allergies è l'unica)
SECTION_REGISTRY: dict[str, dict[str, Any]] = {
    "allergies": {
        "output_format": AllergyOutputFormat,
        "prompt_template": ALLERGIES_EXTRACTION_PROMPT_TEMPLATE,
        "prior_attr": "allergies",
        "prior_kwarg": "allergies",
        "document_as_content": True,
    },
    "substances": {
        "output_format": SubstanceUseOutputFormat,
        "prompt_template": SUBSTANCE_USE_EXTRACTION_PROMPT_TEMPLATE,
        "prior_attr": "substances",
        "prior_kwarg": "substances",
        "document_as_content": False,
    },
    "family_history": {
        "output_format": FamilyHistoryOutputFormat,
        "prompt_template": FAMILY_HISTORY_EXTRACTION_PROMPT_TEMPLATE,
        "prior_attr": "family_history",
        "prior_kwarg": "family_history",
        "document_as_content": False,
    },
    "clinical_problems": {
        "output_format": ClinicalProblemOutputFormat,
        "prompt_template": CLINICAL_PROBLEM_HISTORY_PROMPT_TEMPLATE,
        "prior_attr": "clinical_problems",
        "prior_kwarg": "clinical_problems",
        "document_as_content": False,
    },
    "procedures": {
        "output_format": ProcedureOutputFormat,
        "prompt_template": PROCEDURE_EXTRACTION_PROMPT_TEMPLATE,
        "prior_attr": None,
        "prior_kwarg": None,
        "document_as_content": False,
    },
    "medications": {
        "output_format": MedicationHistoryOutputFormat,
        "prompt_template": MEDICATION_EVENT_PROMPT_TEMPLATE,
        "prior_attr": "medications",
        "prior_kwarg": "medications",
        "document_as_content": False,
    },
    "measurements": {
        "output_format": MeasurementEventOutputFormat,
        "prompt_template": MEASUREMENT_EXTRACTION_PROMPT_TEMPLATE,
        "prior_attr": "measurements",
        "prior_kwarg": "measurements",
        "document_as_content": False,
    },
}

ALL_SECTIONS = list(SECTION_REGISTRY.keys())


# -------------------------------------------------------------------------
# Prompt builder
# -------------------------------------------------------------------------

def _build_verify_prompt(
    section: str,
    document: EncounterDocument,
    prior_summary: PatientSummary,
    v1_section_dict: dict,
) -> str:
    spec = SECTION_REGISTRY[section]
    template = spec["prompt_template"]

    fmt_kwargs: dict[str, Any] = {}
    if spec["document_as_content"]:
        fmt_kwargs["document"] = document.content
    else:
        fmt_kwargs["document"] = document

    if spec["prior_kwarg"] is not None:
        prior_list = getattr(prior_summary, spec["prior_attr"])
        fmt_kwargs[spec["prior_kwarg"]] = prior_list

    base_prompt = template.format(**fmt_kwargs)
    v1_json = json.dumps(v1_section_dict, ensure_ascii=False, indent=2, default=str)
    return base_prompt + SELF_REVIEW_BLOCK.format(v1_json=v1_json)


# -------------------------------------------------------------------------
# I/O
# -------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")


def _filter_records(
    records: list[dict],
    max_patients: int | None,
    max_encounters: int | None,
) -> list[dict]:
    if max_patients is None and max_encounters is None:
        return records

    by_patient: dict[str, list[dict]] = defaultdict(list)
    order: list[str] = []
    for r in records:
        pid = r["summary"]["patient_id"]
        if pid not in by_patient:
            order.append(pid)
        by_patient[pid].append(r)

    if max_patients is not None:
        order = order[:max_patients]

    filtered = []
    for pid in order:
        recs = by_patient[pid]
        if max_encounters is not None:
            recs = recs[:max_encounters]
        filtered.extend(recs)
    return filtered


# -------------------------------------------------------------------------
# Per-section verify
# -------------------------------------------------------------------------

async def _verify_section(
    client: OpenAIClient,
    section: str,
    document: EncounterDocument,
    prior_summary: PatientSummary,
    v1_section_dict: dict,
    dry_run: bool,
) -> dict:
    """Ritorna il dict v2 per la sezione (o v1 se il verify fallisce)."""
    prompt = _build_verify_prompt(section, document, prior_summary, v1_section_dict)
    output_format: type[BaseModel] = SECTION_REGISTRY[section]["output_format"]

    if dry_run:
        return v1_section_dict

    try:
        result = await client.structured_output(prompt, output_format)
        if result is None:
            return v1_section_dict
        return result.model_dump(mode="json")
    except Exception as exc:
        console.log(f"      ⚠ {section} verify fallito ({exc.__class__.__name__}) → tengo v1")
        return v1_section_dict


# -------------------------------------------------------------------------
# Per-record processing
# -------------------------------------------------------------------------

async def _process_record(
    record: dict,
    client: OpenAIClient,
    sections: list[str],
    dry_run: bool,
) -> tuple[dict, dict[str, bool]]:
    document = EncounterDocument.model_validate(record["document"])
    prior_summary = PatientSummary.model_validate(record["summary"])
    v1_delta = record["pred_summary_delta"]

    v2_delta = dict(v1_delta)  # copia (meta + sezioni non toccate restano invariate)
    changed: dict[str, bool] = {}

    coros = [
        _verify_section(client, sec, document, prior_summary, v1_delta[sec], dry_run)
        for sec in sections
    ]
    results = await asyncio.gather(*coros)

    for sec, v2_section in zip(sections, results):
        v2_delta[sec] = v2_section
        changed[sec] = json.dumps(v2_section, sort_keys=True, default=str) != \
                       json.dumps(v1_delta[sec], sort_keys=True, default=str)

    out_record = {
        "summary": record["summary"],
        "document": record["document"],
        "target_summary_delta": record["target_summary_delta"],
        "pred_summary_delta": v2_delta,
    }
    return out_record, changed


# -------------------------------------------------------------------------
# Scenario runner
# -------------------------------------------------------------------------

async def _process_scenario(
    scenario: str,
    model_slug: str,
    eval_dir: Path,
    client: OpenAIClient,
    sections: list[str],
    max_patients: int | None,
    max_encounters: int | None,
    dry_run: bool,
) -> None:
    in_path = eval_dir / model_slug / f"scenario_{scenario}" / "preds.jsonl"
    out_path = eval_dir / model_slug / f"scenario_{scenario}_selfcorrect" / "preds.jsonl"

    if not in_path.exists():
        console.log(f"[red]Mancante: {in_path} — scenario {scenario.upper()} skippato[/red]")
        return

    records = _load_jsonl(in_path)
    records = _filter_records(records, max_patients, max_encounters)

    if out_path.exists():
        out_path.unlink()
        console.log(f"Rimosso preds precedente: {out_path}")

    console.log(f"\n[bold cyan]Scenario {scenario.upper()}[/bold cyan]: "
                f"{len(records)} record × {len(sections)} sezioni "
                f"= {len(records) * len(sections)} chiamate")

    section_changes: dict[str, int] = defaultdict(int)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("{task.fields[status]}"),
    ) as progress:
        task = progress.add_task(
            f"Self-correct scenario {scenario.upper()}",
            total=len(records),
            status="",
        )

        for record in records:
            out_record, changed = await _process_record(record, client, sections, dry_run)
            _append_jsonl(out_path, out_record)
            for sec, did_change in changed.items():
                if did_change:
                    section_changes[sec] += 1
            n_changed = sum(1 for v in changed.values() if v)
            progress.update(
                task,
                status=f"[cyan]{n_changed}/{len(sections)} sezioni modificate",
            )
            progress.advance(task)

    console.log(f"\nScenario {scenario.upper()} - sezioni modificate (su {len(records)} record):")
    for sec in sections:
        n = section_changes[sec]
        pct = 100 * n / len(records) if records else 0
        console.log(f"  {sec:20s}: {n:4d}  ({pct:5.1f}%)")
    console.log(f"\nOutput → {out_path}")


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Self-correction (verify-and-rewrite) sui preds di run_batch_scenarios.py."
    )
    parser.add_argument("--model", required=True,
                        help="Nome modello vLLM (es. Qwen/Qwen3-1.7B). "
                             "Lo slug per la directory viene derivato sostituendo / e . con -.")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--eval-dir", type=Path, default=Path("outputs/summarization/eval"))
    parser.add_argument("--scenarios", nargs="+", default=["a", "b"], choices=["a", "b"],
                        help="Scenari da processare (default: a b).")
    parser.add_argument("--sections", nargs="+", default=ALL_SECTIONS, choices=ALL_SECTIONS,
                        help=f"Sezioni da verificare (default: tutte le 7). Disponibili: {ALL_SECTIONS}")
    parser.add_argument("--max-patients", type=int, default=None,
                        help="PoC: limita ai primi N pazienti (default: tutti).")
    parser.add_argument("--max-encounters", type=int, default=None,
                        help="PoC: limita ai primi N encounter per paziente (default: tutti).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Costruisce i prompt e attraversa la pipeline senza chiamare l'LLM. "
                             "L'output sarà identico a v1 — usalo per validare il flusso.")
    parser.add_argument("--frequency-penalty", type=float, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO if args.verbose else logging.WARNING)

    model_slug = args.model.replace("/", "-").replace(".", "-")

    extra_body: dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": False}}
    if args.frequency_penalty is not None:
        extra_body["frequency_penalty"] = args.frequency_penalty

    openai = AsyncOpenAI(api_key="EMPTY", base_url=args.base_url)
    client = OpenAIClient(
        openai, model=args.model, max_output_tokens=args.max_tokens, extra_body=extra_body,
    )

    console.log(f"[bold]Self-correction[/bold]")
    console.log(f"  Modello:      {args.model}  (slug: {model_slug})")
    console.log(f"  Scenari:      {args.scenarios}")
    console.log(f"  Sezioni:      {args.sections}")
    console.log(f"  Max pazienti: {args.max_patients or 'tutti'}")
    console.log(f"  Max encounter/paziente: {args.max_encounters or 'tutti'}")
    if args.dry_run:
        console.log("  [yellow]DRY RUN[/yellow] — nessuna chiamata all'LLM")

    for scenario in args.scenarios:
        await _process_scenario(
            scenario=scenario,
            model_slug=model_slug,
            eval_dir=args.eval_dir,
            client=client,
            sections=args.sections,
            max_patients=args.max_patients,
            max_encounters=args.max_encounters,
            dry_run=args.dry_run,
        )

    console.log("\n[bold green]Completato.[/bold green]")
    console.log("Prossimi passi:")
    for scenario in args.scenarios:
        console.log(f"  python scripts/summarization/report_deltas.py "
                    f"--model {model_slug}/scenario_{scenario}_selfcorrect")


if __name__ == "__main__":
    asyncio.run(main())
