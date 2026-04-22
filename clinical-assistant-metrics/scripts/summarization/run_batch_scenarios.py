"""
run_batch_scenarios.py

Esegue SOLO l'estrazione degli scenari A e B per tutti i pazienti presenti
nel file gold JSONL. La valutazione è delegata a report_deltas.py.

Scenario A — modello + gold history (inferenza live):
  Il modello riceve il documento + storia gold Claude come contesto.

Scenario B — modello + self history (inferenza live iterativa):
  Il modello riceve il documento + la propria storia accumulata fino al
  documento precedente (aggiornata ad ogni passo).

Per ogni paziente produce:
  - JSON ricchi di debug per documento:
      outputs/summarization/scenario_comparison/{model_slug}/{patient_id}/
        doc_01_scenario_a.json
        doc_01_scenario_b.json
  - Preds JSONL nel formato atteso da report_deltas.py:
      outputs/summarization/eval/{model_slug}/scenario_a/preds.jsonl
      outputs/summarization/eval/{model_slug}/scenario_b/preds.jsonl

Ogni riga dei preds.jsonl contiene:
  {
    "summary":              {...}   # prior state (gold per A, self per B)
    "document":             {...}   # encounter completo
    "target_summary_delta": {...}   # delta gold
    "pred_summary_delta":   {...}   # delta predetto dal modello
  }

Flusso completo:
  1. python run_batch_scenarios.py --model ...
  2. python report_deltas.py --model {slug}/scenario_a
  3. python report_deltas.py --model {slug}/scenario_b
  4. python merge_reports.py --model {slug}
  5. python grafici/plot_aggregate_scores.py --models {slug}

Esecuzione:
  vllm serve Qwen/Qwen3-1.7B --chat-template ~/qwen3_nonthinking.jinja --max-model-len 32768
  cd clinical-assistant-metrics
  PYTHONPATH=src uv run python scripts/summarization/run_batch_scenarios.py \\
    --gold-path data/claude-sonnet-4-6-test.jsonl \\
    --model Qwen/Qwen3-1.7B
"""

import argparse
import asyncio
import json
import logging
import re
import warnings
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(name)s — %(levelname)s — %(message)s")

from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError
from rich.console import Console

from clinical_assistant.summarization.allergies import AllergyOutputFormat
from clinical_assistant.summarization.clinical_problems import ClinicalProblemOutputFormat
from clinical_assistant.summarization.encounter import EncounterDocument
from clinical_assistant.summarization.family_history import FamilyHistoryOutputFormat
from clinical_assistant.summarization.measurements import MeasurementEventOutputFormat
from clinical_assistant.summarization.medications import MedicationHistoryOutputFormat
from clinical_assistant.summarization.patient import PatientSummary, PatientSummaryDelta, PatientSummaryExtractor
from clinical_assistant.summarization.procedures import ProcedureOutputFormat
from clinical_assistant.summarization.structured_output import LlamaCppClient, OpenAIClient
from clinical_assistant.summarization.substances import SubstanceUseOutputFormat


console = Console()


# ---------------------------------------------------------------------------
# Noop client
# ---------------------------------------------------------------------------
class _NoopClient:
    async def structured_output(self, prompt, output_format):
        return None


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_patient_ids(path: Path) -> list[str]:
    ids = []
    seen = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pid = row.get("summary", {}).get("patient_id")
            if pid and pid not in seen:
                seen.add(pid)
                ids.append(pid)
    return ids


def _load_rows_by_patient(path: Path, patient_id: str) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("summary", {}).get("patient_id") == patient_id:
                rows.append(row)
    return sorted(rows, key=lambda r: (
        r["document"]["meta"].get("start_date", ""),
        r["document"]["meta"].get("id", ""),
    ))


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str, ensure_ascii=False)


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Estrazione con retry
# ---------------------------------------------------------------------------

def _strip_thinking_and_parse(text: str, fallback_type: type[BaseModel]) -> BaseModel | None:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"```(?:json)?\s*", "", cleaned)
    cleaned = cleaned.strip()
    match = re.search(r"(\{.*\}|\[.*\])", cleaned, flags=re.DOTALL)
    if match:
        return fallback_type.model_validate_json(match.group(1))
    return None


def _try_recover_truncated_events(raw_text: str, list_field: str) -> list:
    """Recupera oggetti JSON completi da un array troncato (output token limit raggiunto)."""
    match = re.search(r'"' + re.escape(list_field) + r'"\s*:\s*\[', raw_text)
    if not match:
        return []
    array_content = raw_text[match.end():]
    events = []
    depth = 0
    start = None
    in_string = False
    escape = False
    for i, c in enumerate(array_content):
        if escape:
            escape = False
            continue
        if c == "\\" and in_string:
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    events.append(json.loads(array_content[start : i + 1]))
                except Exception:
                    pass
                start = None
    return events


async def _safe(
    extractor_fn,
    fallback_type: type[BaseModel],
    client=None,
    retry_lock: asyncio.Lock | None = None,
    fallbacks_out: set | None = None,
    section_name: str | None = None,
):
    def _register_fallback():
        if fallbacks_out is not None and section_name is not None:
            fallbacks_out.add(section_name)

    try:
        result = await extractor_fn()
        if result is None:
            list_field = next(iter(fallback_type.model_fields))
            console.log(f"      ⚠ {fallback_type.__name__} fallback (None dal modello)")
            _register_fallback()
            return fallback_type(**{list_field: []})
        return result
    except Exception as exc:
        exc_str = str(exc)

        # Retry con meno token se vLLM segnala context length superato.
        # Il lock evita race condition su client.max_output_tokens quando le sezioni
        # girano in parallelo con asyncio.gather.
        if "maximum context length" in exc_str and client is not None:
            input_match = re.search(r"prompt contains at least (\d+) input tokens", exc_str)
            ctx_match = re.search(r"maximum context length is (\d+) tokens", exc_str)
            if input_match and ctx_match:
                input_tokens = int(input_match.group(1))
                max_ctx = int(ctx_match.group(1))
                safe_tokens = max_ctx - input_tokens - 200
                if safe_tokens > 50:
                    lock = retry_lock or asyncio.Lock()
                    async with lock:
                        original = client.max_output_tokens
                        client.max_output_tokens = safe_tokens
                        try:
                            result = await extractor_fn()
                            if result is None:
                                list_field = next(iter(fallback_type.model_fields))
                                _register_fallback()
                                return fallback_type(**{list_field: []})
                            return result
                        except Exception as exc2:
                            exc = exc2
                        finally:
                            client.max_output_tokens = original

        # Fallback per modelli thinking + recupero JSON troncato
        if "json_invalid" in exc_str or "Invalid JSON" in exc_str:
            list_field = next(iter(fallback_type.model_fields))
            raw_text = None

            # Preferenza: legge il testo completo dall'oggetto ValidationError
            # (pydantic tronca input_value nel messaggio str, ma .errors() ha il testo pieno)
            if isinstance(exc, ValidationError):
                for err in exc.errors():
                    if err.get("type") == "json_invalid" and isinstance(err.get("input"), str):
                        raw_text = err["input"]
                        break

            # Fallback: estrai dal repr stringa (può essere troncato)
            if raw_text is None:
                raw_match = re.search(r"input_value='(.*?)'(?:,\s*input_type=)", exc_str, re.DOTALL)
                if raw_match:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", DeprecationWarning)
                        raw_text = raw_match.group(1).encode().decode("unicode_escape")

            if raw_text is not None:
                # Prova strip <think> e parse (modelli thinking)
                try:
                    parsed = _strip_thinking_and_parse(raw_text, fallback_type)
                    if parsed is not None:
                        return parsed
                except Exception:
                    pass
                # Recupero da JSON troncato (output token limit raggiunto).
                # Valida ogni elemento individualmente: salta solo quelli con campi
                # non validi (es. status enum fuori range), non l'intero batch.
                try:
                    recovered = _try_recover_truncated_events(raw_text, list_field)
                    if recovered:
                        valid_items = []
                        for item_dict in recovered:
                            try:
                                single = fallback_type.model_validate({list_field: [item_dict]})
                                valid_items.extend(getattr(single, list_field))
                            except Exception:
                                pass
                        if valid_items:
                            console.log(f"      ↩ {fallback_type.__name__} recuperato parzialmente "
                                        f"({len(valid_items)}/{len(recovered)} elementi validi)")
                            return fallback_type(**{list_field: valid_items})
                except Exception:
                    pass

        list_field = next(iter(fallback_type.model_fields))
        console.log(f"      ⚠ {fallback_type.__name__} fallback (lista vuota): {exc}")
        _register_fallback()
        return fallback_type(**{list_field: []})


async def _extract_sequential(
    extractor: PatientSummaryExtractor,
    document: EncounterDocument,
    summary: PatientSummary,
    client=None,
) -> tuple[PatientSummaryDelta, set[str]]:
    # Lock condiviso tra le 7 sezioni per serializzare i retry su max_output_tokens.
    # Le sezioni normali girano in parallelo (come in patient.py); solo i retry
    # vengono serializzati per evitare race condition sullo stato del client.
    lock = asyncio.Lock()
    fallbacks: set[str] = set()

    def safe(fn, fmt, section):
        return _safe(fn, fmt, client=client, retry_lock=lock, fallbacks_out=fallbacks, section_name=section)

    (
        allergies,
        substances,
        family_history,
        clinical_problems,
        procedures,
        medications,
        measurements,
    ) = await asyncio.gather(
        safe(lambda: extractor.allergy_extractor.extract_single_raw(document, summary.allergies), AllergyOutputFormat, "allergies"),
        safe(lambda: extractor.substance_use_extractor.extract_single_raw(document, summary.substances), SubstanceUseOutputFormat, "substances"),
        safe(lambda: extractor.family_history_extractor.extract_single_raw(document, summary.family_history), FamilyHistoryOutputFormat, "family_history"),
        safe(lambda: extractor.clinical_problem_extractor.extract_single_raw(document, summary.clinical_problems), ClinicalProblemOutputFormat, "clinical_problems"),
        safe(lambda: extractor.procedure_extractor.extract_single_raw(document), ProcedureOutputFormat, "procedures"),
        safe(lambda: extractor.medication_extractor.extract_single_raw(document, summary.medications), MedicationHistoryOutputFormat, "medications"),
        safe(lambda: extractor.measurement_extractor.extract_single_raw(document, summary.measurements), MeasurementEventOutputFormat, "measurements"),
    )
    delta = PatientSummaryDelta(
        meta=document.meta,
        allergies=allergies,
        substances=substances,
        family_history=family_history,
        clinical_problems=clinical_problems,
        procedures=procedures,
        medications=medications,
        measurements=measurements,
    )
    return delta, fallbacks


async def _extract_with_retry(
    extractor: PatientSummaryExtractor,
    document: EncounterDocument,
    summary: PatientSummary,
    client=None,
    max_attempts: int = 3,
) -> tuple[PatientSummaryDelta, set[str]]:
    for attempt in range(1, max_attempts + 1):
        try:
            return await _extract_sequential(extractor, document, summary, client)
        except Exception as exc:
            console.log(f"    [attempt {attempt}/{max_attempts}] Estrazione fallita: {exc}")
            if attempt == max_attempts:
                raise


# ---------------------------------------------------------------------------
# Elaborazione singolo paziente
# ---------------------------------------------------------------------------

async def process_patient(
    patient_id: str,
    gold_path: Path,
    extractor: PatientSummaryExtractor,
    noop_extractor: PatientSummaryExtractor,
    llm_client,
    model_name: str,
    patient_dir: Path,
    preds_a_path: Path,
    preds_b_path: Path,
    max_doc_chars: int,
    no_history: bool = False,
) -> None:
    gold_rows = _load_rows_by_patient(gold_path, patient_id)

    if not gold_rows:
        console.log(f"  [SKIP] Nessuna riga gold per {patient_id}")
        return

    console.log(f"\nPaziente {patient_id}: {len(gold_rows)} documenti"
                + (" [ABLATION: no-history]" if no_history else ""))

    first_gold = PatientSummary.model_validate(gold_rows[0]["summary"])

    def _empty_summary() -> PatientSummary:
        return PatientSummary(
            patient_id=patient_id, document_ids=[],
            sex=first_gold.sex, age=first_gold.age,
        )

    # Stato accumulato per scenario B (self history, aggiornato ad ogni documento)
    self_accumulated = _empty_summary()

    for idx, gold_row in enumerate(gold_rows, start=1):
        doc_id = str(gold_row["document"]["meta"]["id"])
        doc_meta = gold_row["document"]["meta"]
        document = EncounterDocument.model_validate(gold_row["document"])

        if max_doc_chars > 0 and len(document.content) > max_doc_chars:
            truncated = document.content[:max_doc_chars] + "\n[DOCUMENT TRUNCATED]"
            document = document.model_copy(update={"content": truncated})

        console.log(f"  doc_{idx:02d} ({doc_id})  start={doc_meta.get('start_date')}")

        # Target gold
        gold_summary_before = PatientSummary.model_validate(gold_row["summary"])
        gold_delta = PatientSummaryDelta.model_validate(gold_row["summary_delta"])
        if idx < len(gold_rows):
            gold_summary_after = PatientSummary.model_validate(gold_rows[idx]["summary"])
        else:
            gold_summary_after = noop_extractor.update(gold_summary_before, gold_delta)

        # ----------------------------------------------------------------
        # Scenario A: modello + gold history (o empty in ablation)
        # ----------------------------------------------------------------
        prior_a = _empty_summary() if no_history else gold_summary_before
        history_tag = "EMPTY (ablation)" if no_history else "gold history"
        console.log(f"    [A] Inferenza con {history_tag}...")
        delta_a, fallbacks_a = await _extract_with_retry(extractor, document, prior_a, client=llm_client)
        accumulated_a = noop_extractor.update(prior_a, delta_a)

        # ----------------------------------------------------------------
        # Scenario B: modello + self history accumulata (saltato in ablation)
        # ----------------------------------------------------------------
        if no_history:
            delta_b = None
            fallbacks_b = set()
            self_before = _empty_summary()
        else:
            console.log(f"    [B] Inferenza con self history...")
            self_before = self_accumulated
            delta_b, fallbacks_b = await _extract_with_retry(extractor, document, self_before, client=llm_client)
            self_accumulated = noop_extractor.update(self_before, delta_b)

        # ----------------------------------------------------------------
        # Salva JSON completi (debug)
        # ----------------------------------------------------------------
        base = {
            "patient_id": patient_id,
            "document_id": doc_id,
            "document_index": idx,
            "document_date": {
                "start": str(doc_meta.get("start_date", "")),
                "end": str(doc_meta.get("end_date", "")),
            },
            "document": gold_row["document"],
            "target_delta": gold_row["summary_delta"],
            "accumulated_target": gold_summary_after.model_dump(),
        }

        _save_json(patient_dir / f"doc_{idx:02d}_scenario_a.json", {
            **base,
            "scenario": "A",
            "description": (f"{model_name} con history vuota (ablation)." if no_history
                            else f"{model_name} con gold history (Claude) come contesto. Inferenza live."),
            "history_source": "empty" if no_history else "gold_claude",
            "model_source": model_name,
            "previous_history": prior_a.model_dump(),
            "predicted_delta": delta_a.model_dump(),
            "accumulated_prediction": accumulated_a.model_dump(),
            "fallbacks": sorted(fallbacks_a),
        })

        if delta_b is not None:
            _save_json(patient_dir / f"doc_{idx:02d}_scenario_b.json", {
                **base,
                "scenario": "B",
                "description": f"{model_name} con self history accumulata (inferenza live iterativa).",
                "history_source": f"self_predicted_{model_name.replace('/', '-')}",
                "model_source": model_name,
                "previous_history": self_before.model_dump(),
                "predicted_delta": delta_b.model_dump(),
                "accumulated_prediction": self_accumulated.model_dump(),
                "fallbacks": sorted(fallbacks_b),
            })

        # ----------------------------------------------------------------
        # Salva preds nel formato atteso da report_deltas.py
        # ----------------------------------------------------------------
        _append_jsonl(preds_a_path, {
            "summary": prior_a.model_dump(),
            "document": gold_row["document"],
            "target_summary_delta": gold_row["summary_delta"],
            "pred_summary_delta": delta_a.model_dump(),
        })

        if delta_b is not None:
            _append_jsonl(preds_b_path, {
                "summary": self_before.model_dump(),
                "document": gold_row["document"],
                "target_summary_delta": gold_row["summary_delta"],
                "pred_summary_delta": delta_b.model_dump(),
            })

    console.log(f"  Completato paziente {patient_id}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera scenari A e B per tutti i pazienti nel file gold."
    )
    parser.add_argument("--gold-path", default="data/claude-sonnet-4-6-test.jsonl",
                        help="File JSONL gold standard (target Claude).")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B",
                        help="Nome modello vLLM.")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--max-doc-chars", type=int, default=0,
                        help="Tronca documenti a N caratteri (0 = nessun troncamento).")
    parser.add_argument("--provider", default="vllm", choices=["vllm", "llamacpp"])
    parser.add_argument("--output-dir", default="outputs/summarization/scenario_comparison")
    parser.add_argument("--eval-dir", default="outputs/summarization/eval",
                        help="Directory dove salvare i preds per report_deltas.py.")
    parser.add_argument("--patient-ids", nargs="*", default=None,
                        help="Lista di patient_id da processare (default: tutti nel gold).")
    parser.add_argument("--verbose", action="store_true",
                        help="Abilita log INFO (check enable_thinking, HTTP requests). Default: solo WARNING+.")
    parser.add_argument("--frequency-penalty", type=float, default=None,
                        help="Penalità proporzionale alla frequenza di ogni token (range [-2, 2]). "
                             "Utile per ridurre loop di ripetizione. Default: non applicata.")
    parser.add_argument("--no-history", action="store_true",
                        help="ABLATION: passa una history vuota invece della gold per scenario A, "
                             "e disabilita scenario B. Usa con --eval-dir separato per non sovrascrivere.")
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO if args.verbose else logging.WARNING)

    gold_path = Path(args.gold_path)
    model_slug = args.model.replace("/", "-").replace(".", "-")
    scenario_dir = Path(args.output_dir) / model_slug
    eval_model_dir = Path(args.eval_dir) / model_slug
    preds_a_path = eval_model_dir / "scenario_a" / "preds.jsonl"
    preds_b_path = eval_model_dir / "scenario_b" / "preds.jsonl"

    # Rimuove i preds precedenti per evitare duplicati
    for path in (preds_a_path, preds_b_path):
        if path.exists():
            path.unlink()
            print(f"Rimosso preds precedente: {path}")

    patient_ids = args.patient_ids or _load_patient_ids(gold_path)
    print(f"Pazienti da processare: {len(patient_ids)}")
    print(f"Modello: {args.model}  |  Provider: {args.provider}  |  URL: {args.base_url}")
    print(f"JSON completi → {scenario_dir}/")
    print(f"Preds A → {preds_a_path}")
    print(f"Preds B → {preds_b_path}\n")

    _openai = AsyncOpenAI(api_key="EMPTY", base_url=args.base_url)
    if args.provider == "llamacpp":
        llm_client = LlamaCppClient(_openai, model=args.model, max_output_tokens=args.max_tokens)
    else:
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        if args.frequency_penalty is not None:
            extra_body["frequency_penalty"] = args.frequency_penalty
            print(f"frequency_penalty = {args.frequency_penalty}")
        llm_client = OpenAIClient(_openai, model=args.model, max_output_tokens=args.max_tokens, extra_body=extra_body)

    extractor = PatientSummaryExtractor(llm_client)
    noop_extractor = PatientSummaryExtractor(_NoopClient())

    for i, patient_id in enumerate(patient_ids, 1):
        print(f"[{i}/{len(patient_ids)}] Paziente {patient_id}")
        try:
            await process_patient(
                patient_id=patient_id,
                gold_path=gold_path,
                extractor=extractor,
                noop_extractor=noop_extractor,
                llm_client=llm_client,
                model_name=args.model,
                patient_dir=scenario_dir / patient_id,
                preds_a_path=preds_a_path,
                preds_b_path=preds_b_path,
                max_doc_chars=args.max_doc_chars,
                no_history=args.no_history,
            )
        except Exception as exc:
            print(f"  [ERRORE] Paziente {patient_id}: {exc}")
            continue

    print(f"\nEstrazione completata.")
    print(f"  Preds A → {preds_a_path}")
    print(f"  Preds B → {preds_b_path}")
    print(f"\nProssimi passi:")
    print(f"  python scripts/summarization/report_deltas.py --model {model_slug}/scenario_a")
    print(f"  python scripts/summarization/report_deltas.py --model {model_slug}/scenario_b")
    print(f"  python scripts/summarization/merge_reports.py --model {model_slug}")


if __name__ == "__main__":
    asyncio.run(main())
