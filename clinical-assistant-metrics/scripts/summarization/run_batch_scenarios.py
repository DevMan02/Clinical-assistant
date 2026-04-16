"""
run_batch_scenarios.py

Esegue la generazione degli scenari A e B per tutti i pazienti presenti
nel file gold JSONL (es. claude-sonnet-4-6-test.jsonl).

Scenario A — modello + gold history (inferenza live):
  Il modello riceve il documento + storia gold Claude come contesto.
  Score = similarità tra accumulated_A e il target gold.

Scenario B — modello + self history (inferenza live iterativa):
  Il modello riceve il documento + la propria storia accumulata fino al
  documento precedente (aggiornata ad ogni passo). Non serve nessun JSONL
  pre-generato: la storia si costruisce durante l'esecuzione.
  Score = similarità tra accumulated_B e il target gold.

  Confronto A vs B = effetto del contesto (gold history vs self history,
  stesso modello).

Per ogni paziente produce:
  - JSON completi per documento in:
      outputs/summarization/scenario_comparison/{model_slug}/{patient_id}/
        doc_01_scenario_a.json
        doc_01_scenario_b.json
  - Una riga nel file JSONL compatto degli score:
      outputs/summarization/scores/{model_slug}.jsonl

Struttura riga JSONL compatto:
  {
    "patient_id": "...",
    "document_index": 1,
    "document_date": "...",
    "score_final_a": 0.85,
    "score_final_b": 0.72,
    "score_delta_a": 0.91,
    "score_delta_b": 0.78,
    "sections_delta_a": {"medications": 0.9, ...},
    "sections_delta_b": {"medications": 0.7, ...}
  }

Esecuzione (esempio con Qwen3-1.7B, con template Jinja per disabilitare thinking):
  vllm serve Qwen/Qwen3-1.7B --chat-template ~/qwen3_nonthinking.jinja --max-model-len 32768
  cd clinical-assistant-metrics
  PYTHONPATH=src uv run python scripts/summarization/run_batch_scenarios.py \\
    --gold-path data/claude-sonnet-4-6-test.jsonl \\
    --model Qwen/Qwen3-1.7B

Processare solo alcuni pazienti:
  PYTHONPATH=src uv run python scripts/summarization/run_batch_scenarios.py \\
    --gold-path data/claude-sonnet-4-6-test.jsonl \\
    --model Qwen/Qwen3-1.7B \\
    --patient-ids 10000032 10000033
"""

import argparse
import asyncio
import json
import logging
import math
import re
import warnings

logging.basicConfig(level=logging.INFO, format="%(name)s — %(levelname)s — %(message)s")
from pathlib import Path

from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from clinical_assistant.eval.summarization.aggregate import compute_aggregate_scores
from clinical_assistant.eval.summarization.allergies import compute_allergy_metrics
from clinical_assistant.eval.summarization.clinical_problems import compute_clinical_problem_metrics
from clinical_assistant.eval.summarization.family_history import compute_family_history_metrics
from clinical_assistant.eval.summarization.measurements import compute_measurement_metrics
from clinical_assistant.eval.summarization.medications import compute_medication_metrics
from clinical_assistant.eval.summarization.procedures import compute_procedure_metrics
from clinical_assistant.eval.summarization.substances import compute_substance_metrics
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

SECTIONS = [
    "medications", "allergies", "substances",
    "clinical_problems", "measurements", "family_history", "procedures",
]


# ---------------------------------------------------------------------------
# Noop client
# ---------------------------------------------------------------------------
class _NoopClient:
    async def structured_output(self, prompt, output_format):
        return None


# ---------------------------------------------------------------------------
# Metriche
# ---------------------------------------------------------------------------

def _age_score(pred_age: int, ref_age: int, half_life: float = 5.0) -> float:
    sigma = half_life / math.sqrt(2 * math.log(2))
    return math.exp(-0.5 * ((pred_age - ref_age) / sigma) ** 2)


def _compute_patient_info_metrics(pred: PatientSummary, ref: PatientSummary) -> dict:
    return {
        "id": ref.patient_id,
        "sex": {
            "reference": ref.sex, "predicted": pred.sex,
            "correct": pred.sex == ref.sex,
            "score": 1.0 if pred.sex == ref.sex else 0.0,
        },
        "age": {
            "reference": ref.age, "predicted": pred.age,
            "correct": pred.age == ref.age,
            "difference": abs(pred.age - ref.age),
            "score": _age_score(pred.age, ref.age),
        },
    }


def _compute_all_metrics(ref: PatientSummary, pred: PatientSummary) -> dict:
    metrics = {"patient_info": _compute_patient_info_metrics(pred, ref)}
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


def _final_score(reference: PatientSummary, predicted: PatientSummary) -> float | None:
    metrics = _compute_all_metrics(reference, predicted)
    scores = compute_aggregate_scores(metrics)
    ps = scores.get("patient_score")
    return float(ps) if ps is not None else None


def _section_scores(reference: PatientSummary, predicted: PatientSummary) -> dict[str, float]:
    metrics = _compute_all_metrics(reference, predicted)
    agg = compute_aggregate_scores(metrics)
    return {
        s: agg["sections"][s]["section_score"]
        for s in SECTIONS
        if s in agg.get("sections", {})
    }


def _round(v: float | None) -> float | None:
    return round(v, 6) if v is not None else None


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
            print(f"      ⚠ {fallback_type.__name__} fallback (None dal modello)")
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
                            print(f"      ↩ {fallback_type.__name__} recuperato parzialmente "
                                  f"({len(valid_items)}/{len(recovered)} elementi validi)")
                            return fallback_type(**{list_field: valid_items})
                except Exception:
                    pass

        list_field = next(iter(fallback_type.model_fields))
        print(f"      ⚠ {fallback_type.__name__} fallback (lista vuota): {exc}")
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
            print(f"    [attempt {attempt}/{max_attempts}] Estrazione fallita: {exc}")
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
    scores_jsonl: Path,
    max_doc_chars: int,
) -> None:
    gold_rows = _load_rows_by_patient(gold_path, patient_id)

    if not gold_rows:
        print(f"  [SKIP] Nessuna riga gold per {patient_id}")
        return

    print(f"\nPaziente {patient_id}: {len(gold_rows)} documenti")

    first_gold = PatientSummary.model_validate(gold_rows[0]["summary"])

    # Stato accumulato per scenario B (self history, aggiornato ad ogni documento)
    self_accumulated = PatientSummary(
        patient_id=patient_id, document_ids=[],
        sex=first_gold.sex, age=first_gold.age,
    )

    for idx, gold_row in enumerate(gold_rows, start=1):
        doc_id = str(gold_row["document"]["meta"]["id"])
        doc_meta = gold_row["document"]["meta"]
        document = EncounterDocument.model_validate(gold_row["document"])

        if max_doc_chars > 0 and len(document.content) > max_doc_chars:
            truncated = document.content[:max_doc_chars] + "\n[DOCUMENT TRUNCATED]"
            document = document.model_copy(update={"content": truncated})

        print(f"  doc_{idx:02d} ({doc_id})  start={doc_meta.get('start_date')}")

        # Target gold
        gold_summary_before = PatientSummary.model_validate(gold_row["summary"])
        gold_delta = PatientSummaryDelta.model_validate(gold_row["summary_delta"])
        if idx < len(gold_rows):
            gold_summary_after = PatientSummary.model_validate(gold_rows[idx]["summary"])
        else:
            gold_summary_after = noop_extractor.update(gold_summary_before, gold_delta)

        # Summary vuoto per confronto delta puro
        empty = PatientSummary(
            patient_id=patient_id, document_ids=[],
            sex=first_gold.sex, age=first_gold.age,
        )
        gold_delta_as_summary = noop_extractor.update(empty, gold_delta)

        # ----------------------------------------------------------------
        # Scenario A: modello + gold history (inferenza live)
        # ----------------------------------------------------------------
        print(f"    [A] Inferenza con gold history...")
        delta_a, fallbacks_a = await _extract_with_retry(extractor, document, gold_summary_before, client=llm_client)
        accumulated_a = noop_extractor.update(gold_summary_before, delta_a)
        delta_a_as_summary = noop_extractor.update(empty, delta_a)
        score_final_a = _round(_final_score(gold_summary_after, accumulated_a))
        score_delta_a = _round(_final_score(gold_delta_as_summary, delta_a_as_summary))
        sections_delta_a = _section_scores(gold_delta_as_summary, delta_a_as_summary)
        print(f"         score_final={score_final_a}  score_delta={score_delta_a}")

        # ----------------------------------------------------------------
        # Scenario B: modello + self history accumulata (inferenza live iterativa)
        # ----------------------------------------------------------------
        print(f"    [B] Inferenza con self history...")
        self_before = self_accumulated
        delta_b, fallbacks_b = await _extract_with_retry(extractor, document, self_before, client=llm_client)
        self_accumulated = noop_extractor.update(self_before, delta_b)
        delta_b_as_summary = noop_extractor.update(empty, delta_b)
        score_final_b = _round(_final_score(gold_summary_after, self_accumulated))
        score_delta_b = _round(_final_score(gold_delta_as_summary, delta_b_as_summary))
        sections_delta_b = _section_scores(gold_delta_as_summary, delta_b_as_summary)
        print(f"         score_final={score_final_b}  score_delta={score_delta_b}")

        # ----------------------------------------------------------------
        # Salva JSON completi
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
            "description": f"{model_name} con gold history (Claude) come contesto. Inferenza live.",
            "history_source": "gold_claude",
            "model_source": model_name,
            "previous_history": gold_row["summary"],
            "predicted_delta": delta_a.model_dump(),
            "accumulated_prediction": accumulated_a.model_dump(),
            "score_final": score_final_a,
            "score_delta": score_delta_a,
        })

        _save_json(patient_dir / f"doc_{idx:02d}_scenario_b.json", {
            **base,
            "scenario": "B",
            "description": f"{model_name} con self history accumulata (inferenza live iterativa).",
            "history_source": f"self_predicted_{model_name.replace('/', '-')}",
            "model_source": model_name,
            "previous_history": self_before.model_dump(),
            "predicted_delta": delta_b.model_dump(),
            "accumulated_prediction": self_accumulated.model_dump(),
            "score_final": score_final_b,
            "score_delta": score_delta_b,
        })

        # Salva riga JSONL compatta
        _append_jsonl(scores_jsonl, {
            "patient_id": patient_id,
            "document_index": idx,
            "document_date": str(doc_meta.get("start_date", "")),
            "score_final_a": score_final_a,
            "score_final_b": score_final_b,
            "score_delta_a": score_delta_a,
            "score_delta_b": score_delta_b,
            "sections_delta_a": sections_delta_a,
            "sections_delta_b": sections_delta_b,
            "fallbacks_a": sorted(fallbacks_a),
            "fallbacks_b": sorted(fallbacks_b),
        })

    print(f"  Completato paziente {patient_id}")


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
    parser.add_argument("--scores-dir", default="outputs/summarization/scores")
    parser.add_argument("--patient-ids", nargs="*", default=None,
                        help="Lista di patient_id da processare (default: tutti nel gold).")
    parser.add_argument("--verbose", action="store_true",
                        help="Abilita log INFO (check enable_thinking, HTTP requests). Default: solo WARNING+.")
    parser.add_argument("--frequency-penalty", type=float, default=None,
                        help="Penalità proporzionale alla frequenza di ogni token (range [-2, 2]). "
                             "Utile per ridurre loop di ripetizione. Default: non applicata.")
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO if args.verbose else logging.WARNING)

    gold_path = Path(args.gold_path)
    model_slug = args.model.replace("/", "-").replace(".", "-")
    scenario_dir = Path(args.output_dir) / model_slug
    scores_jsonl = Path(args.scores_dir) / f"{model_slug}.jsonl"

    # Rimuove il JSONL precedente per evitare duplicati
    if scores_jsonl.exists():
        scores_jsonl.unlink()
        print(f"Rimosso JSONL precedente: {scores_jsonl}")

    patient_ids = args.patient_ids or _load_patient_ids(gold_path)
    print(f"Pazienti da processare: {len(patient_ids)}")
    print(f"Modello: {args.model}  |  Provider: {args.provider}  |  URL: {args.base_url}")
    print(f"JSON completi → {scenario_dir}/")
    print(f"Score aggregati → {scores_jsonl}\n")

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
                scores_jsonl=scores_jsonl,
                max_doc_chars=args.max_doc_chars,
            )
        except Exception as exc:
            print(f"  [ERRORE] Paziente {patient_id}: {exc}")
            continue

    print(f"\nCompletato. Score salvati in: {scores_jsonl}")


if __name__ == "__main__":
    asyncio.run(main())
