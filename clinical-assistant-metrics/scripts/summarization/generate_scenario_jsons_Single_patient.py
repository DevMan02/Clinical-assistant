"""
generate_scenario_jsons.py

Genera N×3 file JSON di confronto (N documenti × 3 scenari) per il paziente target.
Gli output vengono salvati in: <output-dir>/<model-slug>/<patient-id>/

Per ogni documento i il confronto avviene tra il PatientSummary accumulato DOPO
l'elaborazione di quel documento:

  Target (riferimento fisso):
    gold_summary_after_i = gold_rows[i+1]["summary"]   (per i < N-1)
                         = update(gold_before_i, gold_delta_i)  (per l'ultimo doc)
    È il summary accumulato da Claude dopo il documento i.

  Scenario A — modello con storia gold (inferenza live):
    Il modello riceve il documento + gold_rows[i]["summary"] come contesto.
    Produce delta_A. accumulated_A = update(gold_before_i, delta_A).
    Score = similarità(accumulated_A, target).

  Scenario B — modello di riferimento con storia auto-predetta (dal JSONL, zero inferenza):
    È già nel file --qwen-path JSONL — la storia accumulata dal modello di riferimento
    usando la propria storia come contesto ad ogni passo.
    Score = similarità(ref_summary_after_i, target).

  Scenario C — modello con storia auto-predetta (inferenza live iterativa):
    Il modello riceve il documento + la propria storia accumulata fino al passo i-1.
    Confronto C vs B = effetto modello puro (stessa self-history, modelli diversi).
    Confronto A vs C = effetto contesto puro (stesso modello, storia diversa).

Score calcolati per ogni scenario:
  score_final  — similarità tra accumulated_prediction e accumulated_target
                 (include il traino della storia accumulata)
  score_delta  — similarità tra predicted_delta e target_delta convertiti in
                 PatientSummary con storia vuota (estrazione pura, senza rumore storia)

Provider supportati (per Scenario A e C):
  vLLM — OpenAIClient con Responses API + constrained decoding.

Esecuzione (esempio con Qwen FP8):
  vllm serve Qwen/Qwen3-4B-Instruct-2507-FP8 --max-model-len 32768
  cd clinical-assistant-metrics
  PYTHONPATH=src uv run python scripts/summarization/generate_scenario_jsons.py \\
    --model Qwen/Qwen3-4B-Instruct-2507-FP8

Esecuzione (esempio con Gemma 4 GGUF):
  vllm serve unsloth/gemma-4-E4B-it-GGUF \\
    --gguf-file gemma-4-E4B-it-UD-Q4_K_XL.gguf \\
    --tokenizer google/gemma-4-E4B-it \\
    --served-model-name gemma-4-E4B-it-UD-Q4_K_XL.gguf \\
    --max-model-len 32768
  PYTHONPATH=src uv run python scripts/summarization/generate_scenario_jsons.py \\
    --model gemma-4-E4B-it-UD-Q4_K_XL.gguf \\
    --max-doc-chars 16000 --max-tokens 8192"""

import argparse
import asyncio
import json
import math
import re
from pathlib import Path

from openai import AsyncOpenAI
from pydantic import BaseModel

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


# ---------------------------------------------------------------------------
# Noop client — usato solo per extractor.update(), nessuna chiamata LLM
# ---------------------------------------------------------------------------
class _NoopClient:
    async def structured_output(self, prompt, output_format):
        return None


# ---------------------------------------------------------------------------
# Metriche (identiche a report.py)
# ---------------------------------------------------------------------------

def _age_score(pred_age: int, ref_age: int, half_life: float = 5.0) -> float:
    sigma = half_life / math.sqrt(2 * math.log(2))
    return math.exp(-0.5 * ((pred_age - ref_age) / sigma) ** 2)


def _compute_patient_info_metrics(pred: PatientSummary, ref: PatientSummary) -> dict:
    return {
        "id": ref.patient_id,
        "sex": {
            "reference": ref.sex,
            "predicted": pred.sex,
            "correct": pred.sex == ref.sex,
            "score": 1.0 if pred.sex == ref.sex else 0.0,
        },
        "age": {
            "reference": ref.age,
            "predicted": pred.age,
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


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

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


def _strip_thinking_and_parse(text: str, fallback_type: type[BaseModel]) -> BaseModel | None:
    """Prova a recuperare JSON valido quando il modello antepone testo non strutturato.

    Alcuni modelli reasoning (es. Qwen3) possono restituire blocchi <think> o markdown
    anche con structured output attivo; questa funzione tenta il parsing del primo JSON.
    """
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    match = re.search(r"(\{.*\}|\[.*\])", cleaned, flags=re.DOTALL)
    if not match:
        return None

    try:
        return fallback_type.model_validate_json(match.group(1))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Estrazione sequenziale con retry (per Scenario A — inferenza live)
#
# I sub-extractor vengono chiamati uno alla volta invece di asyncio.gather()
# per evitare problemi con Ollama (che gestisce una richiesta alla volta).
# Con vLLM funziona ugualmente bene.
# ---------------------------------------------------------------------------

async def _safe(extractor_fn, fallback_type: type[BaseModel], client=None):
    try:
        result = await extractor_fn()
        if result is None:
            list_field = next(iter(fallback_type.model_fields))
            print(f"      ⚠ {fallback_type.__name__} fallback (None dal modello)")
            return fallback_type(**{list_field: []})
        return result
    except Exception as exc:
        exc_str = str(exc)
        if "maximum context length" in exc_str and client is not None:
            input_match = re.search(r"prompt contains at least (\d+) input tokens", exc_str)
            ctx_match = re.search(r"maximum context length is (\d+) tokens", exc_str)
            if input_match and ctx_match:
                input_tokens = int(input_match.group(1))
                max_ctx = int(ctx_match.group(1))
                safe_tokens = max_ctx - input_tokens - 50
                if safe_tokens > 50:
                    print(f"      ↩ Retry {fallback_type.__name__} con {safe_tokens} output tokens "
                          f"(input={input_tokens}, max={max_ctx})")
                    original = client.max_output_tokens
                    client.max_output_tokens = safe_tokens
                    try:
                        return await extractor_fn()
                    except Exception as exc2:
                        exc = exc2
                    finally:
                        client.max_output_tokens = original

        # Alcuni backend possono restituire testo con prefissi markdown/thinking.
        # Prova a estrarre un JSON valido prima del fallback a lista vuota.
        if "json_invalid" in exc_str or "Invalid JSON" in exc_str:
            raw_match = re.search(r"input_value='(.*?)'(?:,\s*input_type=)", exc_str, re.DOTALL)
            if raw_match:
                raw_text = raw_match.group(1)
                try:
                    raw_text = raw_text.encode().decode("unicode_escape")
                except Exception:
                    pass
                parsed = _strip_thinking_and_parse(raw_text, fallback_type)
                if parsed is not None:
                    print(f"      ↩ Recuperato JSON valido per {fallback_type.__name__} da output testuale")
                    return parsed

        list_field = next(iter(fallback_type.model_fields))
        print(f"      ⚠ {fallback_type.__name__} fallback (lista vuota): {exc}")
        return fallback_type(**{list_field: []})


async def _extract_sequential(
    extractor: PatientSummaryExtractor,
    document: EncounterDocument,
    summary: PatientSummary,
    client=None,
) -> PatientSummaryDelta:
    allergies = await _safe(
        lambda: extractor.allergy_extractor.extract_single_raw(document, summary.allergies),
        AllergyOutputFormat, client,
    )
    substances = await _safe(
        lambda: extractor.substance_use_extractor.extract_single_raw(document, summary.substances),
        SubstanceUseOutputFormat, client,
    )
    family_history = await _safe(
        lambda: extractor.family_history_extractor.extract_single_raw(document, summary.family_history),
        FamilyHistoryOutputFormat, client,
    )
    clinical_problems = await _safe(
        lambda: extractor.clinical_problem_extractor.extract_single_raw(document, summary.clinical_problems),
        ClinicalProblemOutputFormat, client,
    )
    procedures = await _safe(
        lambda: extractor.procedure_extractor.extract_single_raw(document),
        ProcedureOutputFormat, client,
    )
    medications = await _safe(
        lambda: extractor.medication_extractor.extract_single_raw(document, summary.medications),
        MedicationHistoryOutputFormat, client,
    )
    measurements = await _safe(
        lambda: extractor.measurement_extractor.extract_single_raw(document, summary.measurements),
        MeasurementEventOutputFormat, client,
    )
    return PatientSummaryDelta(
        meta=document.meta,
        allergies=allergies,
        substances=substances,
        family_history=family_history,
        clinical_problems=clinical_problems,
        procedures=procedures,
        medications=medications,
        measurements=measurements,
    )


async def _extract_with_retry(
    extractor: PatientSummaryExtractor,
    document: EncounterDocument,
    summary: PatientSummary,
    client=None,
    max_attempts: int = 3,
) -> PatientSummaryDelta:
    for attempt in range(1, max_attempts + 1):
        try:
            return await _extract_sequential(extractor, document, summary, client)
        except Exception as exc:
            print(f"    [attempt {attempt}/{max_attempts}] Estrazione fallita: {exc}")
            if attempt == max_attempts:
                raise


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera i JSON di confronto Scenario A / B per un paziente."
    )
    parser.add_argument("--gold-path", default="data/claude-sonnet-4-6.jsonl")
    parser.add_argument("--qwen-path", default="data/Qwen-Qwen3-4B-Instruct-2507.jsonl",
                        help="JSONL pre-generato per Scenario B. Se il file non esiste, Scenario B viene skippato.")
    parser.add_argument("--patient-id", default="10000032")
    parser.add_argument("--max-docs", type=int, default=0,
                        help="Limita il numero di documenti da processare (0 = tutti).")
    parser.add_argument("--output-dir", default="outputs/summarization/scenario_comparison")
    parser.add_argument(
        "--base-url", default="http://localhost:8000/v1",
        help="URL server vLLM (default: http://localhost:8000/v1).",
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen3-4B-Instruct-2507-FP8",
        help="Nome modello vLLM.",
    )
    parser.add_argument("--max-tokens", type=int, default=16384, help="Max tokens per risposta (default: 16384, riduci se vLLM risponde con errori di context length).")
    parser.add_argument(
        "--max-doc-chars", type=int, default=0,
        help="Tronca il contenuto del documento a N caratteri prima di passarlo agli estrattori "
             "(0 = nessun troncamento). ~4 chars/token: usa 16000 per ~4000 token di documento.",
    )
    parser.add_argument(
        "--provider", default="vllm", choices=["vllm", "llamacpp"],
        help="Provider del server LLM: 'vllm' usa la Responses API (default), "
             "'llamacpp' usa la Chat Completions API (per llama-server con GGUF).",
    )
    parser.add_argument(
        "--enable-thinking", action="store_true",
        help="Abilita reasoning/thinking nel template chat. Di default viene disabilitato "
             "per migliorare affidabilita del JSON strutturato.",
    )
    parser.add_argument(
        "--guided-decoding-backend",
        default="outlines",
        choices=["outlines", "xgrammar", "lm-format-enforcer"],
        help="Backend guided decoding per vLLM. 'outlines' evita errori noti di xgrammar "
             "su alcuni modelli/schema.",
    )
    args = parser.parse_args()

    gold_path = Path(args.gold_path)
    qwen_path = Path(args.qwen_path)
    patient_id = args.patient_id
    model_slug = args.model.replace("/", "-").replace(".", "-")
    output_dir = Path(args.output_dir) / model_slug / patient_id

    gold_rows = _load_rows_by_patient(gold_path, patient_id)
    if not gold_rows:
        raise ValueError(f"Nessuna riga trovata per il paziente {patient_id} in {gold_path}")

    # Scenario B opzionale: skippato se il file qwen non esiste
    run_scenario_b = qwen_path.exists()
    if run_scenario_b:
        qwen_rows = _load_rows_by_patient(qwen_path, patient_id)
        qwen_by_doc_id = {str(r["document"]["meta"]["id"]): r for r in qwen_rows}
        if not qwen_rows:
            print(f"[WARN] Nessuna riga Qwen per paziente {patient_id} — Scenario B skippato.")
            run_scenario_b = False
    else:
        qwen_rows = []
        qwen_by_doc_id = {}
        print(f"[INFO] {qwen_path} non trovato — Scenario B skippato.")

    # Limita il numero di documenti se richiesto
    if args.max_docs > 0:
        gold_rows = gold_rows[: args.max_docs]
        qwen_rows = qwen_rows[: args.max_docs]

    print(f"Paziente {patient_id}: {len(gold_rows)} documenti da processare.")
    print(f"Model: {args.model}  |  Provider: {args.provider}  |  URL: {args.base_url}")
    print(f"Scenario B: {'sì' if run_scenario_b else 'no (file JSONL mancante)'}\n")

    _openai = AsyncOpenAI(api_key="EMPTY", base_url=args.base_url)
    if args.provider == "llamacpp":
        llm_client = LlamaCppClient(_openai, model=args.model, max_output_tokens=args.max_tokens)
    else:
        extra_body: dict = {
            "guided_decoding_backend": args.guided_decoding_backend,
        }
        if not args.enable_thinking:
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}

        llm_client = OpenAIClient(
            _openai,
            model=args.model,
            max_output_tokens=args.max_tokens,
            extra_body=extra_body,
        )
        print(f"Responses extra_body: {extra_body}")

    qwen_extractor = PatientSummaryExtractor(llm_client)
    noop_extractor = PatientSummaryExtractor(_NoopClient())

    # Scenario C: storia accumulata da FP8 su se stesso (aggiornata ad ogni documento)
    def _round(v): return round(v, 6) if v is not None else None

    first_gold = PatientSummary.model_validate(gold_rows[0]["summary"])
    self_accumulated = PatientSummary(
        patient_id=patient_id, document_ids=[],
        sex=first_gold.sex, age=first_gold.age,
    )

    saved_count = 0
    for idx, gold_row in enumerate(gold_rows, start=1):
        doc_id = str(gold_row["document"]["meta"]["id"])
        doc_meta = gold_row["document"]["meta"]
        document = EncounterDocument.model_validate(gold_row["document"])

        if args.max_doc_chars > 0 and len(document.content) > args.max_doc_chars:
            truncated_content = document.content[: args.max_doc_chars] + "\n[DOCUMENT TRUNCATED]"
            document = document.model_copy(update={"content": truncated_content})
            print(f"  ✂ Documento troncato a {args.max_doc_chars} chars "
                  f"(originale: {len(gold_row['document']['content'])} chars)")

        print(f"doc_{idx:02d} ({doc_id})  start={doc_meta.get('start_date')}")

        # Target gold
        gold_summary_before = PatientSummary.model_validate(gold_row["summary"])
        gold_delta = PatientSummaryDelta.model_validate(gold_row["summary_delta"])
        if idx < len(gold_rows):
            gold_summary_after = PatientSummary.model_validate(gold_rows[idx]["summary"])
        else:
            gold_summary_after = noop_extractor.update(gold_summary_before, gold_delta)

        empty_summary = PatientSummary(
            patient_id=patient_id, document_ids=[],
            sex=gold_summary_before.sex, age=gold_summary_before.age,
        )
        gold_delta_as_summary = noop_extractor.update(empty_summary, gold_delta)

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

        # ----------------------------------------------------------------
        # Scenario A: modello + storia gold (inferenza live)
        # ----------------------------------------------------------------
        print(f"  [A] Estrazione con {args.model} e storia gold (inferenza live)...")
        delta_a = await _extract_with_retry(qwen_extractor, document, gold_summary_before, client=llm_client)
        accumulated_a = noop_extractor.update(gold_summary_before, delta_a)
        delta_a_as_summary = noop_extractor.update(empty_summary, delta_a)
        score_a = _round(_final_score(gold_summary_after, accumulated_a))
        score_delta_a = _round(_final_score(gold_delta_as_summary, delta_a_as_summary))
        print(f"      score_A = {score_a}  score_delta_A = {score_delta_a}")

        _save_json(output_dir / f"doc_{idx:02d}_scenario_a.json", {
            **base,
            "scenario": "A",
            "description": (
                f"{args.model} ha ricevuto il documento + storia gold (Claude) come contesto. "
                "Inferenza live."
            ),
            "history_source": "gold_claude",
            "model_source": args.model,
            "previous_history": gold_row["summary"],
            "predicted_delta": delta_a.model_dump(),
            "accumulated_prediction": accumulated_a.model_dump(),
            "score_final": score_a,
            "score_delta": score_delta_a,
        })
        saved_count += 1

        # ----------------------------------------------------------------
        # Scenario B (opzionale): summary pre-generato dal JSONL, zero inferenza
        # ----------------------------------------------------------------
        if run_scenario_b and doc_id in qwen_by_doc_id:
            qwen_row = qwen_by_doc_id[doc_id]
            qwen_summary_before = PatientSummary.model_validate(qwen_row["summary"])
            qwen_delta = PatientSummaryDelta.model_validate(qwen_row["summary_delta"])
            if idx < len(qwen_rows):
                qwen_summary_after = PatientSummary.model_validate(qwen_rows[idx]["summary"])
            else:
                qwen_summary_after = noop_extractor.update(qwen_summary_before, qwen_delta)
            qwen_delta_as_summary = noop_extractor.update(empty_summary, qwen_delta)
            score_b = _round(_final_score(gold_summary_after, qwen_summary_after))
            score_delta_b = _round(_final_score(gold_delta_as_summary, qwen_delta_as_summary))
            print(f"  [B] score_B = {score_b}  score_delta_B = {score_delta_b}  (dal JSONL, zero inferenza)")

            _save_json(output_dir / f"doc_{idx:02d}_scenario_b.json", {
                **base,
                "scenario": "B",
                "description": "Storia auto-predetta dal JSONL pre-generato (zero inferenza).",
                "history_source": "self_predicted_qwen",
                "model_source": args.qwen_path,
                "previous_history": qwen_row["summary"],
                "predicted_delta": qwen_row["summary_delta"],
                "accumulated_prediction": qwen_summary_after.model_dump(),
                "score_final": score_b,
                "score_delta": score_delta_b,
            })
            saved_count += 1

        # ----------------------------------------------------------------
        # Scenario C: modello + storia auto-predetta (inferenza live iterativa)
        # ----------------------------------------------------------------
        print(f"  [C] Estrazione con {args.model} storia self-predetta (inferenza live)...")
        self_before = self_accumulated
        delta_c = await _extract_with_retry(qwen_extractor, document, self_before, client=llm_client)
        self_accumulated = noop_extractor.update(self_before, delta_c)
        delta_c_as_summary = noop_extractor.update(empty_summary, delta_c)
        score_c = _round(_final_score(gold_summary_after, self_accumulated))
        score_delta_c = _round(_final_score(gold_delta_as_summary, delta_c_as_summary))
        print(f"      score_C = {score_c}  score_delta_C = {score_delta_c}")

        _save_json(output_dir / f"doc_{idx:02d}_scenario_c.json", {
            **base,
            "scenario": "C",
            "description": (
                f"{args.model} ha usato la propria storia auto-predetta come contesto (inferenza live iterativa)."
            ),
            "history_source": f"self_predicted_{model_slug}",
            "model_source": args.model,
            "previous_history": self_before.model_dump(),
            "predicted_delta": delta_c.model_dump(),
            "accumulated_prediction": self_accumulated.model_dump(),
            "score_final": score_c,
            "score_delta": score_delta_c,
        })
        saved_count += 1
        print(f"  Salvati: doc_{idx:02d} scenari completati\n")

    print(f"Completato. {saved_count} file JSON salvati in: {output_dir}")
    print(f"Model: {args.model}  |  Output slug: {model_slug}")


if __name__ == "__main__":
    asyncio.run(main())
