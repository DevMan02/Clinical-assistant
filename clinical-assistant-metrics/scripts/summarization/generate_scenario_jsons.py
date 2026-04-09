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


def _final_score(reference: PatientSummary, predicted: PatientSummary) -> float:
    metrics = _compute_all_metrics(reference, predicted)
    scores = compute_aggregate_scores(metrics)
    return float(scores.get("patient_score", 0.0))


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


# ---------------------------------------------------------------------------
# Estrazione sequenziale con retry (per Scenario A — inferenza live)
#
# I sub-extractor vengono chiamati uno alla volta invece di asyncio.gather()
# per evitare problemi con Ollama (che gestisce una richiesta alla volta).
# Con vLLM funziona ugualmente bene.
# ---------------------------------------------------------------------------

async def _safe(extractor_fn, fallback_type: type[BaseModel], client=None):
    """Chiama extractor_fn() con retry automatico su errore 400 context-length.

    Quando vLLM risponde con "maximum context length is X, prompt contains Y tokens",
    calcoliamo i token disponibili (X - Y - 50) e riproviamo con max_output_tokens
    ridotto. Se anche il retry fallisce, restituiamo lista vuota.
    """
    try:
        return await extractor_fn()
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
    parser.add_argument("--qwen-path", default="data/Qwen-Qwen3-4B-Instruct-2507.jsonl")
    parser.add_argument("--patient-id", default="10000032")
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
    args = parser.parse_args()

    gold_path = Path(args.gold_path)
    qwen_path = Path(args.qwen_path)
    patient_id = args.patient_id
    model_slug = args.model.replace("/", "-").replace(".", "-")
    output_dir = Path(args.output_dir) / model_slug / patient_id

    gold_rows = _load_rows_by_patient(gold_path, patient_id)
    qwen_rows = _load_rows_by_patient(qwen_path, patient_id)
    qwen_by_doc_id = {str(r["document"]["meta"]["id"]): r for r in qwen_rows}

    if not gold_rows:
        raise ValueError(f"Nessuna riga trovata per il paziente {patient_id} in {gold_path}")
    if not qwen_rows:
        raise ValueError(f"Nessuna riga trovata per il paziente {patient_id} in {qwen_path}")

    print(f"Paziente {patient_id}: {len(gold_rows)} documenti.")
    print(f"Model: {args.model}  |  Provider: {args.provider}  |  URL: {args.base_url}\n")

    _openai = AsyncOpenAI(api_key="EMPTY", base_url=args.base_url)
    if args.provider == "llamacpp":
        llm_client = LlamaCppClient(_openai, model=args.model, max_output_tokens=args.max_tokens)
    else:
        llm_client = OpenAIClient(_openai, model=args.model, max_output_tokens=args.max_tokens)

    qwen_extractor = PatientSummaryExtractor(llm_client)
    noop_extractor = PatientSummaryExtractor(_NoopClient())

    # Scenario C: storia accumulata da FP8 su se stesso (aggiornata ad ogni documento)
    first_gold = PatientSummary.model_validate(gold_rows[0]["summary"])
    fp8_accumulated = PatientSummary(
        patient_id=patient_id, document_ids=[],
        sex=first_gold.sex, age=first_gold.age,
    )

    for idx, gold_row in enumerate(gold_rows, start=1):
        doc_id = str(gold_row["document"]["meta"]["id"])
        doc_meta = gold_row["document"]["meta"]
        document = EncounterDocument.model_validate(gold_row["document"])

        # Tronca il documento se richiesto (riduce i token di input per vLLM vincolato)
        if args.max_doc_chars > 0 and len(document.content) > args.max_doc_chars:
            truncated_content = document.content[: args.max_doc_chars] + "\n[DOCUMENT TRUNCATED]"
            document = document.model_copy(update={"content": truncated_content})
            print(f"  ✂ Documento troncato a {args.max_doc_chars} chars "
                  f"(originale: {len(gold_row['document']['content'])} chars)")

        if doc_id not in qwen_by_doc_id:
            raise ValueError(f"Documento {doc_id} non trovato nel file Qwen.")

        qwen_row = qwen_by_doc_id[doc_id]

        print(f"doc_{idx:02d} ({doc_id})  start={doc_meta.get('start_date')}")

        # ----------------------------------------------------------------
        # Target: summary accumulato da Claude DOPO questo documento.
        # Per i < len: è il "summary" della riga successiva nel JSONL gold.
        # Per l'ultimo documento: si calcola con update().
        # ----------------------------------------------------------------
        gold_summary_before = PatientSummary.model_validate(gold_row["summary"])
        gold_delta = PatientSummaryDelta.model_validate(gold_row["summary_delta"])
        if idx < len(gold_rows):
            # Già presente nel JSONL come "summary" del documento successivo
            gold_summary_after = PatientSummary.model_validate(gold_rows[idx]["summary"])
        else:
            gold_summary_after = noop_extractor.update(gold_summary_before, gold_delta)

        # ----------------------------------------------------------------
        # Scenario B: summary accumulato da Qwen DOPO questo documento.
        # Per i < len: è il "summary" della riga Qwen successiva (zero inferenza).
        # Per l'ultimo documento: si calcola con update().
        # ----------------------------------------------------------------
        qwen_summary_before = PatientSummary.model_validate(qwen_row["summary"])
        qwen_delta = PatientSummaryDelta.model_validate(qwen_row["summary_delta"])
        if idx < len(qwen_rows):
            qwen_summary_after = PatientSummary.model_validate(qwen_rows[idx]["summary"])
        else:
            qwen_summary_after = noop_extractor.update(qwen_summary_before, qwen_delta)

        # score_delta B: confronto puro delta Qwen(qwen_history) vs delta Claude
        # — converte entrambi i delta in PatientSummary con storia vuota
        empty_summary = PatientSummary(
            patient_id=patient_id, document_ids=[], sex=gold_summary_before.sex, age=gold_summary_before.age
        )
        gold_delta_as_summary = noop_extractor.update(empty_summary, gold_delta)
        qwen_delta_as_summary = noop_extractor.update(empty_summary, qwen_delta)
        score_b = round(_final_score(gold_summary_after, qwen_summary_after), 6)
        score_delta_b = round(_final_score(gold_delta_as_summary, qwen_delta_as_summary), 6)
        print(f"  [B] score_B = {score_b:.4f}  score_delta_B = {score_delta_b:.4f}  (dal JSONL Qwen, zero inferenza)")

        # ----------------------------------------------------------------
        # Scenario A: Qwen riceve il documento + storia gold come contesto.
        # Inferenza live — produce delta_A, poi accumulated_A = update().
        # ----------------------------------------------------------------
        print(f"  [A] Estrazione con {args.model} e storia gold (inferenza live)...")
        delta_a = await _extract_with_retry(qwen_extractor, document, gold_summary_before, client=llm_client)
        accumulated_a = noop_extractor.update(gold_summary_before, delta_a)
        delta_a_as_summary = noop_extractor.update(empty_summary, delta_a)
        score_a = round(_final_score(gold_summary_after, accumulated_a), 6)
        score_delta_a = round(_final_score(gold_delta_as_summary, delta_a_as_summary), 6)
        print(f"      score_A = {score_a:.4f}  score_delta_A = {score_delta_a:.4f}")

        # ----------------------------------------------------------------
        # Scenario C: FP8 con storia auto-predetta da se stesso (inferenza live).
        # Replica la struttura di dataset.py ma con FP8 invece di 2507.
        # fp8_accumulated viene aggiornato ad ogni documento e riusato al prossimo.
        # Confronto C vs B = effetto modello puro (stessa self-history, modelli diversi).
        # ----------------------------------------------------------------
        print(f"  [C] Estrazione con {args.model} storia self-predetta (inferenza live)...")
        fp8_before = fp8_accumulated  # salva lo stato PRIMA dell'update
        delta_c = await _extract_with_retry(qwen_extractor, document, fp8_before, client=llm_client)
        fp8_accumulated = noop_extractor.update(fp8_before, delta_c)
        delta_c_as_summary = noop_extractor.update(empty_summary, delta_c)
        score_c = round(_final_score(gold_summary_after, fp8_accumulated), 6)
        score_delta_c = round(_final_score(gold_delta_as_summary, delta_c_as_summary), 6)
        print(f"      score_C = {score_c:.4f}  score_delta_C = {score_delta_c:.4f}")

        # ----------------------------------------------------------------
        # Salvataggio JSON
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

        scenario_a = {
            **base,
            "scenario": "A",
            "description": (
                f"{args.model} ha ricevuto il documento + storia gold (Claude) come contesto. "
                "Inferenza live. Score = similarità tra accumulated_A e il target gold."
            ),
            "history_source": "gold_claude",
            "model_source": args.model,
            "previous_history": gold_row["summary"],
            "predicted_delta": delta_a.model_dump(),
            "accumulated_prediction": accumulated_a.model_dump(),
            "score_final": score_a,
            "score_delta": score_delta_a,
        }

        scenario_b = {
            **base,
            "scenario": "B",
            "description": (
                "Qwen ha usato la propria storia auto-predetta come contesto (dal JSONL, zero inferenza). "
                "Score = similarità tra accumulated_B e il target gold."
            ),
            "history_source": "self_predicted_qwen",
            "model_source": "Qwen-Qwen3-4B-Instruct-2507",
            "previous_history": qwen_row["summary"],
            "predicted_delta": qwen_row["summary_delta"],
            "accumulated_prediction": qwen_summary_after.model_dump(),
            "score_final": score_b,
            "score_delta": score_delta_b,
        }

        scenario_c = {
            **base,
            "scenario": "C",
            "description": (
                f"{args.model} ha usato la propria storia auto-predetta come contesto (inferenza live iterativa). "
                "Confronto C vs B = effetto modello puro (stesso tipo self-history, modelli diversi)."
            ),
            "history_source": f"self_predicted_{model_slug}",
            "model_source": args.model,
            "previous_history": fp8_before.model_dump(),
            "predicted_delta": delta_c.model_dump(),
            "accumulated_prediction": fp8_accumulated.model_dump(),
            "score_final": score_c,
            "score_delta": score_delta_c,
        }

        _save_json(output_dir / f"doc_{idx:02d}_scenario_a.json", scenario_a)
        _save_json(output_dir / f"doc_{idx:02d}_scenario_b.json", scenario_b)
        _save_json(output_dir / f"doc_{idx:02d}_scenario_c.json", scenario_c)
        print(f"  Salvati: doc_{idx:02d}_scenario_a/b/c.json\n")

    print(f"Completato. {len(gold_rows) * 3} file JSON salvati in: {output_dir}")
    print(f"Model: {args.model}  |  Output slug: {model_slug}")


if __name__ == "__main__":
    asyncio.run(main())
