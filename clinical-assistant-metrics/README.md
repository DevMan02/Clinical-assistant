# Clinical Assistant — Pipeline Tesi

Pipeline end-to-end per valutare l'estrazione iterativa di summary clinici da
documenti italiani con LLM open-source (Qwen3 0.6B / 1.7B / 4B), confrontati
contro un gold standard prodotto da Claude Sonnet 4.6.

## Setup

```bash
cd clinical-assistant-metrics
uv sync
```

Il file gold di riferimento è `data/claude-sonnet-4-6-test.jsonl` (50 pazienti,
82 documenti). Tutti i comandi vanno eseguiti dalla cartella
`clinical-assistant-metrics/` con `PYTHONPATH=src`.

## Server vLLM

Sul server remoto (es. barney), avvia vLLM **senza** `--structured-outputs-config`
(altrimenti gli output diventano vuoti su schemi complessi):

```bash
vllm serve Qwen/Qwen3-1.7B \
  --chat-template ~/qwen3_nonthinking.jinja \
  --max-model-len 32768
```

Apri tunnel SSH: `ssh -L 8000:localhost:8000 barney`.

## Pipeline standard A vs B

Per ogni modello da valutare, esegui in ordine:

### 1. Estrazione (live, ~30-60 min/modello)

Genera `preds.jsonl` per scenario A (gold history) e B (self history):

```bash
PYTHONPATH=src uv run python scripts/summarization/run_batch_scenarios.py \
  --gold-path data/claude-sonnet-4-6-test.jsonl \
  --model Qwen/Qwen3-1.7B
```

Output:
- `outputs/summarization/eval/{slug}/scenario_a/preds.jsonl`
- `outputs/summarization/eval/{slug}/scenario_b/preds.jsonl`
- `outputs/summarization/scenario_comparison/{slug}/{patient_id}/doc_*.json` (debug + fallback tracking)

### 2. Valutazione per scenario

```bash
PYTHONPATH=src uv run python scripts/summarization/report_deltas.py \
  --model Qwen-Qwen3-1-7B/scenario_a

PYTHONPATH=src uv run python scripts/summarization/report_deltas.py \
  --model Qwen-Qwen3-1-7B/scenario_b
```

Output per scenario:
- `report_deltas/cumulative_scores.jsonl` — un record per (paziente, documento)
- `report_deltas/delta_report.json` — contiene tre metriche:
  - `global` (micro-avg sui documenti)
  - `global_by_patient` (macro-avg: ogni paziente pesa uguale)
  - `by_encounter_index` (score per posizione del documento)

### 3. Merge A+B per i grafici

```bash
PYTHONPATH=src uv run python scripts/summarization/merge_reports.py \
  --model Qwen-Qwen3-1-7B
```

Output: `outputs/summarization/scores/Qwen-Qwen3-1-7B.jsonl` (formato consumato
dai plot, include `fallbacks_a`/`fallbacks_b` letti dai debug JSON).

## Self-correction (opzionale, capitolo dedicato)

Secondo passaggio "verify-and-rewrite" sui preds della pipeline standard.

**Proof of concept** (5 min, valida i prompt):

```bash
PYTHONPATH=src uv run python scripts/summarization/run_self_correction.py \
  --model Qwen/Qwen3-1.7B \
  --scenarios a \
  --max-patients 3 --max-encounters 2 \
  --sections medications clinical_problems
```

**Full run** (entrambi gli scenari, tutte le sezioni):

```bash
PYTHONPATH=src uv run python scripts/summarization/run_self_correction.py \
  --model Qwen/Qwen3-1.7B --scenarios a b
```

Riusa la pipeline di valutazione standard:

```bash
PYTHONPATH=src uv run python scripts/summarization/report_deltas.py \
  --model Qwen-Qwen3-1-7B/scenario_a_selfcorrect
PYTHONPATH=src uv run python scripts/summarization/report_deltas.py \
  --model Qwen-Qwen3-1-7B/scenario_b_selfcorrect
```

## Drift analysis

Regressione lineare di `patient_score` vs `encounter_index` per quantificare
quanto la self-history degrada lo score (slope negativo in B, ≈ 0 in A):

```bash
PYTHONPATH=src uv run python scripts/summarization/drift_analysis.py \
  --models Qwen-Qwen3-0-6B Qwen-Qwen3-1-7B Qwen-Qwen3-4B-Instruct-2507
```

Output: `outputs/summarization/drift_analysis/`
- `drift_table.csv` / `drift_table.md` (slope, R², p-value per modello × scenario)
- `drift_plot_{models}.png`

## Ablation no-history

Per isolare l'effetto del contesto storico, riesegui solo lo scenario A con
history vuota (B viene saltato):

```bash
PYTHONPATH=src uv run python scripts/summarization/run_batch_scenarios.py \
  --model Qwen/Qwen3-1.7B \
  --no-history \
  --eval-dir outputs/summarization/eval_no_history
```

> **Caveat**: i delta gold di Claude sono stati generati *con* history del
> paziente. Confronto sporco — utile per analisi qualitativa o limitato ai
> primi documenti di ogni paziente.

## Grafici

Tutti accettano `--models` con uno o più slug e producono PNG in
`outputs/summarization/grafici_aggregati/`.

| Script | Cosa mostra |
|---|---|
| `grafici/plot_aggregate_scores.py` | score_delta medio per posizione documento (A vs B, multi-modello) |
| `grafici/plot_section_scores.py` | section_score per sezione (A vs B) |
| `grafici/plot_fallback_rates.py` | % documenti con lista vuota per sezione e scenario |
| `grafici/plot_model_comparison.py` | confronto a 3 scenari (A/B/C) — richiede setup C |

Esempio:

```bash
PYTHONPATH=src uv run python scripts/summarization/grafici/plot_aggregate_scores.py \
  --models Qwen-Qwen3-0-6B Qwen-Qwen3-1-7B Qwen-Qwen3-4B-Instruct-2507

PYTHONPATH=src uv run python scripts/summarization/grafici/plot_fallback_rates.py \
  --models Qwen-Qwen3-0-6B Qwen-Qwen3-1-7B Qwen-Qwen3-4B-Instruct-2507
```

## Workflow tipico per un nuovo modello

```bash
# 1. Avvia server vLLM sul modello target
# 2. Estrazione + valutazione
python scripts/summarization/run_batch_scenarios.py --model <name>
python scripts/summarization/report_deltas.py    --model <slug>/scenario_a
python scripts/summarization/report_deltas.py    --model <slug>/scenario_b
python scripts/summarization/merge_reports.py    --model <slug>

# 3. (opzionale) Self-correction
python scripts/summarization/run_self_correction.py --model <name> --scenarios a b
python scripts/summarization/report_deltas.py    --model <slug>/scenario_a_selfcorrect
python scripts/summarization/report_deltas.py    --model <slug>/scenario_b_selfcorrect

# 4. Una volta accumulati N modelli: drift + grafici comparativi
python scripts/summarization/drift_analysis.py --models <slug1> <slug2> <slug3>
python scripts/summarization/grafici/plot_aggregate_scores.py --models <slug1> <slug2> <slug3>
python scripts/summarization/grafici/plot_fallback_rates.py    --models <slug1> <slug2> <slug3>
```

(Per tutti i comandi sopra: prefiggere `PYTHONPATH=src uv run`.)

## Output prodotti per la tesi

```
outputs/summarization/
├── eval/{slug}/
│   ├── scenario_a/                     # baseline gold-history
│   │   ├── preds.jsonl
│   │   └── report_deltas/
│   │       ├── cumulative_scores.jsonl
│   │       ├── delta_report.json       # global + global_by_patient + by_encounter_index
│   │       └── patients/{pid}_*.json
│   ├── scenario_b/                     # baseline self-history
│   ├── scenario_a_selfcorrect/         # self-correction
│   └── scenario_b_selfcorrect/
├── scores/{slug}.jsonl                 # merge A+B per i grafici
├── scenario_comparison/{slug}/         # debug per documento (con fallbacks)
├── drift_analysis/                     # slope drift per modello
└── grafici_aggregati/                  # PNG finali
```

Per ogni modello la tesi dispone di: tre metriche sintetiche
(`global`, `global_by_patient`, drift slope), breakdown per sezione,
fallback rate, e — se applicato — confronto baseline vs self-correct.

## Slug convention

Il nome del modello vLLM viene normalizzato sostituendo `/` e `.` con `-`:

| Modello | Slug |
|---|---|
| `Qwen/Qwen3-0.6B` | `Qwen-Qwen3-0-6B` |
| `Qwen/Qwen3-1.7B` | `Qwen-Qwen3-1-7B` |
| `Qwen/Qwen3-4B-Instruct-2507` | `Qwen-Qwen3-4B-Instruct-2507` |

Usa il **nome originale** con `--model` negli script di estrazione, lo **slug**
nei comandi che leggono cartelle (`report_deltas.py`, `merge_reports.py`,
`drift_analysis.py`, plot).
