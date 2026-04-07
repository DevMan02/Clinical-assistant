patient_id, document_id, document_index   → identità del record
document_date                             → date ricovero/dimissione
document                                  → testo completo documento clinico

target_delta          → delta estratto da Claude da questo documento
                        (ground truth: cosa si doveva estrarre)
accumulated_target    → summary Claude DOPO questo documento
                        (gold reference per le metriche)

previous_history      → summary passato come contesto al modello PRIMA del doc
predicted_delta       → delta estratto da Qwen (la predizione)
accumulated_prediction → update(previous_history, predicted_delta)
                         questo è ciò che viene confrontato con accumulated_target

score_final           → similarità(accumulated_prediction, accumulated_target)

Definizioni precise

accumulated_target = il summary che Claude ha costruito accumulando tutti i documenti fino a i incluso. È il gold standard — la verità di riferimento.

accumulated_prediction = il summary che Qwen produce per lo stesso punto temporale. È ciò che confrontiamo col target.

Sì, esattamente. Il predicted_delta viene calcolato così:


delta_a = await extract(document, gold_summary_before)
Qwen riceve due input:

document — il testo clinico del documento i
gold_summary_before — il summary gold di Claude prima di questo documento (usato come contesto/memoria)
E produce un output:

predicted_delta — solo le nuove informazioni estratte dal documento i, non il summary completo
Poi:


accumulated_prediction = update(gold_summary_before, predicted_delta)
Quindi il flusso completo è:


gold_summary_before  ──┐
                       ├──► Qwen ──► predicted_delta (solo le novità del doc_i)
document_i           ──┘
                                           │
                                           ▼
                       update(gold_summary_before, predicted_delta)
                                           │
                                           ▼
                                accumulated_prediction 


accumulated_prediction = update(gold_summary_before, delta_a)

gold_summary_before contiene già tutta la storia accumulata da Claude fino al documento i-1. Un fallback su delta_a del documento i significa che perdi solo le aggiunte incrementali di quel documento — la storia precedente è integra. Per doc_04 con score=0.93, il gold_summary_before aveva già 3 documenti ricchi, e Qwen doveva aggiungere solo le misurazioni/farmaci nuovi del quarto documento.

Per doc_01 (score=0.53), invece, gold_summary_before è vuoto → tutto dipende da delta_a → i fallback fanno male davvero.