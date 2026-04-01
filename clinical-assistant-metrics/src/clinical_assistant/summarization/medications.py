import logging
from collections import defaultdict
from copy import deepcopy
from enum import StrEnum
from typing import Optional

from pydantic import BaseModel

from clinical_assistant.summarization.encounter import EncounterDocument, EncounterMeta
from clinical_assistant.summarization.structured_output import AsyncClient
from clinical_assistant.summarization.utils import format_dates

logger = logging.getLogger(__name__)


class TherapeuticClass(StrEnum):
    CARDIOVASCULAR = "Cardiovascular"
    ENDOCRINE_METABOLIC = "Endocrine/Metabolic"
    RESPIRATORY = "Respiratory"
    NEUROLOGICAL_PSYCHIATRIC = "Neurological/Psychiatric"
    PAIN_INFLAMMATION = "Pain/Inflammation"
    ANTI_INFECTIVE = "Anti-Infective"
    GASTROINTESTINAL = "Gastrointestinal"
    HEMATOLOGICAL = "Hematological"
    RENAL = "Renal"
    ONCOLOGICAL = "Oncological"
    OTHER = "Other"


class MedicationEventType(StrEnum):
    STARTED = "started"
    CONTINUED = "continued"
    DOSE_CHANGED = "dose changed"
    HELD = "held"
    RESUMED = "resumed"
    DISCONTINUED = "discontinued"


class MedicationEvent(BaseModel):
    medication_name: str
    dosage: Optional[str]
    reason: Optional[str]
    category: TherapeuticClass
    status: MedicationEventType
    detail: Optional[str]


class MedicationHistoryOutputFormat(BaseModel):
    events: list[MedicationEvent]


class MedicationHistory(BaseModel):
    name: str
    events: list[tuple[EncounterMeta, MedicationEvent]]


MEDICATION_EVENT_PROMPT_TEMPLATE = """
You are a clinical data extraction system for a Clinical Decision Support System (CDSS).

## Task

You receive:
1. An existing **Medication list** (JSON). On the first encounter this will be empty.
2. A **clinical document** from a single encounter (e.g. discharge summary, consultation report, outpatient visit note).

Your job is to update the **Medication list** incorporating any new information from the encounter document.

## Core Principles

- Extract, Do not Interpret: only extract information that is explicitly stated in the document.
- Stay close to the source language: preserve medical terminology as written in the document.
- Minimal updates: only modify a medication entry when the document provides new or changed information.

## Extraction Rules

### General
1. Extract every medication mentioned in the document.
2. If a medication already exists in the list with the same name (or a clearly equivalent name, e.g. brand/generic), update it rather than creating a duplicate.
3. Only update a medication if there is genuinely new information about it.

### Medication Name
- Use the name as written in the document. If both brand and generic names appear, prefer the generic name and note the brand name in the detail field.

### Dosage
- Extract the dose amount and unit (e.g. "100 mcg", "10 mg") when explicitly stated.
- Extract the route (e.g. "PO", "IV", "SQ", "topical") when explicitly stated.
- Extract the frequency (e.g. "daily", "BID", "PRN", "once weekly") when explicitly stated.
- Leave any dosage subfield null if it is not stated in the document. Do not infer default values.

### Reason
- Extract the clinical reason for the medication when explicitly stated (e.g. "for hypertension", "DVT prophylaxis").
- This may be a diagnosis, symptom, or clinical indication.
- Leave null if no reason is stated or implied by direct proximity in the text.

### Event Type
- Classify what happened to this medication in this encounter based on explicit documentation:
  - STARTED: newly prescribed or initiated in this encounter.
  - CONTINUED: documented as continued, maintained, or listed as an active home medication without change.
  - HELD: explicitly held or temporarily stopped.
  - RESUMED: restarted after being previously held.
  - DISCONTINUED: explicitly stopped or removed.
  - DOSE_CHANGED: dosage modified from a previously documented dose.
- If the event type is ambiguous, default to CONTINUED for home medications that appear on an active medication list.

### Detail
- Write like a clinician's shorthand note, not a narrative summary.
- Be maximally concise: capture the clinical fact, drop the filler.
- Omit words like "during admission", "on admission", "prior to discharge",
  "listed on medication list", "patient was", "documented as" — these add
  no clinical value.
- Use semicolons to separate distinct facts.
- Include relevant lab values parenthetically.
- Do not explain or narrate what happened — just state it.

Good examples:
  - "Held for AKI (Cr 1.5, baseline 1.0); resumed at discharge (Cr 1.2)"
  - "Dose decreased 100 mg → 50 mg for sinus bradycardia"
  - "Switched from warfarin after 2 doses; also listed as Xarelto"
  - "Started for Afib with RVR"
  - "Held for supratherapeutic INR (3.8); restart on scheduled date"

Bad examples (too verbose):
  - "Held during admission in setting of AKI (Cr 1.5 on admission, baseline 1.0); listed on discharge medication list suggesting resumption at discharge once Cr improved to 1.2."
  - "Newly started during admission for Afib with RVR; 30 tablets dispensed, no refills."
  - "Patient received 2 doses of warfarin during admission; transitioned to rivaroxaban prior to discharge."

Better versions of the bad examples:
  - "Held for AKI (Cr 1.5, baseline 1.0); resumed at discharge (Cr 1.2)"
  - "Started for Afib with RVR"
  - "2 doses given; switched to rivaroxaban at discharge"

---

## Input Data

<medications>
{medications}
</medications>

<encounter_document>
{document}
</encounter_document>
"""


class MedicationExtractor:
    def __init__(
        self,
        client: AsyncClient,
        prompt_template: str = MEDICATION_EVENT_PROMPT_TEMPLATE,
    ):
        self.client = client
        self.prompt_template = prompt_template

    async def extract_single_raw(
        self, document: EncounterDocument, medications: list[MedicationHistory]
    ) -> MedicationHistoryOutputFormat | None:
        prompt = self.prompt_template.format(document=document, medications=medications)
        return await self.client.structured_output(
            prompt,
            output_format=MedicationHistoryOutputFormat,
        )

    def update(
        self, medications: list[MedicationHistory], delta: MedicationHistoryOutputFormat | None, meta: EncounterMeta
    ) -> list[MedicationHistory]:
        medications = deepcopy(medications)

        if delta is None:
            return medications

        index = {m.name: m for m in medications}
        for e in delta.events:
            if e.medication_name in index:
                index[e.medication_name].events.append((meta, e))
            else:
                new_med = MedicationHistory(name=e.medication_name, events=[(meta, e)])
                index[e.medication_name] = new_med
                medications.append(new_med)
        return medications

    async def extract_single(
        self, document: EncounterDocument, medications: list[MedicationHistory]
    ) -> list[MedicationHistory]:
        response = await self.extract_single_raw(document, medications)

        if response is None:
            logger.warning(f"Failed medication extraction step for document {document.meta.id}")

        return self.update(medications, response, document.meta)

    async def extract(self, documents: list[EncounterDocument]) -> list[MedicationHistory]:
        medications = []
        for doc in documents:
            medications = await self.extract_single(doc, medications)
        return medications


def to_markdown(medications: list[MedicationHistory]) -> str:
    active: dict[TherapeuticClass, list[MedicationHistory]] = defaultdict(list)
    discontinued: dict[TherapeuticClass, list[MedicationHistory]] = defaultdict(list)

    for m in medications:
        last_event = m.events[-1][1]
        if m.events[-1][1].status is MedicationEventType.DISCONTINUED:
            discontinued[last_event.category].append(m)
        else:
            active[last_event.category].append(m)

    output = ["## Medications"]
    for i, group in enumerate([active, discontinued]):
        if i == 0:
            output.append("### Active")
        else:
            output.append("### Discontinued")
        for category in group:
            output.append(f"#### {category}")
            for m in group[category]:
                last = m.events[-1][1]
                item = []
                # if medication is active but not prescribed during the last known doctor encounter should also add the date to the info
                item.append(f"- **{m.name} {last.dosage}** - {last.reason}")
                if not all(e.status is MedicationEventType.CONTINUED for meta, e in m.events):
                    item.append("<details>")
                    item.append("<summary>See History</summary>\n")
                    for meta, e in m.events:
                        if e.status is not MedicationEventType.CONTINUED:
                            dates_str = format_dates(meta.start_date, meta.end_date)
                            detail_str = f"    - {dates_str} - {e.status}"
                            if e.detail:
                                detail_str += f" - {e.detail}\n"
                            item.append(detail_str)
                    item.append("</details>")
                output.append("\n".join(item))
    return "\n\n".join(output)
