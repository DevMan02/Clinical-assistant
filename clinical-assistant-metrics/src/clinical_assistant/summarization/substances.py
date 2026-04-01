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


class SubstanceUseStatus(StrEnum):
    ACTIVE = "Active"
    FORMER = "Former"
    NEVER = "Never"
    UNKNOWN = "Unknown"


class SubstanceUseEvent(BaseModel):
    name: str
    status: SubstanceUseStatus
    quantity: Optional[str]
    new_info: bool
    detail: Optional[str]


class SubstanceUseOutputFormat(BaseModel):
    substances: list[SubstanceUseEvent]


class SubstanceUseHistory(BaseModel):
    name: str
    events: list[tuple[EncounterMeta, SubstanceUseEvent]]


SUBSTANCE_USE_EXTRACTION_PROMPT_TEMPLATE = """
You are a clinical data extraction system for a Clinical Decision Support System (CDSS).

## Task

You receive:
1. An existing **Substance Use list** (JSON). On the first encounter this will be empty.
2. A **clinical document** from a single encounter (e.g. discharge summary, consultation report, outpatient visit note).

Your job is to update the **Substance Use list** incorporating any new information from the encounter document.

## Core Principles

- Extract, Do not Interpret: only extract information that is explicitly stated in the document.
- Stay close to the source language: preserve medical terminology as written in the document.
- Minimal updates: only modify an entry when the document provides new or changed information.

## Extraction Rules

### General
1. Extract every substance use mention in the document, including documented denials (e.g. "denies tobacco use").
2. If a substance already exists in the list with the same name, update it rather than creating a duplicate.
3. Only update a substance entry if there is genuinely new information about it.

### Name
- Use a normalized substance name: "tobacco", "alcohol", "cannabis", "cocaine", "opioids", "IV drugs", etc.
- Do not use brand names or slang unless no standard term applies.

### Status
- Classify based on what the document explicitly states:
  - ACTIVE: currently using (e.g. "current smoker", "drinks daily", "active cocaine use")
  - FORMER: previously used but stopped (e.g. "quit smoking 2010", "former drinker", "remote history of cocaine use")
  - NEVER: explicitly denied (e.g. "denies tobacco use", "never smoked", "no alcohol use")
  - UNKNOWN: mentioned but status unclear
- If a substance was previously ACTIVE and the document now says "quit" or "former", update to FORMER.

### Quantity
- Extract quantity or frequency when explicitly stated (e.g. "2 PPD", "30 pack-years", "6 beers/week", "occasional", "heavy use").
- Use the document's own phrasing.
- Leave null if no quantity information is stated.

### Detail
- Write like a clinician's shorthand note, not a narrative summary.
- Be maximally concise: capture the clinical fact, drop the filler.
- Do not explain your reasoning or justify your classification choices.
- Do not speculate about what findings might suggest.
- Just state what the document says.

Good examples:
  - "quit 2010; 30 pack-year history"
  - "2 PPD; counseled to quit"
  - "social use; 6 beers/week"
  - "discharge instructions say do not smoke; emphysema on imaging"

Bad examples:
  - "Imaging showed emphysema, suggesting possible smoking history, 
    but explicit smoking status is not documented in the note."
  - "Patient reports occasional alcohol use, primarily in social 
    settings on the weekends."

Better versions:
  - "emphysema on imaging; smoking status not documented"
  - "occasional; social use on weekends"

  ### New Information Flag
- Set new_info to true when this encounter documents any of:
  - First time the substance is mentioned
  - Status change (e.g. active → former, active → never)
  - New or updated quantity information (e.g. "now 1 PPD" vs prior "2 PPD")
  - New clinical context (e.g. quit date, counseling, related complications)
  - New detail not previously captured
- Set new_info to false when:
  - The substance is merely listed again with the same status and no new detail
  - The only information is a repeat of prior documentation (e.g. "former smoker" restated)
  - The detail would be identical or nearly identical to a prior event
- When new_info is false:
  - Still set the correct current status
  - Still set the current quantity if documented
  - Leave detail empty

---

## Input Data

<substance_use>
{substances}
</substance_use>

<encounter_document>
{document}
</encounter_document>
"""


class SubstanceUseExtractor:
    def __init__(
        self,
        client: AsyncClient,
        prompt_template: str = SUBSTANCE_USE_EXTRACTION_PROMPT_TEMPLATE,
    ):
        self.client = client
        self.prompt_template = prompt_template

    async def extract_single_raw(
        self, document: EncounterDocument, substances: list[SubstanceUseHistory]
    ) -> SubstanceUseOutputFormat | None:
        prompt = self.prompt_template.format(document=document, substances=substances)
        return await self.client.structured_output(
            prompt,
            output_format=SubstanceUseOutputFormat,
        )

    def update(
        self, substances: list[SubstanceUseHistory], delta: SubstanceUseOutputFormat | None, meta: EncounterMeta
    ) -> list[SubstanceUseHistory]:
        substances = deepcopy(substances)

        if delta is None:
            return substances

        index = {s.name: s for s in substances}
        for e in delta.substances:
            if e.name in index:
                index[e.name].events.append((meta, e))
            else:
                new_substance = SubstanceUseHistory(name=e.name, events=[(meta, e)])
                index[e.name] = new_substance
                substances.append(new_substance)
        return substances

    async def extract_single(
        self, document: EncounterDocument, substances: list[SubstanceUseHistory]
    ) -> list[SubstanceUseHistory]:
        response = await self.extract_single_raw(document, substances)

        if response is None:
            logger.warning(f"Failed substance extraction step for document {document.meta.id}")

        return self.update(substances, response, document.meta)

    async def extract(self, documents: list[EncounterDocument]) -> list[SubstanceUseHistory]:
        substances = []
        for doc in documents:
            substances = await self.extract_single(doc, substances)
        return substances


def to_markdown(substances: list[SubstanceUseHistory]) -> str:
    status2substance: dict[SubstanceUseStatus, list[SubstanceUseHistory]] = defaultdict(list)
    for s in substances:
        last = s.events[-1][1]
        status2substance[last.status].append(s)

    output = ["## Substances"]
    # the order in which substances should appear
    for status in [
        SubstanceUseStatus.ACTIVE,
        SubstanceUseStatus.FORMER,
        SubstanceUseStatus.NEVER,
        SubstanceUseStatus.UNKNOWN,
    ]:
        if status not in status2substance:
            continue

        output.append(f"### {status}")
        for substance in status2substance[status]:
            item_str = f"- **{substance.name}**"
            last = substance.events[-1][1]
            if last.quantity:
                item_str += f" - **{last.quantity}**"
            details = [item_str, "<details>", "<summary>See History</summary>\n"]
            for meta, e in substance.events:
                if e.new_info and e.detail:
                    dates_str = format_dates(meta.start_date, meta.end_date)
                    details_str = f"    - {dates_str} - {e.detail}\n"
                    details.append(details_str)
            details.append("</details>")
            output.append("\n".join(details))

    return "\n\n".join(output)
