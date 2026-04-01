import logging
from collections import defaultdict
from copy import deepcopy
from enum import StrEnum
from typing import Optional

from pydantic import BaseModel

from clinical_assistant.summarization.encounter import EncounterDocument
from clinical_assistant.summarization.structured_output import AsyncClient

logger = logging.getLogger(__name__)


class FamilyRelationship(StrEnum):
    FATHER = "Father"
    MOTHER = "Mother"
    BROTHER = "Brother"
    SISTER = "Sister"
    GRANDPARENT = "Grandparent"
    UNCLE_AUNT = "Uncle/Aunt"
    UNSPECIFIED = "Unspecified"


class FamilyHistoryStatus(StrEnum):
    PRESENT = "present"
    DENIED = "denied"


class FamilyHistoryEvent(BaseModel):
    relationship: FamilyRelationship
    condition: str
    status: FamilyHistoryStatus
    detail: Optional[str]


class FamilyHistoryOutputFormat(BaseModel):
    family_history: list[FamilyHistoryEvent]


FAMILY_HISTORY_EXTRACTION_PROMPT_TEMPLATE = """
You are a clinical data extraction system for a Clinical Decision Support System (CDSS).

## Task

You receive:
1. An existing **Family History list** (JSON). On the first encounter this will be empty.
2. A **clinical document** from a single encounter (e.g. discharge summary, consultation report, outpatient visit note).

Your job is to return the complete updated **Family History list** incorporating any new information from the encounter document.

## Core Principles

- Extract, Do not Interpret: only extract information that is explicitly stated in the document.
- Stay close to the source language: preserve medical terminology as written in the document.
- Never downgrade specificity: do not replace existing detailed information with vaguer information from a new document.

## Extraction Rules

### General
1. Return the complete family history list — both existing entries and any new ones.
2. If a condition for the same relative already exists, update that entry rather than creating a duplicate.
3. If an existing entry has no new information in the current document, copy it through unchanged.
4. One entry per condition per relative. If a father has both CAD and diabetes, that is two entries.
5. Documented denials are important — extract them (e.g. "no family history of sudden cardiac death").

### Relationship
- Classify into one of these values:
  - FATHER
  - MOTHER
  - BROTHER
  - SISTER
  - GRANDPARENT: any grandparent
  - UNCLE_AUNT: any uncle or aunt
  - UNSPECIFIED: when the document says "family history of X" without naming who
- If the document says "parents" or "both parents", create separate entries for FATHER and MOTHER.

### Condition
- Use the condition name as written in the document.
- Prefer full terms over abbreviations (e.g. "coronary artery disease" not "CAD"), but preserve clinical terminology.

### Denials
- When the document states "no family history of X" or "denies family 
  history of X", create an entry with:
  - relationship: UNSPECIFIED
  - condition: the condition being denied (e.g. "sudden cardiac death")
  - status: DENIED
  - detail: empty (the status field captures the denial; do not write 
    "no family history" in detail)
- If the denial is specific to a relative (e.g. "father had no history 
  of cardiac disease"), use that relationship instead of UNSPECIFIED.

### Detail
- Write like a clinician's shorthand note, not a narrative summary.
- Be maximally concise: capture the clinical fact, drop the filler.
- Use semicolons to separate distinct facts.
- Include age of onset or age at death when stated.
- Use this field to capture disambiguation when the document provides it:
  - Maternal/paternal side (e.g. "maternal", "paternal side")
  - Sibling identifiers (e.g. "older brother", "younger sister")
  - Specific grandparent (e.g. "paternal grandmother")
- Leave empty if there is nothing notable beyond the relationship and condition.

Good examples:
  - "MI at age 52"
  - "paternal grandmother; died age 68"
  - "older brother; diagnosed age 45"
  - "maternal side"

Bad examples (too verbose):
  - "Patient's father was diagnosed with coronary artery disease at the age of 52 and subsequently underwent bypass surgery."
  - "The patient reports that her maternal grandmother died of a stroke at age 68."

Better versions of the bad examples:
  - "Diagnosed age 52; CABG"
  - "paternal grandmother; died age 68"

---

## Input Data

<family_history>
{family_history}
</family_history>

<encounter_document>
{document}
</encounter_document>
"""


class FamilyHistoryExtractor:
    def __init__(self, client: AsyncClient, prompt_template: str = FAMILY_HISTORY_EXTRACTION_PROMPT_TEMPLATE):
        self.client = client
        self.prompt_template = prompt_template

    async def extract_single_raw(
        self, document: EncounterDocument, family_history: list[FamilyHistoryEvent]
    ) -> FamilyHistoryOutputFormat | None:
        prompt = self.prompt_template.format(document=document, family_history=family_history)
        return await self.client.structured_output(prompt, FamilyHistoryOutputFormat)

    def update(
        self, family_history: list[FamilyHistoryEvent], delta: FamilyHistoryOutputFormat | None
    ) -> list[FamilyHistoryEvent]:
        if delta is None:
            return deepcopy(family_history)

        return delta.family_history

    async def extract_single(
        self, document: EncounterDocument, family_history: list[FamilyHistoryEvent]
    ) -> list[FamilyHistoryEvent]:
        response = await self.extract_single_raw(document, family_history)

        if response is None:
            logger.warning(f"Failed family history extraction step for document {document.meta.id}")

        return self.update(family_history, response)

    async def extract(self, documents: list[EncounterDocument]) -> list[FamilyHistoryEvent]:
        family_history = []
        for doc in documents:
            family_history = await self.extract_single(doc, family_history)
        return family_history


def to_markdown(family_history: list[FamilyHistoryEvent]) -> str:
    relationship: dict[FamilyRelationship, list[FamilyHistoryEvent]] = defaultdict(list)
    denials: dict[FamilyRelationship, list[FamilyHistoryEvent]] = defaultdict(list)
    for item in family_history:
        if item.status is FamilyHistoryStatus.PRESENT:
            relationship[item.relationship].append(item)
        else:
            denials[item.relationship].append(item)

    output = ["## Family History"]
    for r in relationship:
        output.append(f"### {r}")
        for item in relationship[r]:
            item_str = f"- **{item.condition}**"
            if item.detail:
                item_str += f" - {item.detail}"
            output.append(item_str)

    output.append("### Denials")
    for r in denials:
        for item in denials[r]:
            item_str = f"- {item.condition}"
            if item.relationship is not FamilyRelationship.UNSPECIFIED:
                item_str += f" ({item.relationship.lower()})"
            if item.detail is not None:
                item_str += f" - {item.detail}"
            output.append(item_str)

    return "\n\n".join(output)
