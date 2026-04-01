import logging
from collections import defaultdict
from copy import deepcopy
from enum import StrEnum
from typing import Optional

from pydantic import BaseModel

from clinical_assistant.summarization.encounter import EncounterDocument
from clinical_assistant.summarization.structured_output import AsyncClient

logger = logging.getLogger(__name__)


class AllergyCategory(StrEnum):
    DRUG = "drug"
    FOOD = "food"
    ENVIRONMENT = "environment"
    LATEX = "latex"
    INSECT_VENOM = "insect_venom"
    VACCINE = "vaccine"


class AllergySeverity(StrEnum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class Allergy(BaseModel):
    allergen: str
    category: AllergyCategory
    severity: Optional[AllergySeverity]
    reaction: Optional[str]


class AllergyOutputFormat(BaseModel):
    allergies: list[Allergy]


ALLERGIES_EXTRACTION_PROMPT_TEMPLATE = """
You are a clinical data extraction system for a Clinical Decision Support System (CDSS).

## Task

You receive:
1. An existing **Allergy list** (JSON). On the first encounter this will be empty.
2. A **clinical document** from a single encounter (e.g. discharge summary, consultation report, outpatient visit note).

Your job is to return the complete updated **Allergy list** incorporating any new information from the encounter document.

## Core Principles

- Extract, Do not Interpret: only extract information that is explicitly stated in the document.
- Stay close to the source language: preserve medical terminology as written in the document.
- Never downgrade specificity: do not replace existing detailed information with vaguer information from a new document.

## Extraction Rules

### General
1. Return the complete allergy list — both existing entries and any new ones.
2. If an allergen in the document matches an existing entry (same substance, including abbreviations like "PCN" for "penicillin"), update that entry rather than creating a duplicate.
3. If an existing entry has no new information in the current document, copy it through unchanged.
4. If the document states "No Known Allergies" or "NKDA" and the existing list is empty, return an empty list.

### Allergen Name
- Prefer the full generic name over abbreviations (e.g. "penicillin" not "PCN", "aspirin" not "ASA").
- If the document uses an abbreviation and the existing list has the full name, keep the full name.

### Category
- Classify each allergen:
  - DRUG: medication allergies (penicillin, sulfa, aspirin, contrast dye)
  - FOOD: food allergies (shellfish, peanuts, eggs)
  - ENVIRONMENT: environmental allergens (pollen, dust, mold, pet dander)
  - LATEX: latex allergy
  - INSECT_VENOM: bee stings, wasp stings, insect bites
  - VACCINE: vaccine-related allergic reactions

### Severity
- Classify when explicitly stated or clearly inferable from the documented reaction:
  - MILD: localized reaction, single organ system (e.g. rash, itching, nausea)
  - MODERATE: systemic reaction, multiple organ systems (e.g. hives with GI symptoms)
  - SEVERE: life-threatening reaction (e.g. anaphylaxis, angioedema, laryngeal edema)
- Never downgrade an existing severity. If a prior entry says "severe" and the current document says "allergy" with no further detail, keep "severe".
- Leave null if no severity information is available.

### Reaction
- Extract the specific reaction when stated (e.g. "rash", "anaphylaxis", "hives", "GI upset").
- Never replace a specific existing reaction with a vaguer one. If a prior entry says "anaphylaxis" and the current document just says "allergy", keep "anaphylaxis".
- Leave null if no reaction is documented.

---

## Input Data

<allergies>
{allergies}
</allergies>

<encounter_document>
{document}
</encounter_document>
"""


class AllergyExtractor:
    def __init__(self, client: AsyncClient, prompt_template: str = ALLERGIES_EXTRACTION_PROMPT_TEMPLATE):
        self.client = client
        self.prompt_template = prompt_template

    async def extract_single_raw(
        self, document: EncounterDocument, allergies: list[Allergy]
    ) -> AllergyOutputFormat | None:
        prompt = self.prompt_template.format(document=document.content, allergies=allergies)
        return await self.client.structured_output(prompt, AllergyOutputFormat)

    async def extract_single(self, document: EncounterDocument, allergies: list[Allergy]) -> list[Allergy]:
        response = await self.extract_single_raw(document, allergies)

        if response is None:
            logger.warning(f"Failed allergy extraction step for document {document.meta.id}")

        return self.update(allergies, response)

    def update(self, allergies: list[Allergy], delta: AllergyOutputFormat | None) -> list[Allergy]:
        if delta is None:
            return deepcopy(allergies)

        return delta.allergies

    async def extract(self, documents: list[EncounterDocument]) -> list[Allergy]:
        allergies = []
        for doc in documents:
            allergies = await self.extract_single(doc, allergies)
        return allergies


def to_markdown(allergies: list[Allergy]) -> str:
    output = ["## Allergies"]
    groups: dict[AllergyCategory, list[Allergy]] = defaultdict(list)
    for a in allergies:
        groups[a.category].append(a)

    for cat in groups:
        output.append(f"### {cat.replace('_', ' ').title()}")
        for a in groups[cat]:
            a_str = f"- {a.allergen}"
            if a.severity:
                a_str += f" - {a.severity}"
            if a.reaction:
                a_str += f" - {a.reaction}"
            output.append(a_str)

    return "\n\n".join(output)
