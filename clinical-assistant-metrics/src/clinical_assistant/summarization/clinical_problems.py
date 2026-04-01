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


class ClinicalProblemCategory(StrEnum):
    CARDIOVASCULAR = "cardiovascular"
    RESPIRATORY = "respiratory"
    RENAL = "renal"
    ENDOCRINE_METABOLIC = "endocrine_metabolic"
    NEUROLOGICAL_PSYCHIATRIC = "neurological_psychiatric"
    GASTROINTESTINAL = "gastrointestinal"
    HEMATOLOGICAL = "hematological"
    MUSCULOSKELETAL = "musculoskeletal"
    INFECTIOUS = "infectious"
    ONCOLOGICAL = "oncological"
    VASCULAR = "vascular"
    OTHER = "other"


class ClinicalProblemStatus(StrEnum):
    ACTIVE = "active"
    RESOLVED = "resolved"
    CHRONIC = "chronic"
    UNKNOWN = "unknown"


class ClinicalProblemEvent(BaseModel):
    name: str
    category: ClinicalProblemCategory
    status: ClinicalProblemStatus
    detail: Optional[str]
    new_info: bool


class ClinicalProblemOutputFormat(BaseModel):
    problems: list[ClinicalProblemEvent]


class ClinicalProblem(BaseModel):
    name: str
    events: list[tuple[EncounterMeta, ClinicalProblemEvent]]


CLINICAL_PROBLEM_HISTORY_PROMPT_TEMPLATE = """
You are a clinical data extraction system for a Clinical Decision Support System (CDSS).

## Task

You receive:
1. An existing **Clinical Problem list** (JSON). On the first encounter this will be empty.
2. A **clinical document** from a single encounter (e.g. discharge summary, consultation report, outpatient visit note).

Your job is to update the **Clinical Problem list** incorporating any new information from the encounter document.

## Core Principles

- Extract, Do not Interpret: only extract information that is explicitly stated in the document.
- Stay close to the source language: preserve medical terminology as written in the document.
- Minimal updates: only modify a problem entry when the document provides new or changed information.

## Extraction Rules

### General
1. Extract every clinical problem, diagnosis, and condition mentioned in the document.
2. If a problem already exists in the list with the same name (or a clearly equivalent name), update it rather than creating a duplicate.
3. Only update a problem if there is genuinely new information about it.
4. Include both active problems and past medical history items.
5. Do not extract symptoms as standalone problems unless the document treats them as a diagnosis (e.g. "dyspnea" as a symptom of CHF is not a separate problem; "chronic dyspnea under investigation" is).

### Name
- Use the condition name as written in the document.
- Prefer full diagnostic terms over abbreviations (e.g. "atrial fibrillation" not "Afib", "coronary artery disease" not "CAD"), but preserve clinical terminology.
- If both a specific and general name appear (e.g. "HFrEF" and "heart failure"), use the more specific one.
- If a problem already exists in the list under a specific name, reuse that name exactly.

### Category
- Classify each problem:
  - CARDIOVASCULAR: heart failure, CAD, arrhythmias, valvular disease, hypertension, pericardial disease
  - RESPIRATORY: COPD, asthma, pneumonia, pulmonary embolism, pleural effusion
  - RENAL: AKI, CKD, electrolyte disorders, nephrotic syndrome
  - ENDOCRINE_METABOLIC: diabetes, thyroid disorders, adrenal disorders, obesity, metabolic syndrome
  - NEUROLOGICAL_PSYCHIATRIC: stroke, seizure, dementia, depression, anxiety, delirium
  - GASTROINTESTINAL: GERD, GI bleeding, liver disease, pancreatitis, cholecystitis, IBD
  - HEMATOLOGICAL: anemia, coagulopathy, DVT, PE, thrombocytopenia
  - MUSCULOSKELETAL: fractures, arthritis, osteoporosis, back pain
  - INFECTIOUS: pneumonia, UTI, sepsis, endocarditis, cellulitis
  - ONCOLOGICAL: any malignancy, tumor, or cancer-related condition
  - VASCULAR: peripheral artery disease, aortic aneurysm, carotid stenosis
  - OTHER: anything that does not fit the above categories

- When a problem fits multiple categories (e.g. pulmonary embolism could be respiratory or hematological), prefer the category most relevant to its clinical management. PE is primarily hematological, pneumonia is primarily infectious even though it's respiratory.

### Status
- Classify based on what the document explicitly states:
  - ACTIVE: the problem is the reason for this encounter or is being acutely managed (e.g. "admitted for CHF exacerbation", "started antibiotics for pneumonia")
  - CHRONIC: ongoing long-term condition listed as background context, not acutely managed this encounter (e.g. "history of CHF with reduced EF", "type 2 diabetes on metformin")
  - RESOLVED: explicitly stated as resolved, cured, or no longer present (e.g. "AKI resolved", "pneumonia cleared")
  - UNKNOWN: mentioned but status unclear
- If a problem was previously ACTIVE and this document does not mention it, do not change its status. Copy it through unchanged.
- If a problem was previously ACTIVE and this document describes it as resolved, update to RESOLVED.
- A problem can transition: ACTIVE → CHRONIC (e.g. acute HF stabilized, now managed long-term), ACTIVE → RESOLVED, CHRONIC → RESOLVED.

### Detail
- Write like a clinician's shorthand note, not a narrative summary.
- Be maximally concise: capture the clinical fact, drop the filler.
- Use semicolons to separate distinct facts.
- Do not explain your reasoning or justify your classification choices.

Good examples:
  - "EF 35%; on guideline-directed medical therapy"
  - "paroxysmal; rate-controlled on metoprolol"
  - "resolved after IV fluids; Cr back to baseline 1.0"
  - "90% LAD stenosis; PCI with DES"
  - "new diagnosis this admission"

Bad examples (too verbose):
  - "Patient was admitted with acute decompensated heart failure and was found to have an ejection fraction of 35% on echocardiogram performed during this hospitalization."
  - "The patient has a long-standing history of atrial fibrillation which has been managed with rate control strategy."

Better versions:
  - "acute decompensation; EF 35% on echo"
  - "long-standing; rate-controlled"
  
### New Information Flag
- Set new_info to true when this encounter documents any of:
  - First time the problem appears
  - Status change (e.g. active → resolved, active → chronic)
  - New findings, test results, or imaging related to the problem
  - New treatment or intervention for the problem
  - Complications or progression
  - Change in severity or clinical characterization
- Set new_info to false when:
  - The problem is merely listed in past medical history without elaboration
  - The problem is carried forward unchanged from prior encounters
  - The only information is "continued on current management"
  - The detail would be identical or nearly identical to a prior event
- When new_info is false:
  - Still set the correct current status
  - Leave detail empty

---

## Input Data

<clinical_problems>
{clinical_problems}
</clinical_problems>

<encounter_document>
{document}
</encounter_document>
"""


class ClinicalProblemExtractor:
    def __init__(
        self,
        client: AsyncClient,
        prompt_template: str = CLINICAL_PROBLEM_HISTORY_PROMPT_TEMPLATE,
    ):
        self.client = client
        self.prompt_template = prompt_template

    async def extract_single_raw(
        self, document: EncounterDocument, clinical_problems: list[ClinicalProblem]
    ) -> ClinicalProblemOutputFormat | None:
        prompt = self.prompt_template.format(document=document, clinical_problems=clinical_problems)
        return await self.client.structured_output(
            prompt,
            output_format=ClinicalProblemOutputFormat,
        )

    def update(
        self, clinical_problems: list[ClinicalProblem], delta: ClinicalProblemOutputFormat | None, meta: EncounterMeta
    ) -> list[ClinicalProblem]:
        clinical_problems = deepcopy(clinical_problems)

        if delta is None:
            return clinical_problems

        index = {cp.name: cp for cp in clinical_problems}
        for e in delta.problems:
            if e.name in index:
                index[e.name].events.append((meta, e))
            else:
                new_problem = ClinicalProblem(name=e.name, events=[(meta, e)])
                index[e.name] = new_problem
                clinical_problems.append(new_problem)
        return clinical_problems

    async def extract_single(
        self, document: EncounterDocument, clinical_problems: list[ClinicalProblem]
    ) -> list[ClinicalProblem]:
        response = await self.extract_single_raw(document, clinical_problems)

        if response is None:
            logger.warning(f"Failed clinical problem extraction step for document {document.meta.id}")

        return self.update(clinical_problems, response, document.meta)

    async def extract(self, documents: list[EncounterDocument]) -> list[ClinicalProblem]:
        clinical_problems = []
        for doc in documents:
            clinical_problems = await self.extract_single(doc, clinical_problems)
        return clinical_problems


def to_markdown(clinical_problems: list[ClinicalProblem]) -> str:
    groups: dict[ClinicalProblemStatus, dict[ClinicalProblemCategory, list[ClinicalProblem]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for p in clinical_problems:
        last = p.events[-1][1]
        groups[last.status][last.category].append(p)

    output = ["## Clinical Problems"]
    for status in [
        ClinicalProblemStatus.ACTIVE,
        ClinicalProblemStatus.CHRONIC,
        ClinicalProblemStatus.RESOLVED,
        ClinicalProblemStatus.UNKNOWN,
    ]:
        if status not in groups:
            continue

        output.append(f"### {status.title()}")
        for category, problems in groups[status].items():
            output.append(f"#### {category.replace('_', ' ').title()}")
            for p in problems:
                item_str = f"- **{p.name.title()}**"
                history = ["<details>", "<summary>See History</summary>\n"]
                for meta, e in p.events:
                    if e.new_info:
                        date_str = format_dates(meta.start_date, meta.end_date)
                        history.append(f"    - {date_str} - {e.detail}")
                history.append("</details>")
                if len(history) > 3:
                    output.append("\n".join([item_str] + history))
                else:
                    output.append(item_str)

    return "\n\n".join(output)
