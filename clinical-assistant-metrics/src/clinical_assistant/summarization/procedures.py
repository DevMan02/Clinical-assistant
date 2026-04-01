import asyncio
import logging
from collections import defaultdict
from enum import StrEnum
from typing import Optional

from pydantic import BaseModel

from clinical_assistant.summarization.encounter import EncounterDocument, EncounterMeta
from clinical_assistant.summarization.structured_output import AsyncClient
from clinical_assistant.summarization.utils import format_dates

logger = logging.getLogger(__name__)


class ProcedureCategory(StrEnum):
    INTERVENTIONAL = "interventional"
    DIAGNOSTIC = "diagnostic"


class ProcedureSignificance(StrEnum):
    MAJOR = "major"
    MINOR = "minor"


class Procedure(BaseModel):
    name: str
    category: ProcedureCategory
    significance: ProcedureSignificance
    detail: Optional[str]


class EncounterProcedures(BaseModel):
    encounter: EncounterMeta
    procedures: list[Procedure]


class ProcedureOutputFormat(BaseModel):
    procedures: list[Procedure]


PROCEDURE_EXTRACTION_PROMPT_TEMPLATE = """
You are a clinical data extraction system for a cardiology-focused Clinical Decision Support System (CDSS).

## Task

You receive a **clinical document** from a single encounter (e.g. discharge summary, consultation report, outpatient visit note).

Extract all procedures mentioned in the document.

## Core Principles

- Extract, Do not Interpret: only extract information that is explicitly stated in the document.
- Stay close to the source language: preserve medical terminology as written in the document.

## Extraction Rules

### General
1. Extract every procedure mentioned in the document, including those referenced as historical (e.g. "prior CABG in 2019").
2. Each distinct occurrence of a procedure is a separate entry. If two echocardiograms were performed on different days, extract both.
3. Do not extract medications, diagnoses, or symptoms as procedures.

### Name
- Use the procedure name as written in the document.
- Include relevant anatomical or technical specifics when stated (e.g. "left heart catheterization", "transthoracic echocardiogram", "laparoscopic cholecystectomy", not just "catheterization", "echo", or "cholecystectomy").

### Category
- Classify each procedure by intent:
  - INTERVENTIONAL: the procedure changes the patient — surgery, catheterization with intervention, device implant, ablation, chest tube placement, cardioversion, chemotherapy, dialysis, transfusion, percutaneous drain placement.
  - DIAGNOSTIC: the procedure gathers information — imaging, ECG, echocardiogram, stress test, Holter monitor, blood draw, biopsy, diagnostic catheterization without intervention, pulmonary function test.
- When a procedure is both diagnostic and interventional (e.g. cardiac catheterization with PCI), classify as INTERVENTIONAL.

### Significance
- Classify each procedure:
  - MAJOR: requires anesthesia, sedation, sterile/operative setting, or formal informed consent. Carries meaningful procedural risk. Examples: surgery, catheterization, device implant, ablation, biopsy, bronchoscopy, lumbar puncture, chest tube, cardioversion, dialysis initiation, chemotherapy administration.
  - MINOR: routine, low-risk, performed at bedside or in standard clinical settings without sedation or formal consent. Examples: ECG, chest X-ray, CT scan, MRI, echocardiogram, blood draw, urinalysis, vital signs measurement, Holter monitor, stress test.

### Detail
- Write like a clinician's shorthand note, not a narrative summary.
- Be maximally concise: capture the clinical fact, drop the filler.
- Use semicolons to separate distinct facts.
- Include findings or results when stated (e.g. "EF 35%, moderate MR", "90% LAD stenosis", "no fracture").
- Include relevant clinical context when notable (e.g. "emergent for STEMI", "pre-op clearance", "INR reversed with FFP pre-procedure").
- Do not explain your reasoning or justify your classifications.
- Leave empty if there is nothing notable beyond the procedure name.

Good examples:
  - "EF 35%; moderate MR; mild pulmonary hypertension"
  - "90% LAD stenosis; PCI with DES to LAD"
  - "displaced posterior R 10th/11th rib fractures; diminutive R pneumothorax"
  - "no acute intracranial process; acute paranasal sinusitis"
  - "uncomplicated; drain removed POD1"

Bad examples (too verbose):
  - "Echocardiogram was performed during this admission and showed an ejection fraction of 35% with moderate mitral regurgitation and mild pulmonary hypertension."
  - "CT scan of the torso was obtained which demonstrated displaced posterior right tenth and eleventh rib fractures."

Better versions:
  - "EF 35%; moderate MR; mild pulmonary hypertension"
  - "displaced posterior R 10th/11th rib fractures; diminutive R pneumothorax"

---

## Input Data

<encounter_document>
{document}
</encounter_document>
"""


class ProcedureExtractor:
    def __init__(
        self,
        client: AsyncClient,
        prompt_template: str = PROCEDURE_EXTRACTION_PROMPT_TEMPLATE,
        max_concurrency: int = 5,
    ):
        self.client = client
        self.prompt_template = prompt_template
        self._sem = asyncio.Semaphore(max_concurrency)

    async def extract_single_raw(self, document: EncounterDocument) -> ProcedureOutputFormat | None:
        prompt = self.prompt_template.format(document=document)
        return await self.client.structured_output(prompt, ProcedureOutputFormat)

    def update(self, delta: ProcedureOutputFormat | None, meta: EncounterMeta) -> EncounterProcedures:
        if delta is None:
            return EncounterProcedures(encounter=meta, procedures=[])

        return EncounterProcedures(encounter=meta, procedures=delta.procedures)

    async def extract_single(self, document: EncounterDocument) -> EncounterProcedures:
        response = await self.extract_single_raw(document)

        if response is None:
            logger.warning(f"Failed procedure extraction step for document {document.meta.id}")

        return self.update(response, document.meta)

    async def extract(self, documents: list[EncounterDocument]) -> list[EncounterProcedures]:
        return await asyncio.gather(self.extract_single(doc) for doc in documents)


def to_markdown(encounter_procedures: list[EncounterProcedures]) -> str:
    if not encounter_procedures:
        return ""

    output = ["## Procedures"]
    for encounter in reversed(encounter_procedures):
        groups: dict[ProcedureSignificance, dict[ProcedureCategory, list[Procedure]]] = defaultdict(
            lambda: defaultdict(list)
        )

        dates_str = format_dates(encounter.encounter.start_date, encounter.encounter.end_date)
        output.append(f"### {dates_str}")
        for p in encounter.procedures:
            groups[p.significance][p.category].append(p)

        for sig in [ProcedureSignificance.MAJOR, ProcedureSignificance.MINOR]:
            if sig not in groups:
                continue

            output.append(f"#### {sig.title()}")
            for cat in [ProcedureCategory.INTERVENTIONAL, ProcedureCategory.DIAGNOSTIC]:
                if cat not in groups[sig]:
                    continue

                output.append(f"##### {cat.title()}")
                for proc in groups[sig][cat]:
                    item_str = f"- **{proc.name}**"
                    if p.detail:
                        item_str += f" - {p.detail}"
                    output.append(item_str)

    return "\n\n".join(output)
