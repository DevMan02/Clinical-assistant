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


class MeasurementCategory(StrEnum):
    VITAL_SIGNS = "Vital Signs"
    BLOOD = "Blood"
    URINE = "Urine"
    BODY_FLUID = "Body Fluids"
    OTHER = "Other"


class MeasurementFlag(StrEnum):
    HIGH = "high"
    LOW = "low"
    CRITICAL = "critical"


class MeasurementEvent(BaseModel):
    name: str
    category: MeasurementCategory
    value: float
    unit: str
    flag: Optional[MeasurementFlag]


class MeasurementEventOutputFormat(BaseModel):
    measurements: list[MeasurementEvent]


class Measurement(BaseModel):
    name: str
    data_points: list[tuple[EncounterMeta, MeasurementEvent]]


MEASUREMENT_EXTRACTION_PROMPT_TEMPLATE = """
You are a clinical data extraction system for a Clinical Decision Support System (CDSS).

## Task

You receive:
1. An existing **Measurement list** (JSON). This contains measurement names already extracted from prior encounters. On the first encounter this will be empty.
2. A **clinical document** from a single encounter (e.g. discharge summary, consultation report, outpatient visit note).

Extract all vital signs and laboratory values mentioned in the document.

## Core Principles

- Extract, Do not Interpret: only extract values that are explicitly stated in the document.
- Do not calculate or derive values that are not directly stated.
- Do not infer normal/abnormal flags unless the document explicitly states them.
- Name consistency: if a measurement already exists in the measurement list under a specific name, use that same name rather than introducing a synonym or abbreviation.

## Extraction Rules

### General
1. Extract every numeric vital sign and lab value mentioned in the document.
2. For measurements with multiple readings, extract the most clinically relevant:
   - The most recent or discharge value
   - Peak or trough values when explicitly noted (e.g. "Tmax 39.2", "Cr peaked at 2.1")
   - Admission or baseline values when explicitly noted
3. Do not extract qualitative results (e.g. "blood culture positive", "urine cloudy"). Only extract numeric values with units.
4. If the same measurement appears in multiple contexts (e.g. "Cr 1.5 on admission" and "Cr 1.0 at discharge"), extract both as separate entries.

### Name
- Before assigning a name, check the existing measurement list. If the measurement already exists under a standardized name, reuse that name exactly.
- If the measurement is new, prefer the full standard name over abbreviations (e.g. "creatinine" not "Cr", "hemoglobin" not "Hgb").
- Common aliases to normalize:
  - Cr, creat → creatinine
  - Hgb, Hb → hemoglobin
  - plt → platelets
  - WBC → white blood cells
  - K → potassium
  - Na → sodium
  - Tmax, T → temperature
  - HR, pulse → heart rate
  - BP → blood pressure

### Category
- Classify each measurement:
  - VITAL_SIGNS: heart rate, blood pressure, temperature, respiratory rate, SpO2, weight, BMI
  - BLOOD: any lab value from a blood draw (CBC, BMP, LFTs, cardiac biomarkers, coagulation, lipids, thyroid, etc.)
  - URINE: urinalysis values (specific gravity, pH, protein, glucose)
  - BODY_FLUID: CSF, pleural fluid, peritoneal fluid, synovial fluid analysis values
  - OTHER: anything that does not fit the above

### Value and Unit
- Extract the numeric value and its unit separately.
- Use the unit as written in the document (e.g. "mg/dL", "mmol/L", "bpm", "mmHg").
- For blood pressure, extract systolic and diastolic as separate entries.

### Flag
- Only flag a value if the document explicitly marks it as abnormal:
  - HIGH: explicitly noted as elevated, high, or above normal
  - LOW: explicitly noted as low, decreased, or below normal
  - CRITICAL: explicitly noted as critical, panic, or dangerous
- Leave null if the document does not explicitly comment on the value being abnormal.

---

## Input Data

<measurements>
{measurements}
</measurements>

<encounter_document>
{document}
</encounter_document>
"""


class MeasurementExtractor:
    def __init__(
        self,
        client: AsyncClient,
        prompt_template: str = MEASUREMENT_EXTRACTION_PROMPT_TEMPLATE,
    ):
        self.client = client
        self.prompt_template = prompt_template

    async def extract_single_raw(
        self, document: EncounterDocument, measurements: list[Measurement]
    ) -> MeasurementEventOutputFormat | None:
        prompt = self.prompt_template.format(document=document, measurements=measurements)
        return await self.client.structured_output(
            prompt,
            output_format=MeasurementEventOutputFormat,
        )

    def update(
        self, measurements: list[Measurement], delta: MeasurementEventOutputFormat | None, meta: EncounterMeta
    ) -> list[Measurement]:
        measurements = deepcopy(measurements)

        if delta is None:
            return measurements

        index = {m.name: m for m in measurements}
        for m in delta.measurements:
            if m.name in index:
                index[m.name].data_points.append((meta, m))
            else:
                new_measurement = Measurement(name=m.name, data_points=[(meta, m)])
                index[m.name] = new_measurement
                measurements.append(new_measurement)
        return measurements

    async def extract_single(self, document: EncounterDocument, measurements: list[Measurement]) -> list[Measurement]:
        response = await self.extract_single_raw(document, measurements)

        if response is None:
            logger.warning(f"Failed measurement extraction step for document {document.meta.id}")

        return self.update(measurements, response, document.meta)

    async def extract(self, documents: list[EncounterDocument]) -> list[Measurement]:
        measurements = []
        for doc in documents:
            measurements = await self.extract_single(doc, measurements)
        return measurements


def to_markdown(measurements: list[Measurement]) -> str:
    # visualization idea is to have a table of measurements by category with most recent value and expand history on click
    category2meas: dict[MeasurementCategory, list[Measurement]] = defaultdict(list)
    for m in measurements:
        category2meas[m.data_points[-1][1].category].append(m)

    output = ["## Labs & Vitals"]
    for category, cat_measurements in category2meas.items():
        output.append(f"### {category.replace('_', ' ').title()}")
        header = """|Name|Value|Trend|Flag|Date|
|:---:|:---:|:---:|:---:|:---:|"""
        rows = [header]
        for m in cat_measurements:
            meta, last = m.data_points[-1]
            date_str = format_dates(meta.start_date, meta.end_date)
            item_str = f"|{m.name.title()}|{last.value}"
            item_str += f" {last.unit}|" if last.unit else "|"
            if len(m.data_points) > 1:
                _, prev = m.data_points[-2]
                if last.value - prev.value > 0:
                    trend = "\u2191"  # up arrow
                elif last.value == prev.value:
                    trend = "="
                else:
                    trend = "\u2193"  # down arrow
            else:
                trend = ""
            item_str += f"{trend}|"
            item_str += f" {last.flag}|" if last.flag else " |"
            item_str += f"{meta.end_date.day} {date_str}|"
            rows.append(item_str)
        output.append("\n".join(rows))
    return "\n\n".join(output)
