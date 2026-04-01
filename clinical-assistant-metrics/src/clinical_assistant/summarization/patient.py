import asyncio
from enum import StrEnum

from pydantic import BaseModel, Field

from clinical_assistant.summarization import (
    allergies,
    clinical_problems,
    family_history,
    measurements,
    medications,
    procedures,
    substances,
)
from clinical_assistant.summarization.allergies import Allergy, AllergyExtractor, AllergyOutputFormat
from clinical_assistant.summarization.clinical_problems import (
    ClinicalProblem,
    ClinicalProblemExtractor,
    ClinicalProblemOutputFormat,
)
from clinical_assistant.summarization.encounter import EncounterDocument, EncounterMeta
from clinical_assistant.summarization.family_history import (
    FamilyHistoryEvent,
    FamilyHistoryExtractor,
    FamilyHistoryOutputFormat,
)
from clinical_assistant.summarization.measurements import (
    Measurement,
    MeasurementEventOutputFormat,
    MeasurementExtractor,
)
from clinical_assistant.summarization.medications import (
    MedicationExtractor,
    MedicationHistory,
    MedicationHistoryOutputFormat,
)
from clinical_assistant.summarization.procedures import EncounterProcedures, ProcedureExtractor, ProcedureOutputFormat
from clinical_assistant.summarization.structured_output import AsyncClient
from clinical_assistant.summarization.substances import (
    SubstanceUseExtractor,
    SubstanceUseHistory,
    SubstanceUseOutputFormat,
)


class Sex(StrEnum):
    MALE = "male"
    FEMALE = "female"


class Patient(BaseModel):
    id: str
    sex: Sex
    age: int
    documents: list[EncounterDocument]


class PatientSummary(BaseModel):
    patient_id: str
    document_ids: list[str]
    sex: Sex
    age: int
    allergies: list[Allergy] = Field(default_factory=list)
    substances: list[SubstanceUseHistory] = Field(default_factory=list)
    family_history: list[FamilyHistoryEvent] = Field(default_factory=list)
    clinical_problems: list[ClinicalProblem] = Field(default_factory=list)
    procedures: list[EncounterProcedures] = Field(default_factory=list)
    medications: list[MedicationHistory] = Field(default_factory=list)
    measurements: list[Measurement] = Field(default_factory=list)

    def to_markdown(self) -> str:
        output = [
            "# Patient Summary",
            f"**Sex** {self.sex}",
            f"**Age** {self.age} years old",
            allergies.to_markdown(self.allergies),
            substances.to_markdown(self.substances),
            family_history.to_markdown(self.family_history),
            clinical_problems.to_markdown(self.clinical_problems),
            procedures.to_markdown(self.procedures),
            medications.to_markdown(self.medications),
            measurements.to_markdown(self.measurements),
        ]
        return "\n\n".join(output)


class PatientSummaryDelta(BaseModel):
    meta: EncounterMeta
    allergies: AllergyOutputFormat
    substances: SubstanceUseOutputFormat
    family_history: FamilyHistoryOutputFormat
    clinical_problems: ClinicalProblemOutputFormat
    procedures: ProcedureOutputFormat
    medications: MedicationHistoryOutputFormat
    measurements: MeasurementEventOutputFormat


class PatientSummaryExtractor:
    def __init__(self, client: AsyncClient):
        self.client = client
        self.allergy_extractor = AllergyExtractor(client)
        self.substance_use_extractor = SubstanceUseExtractor(client)
        self.family_history_extractor = FamilyHistoryExtractor(client)
        self.clinical_problem_extractor = ClinicalProblemExtractor(client)
        self.procedure_extractor = ProcedureExtractor(client)
        self.medication_extractor = MedicationExtractor(client)
        self.measurement_extractor = MeasurementExtractor(client)

    async def extract_single_raw(self, document: EncounterDocument, summary: PatientSummary) -> PatientSummaryDelta:
        (
            allergies,
            substances,
            family_history,
            clinical_problems,
            procedures,
            medications,
            measurements,
        ) = await asyncio.gather(
            self.allergy_extractor.extract_single_raw(document, summary.allergies),
            self.substance_use_extractor.extract_single_raw(document, summary.substances),
            self.family_history_extractor.extract_single_raw(document, summary.family_history),
            self.clinical_problem_extractor.extract_single_raw(document, summary.clinical_problems),
            self.procedure_extractor.extract_single_raw(document),
            self.medication_extractor.extract_single_raw(document, summary.medications),
            self.measurement_extractor.extract_single_raw(document, summary.measurements),
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

    def update(self, summary: PatientSummary, delta: PatientSummaryDelta) -> PatientSummary:
        return PatientSummary(
            patient_id=summary.patient_id,
            document_ids=summary.document_ids + [delta.meta.id],
            sex=summary.sex,
            age=summary.age,
            allergies=self.allergy_extractor.update(summary.allergies, delta.allergies),
            substances=self.substance_use_extractor.update(summary.substances, delta.substances, delta.meta),
            family_history=self.family_history_extractor.update(summary.family_history, delta.family_history),
            clinical_problems=self.clinical_problem_extractor.update(
                summary.clinical_problems, delta.clinical_problems, delta.meta
            ),
            procedures=summary.procedures + [self.procedure_extractor.update(delta.procedures, delta.meta)],
            medications=self.medication_extractor.update(summary.medications, delta.medications, delta.meta),
            measurements=self.measurement_extractor.update(summary.measurements, delta.measurements, delta.meta),
        )

    async def extract_single(self, document: EncounterDocument, summary: PatientSummary) -> PatientSummary:
        (
            allergies,
            substances,
            family_history,
            clinical_problems,
            procedures,
            medications,
            measurements,
        ) = await asyncio.gather(
            self.allergy_extractor.extract_single(document, summary.allergies),
            self.substance_use_extractor.extract_single(document, summary.substances),
            self.family_history_extractor.extract_single(document, summary.family_history),
            self.clinical_problem_extractor.extract_single(document, summary.clinical_problems),
            self.procedure_extractor.extract_single(document),
            self.medication_extractor.extract_single(document, summary.medications),
            self.measurement_extractor.extract_single(document, summary.measurements),
        )
        return PatientSummary(
            patient_id=summary.patient_id,
            document_ids=summary.document_ids + [document.meta.id],
            sex=summary.sex,
            age=summary.age,
            allergies=allergies,
            substances=substances,
            family_history=family_history,
            clinical_problems=clinical_problems,
            procedures=procedures,
            medications=medications,
            measurements=measurements,
        )

    async def extract(self, patient: Patient) -> PatientSummary:
        (
            allergies,
            substances,
            family_history,
            clinical_problems,
            procedures,
            medications,
            measurements,
        ) = await asyncio.gather(
            self.allergy_extractor.extract(patient.documents),
            self.substance_use_extractor.extract(patient.documents),
            self.family_history_extractor.extract(patient.documents),
            self.clinical_problem_extractor.extract(patient.documents),
            self.procedure_extractor.extract(patient.documents),
            self.medication_extractor.extract(patient.documents),
            self.measurement_extractor.extract(patient.documents),
        )

        return PatientSummary(
            patient_id=patient.id,
            document_ids=[d.meta.id for d in patient.documents],
            sex=patient.sex,
            age=patient.age,
            allergies=allergies,
            substances=substances,
            family_history=family_history,
            clinical_problems=clinical_problems,
            procedures=procedures,
            medications=medications,
            measurements=measurements,
        )
