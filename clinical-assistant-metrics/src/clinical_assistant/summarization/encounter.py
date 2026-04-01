from datetime import date
from enum import StrEnum
from typing import Optional

from pydantic import BaseModel


class EncounterType(StrEnum):
    INPATIENT = "Inpatient"
    OUTPATIENT = "Outpatient"
    EMERGENCY = "Emergency"
    PROCEDURE = "Procedure"
    IMAGING = "Imaging"
    LABORATORY = "Laboratory"
    TELEHEALTH = "Telehealth"
    REHABILITATION = "Rehabilitation"
    OTHER = "Other"


class EncounterMeta(BaseModel):
    id: str
    type: EncounterType
    start_date: Optional[date]
    end_date: Optional[date]


class EncounterDocument(BaseModel):
    content: str
    meta: EncounterMeta
