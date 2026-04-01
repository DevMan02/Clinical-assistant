import asyncio
import json
import pathlib
from datetime import date
from typing import Literal, Optional

import pandas as pd
import yaml
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel

from clinical_assistant.summarization.encounter import EncounterDocument, EncounterMeta
from clinical_assistant.summarization.patient import Patient, PatientSummary, PatientSummaryExtractor
from clinical_assistant.summarization.structured_output import ClaudeClient, OpenAIClient


class Config(BaseModel):
    provider: Literal["anthropic", "openai", "local"]
    model: str
    num_patients: int
    max_output_tokens: int
    local_model_url: Optional[str] = None


if __name__ == "__main__":
    load_dotenv()

    with open("./config/summarization/dataset.yaml") as f:
        cfg = Config(**yaml.safe_load(f))

    output_dir = pathlib.Path("outputs/summarization/datasets")
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    output_path = output_dir / f"{cfg.model.replace('/', '-').replace('.', '-')}.jsonl"

    patients_df = pd.read_csv("data/mimiciv/2.2/hosp/patients.csv.gz")
    discharges_df = pd.read_csv("data/mimiciv/2.2/hosp/discharge.csv.gz")
    admissions = pd.read_csv("data/mimiciv/2.2/hosp/admissions.csv.gz")

    patients: list[Patient] = []
    processed_patients = set()
    if output_path.exists():
        with output_path.open() as f:
            for line in f:
                patient = json.loads(line)
                processed_patients.add(patient["summary"]["patient_id"])

    for _, patient_row in patients_df.iterrows():
        if str(patient_row["subject_id"]) in processed_patients:
            continue

        if len(patients) < cfg.num_patients:
            patient_discharges: pd.DataFrame = discharges_df[
                discharges_df["subject_id"] == patient_row["subject_id"]
            ].sort_values("storetime")
            documents = []
            for _, doc in patient_discharges.iterrows():
                hadm_id = doc["hadm_id"]
                hadm_info = admissions[admissions["hadm_id"] == hadm_id].iloc[0].to_dict()
                year, month, day = map(int, hadm_info["admittime"][:10].split("-"))
                start_date = date(year, month, day)
                year, month, day = map(int, hadm_info["dischtime"][:10].split("-"))
                end_date = date(year, month, day)
                meta = EncounterMeta(
                    id=str(hadm_info["hadm_id"]), type="Inpatient", start_date=start_date, end_date=end_date
                )
                document = EncounterDocument(content=doc["text"], meta=meta)
                documents.append(document)

            if documents:
                patients.append(
                    Patient(
                        id=str(patient_row["subject_id"]),
                        sex={"M": "male", "F": "female"}[patient_row["gender"]],
                        age=patient_row["anchor_age"],
                        documents=documents,
                    )
                )
        else:
            break

    if not patients:
        print("No more patients to process")
        exit(1)

    if cfg.provider in ["openai", "local"]:
        client = OpenAIClient(
            AsyncOpenAI(
                api_key="EMPTY" if cfg.provider == "local" else None,
                base_url=cfg.local_model_url if cfg.provider == "local" else None,
            ),
            model=cfg.model,
            max_output_tokens=cfg.max_output_tokens,
        )
    elif cfg.provider == "anthropic":
        client = ClaudeClient(AsyncAnthropic(), model=cfg.model, max_output_tokens=cfg.max_output_tokens)
    else:
        raise ValueError(f"Unknown provider {cfg.provider}")
    summary_extractor = PatientSummaryExtractor(client)

    for patient in patients:
        print(f"Processing patient {patient.id}")
        summary = PatientSummary(patient_id=patient.id, document_ids=[], sex=patient.sex, age=patient.age)
        for document in patient.documents:
            print(f"  Processing document {document.meta.id}")
            is_extracted = False
            attempt = 1
            while not is_extracted:
                try:
                    summary_delta = asyncio.run(summary_extractor.extract_single_raw(document, summary))
                    is_extracted = True
                except Exception:
                    print(f"Failed extraction. Attempt {attempt}")
                    attempt += 1
            with output_path.open("a") as f:
                to_save = {
                    "summary": summary.model_dump(),
                    "document": document.model_dump(),
                    "summary_delta": summary_delta.model_dump(),
                }
                f.write(json.dumps(to_save, default=str) + "\n")
            summary = summary_extractor.update(summary, summary_delta)
