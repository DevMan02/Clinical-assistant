from clinical_assistant.summarization.patient import PatientSummary
from clinical_assistant.summarization.medications import MedicationHistory, MedicationEvent, MedicationHistoryOutputFormat
from clinical_assistant.summarization.substances import SubstanceUseHistory, SubstanceUseEvent, SubstanceUseOutputFormat
from clinical_assistant.summarization.clinical_problems import ClinicalProblem, ClinicalProblemEvent, ClinicalProblemOutputFormat
from clinical_assistant.summarization.measurements import Measurement, MeasurementEvent, MeasurementEventOutputFormat
from clinical_assistant.summarization.procedures import EncounterProcedures, Procedure, ProcedureOutputFormat
from clinical_assistant.summarization.allergies import Allergy, AllergyOutputFormat
from clinical_assistant.summarization.family_history import FamilyHistoryEvent, FamilyHistoryOutputFormat
from clinical_assistant.summarization.encounter import EncounterMeta


def filter_summary_at_encounter(
    summary: PatientSummary,
    enc_ids: set[str],
) -> PatientSummary:
    """
    Returns a new PatientSummary containing only events/data from the given encounter ids.
    Allergies and family_history are not filterable by encounter id — they are excluded.
    """

    # --- Medications ---
    medications = [
        MedicationHistory(
            name=m.name,
            events=[(meta, e) for meta, e in m.events if meta.id in enc_ids]
        )
        for m in summary.medications
        if any(meta.id in enc_ids for meta, _ in m.events)
    ]

    # --- Substances ---
    substances = [
        SubstanceUseHistory(
            name=s.name,
            events=[(meta, e) for meta, e in s.events if meta.id in enc_ids]
        )
        for s in summary.substances
        if any(meta.id in enc_ids for meta, _ in s.events)
    ]

    # --- Clinical problems ---
    clinical_problems = [
        ClinicalProblem(
            name=p.name,
            events=[(meta, e) for meta, e in p.events if meta.id in enc_ids]
        )
        for p in summary.clinical_problems
        if any(meta.id in enc_ids for meta, _ in p.events)
    ]

    # --- Measurements ---
    measurements = [
        Measurement(
            name=m.name,
            data_points=[(meta, e) for meta, e in m.data_points if meta.id in enc_ids]
        )
        for m in summary.measurements
        if any(meta.id in enc_ids for meta, _ in m.data_points)
    ]

    # --- Procedures ---
    procedures = [
        ep for ep in summary.procedures
        if ep.encounter.id in enc_ids
    ]

    return PatientSummary(
        patient_id=summary.patient_id,
        document_ids=[id_ for id_ in summary.document_ids if id_ in enc_ids],
        sex=summary.sex,
        age=summary.age,
        allergies=summary.allergies,
        family_history=summary.family_history,
        clinical_problems=clinical_problems,
        substances=substances,
        procedures=procedures,
        medications=medications,
        measurements=measurements,
    )


def merge_summary(
    prior: PatientSummary,
    delta: PatientSummary,
) -> PatientSummary:
    """
    Merge a prior summary with a delta summary.
    The delta contains events from a single encounter — they are appended to the prior.
    Allergies and family_history come from the prior (delta has empty lists).
    """

    # --- Medications ---
    prior_med_map = {m.name: m for m in prior.medications}
    merged_meds = []
    delta_med_names = {m.name for m in delta.medications}

    for m in prior.medications:
        if m.name in delta_med_names:
            delta_m = next(d for d in delta.medications if d.name == m.name)
            merged_meds.append(MedicationHistory(
                name=m.name,
                events=m.events + delta_m.events,
            ))
        else:
            merged_meds.append(m)
    for m in delta.medications:
        if m.name not in prior_med_map:
            merged_meds.append(m)

    # --- Substances ---
    prior_sub_map = {s.name: s for s in prior.substances}
    merged_subs = []
    delta_sub_names = {s.name for s in delta.substances}

    for s in prior.substances:
        if s.name in delta_sub_names:
            delta_s = next(d for d in delta.substances if d.name == s.name)
            merged_subs.append(SubstanceUseHistory(
                name=s.name,
                events=s.events + delta_s.events,
            ))
        else:
            merged_subs.append(s)
    for s in delta.substances:
        if s.name not in prior_sub_map:
            merged_subs.append(s)

    # --- Clinical problems ---
    prior_prob_map = {p.name: p for p in prior.clinical_problems}
    merged_probs = []
    delta_prob_names = {p.name for p in delta.clinical_problems}

    for p in prior.clinical_problems:
        if p.name in delta_prob_names:
            delta_p = next(d for d in delta.clinical_problems if d.name == p.name)
            merged_probs.append(ClinicalProblem(
                name=p.name,
                events=p.events + delta_p.events,
            ))
        else:
            merged_probs.append(p)
    for p in delta.clinical_problems:
        if p.name not in prior_prob_map:
            merged_probs.append(p)

    # --- Measurements ---
    prior_meas_map = {m.name: m for m in prior.measurements}
    merged_meas = []
    delta_meas_names = {m.name for m in delta.measurements}

    for m in prior.measurements:
        if m.name in delta_meas_names:
            delta_m = next(d for d in delta.measurements if d.name == m.name)
            merged_meas.append(Measurement(
                name=m.name,
                data_points=m.data_points + delta_m.data_points,
            ))
        else:
            merged_meas.append(m)
    for m in delta.measurements:
        if m.name not in prior_meas_map:
            merged_meas.append(m)

    # --- Procedures ---
    merged_procs = prior.procedures + delta.procedures

    return PatientSummary(
        patient_id=prior.patient_id,
        document_ids=prior.document_ids + delta.document_ids,
        sex=prior.sex,
        age=prior.age,
        allergies=delta.allergies if delta.allergies else prior.allergies,
        family_history=delta.family_history if delta.family_history else prior.family_history,
        clinical_problems=merged_probs,
        substances=merged_subs,
        procedures=merged_procs,
        medications=merged_meds,
        measurements=merged_meas,
    )


def delta_to_patient_summary(
    prior: PatientSummary,
    delta_data: dict,
    meta: EncounterMeta,
) -> PatientSummary:
    """
    Convert a raw summary_delta dict (as stored in the JSONL) into a PatientSummary
    containing only the events from that single encounter, suitable for use with merge_summary.

    - allergies and family_history: the delta replaces the full list (stateless extractors)
    - substances, clinical_problems, medications, measurements: items are wrapped with EncounterMeta
    - procedures: wrapped into a single EncounterProcedures
    """

    # --- Allergies (full replacement) ---
    allergies_data = delta_data.get("allergies", {})
    allergies = AllergyOutputFormat.model_validate(allergies_data).allergies

    # --- Family history (full replacement) ---
    fh_data = delta_data.get("family_history", {})
    family_history = FamilyHistoryOutputFormat.model_validate(fh_data).family_history

    # --- Substances ---
    sub_data = delta_data.get("substances", {})
    sub_output = SubstanceUseOutputFormat.model_validate(sub_data)
    substances = [
        SubstanceUseHistory(name=e.name, events=[(meta, e)])
        for e in sub_output.substances
    ]

    # --- Clinical problems ---
    cp_data = delta_data.get("clinical_problems", {})
    cp_output = ClinicalProblemOutputFormat.model_validate(cp_data)
    clinical_problems = [
        ClinicalProblem(name=e.name, events=[(meta, e)])
        for e in cp_output.problems
    ]

    # --- Medications ---
    med_data = delta_data.get("medications", {})
    med_output = MedicationHistoryOutputFormat.model_validate(med_data)
    medications = [
        MedicationHistory(name=e.medication_name, events=[(meta, e)])
        for e in med_output.events
    ]

    # --- Measurements ---
    meas_data = delta_data.get("measurements", {})
    meas_output = MeasurementEventOutputFormat.model_validate(meas_data)
    measurements = [
        Measurement(name=e.name, data_points=[(meta, e)])
        for e in meas_output.measurements
    ]

    # --- Procedures ---
    proc_data = delta_data.get("procedures", {})
    proc_output = ProcedureOutputFormat.model_validate(proc_data)
    procedures = [EncounterProcedures(encounter=meta, procedures=proc_output.procedures)]

    return PatientSummary(
        patient_id=prior.patient_id,
        document_ids=[meta.id],
        sex=prior.sex,
        age=prior.age,
        allergies=allergies,
        family_history=family_history,
        clinical_problems=clinical_problems,
        substances=substances,
        procedures=procedures,
        medications=medications,
        measurements=measurements,
    )