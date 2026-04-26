from __future__ import annotations

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Shared sub-models
# ---------------------------------------------------------------------------

class Characteristic(BaseModel):
    id: str
    name: str


class Dimension(BaseModel):
    id: str
    name: str
    characteristics: list[Characteristic]


class Taxonomy(BaseModel):
    meta_characteristic: str
    topic: str
    dimensions: list[Dimension]


class DimensionMapping(BaseModel):
    dimension_id: str
    characteristic_id: str  # exactly one per dimension; use "N/A" if no characteristic fits
    reasoning: str


class MappedObject(BaseModel):
    id: str
    name: str
    source_document: str
    dimension_mappings: list[DimensionMapping]


class ObjectMapping(BaseModel):
    iteration: int
    objects: list[MappedObject]


# ---------------------------------------------------------------------------
# Per-node output models
# ---------------------------------------------------------------------------

class EmpiricalOutput(BaseModel):
    taxonomy: Taxonomy
    object_mapping: ObjectMapping
    reasoning_long: str


class ObjectValidationEntry(BaseModel):
    object_name: str
    action: str  # kept / removed / merged / renamed
    reasoning: str


class consolidatorOutput(BaseModel):
    merged_taxonomy: Taxonomy
    merged_object_mapping: ObjectMapping
    object_validation: list[ObjectValidationEntry]
    changes_short: list[str]
    reasoning_long: str
    event_log: list[str]


class ObjectiveEndingConditionResult(BaseModel):
    id: str
    name: str
    met: bool
    evidence: str


class SubjectiveEndingConditionResult(BaseModel):
    id: str
    name: str
    met: bool
    recommendation: str


class ValidatorOutput(BaseModel):
    structure_ok: bool
    structure_msg: str
    objective_ending_conditions: list[ObjectiveEndingConditionResult]
    subjective_ending_conditions: list[SubjectiveEndingConditionResult]
    ending_conditions_met: bool
    summary_recommendation: str
    event_log: list[str]
