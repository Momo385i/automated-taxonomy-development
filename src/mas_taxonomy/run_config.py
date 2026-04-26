from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import yaml


def default_objective_ending_conditions() -> list[dict[str, Any]]:
    """
    Returns the objective ending conditions for taxonomy development 
    according to Nickerson et al.
    """
    return [
        {
            "id": "OEC_01",
            "name": "Mutual Exclusivity",
            "description": (
                "In each dimension, no object can have two different characteristics. "
                "If an object can logically be assigned to more than one characteristic, "
                "the dimension is not mutually exclusive and should be restructured."
            )
        },
        {
            "id": "OEC_02",
            "name": "Collective Exhaustiveness",
            "description": (
                "Each object must have at least one of the characteristics in any dimension. "
                "If a valid object exists that cannot be classified into any existing category, "
                "the dimension is not collectively exhaustive and requires an additional "
                "or a residual characteristic."
            )
        },
        {
            "id": "OEC_03",
            "name": "No object merge/split",
            "description": "No object was merged with a similar object or split into multiple objects in the last iteration. If objects were merged or split, then we need to examine the impact of these changes and determine if changes need to be made in the dimensions or characteristics."
        },
        {
            "id": "OEC_04",
            "name": "No 'null' characteristics",
            "description": "At least one object is classified under every characteristic of every dimension. If at least one object is not found under a characteristic, then the taxonomy has a 'null' characteristic. We must either identify an object with the characteristic or remove the characteristic from the taxonomy."
        },
        {
            "id": "OEC_05",
            "name": "No new dimensions/characteristics",
            "description": "No new dimensions or characteristics were added in the last iteration. If new dimensions were found, then more characteristics of the dimensions may be identified. If new characteristics were found, then more dimensions may be identified that include these characteristics."
        },
        {
            "id": "OEC_06",
            "name": "No dimension/characteristic merge/split",
            "description": "No dimensions or characteristics were merged or split in the last iteration. If dimensions or characteristics were merged or split, then we need to examine the impact of these changes and determine if other dimensions or characteristics need to be merged or split."
        },
        {
            "id": "OEC_07",
            "name": "Unique dimensions",
            "description": "Every dimension is unique and not repeated (i.e., there is no dimension duplication). If dimensions are not unique, then there is redundancy/duplication among dimensions that needs to be eliminated."
        },
        {
            "id": "OEC_08",
            "name": "Unique characteristics",
            "description": "Every characteristic is unique within its dimension (i.e., there is no characteristic duplication within a dimension). If characteristics within a dimension are not unique, then there is redundancy/duplication in characteristics that needs to be eliminated."
        },
        {
            "id": "OEC_09",
            "name": "Unique cells",
            "description": "Each cell (combination of characteristics) is unique and is not repeated (i.e., there is no cell duplication). If cells are not unique, then there is redundancy/duplication in cells that needs to be eliminated."
        },
    ]


def default_subjective_ending_conditions() -> list[dict[str, Any]]:
    """
    Returns the subjective ending conditions for taxonomy development 
    according to Nickerson et al. (2013). 
    These conditions are phrased as questions to be evaluated by the researcher.
    """
    return [
        {
            "id": "SEC_01",
            "name": "Concise",
            "question": "Does the number of dimensions allow the taxonomy to be meaningful without being unwieldy or overwhelming?"
        },
        {
            "id": "SEC_02",
            "name": "Robust",
            "question": "Do the dimensions and characteristics provide for differentiation among objects sufficient to be of interest? Given the characteristics of sample objects, what can we say about the objects?"
        },
        {
            "id": "SEC_03",
            "name": "Comprehensive",
            "question": "Can all objects or a (random) sample of objects within the domain of interest be classified? Are all dimensions of the objects of interest identified?"
        },
        {
            "id": "SEC_04",
            "name": "Extendible",
            "question": "Can a new dimension or a new characteristic of an existing dimension be easily added?"
        },
        {
            "id": "SEC_05",
            "name": "Explanatory",
            "question": "What do the dimensions and characteristic explain about an object?"
        },
        {
            "id": "SEC_06", #extracted from OEC_01
            "name": "Representative sample",
            "question": (
                "Does the examined sample of objects cover a sufficiently diverse range of objects in the domain, "
                "or is it too narrow/biased toward a particular sub-type?"
            )
        }
    ]


@dataclass
class RunConfig: #serves as the run configuration.
    run_id: str
    topic: str
    meta_characteristic: str
    iteration: int = 1

    objective_ending_conditions: list[dict[str, Any]] | None = None
    subjective_ending_conditions: list[dict[str, Any]] | None = None

    status: str = "configured"  # configured | running | waiting_user | finished
    last_user_decision: str | None = None  # edit | next | finish

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if d["objective_ending_conditions"] is None: #if the objective ending conditions are not set, set them to the default.
            d["objective_ending_conditions"] = default_objective_ending_conditions()
        if d["subjective_ending_conditions"] is None: #if the subjective ending conditions are not set, set them to the default.
            d["subjective_ending_conditions"] = default_subjective_ending_conditions()
        return d


def load_run_config(run_dir: Path) -> dict[str, Any]: #loads the run configuration from the run directory.
    path = run_dir / "run_config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"run_config.yaml not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def save_run_config(run_dir: Path, cfg: dict[str, Any]) -> Path: #saves the run configuration to the run directory.
    path = run_dir / "run_config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path
