from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypedDict
from datetime import datetime
import time as _time

import yaml
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver #not needed actually but kept for clarity

import typer

from mas_taxonomy.config import get_settings
from mas_taxonomy.logging.logger import get_logger
from mas_taxonomy.run_config import load_run_config, save_run_config
# Option 1: Explicit imports (RECOMMENDED - clearer, better for IDE support)
from mas_taxonomy.llm_utils import (
    LLMRetryExhausted,
    _accumulate_token_usage,
    _create_llm,
    _extract_token_usage,
    call_llm_with_retry,
    resolve_provider_and_model,
)
from mas_taxonomy.schemas import EmpiricalOutput, consolidatorOutput, ValidatorOutput
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage



class GraphState(TypedDict, total=False): #serves as the state of the graph. + definition of attributes.
    # Run metadata
    run_id: str
    run_dir: str
    iteration: int
    thread_id: str  # For checkpointer persistence

    # Run timing
    run_started_at: str
    run_finished_at: str
    duration_seconds: float

    # Token usage
    token_usage_iteration: dict[str, int]  # Per-iteration token usage (resets each iteration): {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}
    token_usage_total: dict[str, int]  # Cumulative token usage across all iterations: {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}
    token_usage_by_agent: dict[str, dict[str, int]]  # Cumulative per-agent token usage

    errors: list[dict[str, Any]]
    
    # Configuration from run_config
    topic: str
    meta_characteristic: str
    objective_ending_conditions: list[dict[str, Any]]
    subjective_ending_conditions: list[dict[str, Any]]
    
    # User governance – global priority instructions
    important_user_prompt: str  # Global user instruction that all agents must treat as priority constraint
    
    # Consultation agent
    agent_conversation: list[dict[str, Any]]  # Append-only conversation history (in-memory only during execution)
    consultation_completed: bool  # Whether consultation phase is complete
    skip_consultation: bool  # Whether to skip consultation and use provided config
    
    # Interaction agent
    interaction_annotations: list[dict[str, Any]]  # Changes from interaction mode: [{"type": str, "id": str, "name": str, ...}]
    
    # Data flow
    documents: list[dict[str, Any]]  # Input documents with text
    empirical_taxonomy: dict[str, Any]  # Created by empirical_worker
    empirical_object_mapping: dict[str, Any]  # Object-characteristic mapping from empirical_worker
    current_taxonomy: dict[str, Any]  # Created/updated by consolidator
    current_object_mapping: dict[str, Any]  # Object-characteristic mapping from consolidator
    
    # Processing info
    consolidator_changes_short: list[str]  # What consolidator changed
    validation_report: dict[str, Any]  # Detailed validation results
    consolidator_reasoning_long: str #stores the llms reasoning for taxonomy changes
    empirical_reasoning_long: str #stores the llms reasoning for empirical taxonomy creation
    event_log: list[dict[str, Any]]  # Log of important decisions: [{"agent": str, "iteration": int, "event": str}, ...]
    
    # Iteration control (used by iteration_decision routing)
    user_decision: str  # "interaction", "next_iteration", "end", or "" (pending)


@dataclass
class GraphArtifacts: #serves as the artifacts of the graph
    run_dir: Path
    logs_dir: Path
    outputs_dir: Path
    ingest_fn: Callable[[Path, str, int], None] | None = None  # Optional ingest function for next-iteration document reload


def _get_iteration_dir(artifacts: GraphArtifacts, iteration: int) -> Path:
    """Return the per-iteration output subdirectory, creating it if needed."""
    iter_dir = artifacts.outputs_dir / f"iter_{iteration:03d}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    return iter_dir


def _get_iteration_dir_from_path(outputs_dir: Path, iteration: int) -> Path:
    """Return the per-iteration output subdirectory from a raw path, creating it if needed."""
    iter_dir = outputs_dir / f"iter_{iteration:03d}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    return iter_dir


def _accumulate_tokens_from_retry(state: GraphState, usage: dict[str, int], agent_name: str) -> None:
    """Add summed token usage (e.g. all LLM attempts including failed parses) into state."""
    if not (
        usage.get("prompt_tokens", 0)
        or usage.get("completion_tokens", 0)
        or usage.get("total_tokens", 0)
    ):
        return
    cur_i = state.get("token_usage_iteration", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    state["token_usage_iteration"] = _accumulate_token_usage(cur_i, usage)
    cur_t = state.get("token_usage_total", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    state["token_usage_total"] = _accumulate_token_usage(cur_t, usage)
    cur_a = state.get("token_usage_by_agent", {}).get(
        agent_name, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    )
    state.setdefault("token_usage_by_agent", {})[agent_name] = _accumulate_token_usage(cur_a, usage)


def load_run_documents(run_dir: Path) -> list[dict[str, Any]]: #loads the documents from the run directory.
    manifest_path = run_dir / "input_manifest.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"input_manifest.yaml not found: {manifest_path}")

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {} #loads the manifest from the run directory.
    docs = manifest.get("documents", [])

    loaded: list[dict[str, Any]] = []
    for d in docs:
        if "error" in d:
            loaded.append(d)
            continue
        txt_path = Path(d["extracted_text_file"])
        text = txt_path.read_text(encoding="utf-8") if txt_path.exists() else ""
        loaded.append({**d, "text": text})
    return loaded


def _taxonomy_template(meta_characteristic: str, topic: str) -> dict[str, Any]: #creates a minimal stable taxonomy output for the prototype.
    return {
        "meta_characteristic": meta_characteristic,
        "topic": topic,
        "dimensions": [],
        "notes": {
            "format": "prototype_v1",
        },
    }





def _add_event_log_entry(state: GraphState, agent: str, iteration: int, event_text: str) -> None:
    """
    Add an event entry to the event log.
    
    Args:
        state: GraphState to update
        agent: Name of the agent making the decision (e.g., "empirical", "consolidator", "validator")
        iteration: Current iteration number
        event_text: Short description of the important decision
    """
    if "event_log" not in state:
        state["event_log"] = []
    state["event_log"].append({
        "agent": agent,
        "iteration": iteration,
        "event": event_text,
    })


def _save_event_log(artifacts: GraphArtifacts, event_log: list[dict[str, Any]], iteration: int) -> None:
    """
    Save the full event log to a YAML file (overwrite mode) after each iteraton.
    With the looped graph the state accumulates all events across iterations,
    so we always write the complete list.
    """
    if not event_log:
        return
    
    event_log_path = artifacts.outputs_dir / "event_log.yaml"
    event_log_data = {
        "events": event_log,
        "total_events": len(event_log),
        "last_updated_iteration": iteration,
    }
    event_log_path.write_text(
        yaml.safe_dump(event_log_data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _save_iteration_state(state: GraphState, artifacts: GraphArtifacts, iteration: int) -> Path:
    """Save a snapshot of the graph state at the end of an iteration (without document text)."""
    state_to_save = {k: v for k, v in state.items() if k != "documents"}
    docs_clean = []
    for doc in state.get("documents", []):
        if isinstance(doc, dict):
            docs_clean.append({k: v for k, v in doc.items() if k != "text"})
        else:
            docs_clean.append(doc)
    state_to_save["documents"] = docs_clean
    iter_dir = _get_iteration_dir(artifacts, iteration)
    out_path = iter_dir / f"graph_state_iter_{iteration:03d}.yaml"
    out_path.write_text(yaml.safe_dump(state_to_save, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out_path


def _add_conversation_message(state: GraphState, role: str, message: str) -> None:
    """
    Add a message to the agent conversation history.
    
    Args:
        state: GraphState to update
        role: Role of the message sender ("user", "agent", "system")
        message: The message content
    """
    if "agent_conversation" not in state:
        state["agent_conversation"] = []
    state["agent_conversation"].append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "role": role,
        "message": message,
    })


def _log_tool_call(conversation_log: list[dict[str, Any]], agent: str, tool_name: str, target: str, source_type: str = "output_artifact", iteration: int | None = None) -> None:
    """
    Log a tool call to the conversation log AND print it visibly in the CLI.
    
    Args:
        conversation_log: The conversation log list to append to
        agent: Name of the agent making the call (e.g. "consultation", "interaction")
        tool_name: Name of the tool called
        target: File or resource accessed
        source_type: "output_artifact" or "original_pdf" or "state"
        iteration: Current iteration number (if applicable)
    """
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "role": "tool_call",
        "agent": agent,
        "tool": tool_name,
        "target": target,
        "source_type": source_type,
    }
    if iteration is not None:
        entry["iteration"] = iteration
    conversation_log.append(entry)
    typer.echo(typer.style(f"  tool use: {tool_name} {target}", fg=typer.colors.CYAN))


def _is_duplicate_instruction(existing_prompt: str, new_instruction: str) -> bool:
    """Basic exact-substring safety net — the LLM should catch semantic duplicates itself before."""
    if not existing_prompt:
        return False
    return new_instruction.lower().strip() in existing_prompt.lower().strip()


# Keys align with run_taxonomy_intake_questionnaire return dict.
_INTAKE_FIELD_ORDER: list[tuple[str, str, str]] = [
    (
        "topic",
        "A",
        "What is the specific phenomenon or domain of interest that this taxonomy aims to classify (Topic)?",
    ),
    (
        "target_users",
        "B",
        "Who are the primary target users of this taxonomy (e.g., specific researchers, practitioners, or industry experts), and what are their roles?",
    ),
    (
        "primary_purpose",
        "C",
        "What is the primary purpose of the taxonomy? Is it intended for purely structuring existing knowledge or for identifying and structuring constructs of an emerging phenomenon?",
    ),
    (
        "meta_characteristic",
        "D",
        "What is the meta-characteristic that will serve as the basis for all dimensions and characteristics?",
    ),
    (
        "object_type",
        "E",
        "What type of objects should be classified?",
    ),
    (
        "restrictions",
        "F",
        "What restrictions apply? (e.g., only certain technologies, time period, domain)",
    ),
]


def run_taxonomy_intake_questionnaire(
    *,
    default_topic: str = "",
    default_meta_characteristic: str = "",
) -> dict[str, str]:
    """
    CLI form: questions A–F. Topic (A) and meta-characteristic (D) are required; others may be skipped with Enter.
    Returns keys: topic, meta_characteristic, target_users, primary_purpose, object_type, restrictions.
    """
    typer.echo("")
    typer.echo(
        typer.style(
            "Note: Every field below except Topic and Meta-characteristic may be left empty "
            "(press Enter to skip). Topic and Meta-characteristic are required.",
            fg=typer.colors.CYAN,
        )
    )
    typer.echo("")

    out: dict[str, str] = {}

    for key, label, question in _INTAKE_FIELD_ORDER:
        typer.echo(f"{label}. {question}")
        is_required = key in ("topic", "meta_characteristic")
        default = ""
        if key == "topic":
            default = (default_topic or "").strip()
        elif key == "meta_characteristic":
            default = (default_meta_characteristic or "").strip()
        prompt_label = "Your answer" if is_required else "Your answer (optional)"
        while True:
            ans = typer.prompt(f"   {prompt_label}", default=default).strip()
            if is_required and not ans:
                typer.echo(typer.style("   This field is required.", fg=typer.colors.RED))
                default = ""
                continue
            out[key] = ans
            break
        typer.echo("")

    return out


def build_important_user_prompt_intake_no_llm(intake: dict[str, str]) -> str:
    """Option 1: only non-empty B/C/E/F as labeled lines; topic/meta go to config only."""
    parts: list[str] = []
    if intake.get("target_users", "").strip():
        parts.append(f'Primary target users of this taxonomy: "{intake["target_users"].strip()}"')
    if intake.get("primary_purpose", "").strip():
        parts.append(f'Primary purpose of the taxonomy: "{intake["primary_purpose"].strip()}"')
    if intake.get("object_type", "").strip():
        parts.append(f'Type of objects the user wants to classify: "{intake["object_type"].strip()}"')
    if intake.get("restrictions", "").strip():
        parts.append(f'Restrictions: "{intake["restrictions"].strip()}"')
    return "\n".join(parts).strip()


def build_intake_questionnaire_block_for_agent(intake: dict[str, str]) -> str:
    """Path 2 only: full A–F for the topic consultant's system prompt (not stored as IUP)."""
    lines: list[str] = [
        "Pre-consultation questionnaire (answers below were provided before the consultation dialogue; "
        '"Answer: (skipped — no answer)" means the user pressed Enter without typing).',
        "",
    ]
    for key, label, question in _INTAKE_FIELD_ORDER:
        val = (intake.get(key) or "").strip()
        display = val if val else "(skipped — no answer)"
        lines.append(f"{label}. {question}")
        lines.append(f"    Answer: {display}")
        lines.append("")
    return "\n".join(lines).strip()


def build_important_user_prompt_intake_llm_consultation(intake: dict[str, str]) -> str:
    """Path 2: IUP seed — B/C/E/F only (topic A and meta-characteristic D are excluded; they go to the agent prompt only)."""
    lines: list[str] = [
        "Pre-consultation questionnaire excerpts for global user instructions (Topic A and "
        "Meta-characteristic D are intentionally omitted here; they are only in the consultant context). "
        '"Answer: (skipped — no answer)" means the user pressed Enter without typing.',
        "",
    ]
    for key, label, question in _INTAKE_FIELD_ORDER:
        if key in ("topic", "meta_characteristic"):
            continue
        val = (intake.get(key) or "").strip()
        display = val if val else "(skipped — no answer)"
        lines.append(f"{label}. {question}")
        lines.append(f"    Answer: {display}")
        lines.append("")
    return "\n".join(lines).strip()


def _extract_consultation_insights(
    conversation_history: list[dict[str, Any]],
    topic: str,
    meta_characteristic: str,
    existing_iup: str = "",
) -> str:
    """
    Single LLM pass: rebuild the Important User Prompt from the draft IUP plus consultation
    transcript. Merges duplicates, drops empty placeholders, enriches from dialogue.
    Returns the full replacement IUP text, or the trimmed existing_iup if the model returns
    nothing useful or the call fails. Topic and meta-characteristic are passed in for context
    but live in run config.
    """
    try:
        _provider, _model = resolve_provider_and_model()
        llm = _create_llm(provider=_provider, model=_model, temperature=0.3)

        transcript_lines: list[str] = []
        for msg in conversation_history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if not content or role not in ("user", "assistant"):
                continue
            label = "User" if role == "user" else "Agent"
            transcript_lines.append(f"{label}: {content}")
        transcript = "\n".join(transcript_lines)

        draft = (existing_iup or "").strip()
        if not transcript.strip() and not draft:
            return ""

        system = (
            "You consolidate taxonomy consultation material into a single Important User Prompt (IUP) "
            "for downstream AI agents. Topic, meta-characteristic and ending conditions are stored separately in run "
            "configuration — do not restate them verbatim in the IUP unless a short operational nuance "
            "requires it. Output only the final IUP body: no title, no markdown fences, no preamble."
            "Constraint:- only add something to the IUP if actually relevant Information is given in the conversation"
            "- no topic, no meta characteristic and no ending conditions are allowed to be added here. They are saved elsewhere"
        )
        user_prompt = f"""Topic (stored in config): {topic}
Meta-characteristic (stored in config): {meta_characteristic}

Current draft Important User Prompt (may include a pre-form, agent-stored instructions, duplicates, or skipped placeholders):
---
{draft if draft else "(none)"}
---

Consultation transcript (User / Agent messages only):
---
{transcript if transcript.strip() else "(no transcript — rely on draft only)"}
---

Write the complete replacement Important User Prompt as a bullet point list based on the current Important User Prompt. Rules:
- Merge or summarize duplicate / overlapping content; one clear statement beats repeated variants.
- Remove empty sections, skipped placeholders, and lines that add no operational guidance.
- If existent, enrich with concrete constraints, audience, object types, restrictions, and priorities that appear in the conversation transcript and are not already present.
- Do not contradict the user's latest clear statements.
- Prefer concise labeled lines (e.g. "Primary purpose: ...") when it improves clarity.
- Return ONLY the final IUP body. If there is no new or relevant information in the conversation, return an empty string."""

        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user_prompt)])
        raw = _normalize_ai_message_content(resp.content).strip()

        if not raw or raw.upper() in ("NONE", '""', "EMPTY"):
            return draft

        return raw
    except Exception:
        return (existing_iup or "").strip()


def _normalize_ai_message_content(content: Any) -> str:
    """
    Turn AIMessage.content into a single string. LangChain / some providers return a list of
    blocks (e.g. [{'type': 'text', 'text': '...'}]) instead of a plain str; calling .strip() on
    that list raises 'list' object has no attribute 'strip'.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                t = block.get("text")
                if t is not None:
                    parts.append(str(t))
                else:
                    c = block.get("content")
                    if c is not None:
                        parts.append(_normalize_ai_message_content(c))
            else:
                t = getattr(block, "text", None)
                parts.append(str(t) if t is not None else str(block))
        return "".join(parts)
    return str(content)


def _save_conversation_log(artifacts: GraphArtifacts, conversation: list[dict[str, Any]]) -> None:
    """
    Save the conversation log to a YAML file.
    
    Args:
        artifacts: GraphArtifacts containing output directory
        conversation: List of conversation entries
    """
    if not conversation:
        return
    
    conversation_path = artifacts.outputs_dir / "consultation_conversation.yaml"
    conversation_data = {
        "conversation": conversation,
        "total_messages": len(conversation),
    }
    conversation_path.write_text(
        yaml.safe_dump(conversation_data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )



def _save_reasoning_file(artifacts: GraphArtifacts, iteration: int, empirical_reasoning: str, consolidator_reasoning: str) -> None:
    """
    Save both empirical and consolidator reasonings to a combined YAML file.
    
    Args:
        artifacts: GraphArtifacts containing output directory
        iteration: Current iteration number
        empirical_reasoning: Reasoning from empirical agent
        consolidator_reasoning: Reasoning from consolidator agent
    """
    reasoning_data = {
        "iteration": iteration,
        "empirical_reasoning": empirical_reasoning or "",
        "consolidator_reasoning": consolidator_reasoning or "",
    }
    iter_dir = _get_iteration_dir(artifacts, iteration)
    reasoning_path = iter_dir / f"reasoning_iter_{iteration:03d}.yaml"
    reasoning_path.write_text(
        yaml.safe_dump(reasoning_data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _detect_interaction_annotations(snapshot: dict[str, Any], edited: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Compare a taxonomy snapshot (before interaction) with the edited taxonomy to identify changes.
    Returns a list of annotations with stable IDs, element types, and change descriptions.
    Used to protect interaction-modified elements from consolidator overwrite in subsequent iterations.
    """
    annotations = []

    old_dims = {d.get("id"): d for d in snapshot.get("dimensions", [])}
    new_dims = {d.get("id"): d for d in edited.get("dimensions", [])}

    # Detect added dimensions
    for dim_id, new_dim in new_dims.items():
        if dim_id not in old_dims:
            annotations.append({
                "type": "dimension_added",
                "id": dim_id,
                "name": new_dim.get("name", ""),
            })

    # Detect removed dimensions
    for dim_id, old_dim in old_dims.items():
        if dim_id not in new_dims:
            annotations.append({
                "type": "dimension_removed",
                "id": dim_id,
                "name": old_dim.get("name", ""),
            })

    # Detect modifications within existing dimensions
    for dim_id in new_dims:
        if dim_id not in old_dims:
            continue
        old_dim = old_dims[dim_id]
        new_dim = new_dims[dim_id]

        # Dimension renamed
        if old_dim.get("name") != new_dim.get("name"):
            annotations.append({
                "type": "dimension_renamed",
                "id": dim_id,
                "old_name": old_dim.get("name", ""),
                "name": new_dim.get("name", ""),
            })

        # Compare characteristics within this dimension
        old_chars: dict[str, Any] = {}
        for c in old_dim.get("characteristics", []):
            key = c.get("id", str(c)) if isinstance(c, dict) else str(c)
            old_chars[key] = c
        new_chars: dict[str, Any] = {}
        for c in new_dim.get("characteristics", []):
            key = c.get("id", str(c)) if isinstance(c, dict) else str(c)
            new_chars[key] = c

        for char_id, new_c in new_chars.items():
            if char_id not in old_chars:
                annotations.append({
                    "type": "characteristic_added",
                    "dimension_id": dim_id,
                    "id": char_id,
                    "name": new_c.get("name", "") if isinstance(new_c, dict) else str(new_c),
                })
            elif new_c != old_chars[char_id]:
                annotations.append({
                    "type": "characteristic_modified",
                    "dimension_id": dim_id,
                    "id": char_id,
                    "name": new_c.get("name", "") if isinstance(new_c, dict) else str(new_c),
                    "old_name": old_chars[char_id].get("name", "") if isinstance(old_chars[char_id], dict) else str(old_chars[char_id]),
                })

        for char_id, old_c in old_chars.items():
            if char_id not in new_chars:
                annotations.append({
                    "type": "characteristic_removed",
                    "dimension_id": dim_id,
                    "id": char_id,
                    "name": old_c.get("name", "") if isinstance(old_c, dict) else str(old_c),
                })

    return annotations



def _load_cumulative_annotations(outputs_dir: Path) -> list[dict[str, Any]]:
    """Load all interaction annotations across all iterations from the outputs directory."""
    all_annotations: list[dict[str, Any]] = []
    for ann_file in sorted(outputs_dir.glob("iter_*/interaction_annotations_iter_*.yaml")):
        try:
            data = yaml.safe_load(ann_file.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "annotations" in data:
                all_annotations.extend(data["annotations"])
        except Exception:
            pass
    return all_annotations


def _append_to_conversation_file(outputs_dir: Path, entries: list[dict[str, Any]]) -> None:
    """
    Append entries to the cumulative all_agent_conversation.yaml file.
    Each entry must have: timestamp, role, message.
    This file is the single source of truth for all conversation history across all phases.
    """
    conv_path = outputs_dir / "all_agent_conversation.yaml"
    existing: list[dict[str, Any]] = []
    if conv_path.exists():
        try:
            data = yaml.safe_load(conv_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                existing = data.get("conversation", [])
            elif isinstance(data, list):
                existing = data
        except Exception:
            pass

    existing.extend(entries)
    conv_path.write_text(
        yaml.safe_dump({"conversation": existing, "total_messages": len(existing)}, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def run_ending_conditions_consultation(
    topic: str,
    meta_characteristic: str,
    standard_objective_conditions: list[dict[str, Any]],
    standard_subjective_conditions: list[dict[str, Any]],
    conversation_history: list[dict[str, Any]] | None = None,
    existing_token_usage: dict[str, int] | None = None,
    existing_important_user_prompt: str = "",
) -> dict[str, Any]:
    """
    Prompt user to keep standard ending conditions or consult with an agent to adapt them.
    Can be called standalone (manual config path) or from within run_interactive_consultation.

    Returns dict with:
        - objective_ending_conditions: list
        - subjective_ending_conditions: list
        - conversation_history: list
        - token_usage: dict
        - important_user_prompt: str
    """
    try:
        resolve_provider_and_model()
    except RuntimeError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

    conversation_history = conversation_history if conversation_history is not None else []
    token_usage = existing_token_usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    important_user_prompt = existing_important_user_prompt
    finalized_conditions: dict[str, Any] = {"objective": None, "subjective": None}

    @tool
    def save_ending_conditions(
        objective_conditions: list[dict[str, str]] = None,
        subjective_conditions: list[dict[str, str]] = None,
        use_standard: bool = False
    ) -> str:
        """
        Call this tool to save the final ending conditions.

        Args:
            objective_conditions: List of objective ending conditions with structure:
                [{"id": "OEC_01", "name": "condition name", "description": "description"}, ...]
                Only provide if user wants to modify objective conditions.
            subjective_conditions: List of subjective ending conditions with structure:
                [{"id": "SEC_01", "name": "condition name", "question": "question to evaluate"}, ...]
                Only provide if user wants to modify subjective conditions.
            use_standard: Set to True if user wants to keep all standard conditions.

        Returns:
            Confirmation message
        """
        if use_standard:
            finalized_conditions["objective"] = "keep_standard"
            finalized_conditions["subjective"] = "keep_standard"
            return "Using standard ending conditions"

        if objective_conditions:
            finalized_conditions["objective"] = objective_conditions
        if subjective_conditions:
            finalized_conditions["subjective"] = subjective_conditions

        return f"Saved {len(objective_conditions or [])} objective and {len(subjective_conditions or [])} subjective conditions"

    @tool
    def store_user_instruction_ec(instruction: str) -> str:
        """
        Store a NEW global instruction that applies to ALL agents throughout the taxonomy process.
        IMPORTANT: Before calling, check if a similar instruction is already stored (see previous
        tool responses in conversation). Do NOT store duplicates or rephrasings of existing instructions.

        Args:
            instruction: The global user instruction to store
        """
        nonlocal important_user_prompt
        if _is_duplicate_instruction(important_user_prompt, instruction):
            return f"DUPLICATE — not stored. Currently stored instructions: '{important_user_prompt}'"
        if important_user_prompt:
            important_user_prompt += f"\n{instruction}"
        else:
            important_user_prompt = instruction
        return f"Stored. All current instructions: '{important_user_prompt}'"

    # Prompt
    typer.echo("\n" + "-"*60)
    typer.echo("Would you like to keep the standard ending conditions")
    typer.echo("or consult with the agent to adapt them?")
    typer.echo("-"*60)
    typer.echo("1) Keep standard ending conditions")
    typer.echo("2) Consult agent to adapt ending conditions")
    while True:
        ec_choice = typer.prompt("Your choice [1/2]", default="1")
        if ec_choice in ("1", "2"):
            break
        typer.echo("Unexpected answer, try again.")

    ec_agent_used = ec_choice == "2"
    objective_conditions = standard_objective_conditions
    subjective_conditions = standard_subjective_conditions

    if ec_choice == "2":
        _provider, _model = resolve_provider_and_model()
        llm_ec = _create_llm(provider=_provider, model=_model, temperature=0.4)
        llm_ec_with_tools = llm_ec.bind_tools([save_ending_conditions, store_user_instruction_ec])

        ec_system_prompt = f"""You are a consultation agent helping researchers adapt ending conditions for taxonomy development.

ROLE:
- Focused advisor for adapting objective and subjective ending conditions
- Concise, specific, and actionable


TASK:
- Help the user decide whether to keep, modify, add, or remove ending conditions
  for the taxonomy on topic "{topic}" with meta-characteristic "{meta_characteristic}".
- Standard conditions provided: {len(standard_objective_conditions)} objective, {len(standard_subjective_conditions)} subjective. 
- When the user is satisfied, present your proposed changes and ask for explicit confirmation.
  Only call save_ending_conditions AFTER the user explicitly agrees.

CONSTRAINTS:
- Keep responses SHORT (3-5 sentences max). Be specific and actionable.
- If keeping standard conditions: call save_ending_conditions with use_standard=True.
- If modifications: call save_ending_conditions with the modified conditions.
  Use proper structure: objective_conditions=[{{"id": "OEC_01", "name": "...", "description": "..."}}, ...] and
  subjective_conditions=[{{"id": "SEC_01", "name": "...", "question": "..."}}, ...]

TOOLS:
- save_ending_conditions: Save final ending conditions (standard or modified).
- store_user_instruction_ec: Store overarching guidance or hints for all agents if the conversation indicates specific user preferences."""

        ec_initial_message = f"""Now let's adapt the ending conditions for your taxonomy:
- Topic: "{topic}"
- Meta-characteristic: "{meta_characteristic}"
Let me help you adapt the ending conditions if needed. Here are the standard conditions
OBJECTIVE Ending Conditions (must be objectively met):
{yaml.safe_dump(standard_objective_conditions, sort_keys=False, allow_unicode=True)}

SUBJECTIVE Ending Conditions (require expert judgment):
{yaml.safe_dump(standard_subjective_conditions, sort_keys=False, allow_unicode=True)}
I can help you modify, add, or remove conditions. What aspect would you like to adjust?"""

        typer.echo(typer.style("\nAgent: ", fg=typer.colors.BLUE, bold=True) + ec_initial_message + "\n")
        conversation_history.append({"role": "assistant", "content": ec_initial_message})

        ending_conditions_finalized = False

        while not ending_conditions_finalized:
            user_input = typer.prompt(typer.style("You", fg=typer.colors.BRIGHT_YELLOW)).strip()

            if user_input.lower() in ["exit", "quit", "cancel"]:
                typer.echo("\nEnding conditions consultation cancelled. Keeping standard conditions.")
                break

            conversation_history.append({"role": "user", "content": user_input})

            messages = [
                SystemMessage(content=ec_system_prompt),
                *[HumanMessage(content=msg["content"]) if msg["role"] == "user"
                  else AIMessage(content=msg["content"])
                  for msg in conversation_history if "content" in msg]
            ]

            try:
                resp = llm_ec_with_tools.invoke(messages)
                usage = _extract_token_usage(resp)
                token_usage.update(_accumulate_token_usage(token_usage, usage))

                save_called = False
                while hasattr(resp, 'tool_calls') and resp.tool_calls:
                    for tc in resp.tool_calls:
                        t_name = tc['name']
                        t_args = tc.get('args', {})

                        if t_name == 'store_user_instruction_ec':
                            store_user_instruction_ec.invoke(t_args)
                            _log_tool_call(conversation_history, "consultation", "store_user_instruction", t_args.get("instruction", ""), "state")
                            conversation_history.append({"role": "assistant", "content": f"[Instruction stored. All current instructions: {important_user_prompt}]"})
                        elif t_name == 'save_ending_conditions':
                            args = tc['args']
                            _log_tool_call(conversation_history, "consultation", t_name, f"use_standard={args.get('use_standard', False)}", "state")

                            if args.get('use_standard'):
                                typer.echo("\n" + typer.style("✓ Keeping standard ending conditions.", fg=typer.colors.GREEN))
                            else:
                                if args.get('objective_conditions'):
                                    objective_conditions = args['objective_conditions']
                                    typer.echo("\n" + typer.style(f"✓ Updated {len(objective_conditions)} objective ending conditions", fg=typer.colors.GREEN))
                                if args.get('subjective_conditions'):
                                    subjective_conditions = args['subjective_conditions']
                                    typer.echo("\n" + typer.style(f"✓ Updated {len(subjective_conditions)} subjective ending conditions", fg=typer.colors.GREEN))

                            ending_conditions_finalized = True
                            save_called = True
                            conversation_history.append({"role": "assistant", "content": "Ending conditions saved"})

                    if save_called:
                        break

                    messages.append(resp)
                    for tc in resp.tool_calls:
                        messages.append(ToolMessage(content="OK", tool_call_id=tc["id"]))
                    resp = llm_ec_with_tools.invoke(messages)
                    usage = _extract_token_usage(resp)
                    token_usage.update(_accumulate_token_usage(token_usage, usage))

                if save_called:
                    break

                agent_response = _normalize_ai_message_content(resp.content).strip()
                if agent_response:
                    conversation_history.append({"role": "assistant", "content": agent_response})
                    typer.echo(typer.style("\nAgent: ", fg=typer.colors.BLUE) + agent_response + "\n")

            except Exception as e:
                typer.echo(f"\nError communicating with LLM: {e}", err=True)
                typer.echo("Please try again.")
                conversation_history.pop()
                continue

    # Apply finalized conditions if modified by tool
    if finalized_conditions["objective"] and finalized_conditions["objective"] != "keep_standard":
        objective_conditions = finalized_conditions["objective"]
    if finalized_conditions["subjective"] and finalized_conditions["subjective"] != "keep_standard":
        subjective_conditions = finalized_conditions["subjective"]

    return {
        "objective_ending_conditions": objective_conditions,
        "subjective_ending_conditions": subjective_conditions,
        "conversation_history": conversation_history,
        "token_usage": token_usage,
        "important_user_prompt": important_user_prompt,
        "ec_agent_used": ec_agent_used,
    }


def run_interactive_consultation(
    run_dir: Path,
    run_id: str,
    standard_objective_conditions: list,
    standard_subjective_conditions: list,
    intake_agent_context: str = "",
    intake_important_user_prompt: str = "",
) -> dict[str, Any]:
    """
    Run interactive CLI-based consultation to help user define taxonomy parameters.

    Path 2: intake_agent_context = full questionnaire A–F, consultant system message only (includes
    preliminary topic/meta). intake_important_user_prompt = initial IUP (B/C/E/F only); topic and
    meta live in config after finalize_topic_and_meta, not in the IUP. B–F thus appear in both
    strings by design (advisor sees full form; persisted IUP omits A/D).

    Returns dict with:
        - topic: str
        - meta_characteristic: str
        - objective_ending_conditions: list
        - subjective_ending_conditions: list
        - conversation_history: list
        - important_user_prompt: str (global user instruction, if any)
        - token_usage: dict (token usage for consultation)
    """
    try:
        resolve_provider_and_model()
    except RuntimeError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)
    
    # Shared state for storing important user prompt across consultation phases
    shared_state = {"important_user_prompt": (intake_important_user_prompt or "").strip()}
    consultation_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    # Define tools for the topic consultation phase
    @tool
    def finalize_topic_and_meta(topic: str, meta_characteristic: str) -> str:
        """
        Call this tool when you have determined the final topic and meta-characteristic 
        that the user wants to use. Only call this when you are confident the user 
        has agreed to specific formulations.
        
        Args:
            topic: The final topic for the taxonomy
            meta_characteristic: The final meta-characteristic for the taxonomy
            
        Returns:
            Confirmation message
        """
        return f"Finalized: Topic='{topic}', Meta-characteristic='{meta_characteristic}'"

    @tool
    def store_user_instruction(instruction: str) -> str:
        """
        Store a NEW global instruction that applies to ALL agents throughout the taxonomy process.
        Examples: "be more creative", "maximize number of dimensions", "focus on technical aspects".
        IMPORTANT: Before calling, check if a similar instruction is already stored (see previous
        tool responses in conversation). Do NOT store duplicates or rephrasings of existing instructions.
        Only call for genuinely new overarching guidance.
        
        Args:
            instruction: The global user instruction to store
        """
        if _is_duplicate_instruction(shared_state["important_user_prompt"], instruction):
            return f"DUPLICATE — not stored. Currently stored instructions: '{shared_state['important_user_prompt']}'"
        if shared_state["important_user_prompt"]:
            shared_state["important_user_prompt"] += f"\n{instruction}"
        else:
            shared_state["important_user_prompt"] = instruction
        return f"Stored. All current instructions: '{shared_state['important_user_prompt']}'"

    _provider, _model = resolve_provider_and_model()
    llm = _create_llm(provider=_provider, model=_model, temperature=0.4)
    llm_with_tools = llm.bind_tools([finalize_topic_and_meta, store_user_instruction])
    
    topic = None
    meta_characteristic = None
    
    intake_block = (intake_agent_context or "").strip()
    intake_context = ""
    if intake_block:
        intake_context = f"""
PRIOR USER INPUT (pre-consultation questionnaire):
The user completed a structured questionnaire before this chat. The full verbatim content (including preliminary Topic and Meta-characteristic) is below.
You MUST treat this as authoritative starting context. At the moment, all answers of the questionnaire is already saved in the important user prompt.
Only save as new important user prompt, if it is not already existing.

---
{intake_block}
---

QUESTIONNAIRE-AWARE BEHAVIOR:
- You have all of the user's preliminary answers above, including Topic (A) and Meta-characteristic (D). Those two are NOT copied into the global user instructions; only finalize_topic_and_meta writes the agreed final values to run configuration.
- Pay special attention to skipped, empty, or vague items — ask focused follow-ups there.
- Your highest priority is to converge with the user on precise TOPIC and META-CHARACTERISTIC definitions; when both are agreed, call finalize_topic_and_meta with the exact final strings.
- Use store_user_instruction only for additional global guidance beyond what belongs in the IUP excerpt (B/C/E/F and similar), avoiding duplication.
"""

    system_prompt = f"""You are a consultation agent helping the researcher define parameters for taxonomy development using the Nickerson et al. (2013) method. You are the first agent that the researcher will interact with.

ROLE:
- Goal-oriented advisor for defining TOPIC and META-CHARACTERISTIC
- Concise, focused, and constructive
{intake_context}
TASK:
1. Understand the user's research objective through targeted clarifying questions. If not clarified yet, explore:
   - What types of objects or entities should be classified
   - The user's goal, intended audience, or use case
   - Specific priorities, constraints, or emphases
   - The domain perspective or framing the user has in mind
   If a pre-consultation questionnaire was provided, build on it instead of re-asking what is already clear.
2. Based on answers, propose 2-3 concrete topic formulations (when helpful).
3. Once topic is selected, propose 2-3 meta-characteristic options (when helpful).
4. When both are clearly agreed upon, call finalize_topic_and_meta with the exact values.

CONSTRAINTS:
- If the user has clear formulations, confirm and finalize directly instead of asking further questions.
- Wait for explicit user agreement before finalizing (e.g., "yes", "sounds good").
- Keep ALL responses SHORT (3-5 sentences max, 1-2 questions at a time).
- Scope is strictly topic and meta-characteristic. Ending conditions come later.
- Internal coherence check: Verify that the meta-characteristic implies a well-defined object class — not a heterogeneous collection of entities.
  - If not, explicitly warn the user with a brief explanation and suggest narrowing the scope
    before proceeding. Only finalize after you send the warning and the user accepts a broader scope.

TOOLS:
- finalize_topic_and_meta: Call when both topic and meta-characteristic are agreed upon.
- store_user_instruction: 
    - Call if the user provides global instructions that should apply to all agents (e.g., "focus on technical aspects", "maximize number of dimensions").
    - Only use this for overarching guidance, NOT for specific topic/meta-characteristic refinements."""
    
    conversation_history = []
    
    typer.echo("\n" + "="*60)
    typer.echo("      Consultation Agent - Taxonomy Parameter Definition")
    typer.echo("="*60 + "\n")
    
    # Initial agent message
    if intake_block:
        initial_message = (
            "I have your pre-consultation questionnaire in context (including any items you skipped). "
            "I'll help sharpen topic and meta-characteristic — especially where answers were missing or unclear. "
            "What would you like to adjust or confirm first?"
        )
    else:
        initial_message = """Let me help you define your taxonomy parameters (topic and meta-characteristic).

What is your research objective and what types of objects are you trying to classify? Do you already have a topic or ideas for the meta-characteristic in mind?"""
    
    typer.echo(typer.style(f"Agent: ", fg=typer.colors.BLUE) + initial_message + "\n")
    conversation_history.append({"role": "assistant", "content": initial_message})
    
    topic = None
    meta_characteristic = None
    
    # Topic and Meta-Characteristic consultation
    while True:
        user_input = typer.prompt(typer.style("You", fg=typer.colors.BRIGHT_YELLOW)).strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ["exit", "quit", "cancel"]:
            typer.echo("\nConsultation cancelled.")
            raise typer.Exit(code=0)
        
        conversation_history.append({"role": "user", "content": user_input})
        
        # Get agent response with tool calling
        # Filter out tool_call log entries (no "content" key) when building LLM messages
        messages = [
            SystemMessage(content=system_prompt),
            *[HumanMessage(content=msg["content"]) if msg["role"] == "user" 
              else AIMessage(content=msg["content"])
              for msg in conversation_history if "content" in msg]
        ]
        
        try:
            resp = llm_with_tools.invoke(messages)
            usage = _extract_token_usage(resp)
            consultation_token_usage.update(_accumulate_token_usage(consultation_token_usage, usage))
            
            # Process tool calls (may include store_user_instruction and/or finalize)
            finalize_called = False
            while hasattr(resp, 'tool_calls') and resp.tool_calls:
                for tc in resp.tool_calls:
                    t_name = tc['name']
                    t_args = tc.get('args', {})
                    
                    if t_name == 'store_user_instruction':
                        store_user_instruction.invoke(t_args)
                        _log_tool_call(conversation_history, "consultation", t_name, t_args.get("instruction", ""), "state")
                        conversation_history.append({"role": "assistant", "content": f"[Instruction stored. All current instructions: {shared_state['important_user_prompt']}]"})
                    elif t_name == 'finalize_topic_and_meta':
                        topic = tc['args']['topic']
                        meta_characteristic = tc['args']['meta_characteristic']
                        _log_tool_call(conversation_history, "consultation", t_name, f"topic='{topic}', meta='{meta_characteristic}'", "state")
                        finalize_called = True
                    else:
                        typer.echo(f"  ⚠ Unknown tool call: {t_name}")
                
                # If finalize was called, break out of tool loop
                if finalize_called:
                    break
                
                # If only store_user_instruction was called, get the next response
                messages.append(resp)
                for tc in resp.tool_calls:
                    messages.append(ToolMessage(content="OK", tool_call_id=tc["id"]))
                resp = llm_with_tools.invoke(messages)
                usage = _extract_token_usage(resp)
                consultation_token_usage.update(_accumulate_token_usage(consultation_token_usage, usage))
            
            if finalize_called:
                typer.echo("\n" + "="*60)
                typer.echo(typer.style("Agent has finalized the configuration:", fg=typer.colors.GREEN, bold=True))
                typer.echo(f"  Topic: {topic}")
                typer.echo(f"  Meta-characteristic: {meta_characteristic}")
                typer.echo("="*60)
                
                confirm = typer.prompt(typer.style("\nDo you confirm these values? [yes/no]", bold=True), default="yes").strip().lower()
                
                if confirm in ["yes", "y"]:
                    conversation_history.append({"role": "assistant", "content": f"Finalized: Topic='{topic}', Meta-characteristic='{meta_characteristic}'"})
                    break
                else:
                    typer.echo("\nLet's refine the definitions. What would you like to change?")
                    conversation_history.append({"role": "user", "content": "I'd like to refine those definitions."})
                    topic = None
                    meta_characteristic = None
                    continue
            
            # Regular response without tool call
            agent_response = _normalize_ai_message_content(resp.content).strip()
            if agent_response:
                conversation_history.append({"role": "assistant", "content": agent_response})
                typer.echo(typer.style("\nAgent: ", fg=typer.colors.BLUE) + agent_response + "\n")
                
        except Exception as e:
            typer.echo(f"\nError communicating with LLM: {e}", err=True)
            typer.echo("Please try again or type 'exit' to cancel.")
            conversation_history.pop()
            continue
    
    # Ending conditions consultation (delegated to extracted function)
    ec_result = run_ending_conditions_consultation(
        topic=topic,
        meta_characteristic=meta_characteristic,
        standard_objective_conditions=standard_objective_conditions,
        standard_subjective_conditions=standard_subjective_conditions,
        conversation_history=conversation_history,
        existing_token_usage=consultation_token_usage,
        existing_important_user_prompt=shared_state["important_user_prompt"],
    )
    objective_conditions = ec_result["objective_ending_conditions"]
    subjective_conditions = ec_result["subjective_ending_conditions"]
    consultation_token_usage = ec_result["token_usage"]
    shared_state["important_user_prompt"] = ec_result["important_user_prompt"]

    # Rebuild Important User Prompt from draft + full consultation transcript
    typer.echo(typer.style("\n  Consolidating important user prompt from consultation...", fg=typer.colors.CYAN))
    rebuilt = _extract_consultation_insights(
        conversation_history=conversation_history,
        topic=topic,
        meta_characteristic=meta_characteristic,
        existing_iup=shared_state["important_user_prompt"],
    )
    if rebuilt.strip():
        shared_state["important_user_prompt"] = rebuilt.strip()
        typer.echo(typer.style("  ✓ Important user prompt updated.", fg=typer.colors.GREEN))

    # Consultation phase complete
    typer.echo("\n" + "="*60)
    typer.echo(typer.style("✓ Consultation Phase Complete", fg=typer.colors.GREEN, bold=True))
    typer.echo("="*60)
    typer.echo(f"\nConfiguration summary:")
    typer.echo(f"  Topic: {topic}")
    typer.echo(f"  Meta-characteristic: {meta_characteristic}")
    typer.echo(f"  Objective ending conditions: {len(objective_conditions)}")
    typer.echo(f"  Subjective ending conditions: {len(subjective_conditions)}")
    if shared_state["important_user_prompt"]:
        typer.echo(f"  Global user instruction: {shared_state['important_user_prompt']}")
    
    return {
        "topic": topic,
        "meta_characteristic": meta_characteristic,
        "objective_ending_conditions": objective_conditions,
        "subjective_ending_conditions": subjective_conditions,
        "conversation_history": conversation_history,
        "important_user_prompt": shared_state["important_user_prompt"],
        "token_usage": consultation_token_usage,
    }


def consultation_agent_node(state: GraphState, artifacts: GraphArtifacts) -> GraphState:
    """
    Consultation agent for helping users define topic, meta characteristic, and ending conditions.
    This node is skipped if skip_consultation is True.
    """
    logger = get_logger("mas_taxonomy.graph.consultation", log_dir=artifacts.logs_dir)
    
    # Skip if consultation should be bypassed
    if state.get("skip_consultation", False):
        # Distinguish: did the CLI already run a consultation (conversation present), or was no consultation held?
        had_prior_consultation = bool(state.get("agent_conversation"))
        if had_prior_consultation:
            logger.info("Consultation completed via CLI - configuration loaded from run_config")
            _add_conversation_message(state, "system", "Consultation completed via CLI - configuration loaded from run_config")
        else:
            logger.info("Skipping consultation - using provided configuration")
            _add_conversation_message(state, "system", "Consultation skipped - using provided configuration")
        state["consultation_completed"] = True
        return state
    
    # Check if already completed (for resuming with checkpointer)
    if state.get("consultation_completed", False):
        logger.info("Consultation already completed")
        return state
    
    logger.info("Starting consultation agent")
    _add_conversation_message(state, "system", "Starting consultation to define taxonomy parameters")
    
    # Note: The actual interactive consultation happens in the CLI
    # This node primarily validates and stores the final configuration
    # The CLI will update the state with user inputs
    
    # Validate that required fields are set
    topic = state.get("topic", "").strip()
    meta = state.get("meta_characteristic", "").strip()
    
    if not topic or not meta:
        # This shouldn't happen if CLI does its job, but handle gracefully
        logger.warning("Consultation completed but topic or meta_characteristic not set")
        _add_conversation_message(state, "system", "Warning: Topic or meta characteristic not properly set")
    else:
        logger.info(f"Consultation completed - Topic: {topic}, Meta: {meta}")
        _add_conversation_message(state, "system", f"Configuration confirmed - Topic: '{topic}', Meta-characteristic: '{meta}'")
    
    state["consultation_completed"] = True

    # Save conversation log (currently unreachable: this node is skipped whenever
    # topic/meta are already in run_config, i.e. after configure-run. To have this
    # run and write consultation_conversation.yaml, either: (1) start the graph
    # without pre-existing config (no run_config.yaml with topic/meta), or
    # (2) set skip_consultation=False when invoking the graph despite having config.)
    # 
    # conversation = state.get("agent_conversation", [])
    # if conversation:
    #     _save_conversation_log(artifacts, conversation)

    return state


# ---------------------------------------------------------------------------
# Interaction Agent – human-in-the-loop taxonomy and mapping editing
# ---------------------------------------------------------------------------

def run_interactive_interaction(
    run_dir: Path,
    run_id: str,
    iteration: int,
    current_taxonomy: dict[str, Any],
    current_object_mapping: dict[str, Any],
    event_log: list[dict[str, Any]],
    outputs_dir: Path,
    important_user_prompt: str = "",
    meta_characteristic: str = "",
) -> dict[str, Any]:
    """
    Run interactive interaction mode where both the AI agent and the user can edit
    the taxonomy and object mapping. Agent can also change meta characteristic and assists the user with explanations, recommendations, and critical analysis.
    Edits happen in interaction working copies only;
    consolidator originals remain untouched.

    Returns dict with:
        - taxonomy_modified: bool
        - interaction_taxonomy: dict (the current interaction taxonomy)
        - interaction_object_mapping: dict (the current interaction object mapping)
        - interaction_annotations: list of detected changes (with stable IDs)
        - conversation_history: list of conversation entries
        - important_user_prompt: str (updated global user instruction, if changed)
        - meta_characteristic: str (updated meta-characteristic, if changed)
    """
    try:
        resolve_provider_and_model()
    except RuntimeError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

    # --- 1) Snapshot consolidator originals (for diff at session end) ---
    snapshot = yaml.safe_load(yaml.safe_dump(current_taxonomy, sort_keys=False))

    # --- 2) Create interaction working copies (idempotent per iteration) ---
    iter_dir = _get_iteration_dir_from_path(outputs_dir, iteration)
    tax_edit_path = iter_dir / f"interaction_taxonomy_iter_{iteration:03d}.yaml"
    obj_edit_path = iter_dir / f"interaction_object_mapping_iter_{iteration:03d}.yaml"

    if not tax_edit_path.exists():
        tax_edit_path.write_text(
            yaml.safe_dump(current_taxonomy, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    if not obj_edit_path.exists():
        obj_edit_path.write_text(
            yaml.safe_dump(current_object_mapping, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    # --- 3) Shared mutable state for tools ---
    interaction_state = {
        "exit_requested": False,
        "important_user_prompt": important_user_prompt,
        "meta_characteristic": meta_characteristic,
    }
    interaction_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    # --- 4) Helper: load interaction files from disk ---
    def _load_interaction_taxonomy() -> dict[str, Any]:
        try:
            data = yaml.safe_load(tax_edit_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _load_interaction_mapping() -> dict[str, Any]:
        try:
            data = yaml.safe_load(obj_edit_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    # --- 5) Define tools ---
    @tool
    def read_output_file(file_name: str) -> str:
        """
        Read a file from the outputs directory for analysis. Use this tool to access
        output artifacts such as:
        - interaction working copies (e.g. 'interaction_taxonomy_iter_xxx.yaml',
          'interaction_object_mapping_iter_xxx.yaml')
        - consolidator originals (e.g. 'consolidator_taxonomy_iter_xxx.yaml',
          'consolidator_object_mapping_iter_xxx.yaml')
        - reasoning files (e.g. 'reasoning_iter_xxx.yaml')
        - validation reports (e.g. 'validator_report_iter_xxx.yaml')
        - consolidator changes (e.g. 'consolidator_changes_iter_xxx.yaml')
        - event log ('event_log.yaml')
        - prior iteration artifacts

        This is the ONLY way to read files. You do NOT have access to source PDF documents.

        Args:
            file_name: Name of the file in the outputs directory.
        """
        import re
        try:
            target = outputs_dir / file_name
            if not target.exists():
                m = re.search(r'_iter_(\d{3})', file_name)
                if m:
                    target = outputs_dir / f"iter_{m.group(1)}" / file_name
            if not target.exists():
                available: list[str] = sorted(f.name for f in outputs_dir.iterdir() if f.is_file())
                for sub in sorted(outputs_dir.iterdir()):
                    if sub.is_dir() and sub.name.startswith("iter_"):
                        available.extend(f.name for f in sorted(sub.iterdir()) if f.is_file())
                return f"File '{file_name}' not found. Available files:\n" + "\n".join(f"  - {n}" for n in available)
            content = target.read_text(encoding="utf-8")
            if len(content) > 30000:
                content = content[:30000] + "\n\n... (truncated — tell the human that the file is too large)"
            return content
        except Exception as e:
            return f"Error reading file: {e}"

    @tool
    def apply_taxonomy_edit(
        action: str,
        dimension_id: str = "",
        dimension_name: str = "",
        characteristic_id: str = "",
        characteristic_name: str = "",
        new_name: str = "",
        target_dimension_id: str = "",
    ) -> str:
        """
        Apply a single edit to the interaction taxonomy AND automatically propagate
        necessary changes to the interaction object mapping for consistency.

        IMPORTANT: Only apply edits that the user has explicitly discussed and approved.
        After making changes, briefly summarize what was changed in BOTH files.

        Args:
            action: One of: "add_dimension", "remove_dimension", "rename_dimension",
                    "add_characteristic", "remove_characteristic", "rename_characteristic",
                    "move_characteristic" (move to another dimension).
            dimension_id: The dimension ID to operate on (e.g. "D1").
            dimension_name: Name for new dimension (only for add_dimension).
            characteristic_id: The characteristic ID to operate on (e.g. "D1.C2").
            characteristic_name: Name for new characteristic (only for add_characteristic).
            new_name: New name (for rename_dimension or rename_characteristic).
            target_dimension_id: Target dimension for move_characteristic.
        """
        tax = _load_interaction_taxonomy()
        obj_map = _load_interaction_mapping()
        dims = tax.get("dimensions", [])
        dim_index = {d["id"]: i for i, d in enumerate(dims)}
        changes: list[str] = []

        if action == "add_dimension":
            if not dimension_id or not dimension_name:
                return "Error: dimension_id and dimension_name are required for add_dimension."
            if dimension_id in dim_index:
                return f"Error: dimension {dimension_id} already exists."
            dims.append({"id": dimension_id, "name": dimension_name, "characteristics": []})
            na_count = 0
            for obj in obj_map.get("objects", []):
                obj.setdefault("dimension_mappings", []).append({
                    "dimension_id": dimension_id,
                    "characteristic_id": "N/A",
                    "reasoning": f"New dimension {dimension_id} — awaiting classification",
                })
                na_count += 1
            changes.append(f"Added dimension {dimension_id} '{dimension_name}'")
            if na_count > 0:
                changes.append(f"Added N/A mapping entries for {na_count} object(s) in {dimension_id} — assign characteristics via apply_mapping_edit")

        elif action == "remove_dimension":
            if dimension_id not in dim_index:
                return f"Error: dimension {dimension_id} not found."
            dims.pop(dim_index[dimension_id])
            for obj in obj_map.get("objects", []):
                obj["dimension_mappings"] = [
                    m for m in obj.get("dimension_mappings", [])
                    if m.get("dimension_id") != dimension_id
                ]
            changes.append(f"Removed dimension {dimension_id}")
            changes.append(f"Removed all {dimension_id} mappings from object mapping")

            # Re-number remaining dimensions sequentially (D1, D2, …) and update characteristic IDs + mappings
            id_remap: dict[str, str] = {}  # old_dim_id -> new_dim_id
            char_remap: dict[str, str] = {}  # old_char_id -> new_char_id
            for new_idx, dim in enumerate(dims, start=1):
                new_dim_id = f"D{new_idx}"
                old_dim_id = dim["id"]
                if old_dim_id != new_dim_id:
                    id_remap[old_dim_id] = new_dim_id
                    dim["id"] = new_dim_id
                    for ci, c in enumerate(dim.get("characteristics", []), start=1):
                        old_cid = c.get("id", "")
                        new_cid = f"{new_dim_id}.C{ci}"
                        if old_cid != new_cid:
                            char_remap[old_cid] = new_cid
                            c["id"] = new_cid
                else:
                    for ci, c in enumerate(dim.get("characteristics", []), start=1):
                        expected = f"{new_dim_id}.C{ci}"
                        if c.get("id") != expected:
                            char_remap[c.get("id", "")] = expected
                            c["id"] = expected
            if id_remap or char_remap:
                for obj in obj_map.get("objects", []):
                    for m in obj.get("dimension_mappings", []):
                        old_did = m.get("dimension_id", "")
                        if old_did in id_remap:
                            m["dimension_id"] = id_remap[old_did]
                        old_cid = m.get("characteristic_id", "")
                        if old_cid in char_remap:
                            m["characteristic_id"] = char_remap[old_cid]
                rename_parts = [f"{old}→{new}" for old, new in id_remap.items()]
                changes.append(f"Re-numbered dimensions: {', '.join(rename_parts)}. Object mappings updated accordingly.")

        elif action == "rename_dimension":
            if dimension_id not in dim_index or not new_name:
                return "Error: dimension_id and new_name are required."
            old_name = dims[dim_index[dimension_id]]["name"]
            dims[dim_index[dimension_id]]["name"] = new_name
            changes.append(f"Renamed dimension {dimension_id}: '{old_name}' → '{new_name}'")

        elif action == "add_characteristic":
            if dimension_id not in dim_index or not characteristic_id or not characteristic_name:
                return "Error: dimension_id, characteristic_id, and characteristic_name are required."
            dim = dims[dim_index[dimension_id]]
            existing_ids = {c["id"] for c in dim.get("characteristics", [])}
            if characteristic_id in existing_ids:
                return f"Error: characteristic {characteristic_id} already exists in {dimension_id}."
            dim.setdefault("characteristics", []).append({"id": characteristic_id, "name": characteristic_name})
            na_reassigned = 0
            for obj in obj_map.get("objects", []):
                for m in obj.get("dimension_mappings", []):
                    if m.get("dimension_id") == dimension_id and m.get("characteristic_id") == "N/A":
                        na_reassigned += 1
            changes.append(f"Added characteristic {characteristic_id} '{characteristic_name}' to {dimension_id}")
            if na_reassigned > 0:
                changes.append(f"Note: {na_reassigned} object(s) have N/A in {dimension_id} — consider reassigning them")

        elif action == "remove_characteristic":
            if dimension_id not in dim_index or not characteristic_id:
                return "Error: dimension_id and characteristic_id are required."
            dim = dims[dim_index[dimension_id]]
            chars = dim.get("characteristics", [])
            orig_len = len(chars)
            dim["characteristics"] = [c for c in chars if c.get("id") != characteristic_id]
            if len(dim["characteristics"]) == orig_len:
                return f"Error: characteristic {characteristic_id} not found in {dimension_id}."
            affected = 0
            for obj in obj_map.get("objects", []):
                for m in obj.get("dimension_mappings", []):
                    if m.get("dimension_id") == dimension_id and m.get("characteristic_id") == characteristic_id:
                        m["characteristic_id"] = "N/A"
                        m["reasoning"] = f"Set to N/A: original characteristic {characteristic_id} was removed"
                        affected += 1
            changes.append(f"Removed characteristic {characteristic_id} from {dimension_id}")
            if affected > 0:
                changes.append(f"Set {affected} object mapping(s) in {dimension_id} to N/A (characteristic removed)")

        elif action == "rename_characteristic":
            if dimension_id not in dim_index or not characteristic_id or not new_name:
                return "Error: dimension_id, characteristic_id, and new_name are required."
            dim = dims[dim_index[dimension_id]]
            found = False
            for c in dim.get("characteristics", []):
                if c.get("id") == characteristic_id:
                    old_name = c["name"]
                    c["name"] = new_name
                    found = True
                    changes.append(f"Renamed {characteristic_id}: '{old_name}' → '{new_name}'")
                    break
            if not found:
                return f"Error: characteristic {characteristic_id} not found in {dimension_id}."

        elif action == "move_characteristic":
            if not dimension_id or not characteristic_id or not target_dimension_id:
                return "Error: dimension_id, characteristic_id, and target_dimension_id are required."
            if dimension_id not in dim_index or target_dimension_id not in dim_index:
                return f"Error: source or target dimension not found."
            src_dim = dims[dim_index[dimension_id]]
            char_to_move = None
            for i, c in enumerate(src_dim.get("characteristics", [])):
                if c.get("id") == characteristic_id:
                    char_to_move = src_dim["characteristics"].pop(i)
                    break
            if not char_to_move:
                return f"Error: characteristic {characteristic_id} not found in {dimension_id}."
            new_char_id = f"{target_dimension_id}.C{len(dims[dim_index[target_dimension_id]].get('characteristics', [])) + 1}"
            char_to_move["id"] = new_char_id
            dims[dim_index[target_dimension_id]].setdefault("characteristics", []).append(char_to_move)
            moved_count = 0
            for obj in obj_map.get("objects", []):
                for m in obj.get("dimension_mappings", []):
                    if m.get("dimension_id") == dimension_id and m.get("characteristic_id") == characteristic_id:
                        m["characteristic_id"] = "N/A"
                        m["reasoning"] = f"Set to N/A: characteristic {characteristic_id} moved to {target_dimension_id}"
                        has_target = any(
                            em.get("dimension_id") == target_dimension_id
                            for em in obj.get("dimension_mappings", [])
                        )
                        if not has_target:
                            obj["dimension_mappings"].append({
                                "dimension_id": target_dimension_id,
                                "characteristic_id": new_char_id,
                                "reasoning": f"Moved from {dimension_id} ({characteristic_id})",
                            })
                        else:
                            for em in obj.get("dimension_mappings", []):
                                if em.get("dimension_id") == target_dimension_id:
                                    em["characteristic_id"] = new_char_id
                                    em["reasoning"] = f"Reassigned: characteristic moved from {dimension_id}"
                                    break
                        moved_count += 1
            changes.append(f"Moved {characteristic_id} from {dimension_id} to {target_dimension_id} as {new_char_id}")
            if moved_count > 0:
                changes.append(f"Updated {moved_count} object mapping(s): assigned {new_char_id} in {target_dimension_id}, set source {dimension_id} to N/A")

        else:
            return f"Error: unknown action '{action}'. Use one of: add_dimension, remove_dimension, rename_dimension, add_characteristic, remove_characteristic, rename_characteristic, move_characteristic."

        tax["dimensions"] = dims
        tax_edit_path.write_text(yaml.safe_dump(tax, sort_keys=False, allow_unicode=True), encoding="utf-8")
        obj_edit_path.write_text(yaml.safe_dump(obj_map, sort_keys=False, allow_unicode=True), encoding="utf-8")
        for c in changes:
            event_log.append({"agent": "interaction", "iteration": iteration, "event": f"[taxonomy_edit] {c}"})
        return "Changes applied:\n" + "\n".join(f"  - {c}" for c in changes)

    @tool
    def apply_mapping_edit(
        action: str,
        object_id: str,
        dimension_id: str = "",
        new_characteristic_id: str = "",
        reasoning: str = "",
        new_name: str = "",
        source_document: str = "",
        merge_into_id: str = "",
    ) -> str:
        """
        All edits to the interaction object mapping file. Handles both characteristic
        assignments and object-level operations.

        Actions:
        - reassign: Change an object's characteristic in one dimension. Requires
          object_id, dimension_id, new_characteristic_id, reasoning.
          If the object has no mapping for that dimension yet, a new entry is added.
          No-op if the assignment is already correct (returns without writing).
        - add_object: Add a new object with N/A mappings for all dimensions.
          Requires object_id, new_name. Optional: source_document.
        - remove_object: Remove an object entirely. Requires object_id.
        - rename_object: Rename an object. Requires object_id, new_name.
        - merge_objects: Merge object_id INTO merge_into_id. Keeps the target's
          ID, name, and mappings. Combines source_document references.
          Requires object_id, merge_into_id.

        Args:
            action: One of reassign, add_object, remove_object, rename_object, merge_objects.
            object_id: The object to act on (e.g. "O1").
            dimension_id: Dimension for reassign (e.g. "D1").
            new_characteristic_id: Characteristic for reassign (e.g. "D1.C2"), or "N/A".
            reasoning: Justification for reassign.
            new_name: Name for add_object or rename_object.
            source_document: Source reference for add_object.
            merge_into_id: Target object ID for merge_objects.
        """
        obj_map = _load_interaction_mapping()
        tax = _load_interaction_taxonomy()
        objects: list[dict] = obj_map.get("objects", [])
        changes: list[str] = []

        if action == "reassign":
            if not dimension_id or not new_characteristic_id:
                return "Error: dimension_id and new_characteristic_id are required for reassign."
            dim_ids = {d.get("id") for d in tax.get("dimensions", [])}
            if dimension_id not in dim_ids:
                return f"Error: dimension {dimension_id} not found in taxonomy. Valid: {sorted(dim_ids)}"
            if new_characteristic_id != "N/A":
                valid_ids: set[str] = set()
                for dim in tax.get("dimensions", []):
                    if dim.get("id") == dimension_id:
                        valid_ids = {c.get("id") for c in dim.get("characteristics", [])}
                        break
                if new_characteristic_id not in valid_ids:
                    return f"Error: characteristic {new_characteristic_id} does not exist in {dimension_id}. Valid: {sorted(valid_ids)}"
            for obj in objects:
                if obj.get("id") != object_id:
                    continue
                for m in obj.get("dimension_mappings", []):
                    if m.get("dimension_id") == dimension_id:
                        old_cid = m.get("characteristic_id", "?")
                        if old_cid == new_characteristic_id:
                            return f"No change needed: {object_id} in {dimension_id} is already '{old_cid}'."
                        m["characteristic_id"] = new_characteristic_id
                        m["reasoning"] = reasoning
                        obj_edit_path.write_text(
                            yaml.safe_dump(obj_map, sort_keys=False, allow_unicode=True), encoding="utf-8"
                        )
                        msg = f"Updated {object_id} in {dimension_id}: '{old_cid}' → '{new_characteristic_id}'"
                        event_log.append({"agent": "interaction", "iteration": iteration, "event": f"[mapping_edit] {msg}"})
                        return f"{msg}. Reasoning: {reasoning}"
                obj.setdefault("dimension_mappings", []).append({
                    "dimension_id": dimension_id,
                    "characteristic_id": new_characteristic_id,
                    "reasoning": reasoning,
                })
                obj_edit_path.write_text(
                    yaml.safe_dump(obj_map, sort_keys=False, allow_unicode=True), encoding="utf-8"
                )
                msg = f"Added new mapping for {object_id} in {dimension_id}: '{new_characteristic_id}'"
                event_log.append({"agent": "interaction", "iteration": iteration, "event": f"[mapping_edit] {msg}"})
                return f"{msg}. Reasoning: {reasoning}"
            return f"Error: object {object_id} not found in mapping."

        elif action == "add_object":
            if not new_name:
                return "Error: new_name is required for add_object."
            if any(o.get("id") == object_id for o in objects):
                return f"Error: object {object_id} already exists."
            dim_ids_list = [d.get("id") for d in tax.get("dimensions", [])]
            new_obj: dict[str, Any] = {
                "id": object_id,
                "name": new_name,
                "source_document": source_document or "manual",
                "dimension_mappings": [
                    {"dimension_id": did, "characteristic_id": "N/A", "reasoning": "New object — awaiting classification"}
                    for did in dim_ids_list
                ],
            }
            objects.append(new_obj)
            changes.append(f"Added object {object_id} '{new_name}' with N/A mappings for {len(dim_ids_list)} dimensions")

        elif action == "remove_object":
            before = len(objects)
            objects = [o for o in objects if o.get("id") != object_id]
            if len(objects) == before:
                return f"Error: object {object_id} not found."
            obj_map["objects"] = objects
            changes.append(f"Removed object {object_id}")

        elif action == "rename_object":
            if not new_name:
                return "Error: new_name is required for rename_object."
            found = False
            for o in objects:
                if o.get("id") == object_id:
                    old_name = o.get("name", "?")
                    o["name"] = new_name
                    changes.append(f"Renamed {object_id}: '{old_name}' → '{new_name}'")
                    found = True
                    break
            if not found:
                return f"Error: object {object_id} not found."

        elif action == "merge_objects":
            if not merge_into_id:
                return "Error: merge_into_id is required for merge_objects."
            src = next((o for o in objects if o.get("id") == object_id), None)
            tgt = next((o for o in objects if o.get("id") == merge_into_id), None)
            if not src:
                return f"Error: source object {object_id} not found."
            if not tgt:
                return f"Error: target object {merge_into_id} not found."
            src_doc = src.get("source_document", "")
            tgt_doc = tgt.get("source_document", "")
            if src_doc and src_doc not in tgt_doc:
                tgt["source_document"] = f"{tgt_doc}; {src_doc}" if tgt_doc else src_doc
            objects = [o for o in objects if o.get("id") != object_id]
            obj_map["objects"] = objects
            changes.append(f"Merged {object_id} ('{src.get('name', '?')}') into {merge_into_id} ('{tgt.get('name', '?')}')")

        else:
            return f"Error: unknown action '{action}'. Use reassign, add_object, remove_object, rename_object, or merge_objects."

        obj_map["objects"] = objects
        obj_edit_path.write_text(
            yaml.safe_dump(obj_map, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        for c in changes:
            event_log.append({"agent": "interaction", "iteration": iteration, "event": f"[mapping_edit] {c}"})
        summary = "\n".join(f"  - {c}" for c in changes)
        return f"Changes applied:\n{summary}"

    @tool
    def exit_interaction_mode() -> str:
        """
        End this interaction session and return the user to the CLI main menu (where they
        choose e.g. next pipeline iteration or end run). Call when the user is done —
        including phrases like: next iteration, proceed, done, exit, stop, no changes, back to menu.
        """
        interaction_state["exit_requested"] = True
        return "Exiting interaction mode. Changes will be evaluated and saved."

    @tool
    def manage_user_instructions(action: str, instruction: str = "", line_number: int = 0) -> str:
        """
        Manage global instructions that apply to ALL agents throughout the taxonomy process.

        Actions:
        - add: Add a new instruction. Requires 'instruction'.
        - update: Replace an existing instruction by line number. Requires 'line_number' and 'instruction'.
        - delete: Remove an instruction by line number. Requires 'line_number'.
        - list: Show all current instructions with line numbers.

        Args:
            action: One of "add", "update", "delete", "list".
            instruction: The instruction text (for add/update).
            line_number: 1-based line number of the instruction to update/delete.
        """
        current = interaction_state["important_user_prompt"]
        lines = [l for l in current.split("\n") if l.strip()] if current else []

        if action == "list":
            if not lines:
                return "No instructions stored."
            numbered = "\n".join(f"  {i+1}. {l}" for i, l in enumerate(lines))
            return f"Current instructions ({len(lines)}):\n{numbered}"

        elif action == "add":
            if not instruction:
                return "Error: 'instruction' is required for add."
            if _is_duplicate_instruction(current, instruction):
                return f"DUPLICATE — not stored. Current instructions: '{current}'"
            lines.append(instruction)
            interaction_state["important_user_prompt"] = "\n".join(lines)
            return f"Added. All current instructions:\n" + "\n".join(f"  {i+1}. {l}" for i, l in enumerate(lines))

        elif action == "update":
            if not instruction or line_number < 1:
                return "Error: 'instruction' and a valid 'line_number' (>=1) are required for update."
            if line_number > len(lines):
                return f"Error: line_number {line_number} out of range. Only {len(lines)} instruction(s) stored."
            old = lines[line_number - 1]
            lines[line_number - 1] = instruction
            interaction_state["important_user_prompt"] = "\n".join(lines)
            return f"Updated line {line_number}: '{old}' → '{instruction}'.\nAll current instructions:\n" + "\n".join(f"  {i+1}. {l}" for i, l in enumerate(lines))

        elif action == "delete":
            if line_number < 1:
                return "Error: a valid 'line_number' (>=1) is required for delete."
            if line_number > len(lines):
                return f"Error: line_number {line_number} out of range. Only {len(lines)} instruction(s) stored."
            removed = lines.pop(line_number - 1)
            interaction_state["important_user_prompt"] = "\n".join(lines)
            remaining = "\n".join(f"  {i+1}. {l}" for i, l in enumerate(lines)) if lines else "  (none)"
            return f"Deleted line {line_number}: '{removed}'.\nRemaining instructions:\n{remaining}"

        else:
            return f"Error: unknown action '{action}'. Use add, update, delete, or list."

    @tool
    def update_meta_characteristic(new_meta_characteristic: str, rationale: str) -> str:
        """
        Update the meta-characteristic that governs the entire taxonomy.
        Only call this after explicit user approval. The meta-characteristic is
        the single overarching property from which ALL dimensions and characteristics
        must be logical consequences (Nickerson et al., 2013).

        Args:
            new_meta_characteristic: The new meta-characteristic text.
            rationale: Short justification for why the change is necessary.
        """
        old_mc = interaction_state["meta_characteristic"]
        interaction_state["meta_characteristic"] = new_meta_characteristic
        note = f"Meta-characteristic changed from '{old_mc}' to '{new_meta_characteristic}'. Rationale: {rationale}"
        iup = interaction_state["important_user_prompt"]
        mc_note = f"[META-CHARACTERISTIC UPDATED in iteration {iteration}] {note}"
        interaction_state["important_user_prompt"] = f"{iup}\n{mc_note}" if iup else mc_note
        event_log.append({
            "agent": "interaction",
            "iteration": iteration,
            "event": note,
        })
        typer.echo(typer.style(f"\n  ✓ Meta-characteristic updated: '{old_mc}' → '{new_meta_characteristic}'", fg=typer.colors.GREEN))
        typer.echo(typer.style(f"    Rationale: {rationale}", fg=typer.colors.CYAN))
        return f"Meta-characteristic updated successfully.\nOld: '{old_mc}'\nNew: '{new_meta_characteristic}'\nRationale: {rationale}\nThis change will apply to all agents from now on."

    # --- 6) Build LLM with tools ---
    _provider, _model = resolve_provider_and_model()
    llm = _create_llm(provider=_provider, model=_model, temperature=0.3)
    tools = [read_output_file, apply_taxonomy_edit, apply_mapping_edit,
             exit_interaction_mode, manage_user_instructions, update_meta_characteristic]
    llm_with_tools = llm.bind_tools(tools)

    # Filter event_log to current iteration for the system prompt
    iter_events = [e for e in event_log if e.get("iteration") == iteration]

    iup_section = ""
    if important_user_prompt:
        iup_section = f"""
    PRIORITY USER INSTRUCTION:
    {important_user_prompt}
    """

    # --- 7) System prompt ---
    system_prompt = f"""You are an Interaction Agent for taxonomy development (Nickerson et al., 2013 method).

ROLE:
- Critical, constructive partner for reviewing and editing taxonomy and object mapping
- You CAN directly edit taxonomy and object mapping via the provided tools
- You can also read any output file for analysis
- You do NOT modify consolidator originals — all edits happen in interaction working copies

TASK:
1. Provide a brief summary of the current iteration status and key recommendations.
2. Assist the user in reviewing, discussing, and editing the taxonomy and object mapping.
3. Evaluate whether the current meta-characteristic ('{meta_characteristic}') is still appropriate.
   Be conservative — only suggest a change if empirical evidence clearly indicates it would improve
   the taxonomy's usefulness (e.g., focus precision, logical consistency, purpose alignment).
   If you recommend a change, propose specific wording. The user must explicitly approve before
   you call update_meta_characteristic.
4. When the user wants to leave (e.g., "next iteration", "done", "exit"), call exit_interaction_mode immediately.

CONSTRAINTS:
- Only apply edits that the user has explicitly discussed and approved.
  If the user tells you or allows you to act, execute right away — do not ask again.
- For batch operations (e.g., reassign 10 objects): ask once for permission, then apply ALL changes in one response.
- Do NOT call apply_mapping_edit when the assignment would not change (old == new).
- Only re-read files if you need new information.
- apply_taxonomy_edit automatically propagates structural changes to the object mapping
  (e.g., removing a dimension removes its mappings, adding a dimension creates N/A entries).
  However, you still have to verify that the object mapping is still valid after the taxonomy edits.
  After taxonomy edits, check whether N/A assignments should be resolved and proactively suggest fixes.
- Keep responses concise (3-5 sentences unless detailed analysis is requested).

TOOLS:
- read_output_file: Read any output file (interaction copies, consolidator originals, reasoning, validation reports, event log, prior iterations).
- apply_taxonomy_edit: Structural taxonomy edits (add/remove/rename dimension or characteristic, move characteristic). Auto-propagates to object mapping.
- apply_mapping_edit: Object mapping edits (reassign characteristic, add/remove/rename/merge objects).
- exit_interaction_mode: End session and return to CLI menu.
- manage_user_instructions: Add, update, delete, or list global instructions for all agents.
- update_meta_characteristic: Update the meta-characteristic (requires user approval).

CONTEXT:
- Run: {run_id}, Iteration: {iteration}
- Topic: {current_taxonomy.get('topic', 'N/A')}
- Meta-characteristic: {meta_characteristic}
- Current taxonomy: {len(current_taxonomy.get('dimensions', []))} dimensions
{iup_section}
EVENT LOG (current iteration):
{yaml.safe_dump(iter_events, sort_keys=False, allow_unicode=True) if iter_events else 'No events recorded.'}

Start your first message with a brief summary of the current iteration status and key recommendations (including meta-characteristic assessment, but only if actually relevant)."""

    # --- 8) LLM message history ---
    chat_messages: list = [SystemMessage(content=system_prompt)]
    conversation_log: list[dict[str, Any]] = []

    typer.echo("\n" + "=" * 60)
    typer.echo("      Interaction Agent – Taxonomy Editing Mode")
    typer.echo("=" * 60)
    typer.echo(f"\n  Working copies:")
    typer.echo(f"    Taxonomy: {tax_edit_path}")
    typer.echo(f"    Object mapping: {obj_edit_path}")
    typer.echo(typer.style("  The agent can edit these files. You may also edit them manually.", fg=typer.colors.CYAN))
    typer.echo("  Type 'exit', 'done', or 'next iteration' to leave.\n")

    # --- 9) Initial agent summary ---
    initial_prompt_text = "Please provide a brief summary of the current iteration and your key recommendations."
    chat_messages.append(HumanMessage(content=initial_prompt_text))

    try:
        resp = llm_with_tools.invoke(chat_messages)
        usage = _extract_token_usage(resp)
        interaction_token_usage.update(_accumulate_token_usage(interaction_token_usage, usage))
        initial_msg = _normalize_ai_message_content(resp.content).strip()
        chat_messages.append(AIMessage(content=initial_msg))
    except Exception as e:
        initial_msg = "I'm ready to help you review and edit the taxonomy. What would you like to discuss?"
        chat_messages.append(AIMessage(content=initial_msg))

    typer.echo(typer.style("Agent: ", fg=typer.colors.BLUE) + initial_msg + "\n")
    conversation_log.append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "role": "agent",
        "message": initial_msg,
        "phase": "interaction",
        "iteration": iteration,
    })

    # --- 10) Main interaction loop ---
    while not interaction_state["exit_requested"]:
        try:
            user_input = typer.prompt(typer.style("You", fg=typer.colors.BRIGHT_YELLOW)).strip()
        except (KeyboardInterrupt, EOFError):
            interaction_state["exit_requested"] = True
            break

        if not user_input:
            continue

        chat_messages.append(HumanMessage(content=user_input))
        conversation_log.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "role": "user",
            "message": user_input,
            "phase": "interaction",
            "iteration": iteration,
        })

        try:
            resp = llm_with_tools.invoke(chat_messages)
            usage = _extract_token_usage(resp)
            interaction_token_usage.update(_accumulate_token_usage(interaction_token_usage, usage))

            # Process tool calls in a loop (agent may chain multiple tool calls)
            while hasattr(resp, "tool_calls") and resp.tool_calls:
                chat_messages.append(resp)  # Add the AIMessage with tool_calls

                for tc in resp.tool_calls:
                    t_name = tc["name"]
                    t_args = tc.get("args", {})
                    tool_failed = False

                    try:
                        if t_name == "read_output_file":
                            result_text = read_output_file.invoke(t_args)
                            _log_tool_call(conversation_log, "interaction", t_name, t_args.get("file_name", ""), "output_artifact", iteration)
                        elif t_name == "apply_taxonomy_edit":
                            _log_tool_call(conversation_log, "interaction", t_name, f"{t_args.get('action', '')} {t_args.get('dimension_id', '')} {t_args.get('characteristic_id', '')}".strip(), "output_artifact", iteration)
                            result_text = apply_taxonomy_edit.invoke(t_args)
                            typer.echo(typer.style(f"  [Edit] {result_text}", fg=typer.colors.GREEN))
                            typer.echo()  # blank line after edit block (before next tool use / output)
                        elif t_name == "apply_mapping_edit":
                            _log_tool_call(conversation_log, "interaction", t_name, f"{t_args.get('object_id', '')} {t_args.get('dimension_id', '')} → {t_args.get('new_characteristic_id', '')}", "output_artifact", iteration)
                            result_text = apply_mapping_edit.invoke(t_args)
                            typer.echo(typer.style(f"  [Edit] {result_text}", fg=typer.colors.GREEN))
                            typer.echo()  # blank line after edit block (before next tool use / output)
                        elif t_name == "exit_interaction_mode":
                            result_text = exit_interaction_mode.invoke({})
                            _log_tool_call(conversation_log, "interaction", t_name, "", "state", iteration)
                        elif t_name == "manage_user_instructions":
                            result_text = manage_user_instructions.invoke(t_args)
                            _log_tool_call(conversation_log, "interaction", t_name, t_args.get("action", ""), "state", iteration)
                        elif t_name == "update_meta_characteristic":
                            result_text = update_meta_characteristic.invoke(t_args)
                            _log_tool_call(conversation_log, "interaction", t_name, t_args.get("new_meta_characteristic", ""), "state", iteration)
                        else:
                            result_text = f"Unknown tool: {t_name}"
                    except Exception as tool_err:
                        tool_failed = True
                        result_text = f"TOOL ERROR ({t_name}): {tool_err}"
                        typer.echo(typer.style(f"\n  ⚠ Tool call failed: {t_name}({t_args}) — {tool_err}", fg=typer.colors.RED))
                        typer.echo(typer.style("    The agent will be informed. If the issue persists, you may need to apply this change manually.", fg=typer.colors.YELLOW))
                        event_log.append({"agent": "interaction", "iteration": iteration, "event": f"[tool_error] {t_name} failed: {tool_err}"})

                    chat_messages.append(ToolMessage(content=str(result_text), tool_call_id=tc["id"]))
                    conversation_log.append({
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "role": "system",
                        "message": f"[Tool: {t_name}]{' FAILED' if tool_failed else ''} {str(result_text)[:300]}",
                        "phase": "interaction",
                        "iteration": iteration,
                    })

                    if t_name == "exit_interaction_mode" and not tool_failed:
                        interaction_state["exit_requested"] = True

                if interaction_state["exit_requested"]:
                    break

                # Re-invoke the LLM so it can respond after processing tool results
                resp = llm_with_tools.invoke(chat_messages)
                usage = _extract_token_usage(resp)
                interaction_token_usage.update(_accumulate_token_usage(interaction_token_usage, usage))

            # Show the agent's final text response
            agent_text = _normalize_ai_message_content(resp.content).strip()
            if agent_text:
                chat_messages.append(AIMessage(content=agent_text))
                conversation_log.append({
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "role": "agent",
                    "message": agent_text,
                    "phase": "interaction",
                    "iteration": iteration,
                })
                typer.echo(typer.style("\nAgent: ", fg=typer.colors.BLUE) + agent_text + "\n")

        except Exception as e:
            typer.echo(f"\nError communicating with LLM: {e}", err=True)
            typer.echo("Please try again.")
            chat_messages.pop()  # Remove the failed HumanMessage
            conversation_log.pop()  # Remove the failed user entry
            continue

    # --- 11) Post-interaction: load final interaction files and detect changes ---
    edited_taxonomy = _load_interaction_taxonomy() or current_taxonomy
    edited_object_mapping = _load_interaction_mapping() or current_object_mapping

    # Compare with snapshot to detect modifications
    taxonomy_modified = (
        yaml.safe_dump(snapshot, sort_keys=True) != yaml.safe_dump(edited_taxonomy, sort_keys=True)
    )
    interaction_annotations: list[dict[str, Any]] = []

    if taxonomy_modified:
        interaction_annotations = _detect_interaction_annotations(snapshot, edited_taxonomy)

        annotations_path = iter_dir / f"interaction_annotations_iter_{iteration:03d}.yaml"
        annotations_path.write_text(
            yaml.safe_dump({
                "iteration": iteration,
                "annotations": interaction_annotations,
                "annotation_count": len(interaction_annotations),
            }, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

        typer.echo("\n" + typer.style(f"✓ Taxonomy modified. {len(interaction_annotations)} change(s) detected.", fg=typer.colors.GREEN))
        typer.echo(f"  Annotations saved: {annotations_path}")
        for ann in interaction_annotations:
            typer.echo(f"  - {ann['type']}: {ann.get('name', ann.get('id', ''))}")
    else:
        typer.echo("\n" + typer.style("No taxonomy changes detected.", fg=typer.colors.YELLOW))

    return {
        "taxonomy_modified": taxonomy_modified,
        "interaction_taxonomy": edited_taxonomy,
        "interaction_object_mapping": edited_object_mapping,
        "interaction_annotations": interaction_annotations,
        "conversation_history": conversation_log,
        "important_user_prompt": interaction_state["important_user_prompt"],
        "meta_characteristic": interaction_state["meta_characteristic"],
        "token_usage": interaction_token_usage,
    }


def _is_valid_taxonomy_structure(tax: dict[str, Any]) -> tuple[bool, str]: #checks if the taxonomy structure is valid. (for consolidator)
    if "dimensions" not in tax or not isinstance(tax["dimensions"], list): #checks if the taxonomy has a dimensions list.
        return False, "missing or invalid dimensions list"
    for dim in tax["dimensions"]: #iterates through the dimensions and checks if they are valid.
        if not isinstance(dim, dict):
            return False, "dimension is not a dict"
        if "name" not in dim or not isinstance(dim["name"], str) or not dim["name"].strip():
            return False, "dimension missing name"
        # ID is optional but recommended for new schema
        if "characteristics" not in dim or not isinstance(dim["characteristics"], list):
            return False, "dimension missing characteristics list"
        for c in dim["characteristics"]:
            # Support both old format (string) and new format (dict with id and name)
            if isinstance(c, str):
                if not c.strip():
                    return False, "characteristic is not a non-empty string"
            elif isinstance(c, dict):
                if "name" not in c or not isinstance(c["name"], str) or not c["name"].strip():
                    return False, "characteristic dict missing name"
                # ID is optional but recommended
            else:
                return False, "characteristic must be string or dict with name"
    return True, "ok"


def empirical_worker_node(state: GraphState, artifacts: GraphArtifacts) -> GraphState: #creates the empirical taxonomy.
    logger = get_logger("mas_taxonomy.graph.empirical", log_dir=artifacts.logs_dir)


    s = get_settings()

    meta = state.get("meta_characteristic", "").strip()
    topic = state.get("topic", "").strip()
    it = int(state.get("iteration", 1))

    # Default: 100k chars 
    # Set to 0 for no limit, or override via MAX_CHARS_PER_DOC env var
    max_chars_per_doc = s.max_chars_per_doc if s.max_chars_per_doc > 0 else 100000
    doc_blocks = []
    for doc in state.get("documents", []):
        if "error" in doc:
            continue
        name = doc.get("file_name", "unknown")
        text = doc.get("text", "") or "" #gets the text from the document from the "text" key from in memory storage for the llm
        if max_chars_per_doc and len(text) > max_chars_per_doc:
            text = text[:max_chars_per_doc]
            logger.warning(f"Truncated {name} from {len(doc.get('text', ''))} to {max_chars_per_doc} chars")
        doc_blocks.append(f"### SOURCE: {name}\n{text}")

    sources_text = "\n\n".join(doc_blocks) #joins the document blocks into a single string.

    # Build important_user_prompt section for the prompt
    iup = state.get("important_user_prompt", "").strip()
    iup_section = ""
    if iup:
        iup_section = (
            "\n\nIMPORTANT USER INSTRUCTION (applies to your analytical approach, NOT to output format (treat as priority constraint)):\n"
            f"{iup}\n"
        )

    system = (
        "You are an empirical taxonomy worker following the taxonomy development method of Nickerson et al. (2013).\n\n"
        "ROLE:\n"
        "- Methodical, empirically grounded researcher\n"
        "- Conservative in scope: only propose what the empirical material supports\n"
        "- Focused on clarity, parsimony, and explanatory value\n\n"
        "TASK:\n"
        "- Propose a taxonomy (dimensions and characteristics) derived strictly from the provided empirical sources\n"
        "- Identify concrete objects from the sources and map each to exactly one characteristic per dimension\n"
        "- Provide a comprehensive reasoning explaining your analytical process\n\n"
        "CONSTRAINTS:\n"
        "- Only propose dimensions and characteristics that are empirically supported by the provided sources.\n"
        "- Do NOT invent dimensions or characteristics to increase completeness, symmetry, or balance.\n"
        "- If the empirical material supports only few dimensions or characteristics, output only those.\n"
        "- Do NOT revise or compare with previous taxonomies.\n"
        "- Do NOT evaluate ending conditions.\n"
        "- Do NOT merge, split, rename, or delete dimensions across iterations.\n"
        "- All dimensions must be unique, non-redundant, and explain a specific aspect of the classified objects.\n"
        "- All characteristics must be unique within their dimension.\n"
        "- MECE condition (Nickerson et al., 2013): Within each dimension, characteristics must be\n"
        "  (a) Mutually Exclusive — an object is assignable to exactly one characteristic per dimension, and\n"
        "  (b) Collectively Exhaustive — characteristics together cover the relevant feature space so every object can be classified.\n"
        "- Before assigning N/A to any object-dimension pair, check if adding a new characteristic would resolve the gap.\n"
        "   - Only assign N/A if the gap cannot be resolved by adding a new characteristic."
        "- Verify that every characteristic is linked to at least one object.\n"
        "- Verify that dimensions are non-overlapping and allow future extension without restructuring."
        + iup_section
    )


    user = f"""
Topic:
{topic}

Meta-characteristic:
{meta}

Task:
Create a taxonomy using the empirical-to-conceptual approach defined by Nickerson et al. (2013).
Follow these steps:

1) Identify objects
- Identify concrete empirical objects from the provided sources.
- Each object must originate from a specific source document.
- Objects must be sufficiently distinct to justify differentiation.
- CRITICAL — coherent object class: All objects must belong to the same conceptual class as defined by the topic and meta-characteristic.


2) Identify characteristics and group objects
- Identify characteristics that:
  a) are logical consequences of the meta-characteristic
  b) meaningfully discriminate between objects
  c) are grounded in the provided empirical sources
- Exclude characteristics that apply to all objects or do not differentiate.

3) Group characteristics into dimensions
- Group related characteristics into higher-level dimensions.
 - Each dimension must:
  - represent a coherent explanatory concept
  - explain a specific aspect or behavior of the objects
  - provide meaningful differentiation between objects 
- Within each dimension, characteristics must satisfy the MECE condition
- Dimension names and characteristics MUST be concise noun phrases (no sentences).
- Maintain a meaningful number of dimensions that ensures depth without becoming unwieldy.
- Prefer parsimony over coverage. 
- Dimensions are non-overlapping and non-redundant.

4) Object-characteristic mapping
- For each object, assign exactly one characteristic per dimension.
- Each mapping entry has: dimension_id, characteristic_id, reasoning (short logical justification).
- Reasoning DOES NOT require empirical citation — it is sufficient that the object is found in the source
  and the assignment follows logically from the object's described properties or behavior.
- If no characteristic fits, use "N/A" and explain why in reasoning.
- IMPORTANT: An N/A assignment is a signal that the taxonomy may be incomplete. Before assigning N/A,
  check if adding a new characteristic to that dimension would resolve the gap. If so, add it to the taxonomy and use it instead of N/A.

Output:
- Your output structure is defined by the function schema. Fill all fields completely. 
- taxonomy: meta_characteristic, topic, dimensions with characteristics.
- object_mapping: iteration ({it}), objects with id, name, source_document, dimension_mappings.
  dimension_mappings is a list with exactly one characteristic entry per dimension: {{dimension_id, characteristic_id, reasoning}}. 
- reasoning_long: Comprehensive explanation of the analytical process, including how objects were identified, how characteristics were derived and grouped,
  how dimensions were formed, the empirical grounding for each, and any limitations.
- Use IDs consistently: D1, D2, …; D1.C1, D1.C2, …; O1, O2, …

Sources:
{sources_text}
""".strip()


    _provider, _model = resolve_provider_and_model()
    llm = _create_llm(provider=_provider, model=_model, temperature=0.0)
    llm_structured = llm.with_structured_output(EmpiricalOutput, include_raw=True)

    logger.info(f"empirical_worker: iteration={it} provider={_provider} model={_model} sources={len(doc_blocks)}")
    
    try:
        parsed, token_usage = call_llm_with_retry(
            lambda: llm_structured.invoke([{"role": "system", "content": system}, {"role": "user", "content": user}]),
            max_retries=2,
            backoff_seconds=1.5,
            log=logger,
        )

        tax = parsed.taxonomy.model_dump()
        tax["notes"] = {"format": "prototype_v1"}
        obj_mapping = parsed.object_mapping.model_dump()
        reasoning_long = parsed.reasoning_long

        logger.info(
            f"empirical_worker: produced taxonomy dims={len(tax.get('dimensions', []))}"
        )
        obj_count = len(obj_mapping.get("objects", []))
        mapping_count = sum(len(o.get("dimension_mappings", [])) for o in obj_mapping.get("objects", []))
        logger.info(f"empirical_worker: object mapping - objects={obj_count} dimension_assignments={mapping_count}")
        logger.info(
            f"empirical_worker: token usage - prompt={token_usage['prompt_tokens']}, "
            f"completion={token_usage['completion_tokens']}, total={token_usage['total_tokens']}"
        )

        # Accumulate token usage in state (iteration + total + per-agent)
        current_iter_usage = state.get("token_usage_iteration", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        state["token_usage_iteration"] = _accumulate_token_usage(current_iter_usage, token_usage)
        
        current_total_usage = state.get("token_usage_total", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        state["token_usage_total"] = _accumulate_token_usage(current_total_usage, token_usage)
        
        agent_usage = state.get("token_usage_by_agent", {}).get("empirical", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        state.setdefault("token_usage_by_agent", {})["empirical"] = _accumulate_token_usage(agent_usage, token_usage)

        state["empirical_taxonomy"] = tax
        state["empirical_object_mapping"] = obj_mapping
        state["empirical_reasoning_long"] = reasoning_long
    
    except Exception as e:
        if isinstance(e, LLMRetryExhausted):
            _accumulate_tokens_from_retry(state, e.token_usage, "empirical")
        # Structured error handling
        err = {"agent": "empirical", "iteration": it, "error": str(e), "error_type": type(e).__name__}
        state.setdefault("errors", []).append(err)
        error_file = _get_iteration_dir(artifacts, it) / f"error_empirical_iter_{it:03d}.yaml"
        error_file.write_text(yaml.safe_dump(err, sort_keys=False, allow_unicode=True), encoding="utf-8")
        logger.error(f"empirical_worker: error in iteration {it}: {e}")
        
        # Fallback: empty taxonomy structure
        tax = _taxonomy_template(meta, topic)
        tax["dimensions"] = []
        state["empirical_taxonomy"] = tax
        state["empirical_object_mapping"] = {"iteration": it, "objects": []}
        state["empirical_reasoning_long"] = f"Empirical worker encountered an error: {e}. Using empty taxonomy fallback."
        logger.warning(f"empirical_worker: using empty taxonomy fallback due to error")

    # Save taxonomy and object mapping separately
    iter_dir = _get_iteration_dir(artifacts, it)
    out_tax_path = iter_dir / f"empirical_taxonomy_iter_{it:03d}.yaml"
    out_tax_path.write_text(yaml.safe_dump(state["empirical_taxonomy"], sort_keys=False, allow_unicode=True), encoding="utf-8")
    
    out_obj_path = iter_dir / f"empirical_object_mapping_iter_{it:03d}.yaml"
    out_obj_path.write_text(yaml.safe_dump(state.get("empirical_object_mapping", {}), sort_keys=False, allow_unicode=True), encoding="utf-8")
    
    # Save reasoning file (consolidator reasoning will be added later)
    empirical_reasoning = state.get("empirical_reasoning_long", "")
    consolidator_reasoning = state.get("consolidator_reasoning_long", "")
    _save_reasoning_file(artifacts, it, empirical_reasoning, consolidator_reasoning)
    
    return state



def consolidator_node(state: GraphState, artifacts: GraphArtifacts) -> GraphState:
    logger = get_logger("mas_taxonomy.graph.consolidator", log_dir=artifacts.logs_dir)

    it = int(state.get("iteration", 1))
    topic = state.get("topic", "").strip()
    meta = state.get("meta_characteristic", "").strip()

    emp_tax = state.get("empirical_taxonomy", {})
    emp_obj_mapping = state.get("empirical_object_mapping", {})
    ok, msg = _is_valid_taxonomy_structure(emp_tax)
    if not ok: #Get all relevant states. If the empirical taxonomy structure is invalid, it resets the taxonomy structure to the empty template. Save all outputs.
        logger.info(f"consolidator: invalid empirical taxonomy structure reason={msg}")
        fixed = _taxonomy_template(meta, topic)
        fixed["dimensions"] = []
        state["current_taxonomy"] = fixed
        state["current_object_mapping"] = {"iteration": it, "objects": []}
        state["consolidator_changes_short"] = [f"updating: reset taxonomy structure due to invalid empirical format ({msg})"]
        state["consolidator_reasoning_long"] = "Empirical taxonomy was invalid. Reset to empty template."
        _add_event_log_entry(state, "consolidator", it, f"Empirical taxonomy structure invalid ({msg}). Reset to empty template.")
        iter_dir = _get_iteration_dir(artifacts, it)
        out_tax = iter_dir / f"consolidator_taxonomy_iter_{it:03d}.yaml"
        out_tax.write_text(yaml.safe_dump(state["current_taxonomy"], sort_keys=False, allow_unicode=True), encoding="utf-8")
        out_obj = iter_dir / f"consolidator_object_mapping_iter_{it:03d}.yaml"
        out_obj.write_text(yaml.safe_dump(state["current_object_mapping"], sort_keys=False, allow_unicode=True), encoding="utf-8")
        out_changes = iter_dir / f"consolidator_changes_iter_{it:03d}.yaml"
        out_changes.write_text(
            yaml.safe_dump({"iteration": it, "changes_short": state["consolidator_changes_short"]}, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        # Save combined reasoning file
        empirical_reasoning = state.get("empirical_reasoning_long", "")
        consolidator_reasoning = state.get("consolidator_reasoning_long", "")
        _save_reasoning_file(artifacts, it, empirical_reasoning, consolidator_reasoning)
        return state

    prev_tax = None
    prev_obj_mapping = None
    if it > 1: #if the iteration is greater than 1, it loads the previous taxonomy and object mapping from the outputs directory.
        prev_iter_dir = _get_iteration_dir(artifacts, it - 1)
        interaction_prev_path = prev_iter_dir / f"interaction_taxonomy_iter_{it-1:03d}.yaml"
        interaction_prev_obj_path = prev_iter_dir / f"interaction_object_mapping_iter_{it-1:03d}.yaml"
        consol_prev_path = prev_iter_dir / f"consolidator_taxonomy_iter_{it-1:03d}.yaml"
        consol_prev_obj_path = prev_iter_dir / f"consolidator_object_mapping_iter_{it-1:03d}.yaml"
        prev_path = interaction_prev_path if interaction_prev_path.exists() else consol_prev_path
        prev_obj_path = interaction_prev_obj_path if interaction_prev_obj_path.exists() else consol_prev_obj_path
        if prev_path.exists():
            try:
                prev_tax = yaml.safe_load(prev_path.read_text(encoding="utf-8"))
                if prev_tax is None:
                    logger.warning(f"consolidator: previous taxonomy file is empty at {prev_path}")
                    prev_tax = None
                else:
                    # Validate previous taxonomy structure
                    prev_ok, prev_msg = _is_valid_taxonomy_structure(prev_tax)
                    if not prev_ok:
                        logger.warning(f"consolidator: previous taxonomy structure invalid ({prev_msg}), will not use as fallback")
                        prev_tax = None
            except Exception as e:
                logger.warning(f"consolidator: failed to load previous taxonomy from {prev_path}: {e}")
                prev_tax = None
        
        if prev_obj_path.exists():
            try:
                prev_obj_mapping = yaml.safe_load(prev_obj_path.read_text(encoding="utf-8"))
                if prev_obj_mapping is None:
                    prev_obj_mapping = None
            except Exception as e:
                logger.warning(f"consolidator: failed to load previous object mapping from {prev_obj_path}: {e}")
                prev_obj_mapping = None
        else:
            logger.info(f"consolidator: previous object mapping not found at {prev_obj_path}")
        
        if not prev_path.exists():
            logger.info(f"consolidator: previous taxonomy not found at {prev_path}, falling back to pass-through")

    def _get_fallback_taxonomy(error_reason: str) -> tuple[dict[str, Any], dict[str, Any], str]:
        """
        Determine the correct fallback taxonomy and object mapping based on iteration and available taxonomies.
        
        Returns:
            tuple: (fallback_taxonomy, fallback_object_mapping, fallback_source_description)
        """
        if it > 1 and prev_tax is not None:
            fallback_tax = prev_tax
            fallback_obj = prev_obj_mapping if prev_obj_mapping is not None else {"iteration": it, "objects": []}
            source = f"previous taxonomy from iteration {it-1}"
            logger.error(f"consolidator: {error_reason}. Using fallback: {source}")
        else:
            fallback_tax = emp_tax
            fallback_obj = emp_obj_mapping if isinstance(emp_obj_mapping, dict) else {"iteration": it, "objects": []}
            if it > 1:
                source = "empirical taxonomy (previous taxonomy unavailable)"
                logger.error(f"consolidator: {error_reason}. Previous taxonomy unavailable, using empirical taxonomy as fallback.")
            else:
                source = "empirical taxonomy (iteration 1, no previous taxonomy)"
                logger.error(f"consolidator: {error_reason}. Using empirical taxonomy as fallback (iteration 1).")
        return fallback_tax, fallback_obj, source

    _provider, _model = resolve_provider_and_model()
    llm = _create_llm(provider=_provider, model=_model, temperature=0.0)

    # Build important_user_prompt section for the consolidator prompt
    iup = state.get("important_user_prompt", "").strip()
    iup_section = ""
    if iup:
        iup_section = (
            "\n\nIMPORTANT USER INSTRUCTION (applies to your analytical approach, NOT to output format(treat as priority constraint)):\n"
            f"{iup}\n"
        )

    system = (
        "You are the taxonomy consolidator responsible for consolidating and verifying taxonomies "
        "following the taxonomy development method of Nickerson et al. (2013).\n\n"
        "ROLE:\n"
        "- Methodologically rigorous expert in taxonomy development\n"
        "- Focused on coherence and traceability\n"
        "- Responsible for verifying, consolidating, and refining taxonomies across iterations\n"
        "- Responsible for ensuring consistency between taxonomy structure and object-characteristic mappings\n\n"
        "CONSTRAINTS:\n\n"
        "- IMPORTANT:Use the meta characteristic as a filter for relevancy to avoid including unrelated data just because it's availabe.\n"
        "Taxonomy structure:\n"
        "- All dimensions must be unique, non-redundant, and explain a specific aspect of the classified objects.\n"
        "- All characteristics must be unique within their dimension.\n"
        "- Dimension and characteristic names MUST be concise noun phrases (no sentences - maximum 1-3 words).\n"
        "- Enforce consistent IDs: dimensions as D1, D2, …; characteristics as D1.C1, D1.C2, …\n"
        "- When merging or restructuring, renumber IDs sequentially (no gaps, no duplicates).\n"
        "- Aim to keep total dimensions at or below 9. Exceed only if empirically necessary.\n"
        "- MECE condition (Nickerson et al., 2013): Within each dimension, characteristics must be\n"
        "  (a) Mutually Exclusive — an object is assignable to exactly one characteristic per dimension, and\n"
        "  (b) Collectively Exhaustive — characteristics together cover the relevant feature space so every object can be classified.\n\n"
        "Object validation:\n"
        "- For every object, verify it belongs to the same conceptual class defined by the meta-characteristic.\n"
        "  If not, remove it — Retain only those objects that ensures structural purity of the taxonomy.\n"
        "- Detect duplicate objects (same entity under different names) and merge into a single entry.\n"
        "- Every removal, edit, or merge must be documented in reasoning_long, event_log, and the object_validation output field.\n\n"
        "Object mapping (object-centric format):\n"
        "- Each object maps to exactly one characteristic per dimension (via dimension_mappings).\n"
        "- Verify each assignment is plausible. An object may be assigned logically without verbatim text evidence.\n"
        "- If characteristic_id is 'N/A', check if adding a new characteristic would resolve it before accepting.\n"
        "- Flag implausible assignments in event_log and correct them.\n"
        "- After processing, renumber object IDs sequentially (O1, O2, …).\n\n"
        "Interaction annotations:\n"
        "- If interaction_annotations is provided, those elements were edited or confirmed during a prior interaction session.\n"
        "- Do NOT modify, rename, delete, merge, or split any annotated element.\n"
        "- If you recommend changes to annotated elements, document the suggestion in changes_short and reasoning_long without applying it.\n\n"
        "Event logging:\n"
        "- Log every important decision as a short sentence in the event_log output field.\n"
        "- Examples: 'Dimension X rejected — too similar to Y', 'Objects O3 and O7 merged — same entity', 'Added characteristic D2.C4 to resolve N/A gap'.\n\n"
        "Boundaries:\n"
        "- Do NOT invent dimensions or characteristics unrelated to the topic or meta-characteristic.\n"
        "- Do NOT evaluate objective or subjective ending conditions.\n"
        "- Do NOT remove empirically grounded objects unless clearly outside the defined scope."
        + iup_section
    )

    # ---- Build iteration-dependent task instructions ----

    _shared_taxonomy_requirements = (
        "Taxonomy requirements:\n"
        "- Apply all taxonomy structure constraints from your role definition (IDs, naming, MECE, dimension limit).\n"
        "- Merge duplicates and resolve overlaps logically.\n"
    )

    _shared_mapping_requirements = (
        "Object mapping requirements:\n"
        "- Apply all object mapping constraints from your role definition.\n"
        "- Ensure each object has exactly one mapping entry per dimension in the final output.\n"
        "- The final mapping must cover all retained objects against the final set of dimensions.\n"
        "- When an object exhibits multiple behaviors in a dimension, possible solutions are:"
        "   - adding additional characteristic (e.g. \"hybrid...\")\n"
        "   - Define the dimension so that only the primary attribute is classified\n"
    )

    _shared_object_validation = (
        "Object validation:\n"
        "- Apply all object validation constraints from your role definition.\n"
        "- Fill the object_validation output field for every evaluated object (object_name, action, reasoning).\n"
    )

    _shared_output_instructions = (
        "Output:\n"
        "- Your output structure is defined by the function schema. Fill ALL fields completely.\n"
        "- merged_taxonomy: The final taxonomy with meta_characteristic, topic, and dimensions.\n"
        "- merged_object_mapping: The final object-centric mapping with iteration and objects.\n"
        "- object_validation: One entry per evaluated object (object_name, action, reasoning).\n"
        "- changes_short: Short summary of applied operations.\n"
        "- reasoning_long: One coherent explanation justifying all changes and the resulting structure.\n"
        "- event_log: List of short sentences, each describing one important decision.\n"
    )

    _shared_self_reflection = (
        "Self-reflection (internal, do NOT include in output):\n"
        "- Check whether dimensions and characteristics are logically coherent.\n"
        "- Check whether objects are assigned to appropriate characteristics.\n"
        "- Check whether evidence supports each assignment.\n"
        "- Check whether the taxonomy improves explanatory power without unnecessary complexity.\n"
        "- Check whether the structure allows future extension without rework.\n"
    )

    # ---- Shared task blocks (reused across iteration branches) ----

    _task_quality = (
        "== TASK 1: Quality Check, Structure Validation, Object Validation ==\n"
        "- Validate the structure of Taxonomy A against MECE and schema requirements.\n"
        "- Run object validation on all objects: verify scope, detect and merge duplicates.\n"
        "- Verify that every object is a genuine object within the scope of the topic and meta-characteristic. Remove only objects clearly outside scope; keep doubtful cases.\n"
        "- Verify that every object-characteristic assignment is plausible.\n"
        "- Apply minimal normalization (consistent IDs, concise noun-phrase names - no sentences - maximum 1-3 words) where needed.\n\n"
    )

    _task_taxonomy_synthesis = (
        "== TASK 2: Taxonomy Synthesis ==\n"
        "- Synthesize Taxonomy A and Taxonomy B into the best possible taxonomy.\n"
        "- Treat both taxonomies as equal inputs. Compare all dimensions and characteristics side by side. Synthesize the best out of both inputs.\n"
        "- For each element, decide whether it belongs based on its contribution to explanatory power,\n"
        "  measured against the topic and meta-characteristic.\n"
        "- You MAY remove dimensions or characteristics from Taxonomy B if Taxonomy A shows better fitting dimensions or characteristics\n"
        "  Elements that do not logically derive from the meta characteristic should be dropped regardless of which taxonomy they originate from.\n"
        "- An exclusion of any element from either taxonomy must be justified.\n"
        "- Allowed operations: adding, renaming, splitting, merging, deleting.\n"
        "- Every deletion or major restructuring must be justified in reasoning_long and event_log.\n"
    )

    _task_mapping_synthesis = (
        "== TASK 3: Object Mapping Synthesis ==\n"
        "- Merge object mappings from both taxonomies into a single consistent mapping.\n"
        "- Deduplicate: if an object from A describes the same entity as one from B,\n"
        "  merge into a single entry. Keep the most descriptive name. Combine source_document references (semicolon-separated). \n"
        "  Re-evaluate dimension_mappings: prefer the assignment with stronger reasoning; if conflicting, use your judgement.\n"
        "- Reassign object-characteristic relations where characteristics were merged, renamed, split, or moved.\n"
        "- Log each merge decision in event_log.\n\n"
    )

    _task_consistency = (
        "== TASK {n} (final): Consistency Verification ==\n"
        "- Verify that the final taxonomy and object mapping are internally consistent and match each other.\n"
        "- Every dimension_id and characteristic_id referenced in the mapping must exist in the taxonomy.\n"
        "- Every object must have exactly one mapping entry per dimension.\n"
        "- Mutual Exclusivity self-check: For each dimension, verify no two characteristics could logically "
        "co-occur on the same object. If they can, merge them or redefine their boundaries.\n"
        "- Collective Exhaustiveness self-check: Verify every object can be classified in every dimension "
        "without requiring N/A. If a gap exists, add a residual characteristic.\n"
        "- If inconsistencies are found, fix them and document the fix in changes_short and event_log.\n\n"
    )

    _shared_requirements_tail = (
        _shared_taxonomy_requirements + "\n"
        + _shared_mapping_requirements + "\n"
        + _shared_object_validation + "\n"
        + _shared_output_instructions + "\n"
        + _shared_self_reflection
    )

    _preamble_equal_weight = (
        "IMPORTANT: Both taxonomies are EQUALLY VALID inputs to the synthesis. Neither has automatic priority. "
        "Compare all dimensions and characteristics from both taxonomies side by side. "
        "An exclusion of any element from either taxonomy MUST be justified. "
        "The overriding criterion: Does a dimension/characteristic contribute to the taxonomy's usefulness "
        "as measured against the topic and meta-characteristic?\n\n"
    )

    _preamble_stability = (
        "STABILITY NOTE: Taxonomy B has been refined over multiple iterations. "
        "Structural disruptions (deleting entire dimensions, fundamental renames) require strong empirical "
        "evidence from Taxonomy A. However, new dimensions and characteristics from A must still be evaluated "
        "on equal footing — do not dismiss them merely because B is more mature.\n\n"
    )

    # ---- Assemble merge_instructions per iteration ----

    if it == 1:
        merge_instructions = (
            "You are given Taxonomy A (new empirical analysis) — the taxonomy and object-characteristic mapping "
            "produced by the empirical agent in this first iteration.\n"
            "There is no previous taxonomy to merge with.\n\n"
            + _task_quality
            + _task_consistency.format(n=2)
            + _shared_requirements_tail
        ).strip()

    elif it <= 3:
        merge_instructions = (
            "You are given two taxonomies and their object-characteristic mappings.\n"
            "- Both taxonomies are produced by empirical agents with different input.\n"
            "- Taxonomy A \n"
            "- Taxonomy B \n\n"
            + _preamble_equal_weight
            + _task_quality
            + _task_taxonomy_synthesis + "\n"
            + _task_mapping_synthesis
            + _task_consistency.format(n=4)
            + _shared_requirements_tail
        ).strip()

    else:  # it >= 4
        merge_instructions = (
            "You are given two taxonomies and their object-characteristic mappings:\n"
            "- Taxonomy A (new empirical analysis): produced by the empirical agent in this iteration.\n"
            "- Taxonomy B (previous synthesis): the consolidated result from previous iterations.\n\n"
            + _preamble_equal_weight
            + _preamble_stability
            + _task_quality
            + _task_taxonomy_synthesis + "\n"
            + _task_mapping_synthesis
            + _task_consistency.format(n=4)
            + _shared_requirements_tail
        ).strip()

    # Load cumulative user annotations from all prior interaction sessions
    cumulative_annotations = state.get("interaction_annotations", [])
    if not cumulative_annotations:
        cumulative_annotations = _load_cumulative_annotations(artifacts.outputs_dir)

    if it == 1:
        user_payload = {
            "iteration": it,
            "topic": topic,
            "meta_characteristic": meta,
            "Taxonomy A (new empirical analysis)": emp_tax,
            "Object Mapping A (new empirical analysis)": emp_obj_mapping,
        }
    else:
        user_payload = {
            "iteration": it,
            "topic": topic,
            "meta_characteristic": meta,
            "Taxonomy A": emp_tax,
            "Object Mapping A": emp_obj_mapping,
            "Taxonomy B": prev_tax,
            "Object Mapping B": prev_obj_mapping,
        }
    if cumulative_annotations:
        user_payload["interaction_annotations (DO NOT MODIFY these elements)"] = cumulative_annotations

    logger.info(f"consolidator: iteration={it} provider={_provider} model={_model} prev_present={prev_tax is not None}")

    if get_settings().debug_mode:
        try:
            _prompt_debug = {
                "system": system,
                "merge_instructions": merge_instructions,
                "user_payload": user_payload,
            }
            _prompt_path = artifacts.outputs_dir / f"debug_prompt_consolidator_iter_{it:03d}.yaml"
            _prompt_path.write_text(yaml.safe_dump(_prompt_debug, sort_keys=False, allow_unicode=True), encoding="utf-8")
            logger.info(f"consolidator: full prompt dumped to {_prompt_path}")
        except Exception as _dump_err:
            logger.warning(f"consolidator: could not dump prompt: {_dump_err}")

    try:
        llm_structured = llm.with_structured_output(consolidatorOutput, include_raw=True)
        if get_settings().debug_mode:
            try:
                import json as _json
                _schema_path = artifacts.outputs_dir / f"debug_tool_schema_iter_{it:03d}.json"
                _bound = getattr(llm_structured, 'bound', llm_structured)
                _tools = getattr(_bound, 'kwargs', {}).get('tools') or getattr(llm_structured, 'tools', None)
                if _tools:
                    _schema_path.write_text(_json.dumps(_tools, indent=2, ensure_ascii=False), encoding="utf-8")
                    logger.info(f"consolidator: tool schema dumped to {_schema_path}")
                else:
                    logger.warning("consolidator: could not extract tool schema from llm_structured (no .tools or .bound.kwargs['tools'])")
            except Exception as _schema_err:
                logger.warning(f"consolidator: could not dump tool schema: {_schema_err}")
        invoke_fn = lambda: llm_structured.invoke(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": merge_instructions},
                {"role": "user", "content": yaml.safe_dump(user_payload, sort_keys=False, allow_unicode=True)},
            ]
        )
        parsed, token_usage = call_llm_with_retry(invoke_fn, max_retries=2, backoff_seconds=1.5, log=logger)
        out = parsed.model_dump()
        # Inject notes into merged taxonomy
        if isinstance(out.get("merged_taxonomy"), dict):
            out["merged_taxonomy"]["notes"] = {"format": "prototype_v1"}

        obj_validation_raw: list = []

        # Validate required keys exist
        if not isinstance(out, dict):
            fallback_tax, fallback_obj, fallback_source = _get_fallback_taxonomy(f"LLM output is not a dict, got {type(out)}")
            merged = fallback_tax
            merged_obj = fallback_obj
            changes_short = [f"updating: LLM output format invalid, using fallback ({fallback_source})"]
            reasoning_long = f"LLM output was not in expected format (got {type(out)} instead of dict). Using fallback taxonomy: {fallback_source}."
            _add_event_log_entry(state, "consolidator", it, f"LLM output format invalid (got {type(out)}). Using fallback taxonomy: {fallback_source}.")
        else:
            merged = out.get("merged_taxonomy", {})
            merged_obj = out.get("merged_object_mapping", {})
            changes_short = out.get("changes_short", [])
            reasoning_long = out.get("reasoning_long", "")
            
            # Extract event_log entries from consolidator response
            event_log_raw = out.get("event_log", [])
            if isinstance(event_log_raw, list):
                for event_text in event_log_raw:
                    if isinstance(event_text, str) and event_text.strip():
                        _add_event_log_entry(state, "consolidator", it, event_text.strip())

            # Extract object_validation entries and log non-kept actions
            obj_validation_raw = out.get("object_validation", [])
            if isinstance(obj_validation_raw, list):
                for entry in obj_validation_raw:
                    if isinstance(entry, dict) and entry.get("action", "kept") != "kept":
                        _add_event_log_entry(
                            state, "consolidator", it,
                            f"Object validation: '{entry.get('object_name', '?')}' -> {entry.get('action', '?')} — {entry.get('reasoning', '')}"
                        )

            # Validate merged_taxonomy exists
            if not merged:
                fallback_tax, fallback_obj, fallback_source = _get_fallback_taxonomy("merged_taxonomy missing or empty in LLM output")
                merged = fallback_tax
                merged_obj = fallback_obj
                if not isinstance(changes_short, list):
                    changes_short = []
                changes_short.append(f"updating: merged_taxonomy missing from LLM output, using fallback ({fallback_source})")
                reasoning_long = (reasoning_long or "") + f"\n\nFallback applied: merged_taxonomy was missing or empty in LLM output. Using {fallback_source}."
            
            # Validate merged_object_mapping exists
            if not merged_obj or not isinstance(merged_obj, dict):
                fallback_tax, fallback_obj, fallback_source = _get_fallback_taxonomy("merged_object_mapping missing or invalid in LLM output")
                if not merged:  # Only use fallback for taxonomy if it wasn't already set
                    merged = fallback_tax
                merged_obj = fallback_obj
                if not isinstance(changes_short, list):
                    changes_short = []
                changes_short.append(f"updating: merged_object_mapping missing or invalid, using fallback ({fallback_source})")
                reasoning_long = (reasoning_long or "") + f"\n\nFallback applied: merged_object_mapping was missing or invalid in LLM output. Using {fallback_source}."

        ok2, msg2 = _is_valid_taxonomy_structure(merged) #checks if the merged taxonomy structure is valid.
        if not ok2: #if the merged taxonomy structure is invalid, it uses the fallback taxonomy.
            fallback_tax, fallback_obj, fallback_source = _get_fallback_taxonomy(f"merged taxonomy structure invalid: {msg2}")
            merged = fallback_tax
            merged_obj = fallback_obj
            changes_short = list(changes_short) if isinstance(changes_short, list) else []
            changes_short.append(f"updating: fallback to {fallback_source} due to invalid merged format ({msg2})")
            reasoning_long = (reasoning_long or "") + f"\n\nFallback applied: merged taxonomy structure was invalid ({msg2}). Using {fallback_source}."

        state["current_taxonomy"] = merged
        state["current_object_mapping"] = merged_obj if isinstance(merged_obj, dict) else {"iteration": it, "objects": []}
        state["consolidator_changes_short"] = changes_short if isinstance(changes_short, list) else [str(changes_short)]
        state["consolidator_reasoning_long"] = str(reasoning_long)

        # Accumulate token usage in state (iteration + total + per-agent)
        current_iter_usage = state.get("token_usage_iteration", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        state["token_usage_iteration"] = _accumulate_token_usage(current_iter_usage, token_usage)
        
        current_total_usage = state.get("token_usage_total", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        state["token_usage_total"] = _accumulate_token_usage(current_total_usage, token_usage)
        
        agent_usage = state.get("token_usage_by_agent", {}).get("consolidator", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        state.setdefault("token_usage_by_agent", {})["consolidator"] = _accumulate_token_usage(agent_usage, token_usage)
        
        logger.info(
            f"consolidator: token usage - prompt={token_usage['prompt_tokens']}, "
            f"completion={token_usage['completion_tokens']}, total={token_usage['total_tokens']}"
        )

        iter_dir = _get_iteration_dir(artifacts, it)
        out_tax = iter_dir / f"consolidator_taxonomy_iter_{it:03d}.yaml"
        out_tax.write_text(yaml.safe_dump(merged, sort_keys=False, allow_unicode=True), encoding="utf-8")

        out_obj = iter_dir / f"consolidator_object_mapping_iter_{it:03d}.yaml"
        out_obj.write_text(yaml.safe_dump(state.get("current_object_mapping", {}), sort_keys=False, allow_unicode=True), encoding="utf-8")

        out_changes = iter_dir / f"consolidator_changes_iter_{it:03d}.yaml"
        out_changes.write_text(
            yaml.safe_dump({"iteration": it, "changes_short": state["consolidator_changes_short"]}, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

        # Save object validation results
        if isinstance(obj_validation_raw, list) and obj_validation_raw:
            out_val = iter_dir / f"consolidator_object_validation_iter_{it:03d}.yaml"
            out_val.write_text(
                yaml.safe_dump({"iteration": it, "object_validation": obj_validation_raw}, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )

        # Save combined reasoning file
        empirical_reasoning = state.get("empirical_reasoning_long", "")
        consolidator_reasoning = state.get("consolidator_reasoning_long", "")
        _save_reasoning_file(artifacts, it, empirical_reasoning, consolidator_reasoning)
    
    except Exception as e:
        if isinstance(e, LLMRetryExhausted):
            _accumulate_tokens_from_retry(state, e.token_usage, "consolidator")
        # Structured error handling
        err = {"agent": "consolidator", "iteration": it, "error": str(e), "error_type": type(e).__name__}
        state.setdefault("errors", []).append(err)
        error_file = _get_iteration_dir(artifacts, it) / f"error_consolidator_iter_{it:03d}.yaml"
        error_file.write_text(yaml.safe_dump(err, sort_keys=False, allow_unicode=True), encoding="utf-8")
        logger.error(f"consolidator: error in iteration {it}: {e}")
        
        # Fallback: use previous taxonomy or empirical taxonomy
        fallback_tax, fallback_obj, fallback_source = _get_fallback_taxonomy(f"consolidator failed with exception: {e}")
        state["current_taxonomy"] = fallback_tax
        state["current_object_mapping"] = fallback_obj
        state["consolidator_changes_short"] = [f"updating: consolidator failed, using fallback ({fallback_source})"]
        state["consolidator_reasoning_long"] = f"consolidator encountered an error: {e}. Using fallback taxonomy: {fallback_source}."
        _add_event_log_entry(state, "consolidator", it, f"Error occurred: {e}. Using fallback taxonomy: {fallback_source}.")
        logger.warning(f"consolidator: using fallback taxonomy due to error")
        
        # Save fallback outputs
        iter_dir = _get_iteration_dir(artifacts, it)
        out_tax = iter_dir / f"consolidator_taxonomy_iter_{it:03d}.yaml"
        out_tax.write_text(yaml.safe_dump(fallback_tax, sort_keys=False, allow_unicode=True), encoding="utf-8")
        out_obj = iter_dir / f"consolidator_object_mapping_iter_{it:03d}.yaml"
        out_obj.write_text(yaml.safe_dump(fallback_obj, sort_keys=False, allow_unicode=True), encoding="utf-8")
        
        # Save combined reasoning file
        empirical_reasoning = state.get("empirical_reasoning_long", "")
        consolidator_reasoning = state.get("consolidator_reasoning_long", "")
        _save_reasoning_file(artifacts, it, empirical_reasoning, consolidator_reasoning)

    logger.info(f"consolidator: iteration={it} ops={len(state['consolidator_changes_short'])}")
    logger.info(f"consolidator changes_short: {state['consolidator_changes_short']}")
    return state



def validator_node(state: GraphState, artifacts: GraphArtifacts) -> GraphState: #validates the taxonomy.
    logger = get_logger("mas_taxonomy.graph.validator", log_dir=artifacts.logs_dir)

    it = int(state.get("iteration", 1)) #gets the iteration from the state.
    tax = state.get("current_taxonomy", {}) #gets the current taxonomy from the state.
    obj_mapping = state.get("current_object_mapping", {}) #gets the current object mapping from the state.
    oecs = state.get("objective_ending_conditions", []) #gets the objective ending conditions from the state.
    secs = state.get("subjective_ending_conditions", []) #gets the subjective ending conditions from the state.

    # Skeleton-check: check formal aspects
    structure_ok, structure_msg = _is_valid_taxonomy_structure(tax)

    _provider, _model = resolve_provider_and_model()
    llm = _create_llm(provider=_provider, model=_model, temperature=0.0)

    # Build important_user_prompt section for the validator prompt
    iup = state.get("important_user_prompt", "").strip()
    iup_section = ""
    if iup:
        iup_section = (
            "\n\nIMPORTANT USER INSTRUCTION (applies to your analytical approach, NOT to output format(treat as priority constraint)):\n"
            f"{iup}\n"
        )

    system = (
        "You are a validator for a taxonomy development process following Nickerson et al. (2013).\n\n"
        "ROLE:\n"
        "- Independent evaluator of taxonomy quality\n"
        "- You receive both the taxonomy structure and the object-characteristic mapping to evaluate ending conditions\n"
        "- You do NOT modify the taxonomy or object mapping\n\n"
        "TASK:\n"
        "- Evaluate all objective ending conditions (clear yes/no criteria) with short evidence.\n"
        "- Evaluate all subjective ending conditions with a recommendation (met/not met) and 1-3 sentences of reasoning.\n"
        "- You MUST provide a short summary_recommendation including:\n"
        "  (a) Count of met conditions (e.g., '5 out of 8 OECs met, 4 out of 6 SECs met')\n"
        "  (b) Most important actionable recommendations where human intervention would be useful or necessary regarding the ending conditions\n"
        "  (c) Whether another iteration should be run\n"
        "- The summary should be concise but informative."
        "- Log every important validation decision in the event_log output field as a short sentence (e.g. Objective ending condition OEC1 not met: taxonomy has only 2 dimensions, requires at least 3', 'Subjective ending condition SEC2 met: taxonomy is comprehensive but could benefit from additional characteristics')\n"

        "CONSTRAINTS:\n"
        "- Do NOT modify the taxonomy or object mapping.\n"
        "- N/A assignments: If any object has characteristic_id 'N/A' in any dimension, flag it as a taxonomy gap\n"
        "  in event_log and reflect it in exhaustiveness-related ending condition evaluations.\n"
        "- Evidence for objective conditions must reference specific taxonomy content (e.g., dimension names, overlaps, missing parts).\n"
        "- ending_conditions_met is true only if ALL objective ending conditions are met AND structure_ok is true."
        + iup_section
    )

    changes_short = state.get("consolidator_changes_short", [])
    if it == 1:
        changes_section = (
            "\n This is iteration 1. "
            "The entire taxonomy was created from scratch — all dimensions and characteristics "
            "are new by definition. Affected OECs are therefore NOT met.\n"
        )
    else:
        changes_section = (
            "\nConsolidator changes applied in this iteration "
            "(relevant, if dimensions/characteristics were added, removed, merged, split, or renamed):\n"
            + yaml.safe_dump(changes_short, sort_keys=False, allow_unicode=True)
            + "\n"
        )

    user = f"""
    Iteration: {it}
    Topic: {state.get('topic', '')}
    Meta-characteristic: {state.get('meta_characteristic', '')}

    Objective ending conditions to evaluate:
    {yaml.safe_dump(oecs, sort_keys=False, allow_unicode=True)}

    Subjective ending conditions to evaluate:
    {yaml.safe_dump(secs, sort_keys=False, allow_unicode=True)}

    Taxonomy (do not change):
    {yaml.safe_dump(tax, sort_keys=False, allow_unicode=True)}

    Object-characteristic mapping (do not change):
    {yaml.safe_dump(obj_mapping, sort_keys=False, allow_unicode=True)}
    {changes_section}
    Output:
    - Your output structure is defined by the function schema. Fill all fields completely.
    - For objective conditions: set met true/false with short evidence referencing taxonomy content.
    - For subjective conditions: set met true/false with 1-3 sentence recommendation.
    """.strip()

    logger.info(f"validator: iteration={it} provider={_provider} model={_model} oecs={len(oecs)} secs={len(secs)}")
    
    try:
        llm_structured = llm.with_structured_output(ValidatorOutput, include_raw=True)
        parsed, usage = call_llm_with_retry(
            lambda: llm_structured.invoke([{"role": "system", "content": system}, {"role": "user", "content": user}]),
            max_retries=2,
            backoff_seconds=1.5,
            log=logger,
        )
        report = parsed.model_dump()
        
        # Extract event_log entries and summary_recommendation from validator response
        summary_recommendation = ""
        if isinstance(report, dict):
            # Extract summary_recommendation
            summary_recommendation = str(report.get("summary_recommendation", "")).strip()
            
            # Extract event_log entries
            event_log_raw = report.get("event_log", [])
            if isinstance(event_log_raw, list):
                for event_text in event_log_raw:
                    if isinstance(event_text, str) and event_text.strip():
                        _add_event_log_entry(state, "validator", it, event_text.strip())
            
            # Add summary_recommendation to event_log
            if summary_recommendation:
                _add_event_log_entry(state, "validator", it, f"Summary: {summary_recommendation}")
        
        # Accumulate token usage in state (iteration + total + per-agent)
        current_iter_usage = state.get("token_usage_iteration", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        state["token_usage_iteration"] = _accumulate_token_usage(current_iter_usage, usage)
        
        current_total_usage = state.get("token_usage_total", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        state["token_usage_total"] = _accumulate_token_usage(current_total_usage, usage)
        
        agent_usage = state.get("token_usage_by_agent", {}).get("validator", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        state.setdefault("token_usage_by_agent", {})["validator"] = _accumulate_token_usage(agent_usage, usage)
        
        logger.info(f"validator: token usage - prompt={usage.get('prompt_tokens', 0)}, completion={usage.get('completion_tokens', 0)}, total={usage.get('total_tokens', 0)}")
    
    except Exception as e:
        if isinstance(e, LLMRetryExhausted):
            _accumulate_tokens_from_retry(state, e.token_usage, "validator")
        # Structured error handling
        err = {"agent": "validator", "iteration": it, "error": str(e), "error_type": type(e).__name__}
        state.setdefault("errors", []).append(err)
        error_file = _get_iteration_dir(artifacts, it) / f"error_validator_iter_{it:03d}.yaml"
        error_file.write_text(yaml.safe_dump(err, sort_keys=False, allow_unicode=True), encoding="utf-8")
        logger.error(f"validator: error in iteration {it}: {e}")
        
        # Fallback report
        oec_total = len(oecs)
        sec_total = len(secs)
        summary_recommendation = f"Validator error occurred: {e}. Cannot provide summary recommendation. Using fallback validation report."
        report = {
            "iteration": it,
            "structure_ok": structure_ok,
            "structure_msg": structure_msg,
            "objective_ending_conditions": [
                {"id": ec.get("id"), "name": ec.get("name"), "met": False, "evidence": f"validator failed: {str(e)}"}
                for ec in oecs
            ],
            "subjective_ending_conditions": [
                {"id": sec.get("id"), "name": sec.get("name"), "met": False, "recommendation": f"validator failed: {str(e)}"}
                for sec in secs
            ],
            "ending_conditions_met": False,
            "summary_recommendation": summary_recommendation,
        }
        _add_event_log_entry(state, "validator", it, f"Validator error occurred: {e}. Using fallback validation report.")
        _add_event_log_entry(state, "validator", it, f"Summary: {summary_recommendation}")

    # Save: structure_ok overrides baseline, but consolidate logically
    if isinstance(report, dict):
        report["iteration"] = it
        report["structure_ok"] = bool(report.get("structure_ok", structure_ok)) and structure_ok
        report["structure_msg"] = str(report.get("structure_msg", structure_msg))

        oec_items = report.get("objective_ending_conditions", [])
        if not isinstance(oec_items, list):
            oec_items = []
        report["objective_ending_conditions"] = oec_items

        sec_items = report.get("subjective_ending_conditions", [])
        if not isinstance(sec_items, list):
            sec_items = []
        report["subjective_ending_conditions"] = sec_items

        all_met = True
        for item in oec_items:
            if not isinstance(item, dict) or item.get("met") is not True:
                all_met = False
                break
        report["ending_conditions_met"] = bool(report.get("structure_ok")) and all_met
        
        # Ensure summary_recommendation is present, generate if missing
        if "summary_recommendation" not in report or not report.get("summary_recommendation"):
            # Count met conditions
            oec_met_count = sum(1 for item in oec_items if isinstance(item, dict) and item.get("met") is True)
            sec_met_count = sum(1 for item in sec_items if isinstance(item, dict) and item.get("met") is True)
            oec_total = len(oec_items)
            sec_total = len(sec_items)
            report["summary_recommendation"] = f"{oec_met_count} out of {oec_total} OECs met, {sec_met_count} out of {sec_total} SECs met. Review unmet conditions for recommendations."
        else:
            # Ensure summary_recommendation is a string
            report["summary_recommendation"] = str(report.get("summary_recommendation", ""))
    else:
        oec_total = len(oecs)
        sec_total = len(secs)
        report = {
            "iteration": it,
            "structure_ok": structure_ok,
            "structure_msg": structure_msg,
            "objective_ending_conditions": [
                {"id": ec.get("id"), "name": ec.get("name"), "met": False, "evidence": "validator output invalid"}
                for ec in oecs
            ],
            "subjective_ending_conditions": [
                {"id": sec.get("id"), "name": sec.get("name"), "met": False, "recommendation": "validator output invalid"}
                for sec in secs
            ],
            "ending_conditions_met": False,
            "summary_recommendation": f"0 out of {oec_total} OECs met, 0 out of {sec_total} SECs met. Validator output invalid - cannot provide recommendations.",
        }
        _add_event_log_entry(state, "validator", it, f"Summary: {report['summary_recommendation']}")

    state["validation_report"] = report

    out_report = _get_iteration_dir(artifacts, it) / f"validator_report_iter_{it:03d}.yaml"
    out_report.write_text(yaml.safe_dump(report, sort_keys=False, allow_unicode=True), encoding="utf-8")

    logger.info(
        f"validator: iteration={it} structure_ok={report.get('structure_ok')} ending_conditions_met={report.get('ending_conditions_met')}"
    )
    return state


# ---------------------------------------------------------------------------
# Iteration control – routing, decision, interaction, next-iteration nodes
# ---------------------------------------------------------------------------

def route_iteration_decision(state: GraphState) -> str:
    """Route based on user's iteration decision set by iteration_decision_node."""
    decision = state.get("user_decision", "")
    if decision == "interaction":
        return "interaction_node"
    elif decision == "next_iteration":
        return "prepare_next_iteration"
    # "end" or any fallback → finish
    return "end"


def iteration_decision_node(state: GraphState, artifacts: GraphArtifacts) -> dict:
    """
    Present user with post-iteration options inside the graph.
    Saves iteration state and event log, then prompts for next action.
    Routes to: interaction_node, prepare_next_iteration, or END.
    """
    it = int(state.get("iteration", 1))
    run_dir = Path(state.get("run_dir", ""))

    # Save iteration state snapshot and event log at the end of each iteration
    _save_iteration_state(state, artifacts, it)
    _save_event_log(artifacts, state.get("event_log", []), it)

    # Display iteration summary
    report = state.get("validation_report", {})
    summary = report.get("summary_recommendation", "")

    typer.echo("")
    typer.echo("=" * 60)
    typer.echo(f"  Iteration {it} Complete")
    typer.echo("=" * 60)

    if summary:
        typer.echo(f"\nValidator: {summary}")

    # Display token usage for this iteration and cumulative total
    iter_usage = state.get("token_usage_iteration", {})
    total_usage = state.get("token_usage_total", {})
    if iter_usage.get("total_tokens", 0) > 0:
        typer.echo(f"\nIteration {it} tokens: {iter_usage.get('total_tokens', 0):,}")
    if total_usage.get("total_tokens", 0) > 0:
        typer.echo(f"Cumulative tokens: {total_usage.get('total_tokens', 0):,}")

    # Present options with local validation loop
    while True:
        typer.echo("")
        typer.echo("=" * 60)
        typer.echo("  Next step:")
        typer.echo("=" * 60)
        typer.echo("1) Manual edit with Interaction Agent")
        typer.echo("2) Start next iteration (skip editing)")
        typer.echo("3) End taxonomy development and output final taxonomy")
        choice = typer.prompt("Your choice [1/2/3]", default="1")

        if choice == "1":
            # Update run_config status
            cfg = load_run_config(run_dir)
            cfg["last_user_decision"] = "interaction"
            cfg["status"] = "interaction_mode"
            save_run_config(run_dir, cfg)
            return {"user_decision": "interaction"}

        elif choice == "2":
            # Confirm readiness for next iteration
            typer.echo(f"\nPreparing for iteration {it + 1}...")
            typer.echo("Please add new empirical PDF files to the input directory if needed.")
            confirm = typer.prompt("Ready to continue? [y/n]", default="y").strip().lower()
            if confirm in ("y", "yes"):
                return {"user_decision": "next_iteration"}
            else:
                typer.echo("Cancelled. Choose again.")
                continue

        elif choice == "3":
            # Update run_config status to finished
            cfg = load_run_config(run_dir)
            cfg["last_user_decision"] = "finish"
            cfg["status"] = "finished"
            save_run_config(run_dir, cfg)
            return {"user_decision": "end"}

        else:
            typer.echo("Unexpected answer, try again.")


def interaction_node(state: GraphState, artifacts: GraphArtifacts) -> dict:
    """
    Interaction mode as a graph node. Wraps run_interactive_interaction
    and returns state deltas for taxonomy, object mapping, annotations,
    conversation, token usage, and important_user_prompt.
    """
    it = int(state.get("iteration", 1))
    run_dir = Path(state.get("run_dir", ""))
    run_id = state.get("run_id", "")
    important_user_prompt = state.get("important_user_prompt", "")
    meta_characteristic = state.get("meta_characteristic", "")

    try:
        result = run_interactive_interaction(
            run_dir=run_dir,
            run_id=run_id,
            iteration=it,
            current_taxonomy=state.get("current_taxonomy", {}),
            current_object_mapping=state.get("current_object_mapping", {}),
            event_log=state.get("event_log", []),
            outputs_dir=artifacts.outputs_dir,
            important_user_prompt=important_user_prompt,
            meta_characteristic=meta_characteristic,
        )
    except (KeyboardInterrupt, typer.Exit):
        typer.echo("\nInteraction mode cancelled.")
        return {"user_decision": ""}

    delta: dict[str, Any] = {"user_decision": ""}  # Reset for re-prompt

    # Always write back interaction files to state (may have object mapping changes even without taxonomy changes)
    delta["current_taxonomy"] = result["interaction_taxonomy"]
    delta["current_object_mapping"] = result["interaction_object_mapping"]

    if result.get("taxonomy_modified"):
        existing_annotations = list(state.get("interaction_annotations", []))
        existing_annotations.extend(result.get("interaction_annotations", []))
        delta["interaction_annotations"] = existing_annotations
        typer.echo(f"\n{len(result.get('interaction_annotations', []))} interaction annotation(s) recorded for next iteration.")

    # Update important_user_prompt if changed
    new_iup = result.get("important_user_prompt", "")
    if new_iup and new_iup != important_user_prompt:
        delta["important_user_prompt"] = new_iup
        # Persist to run_config
        run_dir_path = Path(state.get("run_dir", ""))
        cfg = load_run_config(run_dir_path)
        cfg["important_user_prompt"] = new_iup
        save_run_config(run_dir_path, cfg)

    # Update meta_characteristic if changed
    new_mc = result.get("meta_characteristic", "")
    if new_mc and new_mc != meta_characteristic:
        delta["meta_characteristic"] = new_mc
        run_dir_path = Path(state.get("run_dir", ""))
        cfg = load_run_config(run_dir_path)
        cfg["meta_characteristic"] = new_mc
        save_run_config(run_dir_path, cfg)
        # Propagate event_log updates (the tool already appended to the live list)
        delta["event_log"] = state.get("event_log", [])

    # Accumulate conversation history
    interaction_conv = result.get("conversation_history", [])
    if interaction_conv:
        existing_conv = list(state.get("agent_conversation", []))
        existing_conv.extend(interaction_conv)
        delta["agent_conversation"] = existing_conv

    # Accumulate token usage (iteration + total + per-agent)
    interaction_tokens = result.get("token_usage", {})
    if interaction_tokens and interaction_tokens.get("total_tokens", 0) > 0:
        cur_iter = state.get("token_usage_iteration", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        delta["token_usage_iteration"] = _accumulate_token_usage(cur_iter, interaction_tokens)

        cur_total = state.get("token_usage_total", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        delta["token_usage_total"] = _accumulate_token_usage(cur_total, interaction_tokens)

        by_agent = dict(state.get("token_usage_by_agent", {}))
        agent_cur = by_agent.get("interaction", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        by_agent["interaction"] = _accumulate_token_usage(agent_cur, interaction_tokens)
        delta["token_usage_by_agent"] = by_agent

    return delta


def prepare_next_iteration_node(state: GraphState, artifacts: GraphArtifacts) -> dict:
    """
    Prepare state for the next iteration:
    - Run ingest inline to pick up new documents
    - Increment iteration counter
    - Reset per-iteration fields (empirical outputs, validation, timing, token_usage_iteration)
    - Preserve cross-iteration fields (topic, meta, conversation, annotations, token_usage_total, etc.)
    """
    it = int(state.get("iteration", 1))
    run_dir = Path(state.get("run_dir", ""))
    run_id = state.get("run_id", "")
    new_it = it + 1

    # Run ingest inline to pick up any new/changed documents
    if artifacts.ingest_fn:
        typer.echo("\n" + "-" * 60)
        typer.echo("  Running ingest for new documents...")
        typer.echo("-" * 60)
        artifacts.ingest_fn(run_dir, run_id, new_it)
    else:
        typer.echo("Warning: No ingest function provided. Documents will not be reloaded.")

    # Reload documents from manifest, filtered to only those added in this iteration
    docs = [d for d in load_run_documents(run_dir) if d.get("iteration_added") == new_it]

    # Update run_config on disk
    cfg = load_run_config(run_dir)
    cfg["iteration"] = new_it
    cfg["status"] = "running"
    cfg["last_user_decision"] = "next"
    save_run_config(run_dir, cfg)

    typer.echo(f"\nIteration incremented to {new_it}. Starting empirical analysis...")

    # Return state delta: reset per-iteration fields, preserve cross-iteration fields
    return {
        "iteration": new_it,
        "documents": docs,
        # Reset per-iteration fields
        "empirical_taxonomy": {},
        "empirical_object_mapping": {},
        "empirical_reasoning_long": "",
        "consolidator_changes_short": [],
        "consolidator_reasoning_long": "",
        "validation_report": {},
        "token_usage_iteration": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "run_started_at": datetime.now().isoformat(timespec="seconds"),
        "user_decision": "",
        # Cross-iteration fields are preserved automatically (not in this delta):
        # run_id, thread_id, topic, meta_characteristic, ending_conditions,
        # agent_conversation, event_log, interaction_annotations, important_user_prompt,
        # token_usage_total, token_usage_by_agent, current_taxonomy, current_object_mapping
    }


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph(artifacts: GraphArtifacts, checkpointer=None):
    """
    Build the taxonomy development graph with an internal iteration loop.

    Topology:
      consultation → empirical_worker → consolidator → validator → iteration_decision
      iteration_decision → interaction_node → iteration_decision   (edit loop)
      iteration_decision → prepare_next_iteration → empirical_worker  (iteration loop)
      iteration_decision → END                                        (finish)

    Consultation is the entry point and handles skip logic internally.
    For iteration > 1 the loop bypasses consultation entirely
    (prepare_next_iteration routes directly to empirical_worker).
    """
    g = StateGraph(GraphState)

    # Core pipeline nodes
    g.add_node("consultation", lambda s: consultation_agent_node(s, artifacts))
    g.add_node("empirical_worker", lambda s: empirical_worker_node(s, artifacts))
    g.add_node("consolidator", lambda s: consolidator_node(s, artifacts))
    g.add_node("validator", lambda s: validator_node(s, artifacts))

    # Iteration control nodes
    g.add_node("iteration_decision", lambda s: iteration_decision_node(s, artifacts))
    g.add_node("interaction_node", lambda s: interaction_node(s, artifacts))
    g.add_node("prepare_next_iteration", lambda s: prepare_next_iteration_node(s, artifacts))

    # Entry: consultation (handles skip internally for iteration 1)
    g.set_entry_point("consultation")

    # Core pipeline edges
    g.add_edge("consultation", "empirical_worker")
    g.add_edge("empirical_worker", "consolidator")
    g.add_edge("consolidator", "validator")
    g.add_edge("validator", "iteration_decision")

    # Iteration decision routing (conditional edges)
    g.add_conditional_edges("iteration_decision", route_iteration_decision, {
        "interaction_node": "interaction_node",
        "prepare_next_iteration": "prepare_next_iteration",
        "end": END,
    })

    # Interaction loops back to iteration_decision for re-prompt
    g.add_edge("interaction_node", "iteration_decision")

    # Next iteration loops back to empirical_worker (bypasses consultation)
    g.add_edge("prepare_next_iteration", "empirical_worker")

    # Compile with optional checkpointer
    if checkpointer:
        return g.compile(checkpointer=checkpointer)
    return g.compile()


def run_graph_for_run(
    run_dir: Path,
    run_id: str,
    skip_consultation: bool = False,
    checkpointer=None,
    important_user_prompt: str = "",
    consultation_token_usage: dict | None = None,
    consultation_conversation: list | None = None,
    ingest_fn: Callable[[Path, str], None] | None = None,
) -> dict[str, Any]:
    """
    Run the full taxonomy development graph (with internal iteration loop).
    The graph loops internally via iteration_decision → prepare_next_iteration → empirical_worker.
    Returns when the user selects "end taxonomy development" inside the graph.
    """
    artifacts = GraphArtifacts(
        run_dir=run_dir,
        logs_dir=run_dir / "logs",
        outputs_dir=run_dir / "outputs",
        ingest_fn=ingest_fn,
    )
    artifacts.logs_dir.mkdir(parents=True, exist_ok=True)
    artifacts.outputs_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_run_config(run_dir)
    docs = load_run_documents(run_dir)

    it = int(cfg.get("iteration", 1))

    t0 = _time.time()  # Keep as variable for duration calculation
    
    # Derive thread_id from run_id for conversation persistence
    thread_id = f"thread_{run_id}"

    # Load cumulative user annotations from prior interaction sessions
    cumulative_annotations = _load_cumulative_annotations(artifacts.outputs_dir)

    # If interaction files from the previous iteration exist, load them as current state
    prev_iter_dir = artifacts.outputs_dir / f"iter_{it-1:03d}"
    preloaded_taxonomy = None
    preloaded_obj_mapping = None
    if it > 1:
        interaction_tax_path = prev_iter_dir / f"interaction_taxonomy_iter_{it-1:03d}.yaml"
        consol_tax_path = prev_iter_dir / f"consolidator_taxonomy_iter_{it-1:03d}.yaml"
        tax_path = interaction_tax_path if interaction_tax_path.exists() else consol_tax_path
        if tax_path.exists():
            try:
                preloaded_taxonomy = yaml.safe_load(tax_path.read_text(encoding="utf-8"))
            except Exception:
                preloaded_taxonomy = None

        interaction_obj_path = prev_iter_dir / f"interaction_object_mapping_iter_{it-1:03d}.yaml"
        consol_obj_path = prev_iter_dir / f"consolidator_object_mapping_iter_{it-1:03d}.yaml"
        obj_path = interaction_obj_path if interaction_obj_path.exists() else consol_obj_path
        if obj_path.exists():
            try:
                preloaded_obj_mapping = yaml.safe_load(obj_path.read_text(encoding="utf-8"))
            except Exception:
                preloaded_obj_mapping = None

    # Initialize token usage counters (seed with consultation tokens if provided)
    init_token_iter = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    init_token_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if consultation_token_usage:
        for k in init_token_iter:
            init_token_iter[k] += consultation_token_usage.get(k, 0)
            init_token_total[k] += consultation_token_usage.get(k, 0)

    # Resolve important_user_prompt: parameter > config > empty
    iup = important_user_prompt or cfg.get("important_user_prompt", "")

    # Include consultation conversation in initial agent_conversation
    initial_conversation: list[dict[str, Any]] = []
    if consultation_conversation:
        initial_conversation = list(consultation_conversation)

    state: GraphState = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "iteration": it,
        "thread_id": thread_id,
        "topic": cfg.get("topic", ""),
        "meta_characteristic": cfg.get("meta_characteristic", ""),
        "objective_ending_conditions": cfg.get("objective_ending_conditions", []),
        "subjective_ending_conditions": cfg.get("subjective_ending_conditions", []),
        "important_user_prompt": iup,
        "documents": docs,
        "run_started_at": datetime.now().isoformat(timespec="seconds"),
        "token_usage_iteration": init_token_iter,
        "token_usage_total": init_token_total,
        "token_usage_by_agent": {
            "consultation": consultation_token_usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "empirical": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "consolidator": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "validator": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "interaction": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        },
        "errors": [],
        "event_log": [],
        "agent_conversation": initial_conversation,
        "interaction_annotations": cumulative_annotations,
        "skip_consultation": skip_consultation,
        "consultation_completed": skip_consultation,  # If skipping, mark as completed
        "user_decision": "",
    }

    # If interaction/consolidator files from previous iteration exist, pre-load them
    if preloaded_taxonomy:
        state["current_taxonomy"] = preloaded_taxonomy
    if preloaded_obj_mapping:
        state["current_object_mapping"] = preloaded_obj_mapping

    # Mark run as running
    cfg["status"] = "running"
    save_run_config(run_dir, cfg)

    graph = build_graph(artifacts, checkpointer=checkpointer)

    # Invoke graph — it loops internally until user selects "end"
    if checkpointer:
        config = {"configurable": {"thread_id": thread_id}}
        final_state = graph.invoke(state, config=config)
    else:
        final_state = graph.invoke(state) #invokes the graph with the initial state and saves the final state

    end_ts = datetime.now().isoformat(timespec="seconds")
    final_state["run_finished_at"] = end_ts
    final_state["duration_seconds"] = round(_time.time() - t0, 3)

    # Remove large "text" fields from documents before saving final state
    if "documents" in final_state:
        for doc in final_state["documents"]:
            if isinstance(doc, dict) and "text" in doc:
                doc.pop("text", None)

    # Save final state file
    final_it = int(final_state.get("iteration", it))
    out_state = artifacts.outputs_dir / "graph_state_final.yaml"
    out_state.write_text(yaml.safe_dump(final_state, sort_keys=False, allow_unicode=True), encoding="utf-8")

    # Save final event log (full overwrite)
    _save_event_log(artifacts, final_state.get("event_log", []), final_it)

    return {
        "final_state": final_state,
        "state_file": str(out_state),
        "token_usage_total": final_state.get("token_usage_total", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
        "token_usage_iteration": final_state.get("token_usage_iteration", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
    }
