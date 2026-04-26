from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
import yaml

from langgraph.checkpoint.memory import MemorySaver

from mas_taxonomy.graph.skeleton import (
    run_graph_for_run,
    run_interactive_consultation,
    run_ending_conditions_consultation,
    _append_to_conversation_file,
    _extract_consultation_insights,
    run_taxonomy_intake_questionnaire,
    build_important_user_prompt_intake_no_llm,
    build_important_user_prompt_intake_llm_consultation,
    build_intake_questionnaire_block_for_agent,
)
from mas_taxonomy.config import get_settings
from mas_taxonomy.io.pdf_loader import extract_text_from_pdf, list_pdfs, sha256_file
from mas_taxonomy.llm_utils import resolve_provider_and_model, COST_PER_MILLION
from mas_taxonomy.logging.logger import get_logger
from mas_taxonomy.run_config import RunConfig, default_objective_ending_conditions, default_subjective_ending_conditions, load_run_config, save_run_config


app = typer.Typer(add_completion=False)


def _make_run_id(run_id: Optional[str]) -> str: #creates a run id for the run, if no run id is provided
    if run_id and run_id.strip():
        return run_id.strip()
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")



@app.command("list-inputs") #creates cli command to list the input pdfs.
def list_inputs() -> None:
    s = get_settings()
    pdfs = list_pdfs(s.input_pdfs_dir)

    if not pdfs:
        typer.echo(f"no pdfs found in: {s.input_pdfs_dir}")
        return

    typer.echo(f"found pdfs in: {s.input_pdfs_dir}")
    for p in pdfs:
        typer.echo(f"- {p.name}")



@app.command("configure-run") #creates cli command with function to configure the run.
def configure_run(
    run_id: str = typer.Option(..., help="example: run_001"),
    topic: Optional[str] = typer.Option(None, help="Topic of the taxonomy"),
    meta_characteristic: Optional[str] = typer.Option(None, help="Meta-Characteristic after Nickerson"),
    use_defaults: bool = typer.Option(True, help="If true: Nickerson-Defaults (placeholders) use"),
    add_objective_ec: list[str] = typer.Option([], help="Additional objective ending conditions (only name/text)"),
    add_subjective_ec: list[str] = typer.Option([], help="Additional subjective ending conditions (only name/text)"),
) -> None:
    s = get_settings()
    run_dir = s.runs_dir / run_id
    
    # Create run directory if it doesn't exist
    if not run_dir.exists():
        typer.echo(f"Creating run directory: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "logs").mkdir(exist_ok=True)
        (run_dir / "outputs").mkdir(exist_ok=True)
        (run_dir / "extracted").mkdir(exist_ok=True)
    
    # Taxonomy setup path: no-LLM vs LLM-supported criteria consultation (both use the same intake form first)
    typer.echo("\n" + "=" * 60)
    typer.echo("How would you like to start taxonomy setup?")
    typer.echo("=" * 60)
    typer.echo(
        "1) No LLM consultation: A clear understanding of taxonomy goals "
        "(particularly topic and meta-characteristics) is already in place"
    )
    typer.echo(
        "2) LLM consultation: Support for the precise formulation of taxonomy criteria "
        "(topic and meta-characteristic refined with an agent)"
    )
    while True:
        choice = typer.prompt("Your choice [1/2]", default="1")
        if choice in ("1", "2"):
            break
        typer.echo("Unexpected answer, try again.")

    typer.echo("\n" + "-" * 60)
    typer.echo("  Intake questionnaire")
    typer.echo("-" * 60)
    intake = run_taxonomy_intake_questionnaire(
        default_topic=(topic or "").strip(),
        default_meta_characteristic=(meta_characteristic or "").strip(),
    )

    consultation_used = False
    consultation_conv: list = []
    initial_agent_conversation: list = []
    important_user_prompt = ""
    consultation_token_usage: dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    objective = default_objective_ending_conditions() if use_defaults else []
    subjective = default_subjective_ending_conditions() if use_defaults else []

    if choice == "2":
        intake_agent_context = build_intake_questionnaire_block_for_agent(intake)
        intake_iup_seed = build_important_user_prompt_intake_llm_consultation(intake)
        try:
            consultation_result = run_interactive_consultation(
                run_dir=run_dir,
                run_id=run_id,
                standard_objective_conditions=objective,
                standard_subjective_conditions=subjective,
                intake_agent_context=intake_agent_context,
                intake_important_user_prompt=intake_iup_seed,
            )
            topic = consultation_result["topic"]
            meta_characteristic = consultation_result["meta_characteristic"]
            objective = consultation_result.get("objective_ending_conditions", objective)
            subjective = consultation_result.get("subjective_ending_conditions", subjective)
            important_user_prompt = consultation_result.get("important_user_prompt", "")
            consultation_token_usage = consultation_result.get("token_usage", consultation_token_usage)
            consultation_used = True
            consultation_conv = consultation_result.get("conversation_history", [])
            initial_agent_conversation = consultation_conv
        except (KeyboardInterrupt, typer.Exit):
            typer.echo("\nConsultation cancelled.")
            raise typer.Exit(code=0)
    else:
        topic = intake["topic"]
        meta_characteristic = intake["meta_characteristic"]
        important_user_prompt = build_important_user_prompt_intake_no_llm(intake)

        ec_result = run_ending_conditions_consultation(
            topic=topic,
            meta_characteristic=meta_characteristic,
            standard_objective_conditions=objective,
            standard_subjective_conditions=subjective,
            existing_important_user_prompt=important_user_prompt,
        )
        objective = ec_result["objective_ending_conditions"]
        subjective = ec_result["subjective_ending_conditions"]
        important_user_prompt = ec_result.get("important_user_prompt", important_user_prompt)
        consultation_token_usage = ec_result.get("token_usage", consultation_token_usage)
        initial_agent_conversation = ec_result.get("conversation_history", [])

        if ec_result.get("ec_agent_used"):
            typer.echo(typer.style("\n  Consolidating important user prompt from consultation...", fg=typer.colors.CYAN))
            rebuilt = _extract_consultation_insights(
                conversation_history=initial_agent_conversation,
                topic=topic,
                meta_characteristic=meta_characteristic,
                existing_iup=important_user_prompt,
            )
            if rebuilt.strip():
                important_user_prompt = rebuilt.strip()
                typer.echo(typer.style("  ✓ Important user prompt updated.", fg=typer.colors.GREEN))

    # Add custom ending conditions
    for i, txt in enumerate(add_objective_ec, start=1):
        objective.append({
            "id": f"USER_OEC_{i:02d}", 
            "name": txt.strip(),
            "description": "User-provided objective ending condition.",
        })

    for i, txt in enumerate(add_subjective_ec, start=1):
        subjective.append({
            "id": f"USER_SEC_{i:02d}",
            "name": txt.strip(),
            "question": f"User-provided subjective ending condition: {txt.strip()}",
        })

    # Create run configuration with ending conditions
    cfg = RunConfig( #creates the run configuration.
        run_id=run_id,
        topic=topic.strip(),
        meta_characteristic=meta_characteristic.strip(),
        iteration=1,
        objective_ending_conditions=objective,
        subjective_ending_conditions=subjective,
    ).to_dict()

    # Apply important_user_prompt and consultation metadata to run config
    if consultation_used:
        cfg["consultation_used"] = True
    if important_user_prompt:
        cfg["important_user_prompt"] = important_user_prompt

    path = save_run_config(run_dir, cfg)

    typer.echo(f"\nrun_config.yaml saved: {path}")
    typer.echo(f"topic: {cfg['topic']}")
    typer.echo(f"meta_characteristic: {cfg['meta_characteristic']}")
    typer.echo(f"objective_ending_conditions: {len(cfg['objective_ending_conditions'])}")
    typer.echo(f"subjective_ending_conditions: {len(cfg.get('subjective_ending_conditions', []))}")

    # --- Chain: ask user if they want to continue with ingest + run graph ---
    typer.echo("\n" + "=" * 60)
    typer.echo("Configuration complete. What would you like to do next?")
    typer.echo("=" * 60)
    typer.echo("1) Continue with ingest + run graph (recommended)")
    typer.echo("2) Stop here (run ingest and run-graph separately later)")
    while True:
        chain_choice = typer.prompt("Your choice [1/2]", default="1")
        if chain_choice in ("1", "2"):
            break
        typer.echo("Unexpected answer, try again.")

    if chain_choice == "1":
        # --- Ingest phase (reuse existing run_dir, no double creation) ---
        typer.echo("\n" + "-" * 60)
        typer.echo("  Starting Ingest Phase")
        typer.echo("-" * 60)
        _run_ingest_for_dir(run_dir, run_id, iteration=1)

        # --- Run graph phase ---
        typer.echo("\n" + "-" * 60)
        typer.echo("  Starting Graph Execution")
        typer.echo("-" * 60)
        _run_graph_for_dir(
            run_dir=run_dir,
            run_id=run_id,
            use_checkpointer=True,
            important_user_prompt=important_user_prompt,
            consultation_token_usage=consultation_token_usage,
            consultation_conversation=initial_agent_conversation if initial_agent_conversation else None,
        )


def _run_ingest_for_dir(run_dir: Path, run_id: str, iteration: int = 1) -> None:
    """
    Core ingest logic: extract text from PDFs and append to the manifest.
    Existing document entries from previous iterations are preserved; only
    PDFs whose sha256 is not yet in the manifest are extracted and added.
    Each new document entry is tagged with `iteration_added` so the empirical
    worker can filter to only the papers introduced in the current iteration.
    """
    s = get_settings()
    logs_dir = run_dir / "logs"
    extracted_dir = run_dir / "extracted"
    # Ensure subdirs exist (idempotent, no double creation of run_dir itself)
    logs_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "outputs").mkdir(parents=True, exist_ok=True)

    logger = get_logger("mas_taxonomy", log_dir=logs_dir)

    pdfs = list_pdfs(s.input_pdfs_dir)
    if not pdfs:
        typer.echo(f"no pdfs found in: {s.input_pdfs_dir}")
        return

    # Load existing manifest to preserve prior iterations' documents
    manifest_path = run_dir / "input_manifest.yaml"
    if manifest_path.exists():
        existing = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        existing_documents: list[dict] = existing.get("documents", [])
    else:
        existing_documents = []

    # Build a set of already-ingested sha256 hashes to skip re-extraction
    known_hashes: set[str] = {d["sha256"] for d in existing_documents if "sha256" in d}

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_dir": str(s.input_pdfs_dir),
        "documents": existing_documents,
    }

    logger.info(f"ingest started: {run_id}")
    logger.info(f"number of pdfs: {len(pdfs)} | already ingested: {len(existing_documents)}")

    new_count = 0
    for pdf in pdfs:
        try:
            digest = sha256_file(pdf)

            if digest in known_hashes:
                logger.info(f"SKIP (already ingested): {pdf.name}")
                continue

            text, page_count = extract_text_from_pdf(pdf)

            out_txt = extracted_dir / f"{pdf.stem}.txt"
            out_txt.write_text(text, encoding="utf-8")

            manifest["documents"].append(
                {
                    "file_name": pdf.name,
                    "sha256": digest,
                    "pages": page_count,
                    "extracted_text_file": str(out_txt),
                    "iteration_added": iteration,
                }
            )
            known_hashes.add(digest)
            new_count += 1

            logger.info(f"OK: {pdf.name} | pages: {page_count} | text: {out_txt.name}")
        except Exception as e:
            logger.exception(f"error: {pdf.name}: {e}")
            manifest["documents"].append(
                {
                    "file_name": pdf.name,
                    "error": str(e),
                }
            )

    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8")

    existing_count = len(manifest["documents"]) - new_count
    typer.echo(f"extracted text: {extracted_dir}")
    typer.echo(f"manifest: {manifest_path} ({new_count} new, {existing_count} carried over)")


def _run_graph_for_dir(
    run_dir: Path,
    run_id: str,
    use_checkpointer: bool = True,
    important_user_prompt: str = "",
    consultation_token_usage: dict | None = None,
    consultation_conversation: list | None = None,
) -> None:
    """
    Core run-graph logic: execute the LangGraph pipeline.
    The graph loops internally (iteration_decision → prepare_next_iteration → empirical_worker)
    and handles interaction mode as a graph node. No external while-loop needed.
    """
    cfg = load_run_config(run_dir)

    # Determine if we should skip consultation (config already has topic/meta)
    has_config = bool(cfg.get("topic", "").strip() and cfg.get("meta_characteristic", "").strip())
    skip_consultation = has_config

    # Initialize checkpointer if requested
    checkpointer = MemorySaver() if use_checkpointer else None

    # Resolve and display the LLM provider/model for this run
    provider, model = resolve_provider_and_model()
    typer.echo("")
    typer.echo("=" * 60)
    typer.echo(f"  LLM Provider : {provider}")
    typer.echo(f"  Model        : {model}")
    typer.echo("=" * 60)
    typer.echo("")

    if checkpointer:
        typer.echo("Using MemorySaver checkpointer for iteration persistence")
    if skip_consultation:
        typer.echo("Using existing configuration (skipping consultation)")

    typer.echo(f"Running graph for run: {run_id}")

    # Invoke the graph — it loops internally until the user selects "end"
    result = run_graph_for_run(
        run_dir=run_dir,
        run_id=run_id,
        skip_consultation=skip_consultation,
        checkpointer=checkpointer,
        important_user_prompt=important_user_prompt or cfg.get("important_user_prompt", ""),
        consultation_token_usage=consultation_token_usage,
        consultation_conversation=consultation_conversation,
        ingest_fn=_run_ingest_for_dir,
    )

    typer.echo(f"\nGraph run completed for: {run_dir}")
    typer.echo(f"State saved: {result['state_file']}")

    final_state = result.get("final_state", {})
    final_it = int(final_state.get("iteration", 1))

    # Display total token usage (all iterations)
    total_usage = result.get("token_usage_total", {})
    if total_usage and total_usage.get("total_tokens", 0) > 0:
        typer.echo("")
        typer.echo("Total Token Usage (all iterations):")
        typer.echo(f"  Prompt tokens: {total_usage.get('prompt_tokens', 0):,}")
        typer.echo(f"  Completion tokens: {total_usage.get('completion_tokens', 0):,}")
        typer.echo(f"  Total tokens: {total_usage.get('total_tokens', 0):,}")

        pricing = COST_PER_MILLION.get(model, {"input": 0.0, "output": 0.0})
        prompt_cost = (total_usage.get("prompt_tokens", 0) / 1_000_000) * pricing["input"]
        completion_cost = (total_usage.get("completion_tokens", 0) / 1_000_000) * pricing["output"]
        total_cost = prompt_cost + completion_cost
        typer.echo(f"  Model: {model}  (${pricing['input']}/M in, ${pricing['output']}/M out)")
        typer.echo(f"  Estimated cost: ${total_cost:.4f} (prompt: ${prompt_cost:.4f}, completion: ${completion_cost:.4f})")

    # Display per-agent token usage
    token_usage_by_agent = final_state.get("token_usage_by_agent", {})
    if token_usage_by_agent:
        typer.echo("\n  Per-agent token usage:")
        for agent_name, agent_usage in token_usage_by_agent.items():
            if agent_usage.get("total_tokens", 0) > 0:
                typer.echo(f"    {agent_name}: {agent_usage.get('total_tokens', 0):,} tokens")

    # Display final output
    _display_final_output(run_dir, final_it)

    # Export conversation YAML (accumulated naturally in state across all phases)
    outputs_dir = run_dir / "outputs"
    conversation = final_state.get("agent_conversation", [])
    if conversation:
        _append_to_conversation_file(outputs_dir, conversation)
        typer.echo(f"  Conversation log exported: {outputs_dir / 'all_agent_conversation.yaml'}")


@app.command("ingest") #creates cli command with function to extract the text from the input pdfs.
def ingest(
    run_id: Optional[str] = typer.Option(None, help="Optional. example: run_001"),
) -> None:
    """Standalone ingest command. Can also be called as part of the chained configure-run flow."""
    s = get_settings()
    rid = _make_run_id(run_id)
    # Ensure run_dir exists for standalone ingest
    run_dir = s.runs_dir / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    _run_ingest_for_dir(run_dir, rid, iteration=1)
    typer.echo(f"run created: {run_dir}")


def _display_final_output(run_dir: Path, iteration: int) -> None:
    """Display final taxonomy and validator report paths."""
    iter_dir = run_dir / "outputs" / f"iter_{iteration:03d}"
    interaction_tax = iter_dir / f"interaction_taxonomy_iter_{iteration:03d}.yaml"
    consol_tax = iter_dir / f"consolidator_taxonomy_iter_{iteration:03d}.yaml"
    tax_path = interaction_tax if interaction_tax.exists() else consol_tax
    val_path = iter_dir / f"validator_report_iter_{iteration:03d}.yaml"

    typer.echo("")
    typer.echo("Final Output:")
    if tax_path.exists():
        typer.echo(f"  Final taxonomy: {tax_path}")
    else:
        typer.echo("  Final taxonomy not found.")

    if val_path.exists():
        typer.echo(f"  Validator report: {val_path}")
    else:
        typer.echo("  Validator report not found.")


@app.command("run-graph") #creates cli command with function to run the graph.
def run_graph(
    run_id: str = typer.Option(..., help="example: run_001"),
    use_checkpointer: bool = typer.Option(True, help="if true: use MemorySaver checkpointer for conversation persistence"),
) -> None:
    """Standalone run-graph command. Can also be called as part of the chained configure-run flow."""
    try:
        s = get_settings()
        run_dir = s.runs_dir / run_id
        if not run_dir.exists():
            typer.echo(f"Run not found: {run_dir}")
            raise typer.Exit(code=1)

        typer.echo(f"Loading run config from: {run_dir}")
        cfg = load_run_config(run_dir)
        important_user_prompt = cfg.get("important_user_prompt", "")
        consultation_token_usage = cfg.get("consultation_token_usage", None)

        _run_graph_for_dir(
            run_dir=run_dir,
            run_id=run_id,
            use_checkpointer=use_checkpointer,
            important_user_prompt=important_user_prompt,
            consultation_token_usage=consultation_token_usage,
        )
    except FileNotFoundError as e:
        typer.echo(f"Error: Required file not found: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error running graph: {e}", err=True)
        import traceback
        typer.echo(traceback.format_exc(), err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
