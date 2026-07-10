#!/usr/bin/env python3
"""CLI entry point for the local code assistant (Article 6, Part 6)."""
import click

from agent import CodingAgent


@click.command()
@click.option("--model", default="qwen3:32b", help="Main Ollama model (chat + agent); use qwen2.5-coder:7b on smaller machines.")
@click.option("--small-model", default="qwen2.5-coder:1.5b", help="Small model for routing.")
@click.option("--edit-model", default=None, help="Optional stronger model for write/edit requests (e.g. qwen3:32b).")
@click.option("--no-router", is_flag=True, help="Disable the multi-model router.")
@click.option("--rag", is_flag=True, help="Enable the optional RAG-over-code tool.")
@click.option("--web", is_flag=True, help="Enable the optional Exa web search (leaves the machine).")
@click.option("--max-iterations", default=5, help="Agent loop iteration cap.")
@click.option("--temperature", default=0.3, help="Sampling temperature (0.3 measured best for tool-call reliability).")
@click.option("--debug", is_flag=True, help="Print routing and tool-call diagnostics.")
def main(model, small_model, edit_model, no_router, rag, web, max_iterations, temperature, debug):
    """Run the local coding agent against the current working directory."""
    agent = CodingAgent(
        model=model,
        small_model=small_model,
        edit_model=edit_model,
        use_router=not no_router,
        use_rag=rag,
        use_web=web,
        max_iterations=max_iterations,
        temperature=temperature,
        debug=debug,
    )
    agent.run()


if __name__ == "__main__":
    main()
