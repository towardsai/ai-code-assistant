#!/usr/bin/env python3
"""Golden-set evaluation of the upgrades (Article 6, Part 5 / Step 7).

Run a fixed set of repo questions through three configurations and compare
answer quality, tokens, and latency per query. The AST tools should win on the
symbol/cross-file questions; the router should leave quality flat while cutting
tokens and latency.

This is a thin, dependency-light harness: it drives the real agent against a
target repo and records per-query latency. "Quality" here is left for a human
(or a stronger LLM judge) to score from the saved answers -- the point of the
harness is the side-by-side, not an automated grade.

Usage:
    python -m eval.golden_eval --repo /path/to/target/repo
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

from agent import CodingAgent

# A 20-question golden set over THIS repo: 10 symbol lookups, 5 explanations,
# 5 cross-file questions. Swap in questions for your own target repo.
GOLDEN = [
    # --- 10 symbol lookups ---
    {"q": "Where is process_user_input defined?", "type": "symbol"},
    {"q": "Where is the route function defined?", "type": "symbol"},
    {"q": "Where is find_symbol defined?", "type": "symbol"},
    {"q": "Where is fim_complete defined?", "type": "symbol"},
    {"q": "Where is index_repo defined?", "type": "symbol"},
    {"q": "List the methods of the CodingAgent class.", "type": "symbol"},
    {"q": "Where is _safe_path defined and what is its signature?", "type": "symbol"},
    {"q": "Where is _escapes_workspace defined?", "type": "symbol"},
    {"q": "Where is the Spinner class defined?", "type": "symbol"},
    {"q": "Where is ast_chunks defined?", "type": "symbol"},
    # --- 5 explanations ---
    {"q": "What does the run_command tool restrict?", "type": "explain"},
    {"q": "Where is the iteration cap enforced in the loop?", "type": "explain"},
    {"q": "How does the router decide between the small and large model?", "type": "explain"},
    {"q": "Why does autocomplete use the base model instead of the instruct model?", "type": "explain"},
    {"q": "How does the agent recover tool calls that a model emits as text?", "type": "explain"},
    # --- 5 cross-file ---
    {"q": "Which tools touch the filesystem and where are they dispatched?", "type": "cross"},
    {"q": "How does a user request flow from main.py to a model call?", "type": "cross"},
    {"q": "Which modules import from agent/symbols.py and why?", "type": "cross"},
    {"q": "Where is the RAG path wired in, from the CLI flag to search_code?", "type": "cross"},
    {"q": "What are the workspace-containment checks and which files enforce them?", "type": "cross"},
]

CONFIGS = {
    "baseline": dict(use_router=False),                      # file tools + grep only*
    "baseline+ast": dict(use_router=False),                  # + AST symbol tools
    "baseline+ast+router": dict(use_router=True),            # + router
}
# *In this build AST tools ship in the base agent; the "baseline" row stands in
#  for grep-only behavior by instructing the model not to use the symbol tools.


def build_agent(config_name: str, repo: pathlib.Path, model: str) -> CodingAgent:
    opts = CONFIGS[config_name]
    agent = CodingAgent(model=model, **opts)
    agent.current_directory = repo.resolve()
    if config_name == "baseline":
        # Simulate grep-only: drop the AST tools from the surface AND from the
        # system prompt -- advertising a tool the model cannot call makes it
        # emit dead find_symbol calls instead of falling back to grep.
        original = agent.get_tools_definition
        agent.get_tools_definition = lambda: [
            t for t in original() if t["function"]["name"] not in ("find_symbol", "list_symbols")]
        original_prompt = agent.get_system_prompt
        agent.get_system_prompt = lambda: (
            original_prompt()
            .replace("- find_symbol / list_symbols (AST-aware: prefer these over grep for code structure)", "")
            .replace("Prefer find_symbol/list_symbols for 'where is X defined' and 'what's in this file'. ", ""))
    return agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="Target repo to evaluate over")
    # Deliberately NOT the agent's qwen3:32b default: 60 runs on a 32B model
    # takes hours on a laptop. Pass --model qwen3:32b for the full-fidelity run.
    ap.add_argument("--model", default="qwen2.5-coder:7b")
    ap.add_argument("--out", default="eval_results.json")
    args = ap.parse_args()

    repo = pathlib.Path(args.repo)
    records = []
    for config_name in CONFIGS:
        for item in GOLDEN:
            agent = build_agent(config_name, repo, args.model)
            agent.messages.append({"role": "system", "content": agent.get_system_prompt()})
            t0 = time.time()
            answer = agent.process_user_input(item["q"])
            latency = time.time() - t0
            records.append({"config": config_name, "type": item["type"],
                            "q": item["q"], "latency_s": round(latency, 2),
                            "tokens": agent.last_token_count(),
                            "routed_to": agent.last_route[0], "answer": answer})
            print(f"[{config_name}] ({latency:.1f}s) {item['q']}")

    pathlib.Path(args.out).write_text(json.dumps(records, indent=2))
    print(f"\nWrote {len(records)} records to {args.out}")


if __name__ == "__main__":
    main()
