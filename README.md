# Local Code Assistant

> A build-along, privacy-first AI coding agent that runs entirely on your machine — the companion code for **Article 6: Design an AI Code Assistant (Autocomplete, Chat, and Agent)**.

This repo is the practical build from Part 6 of the article, grown across the article's three problems:

1. **Agent over a repo** — file/command tools in the Article 5 think-act-observe loop (Steps 1–3)
2. **Smarter chat exploration** — AST-aware `find_symbol` / `list_symbols` via tree-sitter (Steps 4–5)
3. **Multi-model routing** — a small fast model for short questions, a larger one for tool use (Step 6)
4. **Optional FIM autocomplete** — fill-in-the-middle with a base code model (Step 8)
5. **Optional RAG over code** — AST chunking + Chroma, only when exploration stops scaling (Step 9)

Everything runs locally on [Ollama](https://ollama.com). The one exception is the optional Exa web search,
which is **off by default** and, when enabled with `--web`, redacts common credential patterns and requires
approval for the exact query before sending it to Exa.

> This is **not** the same as the [`local-cursor`](https://github.com/towardsai/local-cursor) repo that the
> earlier draft referenced. It is a separate, hardened, modular build with AST tools, a router, FIM, optional
> RAG, a test suite, and an eval harness.

---

## What's different from `local-cursor`

`local-cursor` is a single `main.py` with file/command tools. This build adds, and corrects:

| Area | `local-cursor` | This repo |
| --- | --- | --- |
| Structure | one file | modular `agent/` package + tests + eval |
| Symbol search | grep only | AST-aware `find_symbol` / `list_symbols` (tree-sitter) |
| Models | one model | rule-based router (small ↔ large) |
| Autocomplete | — | FIM path (`qwen2.5-coder:*-base`) |
| RAG | — | optional AST-chunked Chroma index |
| **Edits** | whole-file overwrite only | targeted `edit_file` (exact-match replace) + **diff approval gate** |
| **Path safety** | `resolve()` only (escapable) | `resolve()` **+ containment check** |
| **Command safety** | whitelist + `shell=True` (**bypassable**) | whitelist + `shlex` argv, **no shell** + sensitive-path block |
| Tests | — | `pytest` suite for the deterministic guards |

The last two rows are real security fixes — see [Security](#security).

---

## Requirements

- **Python ≥ 3.9** (uses `pathlib.Path.is_relative_to`, added in 3.9)
- [Ollama](https://ollama.com) running locally
- ~25 GB disk for the default models (or ~5 GB with the `qwen2.5-coder:7b` fallback)

## Quick start

```bash
git clone https://github.com/towardsai/ai-code-assistant.git
cd ai-code-assistant

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Models (chat+agent / router / autocomplete)
ollama pull qwen3:32b                 # main model (chat + agent) — ~20 GB; went 3/3 on the edit benchmark
ollama pull qwen2.5-coder:1.5b        # small model for the router
ollama pull qwen2.5-coder:1.5b-base   # base model for FIM autocomplete (Step 8)
ollama pull nomic-embed-text          # only for the optional RAG step
# ollama pull qwen2.5-coder:7b        # laptop-friendly fallback (3/5 on edits, much faster)

ollama serve                          # keep running in another terminal

# Drive the agent against the CURRENT directory
python main.py                        # or: python main.py --model qwen2.5-coder:7b on smaller machines
```

> The agent operates on your **current working directory**. `cd` into the repo you want it to work on first.

### CLI flags

```
--model TEXT            Main Ollama model (default qwen3:32b; qwen2.5-coder:7b for smaller machines)
--small-model TEXT      Small model for routing (default qwen2.5-coder:1.5b)
--edit-model TEXT       Optional stronger model for write/edit requests (e.g. qwen3:32b,
                        which went 3/3 on the edit benchmark where the 7B went 3/5)
--no-router             Disable the multi-model router
--rag                   Enable the optional RAG-over-code tool
--web                   Enable the optional Exa web search (off by default; leaves the machine)
--max-iterations INT    Agent loop iteration cap (default 5)
--temperature FLOAT     Sampling temperature (default 0.3 — measured best for tool-call
                        reliability: 0.0 loops on its own errors, 0.8 drifts and typos)
--debug                 Print routing + tool-call diagnostics
```

---

## Layout

```
agent/
  core.py          CodingAgent: the loop, file/command tools, security guards, router + symbol wiring
  symbols.py       tree-sitter find_symbol / list_symbols (Steps 4-5)
  router.py        rule-based multi-model router (Step 6)
  autocomplete.py  fill-in-the-middle (Step 8)
  rag.py           optional AST-chunked Chroma index (Step 9)
main.py            CLI entry point
eval/golden_eval.py  golden-set evaluation harness (Step 7)
tests/             pytest suite for the deterministic guards (no Ollama needed)
examples/sample_project/  a tiny repo for tests and demos
```

## Tests

The deterministic parts (path containment, command whitelist, AST tools, router) are unit-tested and need **no Ollama**:

```bash
pytest -q
```

## Autocomplete (Step 8)

```python
from agent import CodingAgent
a = CodingAgent()
print(a.autocomplete("def factorial(n):\n    if n <= 1:\n        return 1\n    return ",
                      "\n\nprint(factorial(5))\n"))
# -> 'n*factorial(n-1)'
```

FIM uses the **base** model (`qwen2.5-coder:1.5b-base`), not the instruct one — the instruct tag applies a
chat template and explains the code instead of filling the gap.

## RAG (Step 9, optional)

```bash
python main.py --rag        # exposes a `search_code` tool backed by an AST-chunked Chroma index
```

Only worth it once the repo outgrows grep and symbol lookup. Indexing and querying use the **same** embedding
model so the vectors live in one space (a common bug is to `add` custom embeddings but `query` with text, which
silently falls back to Chroma's default embedder).

---

## Security

Two guards make this safe enough to point at a real repo. Both are intentionally minimal — a team deployment
wraps the whole thing in a container.

- **Path containment.** Every file path is resolved and then checked with `is_relative_to(working_dir)`.
  `resolve()` alone normalizes `../` and symlinks but does **not** contain them; the containment check rejects
  `../../etc/passwd`, absolute paths, and symlink escapes.
- **Sensitive-file filtering.** File reads, symbol discovery, RAG indexing, glob results, directory listings,
  and allowed commands reject common credential and private-key paths.
- **Command execution.** `run_command` allows only a read-only whitelist (`ls`, `grep`, `cat`, …), parses the
  command with `shlex` and runs the **argv list without a shell**, so `cat x; rm -rf .` cannot chain a second
  command past the check. Each path-like argument is contained to the workspace too, so `cat /etc/passwd` and
  `find /etc` are rejected — an allowed command can't read outside the project. Sensitive paths (`.env`, keys,
  `.ssh`, …) are blocked even for allowed commands. `find_files` likewise rejects absolute / `..` glob patterns.
- **Outbound approval.** Optional web search is deny-by-default for library use. The CLI redacts common secret
  shapes, displays the exact outbound query, and requires explicit approval before making the request.

These guards make the agent safe to point at a local repo; they are **not** a substitute for a real sandbox.
A team deployment runs the whole thing in a container — see the article's Part 3.

Turning on the test loop from Part 3 means adding `pytest` to the whitelist — the first non-read-only capability
you grant, and the right place to require explicit approval.

## License

MIT — see [LICENSE](LICENSE).
