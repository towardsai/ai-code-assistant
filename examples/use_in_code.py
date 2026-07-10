#!/usr/bin/env python3
"""Use the local code assistant as a library — programmatic examples.

Run from anywhere:
    python examples/use_in_code.py

Needs Ollama serving (`ollama serve`) for the model-backed parts (3, 5, 6, 7);
parts 1, 2, and 4 are pure-Python and need no model.
"""
import pathlib
import sys

# Make `agent` importable no matter where this is run from.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from agent import CodingAgent, router, symbols  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent          # this repo
SAMPLE = REPO / "examples" / "sample_project"


def banner(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


# 1) AST symbol tools — no model needed -------------------------------------
banner("1. find_symbol / list_symbols (no model)")
hits = symbols.find_symbol(SAMPLE, "process_payment")
print("find_symbol('process_payment'):")
for h in hits:
    print(f"  {h['file'].split('/')[-1]}:{h['start']}-{h['end']}")
print("list_symbols(calculator.py):")
for s in symbols.list_symbols(SAMPLE / "calculator.py"):
    print(f"  {s['kind']:8} {s['name']:16} L{s['start']}-{s['end']}")


# 2) The router — no model needed -------------------------------------------
banner("2. router.route(...) (no model)")
for q in ["what is recursion?", "where is process_payment defined?",
          "read config.py and fix the bug"]:
    model, tools = router.route(q, big_model="qwen2.5-coder:7b")
    print(f"  {q!r:45} -> {model}  tools={tools}")


# 3) Drive the agent over a repo --------------------------------------------
banner("3. CodingAgent over a repo (needs Ollama)")
agent = CodingAgent(model="qwen2.5-coder:7b", debug=True)
agent.current_directory = SAMPLE                      # point it at any repo
agent.messages.append({"role": "system", "content": agent.get_system_prompt()})
answer = agent.process_user_input(
    "Use find_symbol to locate process_payment and tell me its line range.")
print("\nANSWER:", answer)


# 4) Call individual tools directly -----------------------------------------
banner("4. Calling tools directly (no model)")
direct = CodingAgent()
direct.current_directory = SAMPLE
print("read_file:", direct.read_file("calculator.py")[:50].replace("\n", " "), "...")
print("list_files:", direct.list_files(".").replace("\n", "  "))
print("run_command('grep def calculator.py'):")
print("  " + direct.run_command("grep def calculator.py").strip().replace("\n", "\n  "))
print("blocked:", direct.run_command("cat /etc/passwd"))


# 5) FIM autocomplete (needs the -base model) -------------------------------
banner("5. autocomplete (FIM, needs Ollama + -base model)")
fill = direct.autocomplete(
    prefix="def factorial(n):\n    if n <= 1:\n        return 1\n    return ",
    suffix="\n\nprint(factorial(5))\n")
print("  prefix: 'def factorial(n): ... return '")
print("  FILL ->", repr(fill))


if __name__ == "__main__":
    print("\nDone.")
