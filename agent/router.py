"""Rule-based multi-model router (Article 6, Part 4 / Step 6).

Cheap, local, single-digit-millisecond routing: short questions with no tool
verb go to the small fast model; anything that reads, writes, or runs goes to
the larger model with the tool set attached. The decision is made, not faked --
but a production router would classify intent far more carefully than a verb
match (it weighs cost, model health, and availability too).
"""
from __future__ import annotations

import re
from typing import Tuple

# Verbs that imply the request needs the tool set (and therefore the big model).
# Includes edit/create verbs ("add a multiply method...") because an agentic
# request with no tools attached can't act -- the model just narrates. In a
# coding REPL a false "use tools" is cheap; a false "no tools" breaks the agent.
TOOL_VERBS = frozenset({
    "read", "write", "create", "edit", "run", "find", "list", "fix", "test",
    "search", "grep", "delete", "refactor", "implement", "open", "show",
    "add", "append", "insert", "remove", "rename", "update", "change",
    "modify", "replace", "make", "generate", "move", "document", "comment",
})

# Code-domain nouns that mark a request as being about *this repo's code* even
# with no tool verb or identifier ("the Calculator class", "a multiply method").
CODE_KEYWORDS = frozenset({
    "class", "method", "function", "def", "module", "import", "decorator",
    "variable", "parameter", "argument", "return", "exception", "attribute",
})

SMALL_MODEL_DEFAULT = "qwen2.5-coder:1.5b"

# Verbs that imply a WRITE (not just tool use). Measured motivation (Article 6
# Step 7): on the edit benchmark the 7B coder scored 3/5 while qwen3:32b went
# 3/3 -- edits are where the strongest available model earns its latency, and
# questions are where it doesn't. Routed to edit_model when one is configured.
WRITE_VERBS = frozenset({
    "write", "create", "edit", "fix", "add", "append", "insert", "remove",
    "rename", "update", "change", "modify", "replace", "refactor",
    "implement", "delete", "make", "generate", "move",
})

_WORD = re.compile(r"[a-z]+")

# Signals that a question is about *this repo's code* even without a tool verb,
# so it still needs the tools (and the big model). Cheap regex, ~microseconds.
_CODE_SIGNALS = (
    re.compile(r"\w+_\w+"),                 # snake_case identifier (process_payment)
    re.compile(r"[a-z]\w*[A-Z]\w*"),        # camelCase / PascalCase (CodingAgent)
    re.compile(r"\.[A-Za-z]{1,5}\b"),       # a file extension (config.py)
    re.compile(r"`[^`]+`"),                 # a `code` span in backticks
)


def looks_like_code_question(text: str) -> bool:
    words = set(_WORD.findall(text.lower()))
    return bool(words & CODE_KEYWORDS) or any(p.search(text) for p in _CODE_SIGNALS)


def route(user_input: str, big_model: str,
          small_model: str = SMALL_MODEL_DEFAULT,
          edit_model: str = None) -> Tuple[str, bool]:
    """Pick a model and decide whether to attach tools.

    Returns ``(model_name, needs_tools)``. Three tiers when ``edit_model`` is
    set: write-implying requests go to the edit model, other repo/tool requests
    go to the big model, and short generic questions go to the small no-tools
    model. Whole-word matching on verbs keeps "address" from tripping "add".
    """
    text = (user_input or "").lower()
    words = set(_WORD.findall(text))
    if edit_model and words & WRITE_VERBS:
        return edit_model, True          # edits: strongest available model
    needs_tools = bool(words & TOOL_VERBS) or looks_like_code_question(user_input or "")
    if not needs_tools and len(text) < 80:
        return small_model, False        # short, generic: small fast model
    return big_model, True               # chat / agent: larger model + tools
