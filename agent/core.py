#!/usr/bin/env python3
"""Local code assistant -- the agent core (Article 6, Steps 1-6, 8-9).

A local CLI coding agent in the Article 5 think-act-observe loop, pointed at a
repo and a terminal. It grows across the article's three problems: file and
command tools (Steps 1-3), AST symbol tools (Steps 4-5), a model router
(Step 6), optional FIM autocomplete (Step 8), and an optional RAG layer
(Step 9). Everything runs locally on Ollama and no source code leaves the
machine -- the one exception is the optional Exa web search, which is off by
default and requires approval for its redacted outbound query when enabled.
"""
from __future__ import annotations

import glob
import json
import os
import pathlib
import re
import shlex
import subprocess
import threading
import time
from typing import Any, Optional

import dotenv
import requests
from colorama import Fore, Style, init
from openai import OpenAI

from . import autocomplete as fim
from . import router
from . import symbols
from .security import is_sensitive_path, redact_outbound_secrets, resolve_in_workspace

init(autoreset=True)
dotenv.load_dotenv()

# --- Execution policy (Part 3) -------------------------------------------------
# Read-only commands that run without a prompt. Nothing here deletes, installs,
# reaches the network, or runs a test suite -- turning on the test loop means
# adding `pytest` here, the first non-read-only capability you grant.
# `find` is deliberately left off: even without a shell it can act (`find . -delete`,
# `find . -exec rm {} +`), so its flags would bypass the whitelist. File finding
# goes through the dedicated `find_files` tool instead.
ALLOWED_COMMANDS = ["ls", "dir", "grep", "cat", "head", "tail", "wc", "echo", "pwd"]

def render_diff(path: str, old: str, new: str, color: bool = True) -> str:
    """A unified diff of a proposed write, optionally colorized for the terminal.

    This is what makes a write reviewable: the developer sees exactly which
    lines change before accepting. A model that returns only a new method (and
    would blow away the rest of the file via the overwrite) shows up here as a
    wall of deletions -- the signal to reject.
    """
    import difflib
    diff = difflib.unified_diff(
        old.splitlines(), new.splitlines(),
        fromfile=f"a/{path}", tofile=f"b/{path}", lineterm="")
    out = []
    for line in diff:
        if not color:
            out.append(line)
        elif line.startswith("+") and not line.startswith("+++"):
            out.append(f"{Fore.GREEN}{line}{Style.RESET_ALL}")
        elif line.startswith("-") and not line.startswith("---"):
            out.append(f"{Fore.RED}{line}{Style.RESET_ALL}")
        elif line.startswith("@@"):
            out.append(f"{Fore.CYAN}{line}{Style.RESET_ALL}")
        else:
            out.append(line)
    return "\n".join(out) if out else "(no changes)"


def deletion_warning(old: str, new: str) -> Optional[str]:
    """A one-line warning when a proposed write drops a large share of the
    file -- the signature of a whole-file rewrite from memory. Shown above the
    diff, because in review the reject signal (a wall of red) is easy to skim
    past; both bad approvals in manual testing happened exactly that way."""
    old_lines = [line for line in old.splitlines() if line.strip()]
    if len(old_lines) < 5:
        return None
    lost = sum(1 for line in old_lines if line not in new)
    if lost / len(old_lines) > 0.3:
        return (f"⚠ WARNING: this change deletes or rewrites {lost} of "
                f"{len(old_lines)} non-empty lines. If you asked for a small "
                "edit, this is a rewrite from memory -- reject it.")
    return None


class Spinner:
    """Minimal terminal spinner shown while the model is thinking."""

    def __init__(self, message: str = "Thinking"):
        self.message = message
        self.spinning = False
        self.spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self.spinner_thread: Optional[threading.Thread] = None

    def spin(self):
        i = 0
        while self.spinning:
            i = (i + 1) % len(self.spinner_chars)
            print(f"\r{Fore.YELLOW}{self.message} {self.spinner_chars[i]} ", end="")
            time.sleep(0.1)
        print("\r" + " " * (len(self.message) + 10) + "\r", end="")

    def start(self):
        self.spinning = True
        self.spinner_thread = threading.Thread(target=self.spin, daemon=True)
        self.spinner_thread.start()

    def stop(self):
        self.spinning = False
        if self.spinner_thread:
            self.spinner_thread.join()


class CodingAgent:
    """The whole agent is one class: client, message history, working directory."""

    def __init__(self, model: str = "qwen3:32b",
                 small_model: str = router.SMALL_MODEL_DEFAULT,
                 edit_model: str = None,
                 fim_model: str = fim.DEFAULT_FIM_MODEL,
                 use_router: bool = True, use_rag: bool = False,
                 use_web: bool = False,
                 max_iterations: int = 5, debug: bool = False,
                 temperature: float = 0.3):
        self.model = model
        self.small_model = small_model
        # Optional third routing tier: write-implying requests go here when
        # set (e.g. qwen3:32b, which went 3/3 on the edit benchmark where the
        # 7B coder went 3/5). None = edits use the main model, no extra pull.
        self.edit_model = edit_model
        self.fim_model = fim_model
        self.use_router = use_router
        self.use_rag = use_rag
        self.use_web = use_web   # off by default: keeps everything local
        self.max_iterations = max_iterations
        self.debug = debug
        # Measured on a 5-run edit benchmark: temp 0.8 -> 2/5 (typos, format
        # drift), temp 0.0 -> 0/5 (deterministic error loops: the same wrong
        # call repeats until the step cap), temp 0.3 -> 3/5. Low-but-nonzero
        # keeps the tool-call format stable while letting a retry after an
        # error be a genuinely different attempt.
        self.temperature = temperature

        # The OpenAI SDK talks to Ollama because Ollama exposes an
        # OpenAI-compatible API; only the base_url changes.
        self.client = OpenAI(base_url="http://localhost:11434/v1/", api_key="ollama")
        self.messages: list[dict] = []
        self._token_count = 0     # total tokens spent on the last user turn
        self.spinner = Spinner()
        self.current_directory = pathlib.Path.cwd().resolve()
        self.last_route: tuple[str, bool] = (self.model, True)
        self._collection = None   # lazily built RAG index
        # Diff/approval gate for writes (Part 3). DENY by default: a write only
        # lands when an approver is installed -- run() installs the interactive
        # diff gate for the CLI, and library users opt in explicitly with
        # `agent._approver = lambda path, old, new: True`. Learned the hard way:
        # under an auto-approve default, an agent asked a read-only question
        # decided to "document" its findings by overwriting the repo's README.
        # Denying never blocks on stdin, so the library still runs headless.
        self._approver = lambda path, old, new: False
        # Network access is deny-by-default too. The CLI installs an interactive
        # gate that shows the redacted query before it leaves the machine.
        self._web_approver = lambda query: False

    # --- Path safety (Part 3) -------------------------------------------------
    def _safe_path(self, path: str) -> pathlib.Path:
        """Resolve a path and reject anything that escapes the working directory.

        `resolve()` normalizes `../` and symlinks; `is_relative_to` is what
        actually contains the result inside the project root.
        """
        # Resolve the root too: on macOS the working dir may itself be a symlink
        # (/var -> /private/var), and comparing a resolved file path against an
        # unresolved root would false-positive as an escape.
        return resolve_in_workspace(self.current_directory, path)

    def _safe_read_path(self, path: str) -> pathlib.Path:
        file_path = self._safe_path(path)
        if is_sensitive_path(file_path):
            raise ValueError(f"path '{path}' is sensitive and cannot be read by the agent")
        return file_path

    def _escapes_workspace(self, arg: str) -> bool:
        """True if a path-like argument would resolve outside the workspace.

        Used to contain command/glob arguments: a whitelisted command like
        `cat` is still a read primitive for `/etc/passwd` if its argument is an
        absolute or `../`-escaping path. An absolute path *inside* the workspace
        is fine, though -- the model often echoes back the absolute paths that
        find_symbol returned -- so we resolve and check containment rather than
        rejecting every absolute path. `/etc/passwd` still escapes; the repo's
        own files do not.
        """
        root = self.current_directory.resolve()
        try:
            candidate = pathlib.Path(arg) if os.path.isabs(arg) else (root / arg)
            return not candidate.resolve().is_relative_to(root)
        except (ValueError, OSError):
            return True

    # --- The loop (Step 1) ----------------------------------------------------
    def process_user_input(self, user_input: str) -> str:
        """Run the think-act-observe loop until completion or the iteration cap."""
        self.messages.append({"role": "user", "content": user_input})
        self._token_count = 0     # per-turn counter, read by last_token_count()
        final_response = ""
        nudges = 0                # corrective retries this turn (bounded below)

        for _ in range(self.max_iterations):          # iteration cap (stop condition 1)
            completion = self.chat(self.messages)
            response_message = completion.choices[0].message
            tool_calls = response_message.tool_calls
            # Robustness: some local models (e.g. qwen2.5-coder via Ollama) emit
            # the tool call as JSON text in `content` instead of the structured
            # tool_calls field. Recover it so the agent loop still works.
            recovered = [] if tool_calls else self._tool_calls_from_content(response_message.content)

            if tool_calls:                              # act (native tool calls)
                self.messages.append(response_message.model_dump())
                for tool_call in tool_calls:
                    name = tool_call.function.name
                    try:
                        args = json.loads(tool_call.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    self._run_and_record(name, args, tool_call.id)
                continue                                # run the model again
            elif recovered:                             # act (recovered from content)
                synth_calls = []
                for i, (name, args) in enumerate(recovered):
                    synth_calls.append({"id": f"call_{i}", "type": "function",
                                        "function": {"name": name, "arguments": json.dumps(args)}})
                # A valid OpenAI sequence: assistant message carrying tool_calls,
                # then one tool message per call with the matching id.
                self.messages.append({"role": "assistant", "content": None, "tool_calls": synth_calls})
                for i, (name, args) in enumerate(recovered):
                    self._run_and_record(name, args, f"call_{i}")
                continue
            else:
                content = response_message.content or ""
                self.messages.append(response_message.model_dump())
                # A "final answer" that is really a tool call written as text
                # (e.g. 'edit_file /path/to/file.py {"old_text": ...}') must not
                # end the turn: nothing was executed. Nudge the model once to
                # use the tool interface, then let the loop continue.
                # Budget of 2: in testing the model narrated once early, burned
                # a single nudge, then narrated again at the end of the turn.
                # The iteration cap still bounds the whole loop.
                if nudges < 2 and self._looks_like_attempted_call(content):
                    nudges += 1
                    self.messages.append({"role": "system", "content": (
                        "Your last message was a tool call written as text, so "
                        "NOTHING was executed. Call the tool through the "
                        "tool-calling interface instead, with a real file path "
                        "from this workspace (use list_files or find_symbol "
                        "first if you are unsure of the path).")})
                    continue
                final_response = self._clean_final_answer(content)
                break                                   # completion signal (stop condition 2)

        return final_response or "(stopped: reached the step limit without a final answer)"

    def _looks_like_attempted_call(self, content: str) -> bool:
        """True when a text answer is really an attempted tool call: it starts
        with a valid tool name and carries an argument payload. A prose answer
        that merely mentions a tool ('I used edit_file to...') does not match,
        because the tool name is not its first token."""
        text = content.strip().lstrip("`").strip()
        first = re.split(r"[\s({]", text, maxsplit=1)[0]
        valid = {t["function"]["name"] for t in self.get_tools_definition()}
        return first in valid and ("{" in text or "(" in text)

    @staticmethod
    def _clean_final_answer(content: str) -> str:
        """Unwrap a pseudo tool call some models emit as their last message --
        '{"name": "final_answer", "arguments": "..."}' -- into plain text."""
        text = content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text.split("\n", 1)[-1] if "\n" in text else text
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return content
        if isinstance(data, dict) and isinstance(data.get("arguments"), str) \
                and data.get("name") in ("final_answer", "answer", "response"):
            return data["arguments"]
        return content

    def _run_and_record(self, name: str, args: dict, call_id: str) -> None:
        """Execute one tool call and append its result as an observation."""
        if self.debug:
            print(f"\n{Fore.YELLOW}DEBUG tool: {name} {args}")
        self.spinner.message = f"Using {name}"
        result = self.execute_tool(name, args)
        self.messages.append({"role": "tool", "tool_call_id": call_id,
                              "name": name, "content": result})

    def _tool_calls_from_content(self, content: Optional[str]) -> list[tuple[str, dict]]:
        """Parse tool calls a model leaked into content instead of the structured
        tool_calls field. Returns [] when the content is a normal answer.

        Two shapes are handled. Fast path: the whole content is a JSON object or
        array (optionally inside one ```json fence). Fallback: a local model
        (e.g. qwen2.5-coder via Ollama) often wraps the call in prose or a ```bash
        fence -- 'To apply this, run: ```bash {"name": "write_file", ...}``` ' --
        so we scan for embedded JSON objects and keep the ones that look like a
        call (valid tool name + an arguments dict). The valid-name filter is what
        keeps an explanatory JSON example from being executed.
        """
        if not content:
            return []
        valid_names = {t["function"]["name"] for t in self.get_tools_definition()}

        # Fast path: whole content is JSON (optionally fenced).
        text = content.strip()
        if text.startswith("```"):                      # strip a ```json fence
            text = text.strip("`")
            text = text.split("\n", 1)[-1] if "\n" in text else text
        try:
            data = json.loads(text)
            candidates = data if isinstance(data, list) else [data]
        except (json.JSONDecodeError, ValueError):
            # Fallback: the call is embedded in prose / a ```bash fence.
            candidates = self._extract_json_objects(content)

        out: list[tuple[str, dict]] = []
        for c in candidates:
            if isinstance(c, dict) and c.get("name") in valid_names:
                args = c.get("arguments", c.get("parameters", {}))
                if isinstance(args, dict):
                    out.append((c["name"], args))
        if not out:
            # Last resort: the model narrated the call in function syntax --
            # `write_file({"path": ..., "content": '...'})` -- often with
            # Python-style quoting that json.loads rejects.
            out = self._calls_from_call_syntax(content, valid_names)
        return out

    @staticmethod
    def _parse_object(text: str) -> Optional[dict]:
        """Parse one ``{...}`` literal as JSON, falling back to a Python dict
        literal (single-quoted strings), which local models often emit."""
        import ast
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            try:
                data = ast.literal_eval(text)   # literals only: safe to eval
            except (ValueError, SyntaxError, MemoryError, RecursionError):
                return None
        return data if isinstance(data, dict) else None

    @staticmethod
    def _balanced_object_span(text: str, start: int) -> int:
        """Given ``text[start] == '{'``, return the index just past the matching
        closing brace, or -1. String-aware for both quote styles, so braces
        inside string values don't break the balance."""
        depth = 0
        quote: Optional[str] = None
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if quote:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == quote:
                    quote = None
                continue
            if ch in ("\"", "'"):
                quote = ch
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return i + 1
        return -1

    def _calls_from_call_syntax(self, text: str,
                                valid_names: set) -> list[tuple[str, dict]]:
        """Recover ``tool_name({...})``-style calls narrated in prose or a fence."""
        out: list[tuple[str, dict]] = []
        for name in valid_names:
            idx = 0
            while (idx := text.find(f"{name}(", idx)) != -1:
                brace = idx + len(name) + 1
                while brace < len(text) and text[brace] in " \n\t":
                    brace += 1
                if brace < len(text) and text[brace] == "{":
                    end = self._balanced_object_span(text, brace)
                    if end != -1:
                        args = self._parse_object(text[brace:end])
                        if args is not None:
                            out.append((name, args))
                idx += len(name)
        return out

    @staticmethod
    def _extract_json_objects(text: str) -> list:
        """Return every top-level ``{...}`` JSON object embedded anywhere in text.

        A brace-depth scan that respects string literals (so braces or quotes
        inside a JSON string value don't break the balance), used to pull a tool
        call out of a larger prose answer.
        """
        objects: list = []
        depth = 0
        start = -1
        in_string = False
        escape = False
        for i, ch in enumerate(text):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}" and depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    try:
                        objects.append(json.loads(text[start:i + 1]))
                    except (json.JSONDecodeError, ValueError):
                        pass
                    start = -1
        return objects

    # --- Routing (Step 6) -----------------------------------------------------
    def route(self, user_input: str) -> tuple[str, bool]:
        """Pick a model and decide whether to attach tools (see agent/router.py)."""
        return router.route(user_input, self.model, self.small_model, self.edit_model)

    def _route_for(self, messages) -> tuple[str, bool]:
        """Routing decision for one request, from the latest user turn.

        The mid-tool-loop pin (keep the big model until the current task
        finishes) must only look at tool messages AFTER the latest user turn.
        Checking the whole history is a session-killing bug: the first tool
        call of a chat session would pin every later question, however
        generic, to the big model for good.
        """
        if not self.use_router:
            return self.model, True
        last_user_idx = next((i for i in range(len(messages) - 1, -1, -1)
                              if messages[i].get("role") == "user"), None)
        if last_user_idx is None:
            return self.model, True
        model, needs_tools = self.route(messages[last_user_idx].get("content") or "")
        if any(m.get("role") == "tool" for m in messages[last_user_idx + 1:]):
            # Mid tool-loop for THIS turn: keep the routed model (an edit that
            # started on the edit model must finish on it), with tools on.
            return (model if needs_tools else self.model), True
        return model, needs_tools

    def chat(self, messages):
        """Send one chat request, routing the model when the router is enabled."""
        model, needs_tools = self._route_for(messages)
        self.last_route = (model, needs_tools)

        kwargs: dict[str, Any] = {"model": model, "messages": messages,
                                  "temperature": self.temperature}
        if needs_tools:
            kwargs["tools"] = self.get_tools_definition()
            kwargs["tool_choice"] = "auto"

        t0 = time.time()
        response = self.client.chat.completions.create(**kwargs)
        if getattr(response, "usage", None):
            self._token_count += response.usage.total_tokens or 0
        if self.debug:
            print(f"\n{Fore.YELLOW}DEBUG routed to {model} "
                  f"(tools={needs_tools}) in {(time.time() - t0) * 1000:.0f}ms")
        return response

    def last_token_count(self) -> int:
        """Tokens (prompt + completion, summed across loop turns) spent on the
        last user turn -- the per-query cost column in the Step 7 eval."""
        return self._token_count

    # --- Autocomplete (Step 8) ------------------------------------------------
    def autocomplete(self, prefix: str, suffix: str) -> str:
        """Fill-in-the-middle completion for the gap at the cursor."""
        return fim.fim_complete(self.client, prefix, suffix, model=self.fim_model)

    # --- Tool dispatch --------------------------------------------------------
    def execute_tool(self, tool_name: str, params: dict[str, Any]) -> str:
        tools = {
            "read_file": self.read_file,
            "write_file": self.write_file,
            "edit_file": self.edit_file,
            "list_files": self.list_files,
            "find_files": self.find_files,
            "run_command": self.run_command,
            "find_symbol": self.find_symbol,
            "list_symbols": self.list_symbols,
        }
        if self.use_rag:
            tools["search_code"] = self.search_code
        if self.use_web:
            tools["web_search"] = self.web_search
        if tool_name not in tools:
            return f"Tool '{tool_name}' not implemented"
        params = self._repair_param_keys(tool_name, params)
        try:
            return tools[tool_name](**params)
        except Exception as e:  # surface tool errors to the model as observations
            return f"Error executing {tool_name}: {e}"

    def _repair_param_keys(self, tool_name: str, params: dict) -> dict:
        """Fix argument keys a model mangled, matched against the tool's schema.

        Seen in testing: the model emits the key '\\new_text' in JSON, which
        parses to a newline + 'ew_text'. Strip whitespace, then map a still-
        unknown key to the schema parameter it is an unambiguous fragment of
        ('ew_text' -> 'new_text'). Anything ambiguous is left alone so the
        TypeError still surfaces to the model as an observation.
        """
        schema = next((t["function"]["parameters"]["properties"]
                       for t in self.get_tools_definition()
                       if t["function"]["name"] == tool_name), {})
        if not schema:
            return params
        repaired = {}
        for key, value in params.items():
            k = key.strip()
            if k not in schema:
                fragment_of = [p for p in schema if p.endswith(k) or k.endswith(p)]
                if len(fragment_of) == 1 and fragment_of[0] not in params:
                    k = fragment_of[0]
            repaired[k] = value
        return repaired

    # --- File tools (Steps 1-3) ----------------------------------------------
    def read_file(self, path: str) -> str:
        try:
            file_path = self._safe_read_path(path)
        except ValueError as e:
            return f"Error: {e}"
        content = file_path.read_text(encoding="utf-8", errors="replace")
        return f"Content of {path}:\n{content}"

    def write_file(self, path: str, content: str) -> str:
        try:
            file_path = self._safe_path(path)
        except ValueError as e:
            return f"Error: {e}"
        # Diff/approval gate: never overwrite blindly. The developer sees the
        # diff and accepts or rejects (Part 3, "Diff-Based Writes"). A rejection
        # is fed back to the model as an observation so it can adjust.
        old = file_path.read_text(encoding="utf-8", errors="replace") if file_path.exists() else ""
        if err := self._syntax_error(path, content):
            return (f"Error: this write would leave {path} with a syntax error "
                    f"({err}). Fix the content and try again. The file is unchanged.")
        if not self._approver(path, old, content):
            return f"Write to {path} was rejected by the developer; the file is unchanged."
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return f"✅ Wrote {len(content)} characters to {path}"

    @staticmethod
    def _syntax_error(path: str, content: str) -> Optional[str]:
        """Return a description of the syntax error a .py write would introduce,
        or None. The cheapest diagnostic in the Part 3 sense: catching a broken
        edit before it lands beats catching it when the tests run -- and beats
        trusting the human to spot a missing newline in the diff."""
        if not str(path).endswith(".py"):
            return None
        try:
            compile(content, path, "exec")
        except SyntaxError as e:
            return f"line {e.lineno}: {e.msg}: {(e.text or '').strip()!r}"
        return None

    def edit_file(self, path: str, old_text: str, new_text: str) -> str:
        """Replace one exact snippet in an existing file.

        This is the preferred write path for modifications: the model produces
        only the changed lines and the rest of the file is preserved
        mechanically, instead of trusting the model to reproduce the whole file
        from memory (where a small model drifts, typos, and drops methods).
        Fails loudly unless old_text matches exactly once, and the resulting
        change still goes through the same diff/approval gate as write_file.
        """
        try:
            file_path = self._safe_path(path)
        except ValueError as e:
            return f"Error: {e}"
        if not file_path.is_file():
            return f"Error: {path} does not exist; use write_file to create a new file"
        content = file_path.read_text(encoding="utf-8", errors="replace")
        if not old_text:
            # A small model's favorite mistake: it wants to APPEND, so it sends
            # old_text="". Teach the anchor pattern in the error, with lines
            # from the actual file, so the retry can succeed.
            anchor = "\n".join(content.splitlines()[-2:])
            return ("Error: old_text must not be empty. To ADD code, anchor on "
                    "existing lines: set old_text to the exact lines the new code "
                    "goes next to (copied verbatim from the file), and set new_text "
                    f"to those same lines plus your addition. For example, to append "
                    f"at the end of this file use old_text={anchor!r} and new_text="
                    f"{anchor!r} + your new code.")
        count = content.count(old_text)
        if count == 0:
            return (f"Error: old_text was not found in {path}. Read the file and "
                    "pass the exact text to replace, including whitespace.")
        if count > 1:
            return (f"Error: old_text occurs {count} times in {path}; include more "
                    "surrounding lines so it matches exactly once.")
        new_content = content.replace(old_text, new_text, 1)
        if err := self._syntax_error(path, new_content):
            return (f"Error: this edit would break {path} with a syntax error "
                    f"({err}). Check newlines and indentation in new_text and try again. "
                    "The file is unchanged.")
        if not self._approver(path, content, new_content):
            return f"Edit to {path} was rejected by the developer; the file is unchanged."
        file_path.write_text(new_content, encoding="utf-8")
        return f"✅ Edited {path}: replaced {len(old_text)} characters with {len(new_text)}"

    def _interactive_approval(self, path: str, old: str, new: str) -> bool:
        """Show a colored diff and ask before a write lands (installed by run())."""
        self.spinner.stop()                       # pause the 'Thinking' spinner
        verb = "create" if not old else "overwrite"
        print(f"\n{Fore.YELLOW}The agent wants to {verb} {path}:{Style.RESET_ALL}")
        warning = deletion_warning(old, new)
        if warning:
            print(f"{Fore.RED}{warning}{Style.RESET_ALL}")
        print(render_diff(path, old, new))
        try:
            answer = input(f"{Fore.YELLOW}Apply this change? [y/N] {Style.RESET_ALL}").strip().lower()
        except EOFError:
            answer = "n"                           # no input available -> safe default
        approved = answer in ("y", "yes")
        if not approved:
            print(f"{Fore.RED}Skipped {path}.{Style.RESET_ALL}")
        self.spinner.start()                       # resume for the next model turn
        return approved

    def list_files(self, path: str = ".") -> str:
        try:
            dir_path = self._safe_path(path)
        except ValueError as e:
            return f"Error: {e}"
        if not dir_path.is_dir():
            return f"Error: {path} is not a directory"
        dirs, files = [], []
        for item in sorted(dir_path.iterdir()):
            try:
                resolve_in_workspace(self.current_directory, item)
            except ValueError:
                continue
            if is_sensitive_path(item):
                continue
            (dirs if item.is_dir() else files).append(
                f"{'📁' if item.is_dir() else '📄'} {item.name}")
        return f"Contents of {path}:\n" + "\n".join(dirs + files)

    def find_files(self, pattern: str) -> str:
        # Reject absolute / `..`-escaping patterns before touching the filesystem.
        if os.path.isabs(pattern) or ".." in pathlib.PurePath(pattern).parts:
            return f"Error: pattern '{pattern}' must be a relative path inside the workspace"
        root = self.current_directory.resolve()
        matches = glob.glob(str(root / pattern), recursive=True)
        inside = [p for p in matches
                  if pathlib.Path(p).resolve().is_relative_to(root)
                  and not is_sensitive_path(pathlib.Path(p))]
        if not inside:
            return f"No files found matching '{pattern}'"
        rel = [str(pathlib.Path(p).resolve().relative_to(root)) for p in inside]
        return f"Found {len(rel)} files matching '{pattern}':\n" + "\n".join(rel)

    def run_command(self, cmd: str) -> str:
        """Run a whitelisted, read-only command -- no shell, so `;`/`|` can't chain.

        Three guards: the whitelist checks the program; running the parsed argv
        list (not a shell string) stops `cat x; rm -rf .` from chaining a second
        command; and argument containment stops `cat /etc/passwd` from reading
        outside the workspace via an absolute or `../` path.
        """
        try:
            parts = shlex.split(cmd)
        except ValueError as e:
            return f"Error: could not parse command: {e}"
        if not parts:
            return "Error: empty command"
        if parts[0] not in ALLOWED_COMMANDS:
            return (f"Error: command '{parts[0]}' is not allowed. "
                    f"Allowed: {', '.join(ALLOWED_COMMANDS)}")
        # Argument containment: an allowed command must not read outside the
        # workspace via an absolute or `../` path (`cat /etc/passwd`, `find /etc`).
        for arg in parts[1:]:
            candidate = arg.split("=", 1)[1] if arg.startswith("-") and "=" in arg else arg
            if is_sensitive_path(candidate):
                return "Error: command references a sensitive path and was blocked"
            if arg.startswith("-"):
                continue                         # a flag, not a path
            if self._escapes_workspace(arg):
                return f"Error: argument '{arg}' points outside the workspace"
        try:
            result = subprocess.run(parts, cwd=self.current_directory,
                                    capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired:
            return "Error: command timed out"
        if result.returncode == 0:
            return result.stdout or "(no output)"
        return f"Command failed:\n{result.stderr}"

    # --- AST symbol tools (Steps 4-5) ----------------------------------------
    def find_symbol(self, name: str, language: str = "python") -> str:
        hits = symbols.find_symbol(self.current_directory, name, language)
        # Return paths relative to the workspace so the model can feed them
        # straight back into read_file / run_command without tripping the
        # workspace-containment guard with a long absolute path.
        root = self.current_directory.resolve()
        for h in hits:
            try:
                h["file"] = str(pathlib.Path(h["file"]).resolve().relative_to(root))
            except ValueError:
                pass
        return json.dumps(hits, indent=2) if hits else f"No symbol named '{name}' found"

    def list_symbols(self, path: str) -> str:
        try:
            file_path = self._safe_read_path(path)
        except ValueError as e:
            return f"Error: {e}"
        return json.dumps(symbols.list_symbols(file_path), indent=2)

    # --- RAG tool (Step 9, optional) -----------------------------------------
    def search_code(self, query: str, k: int = 8) -> str:
        from . import rag
        if self._collection is None:
            self._collection = rag.index_repo(self.current_directory, self.client)
            if self._collection is None:
                return "Code index is empty (no .py files found)."
        return json.dumps(rag.search_code(self._collection, query, self.client, k), indent=2)

    # --- Optional web search (Exa) -------------------------------------------
    def web_search(self, query: str, num_results: int = 5) -> str:
        api_key = os.environ.get("EXA_API_KEY")
        if not api_key:
            return "Error: Exa API key not configured (set EXA_API_KEY in .env)."
        safe_query = redact_outbound_secrets(query)
        if not self._web_approver(safe_query):
            return "Web search was rejected by the developer; no data was sent."
        resp = requests.post(
            "https://api.exa.ai/search",
            headers={"Content-Type": "application/json", "x-api-key": api_key},
            json={"query": safe_query, "num_results": int(num_results), "use_autoprompt": True},
            timeout=30,
        )
        if resp.status_code != 200:
            return f"Error: Exa API returned status {resp.status_code}"
        results = resp.json().get("results", [])
        if not results:
            return f"No results found for '{safe_query}'"
        out = [f"Search results for '{safe_query}':"]
        for i, item in enumerate(results, 1):
            text = (item.get("text") or "")[:150]
            out.append(f"{i}. {item.get('title', 'No title')}\n   {item.get('url', '')}\n   {text}")
        return "\n".join(out)

    # --- Tool schemas ---------------------------------------------------------
    def get_tools_definition(self) -> list[dict]:
        def fn(name, description, properties, required):
            return {"type": "function", "function": {
                "name": name, "description": description,
                "parameters": {"type": "object", "properties": properties, "required": required}}}

        tools = [
            fn("read_file", "Read the contents of a file",
               {"path": {"type": "string", "description": "Path to the file to read"}}, ["path"]),
            fn("write_file", "Create a new file (overwrites the whole file if it exists; use edit_file to modify)",
               {"path": {"type": "string", "description": "Path to the file"},
                "content": {"type": "string", "description": "Content to write"}}, ["path", "content"]),
            fn("edit_file", "Modify an existing file by replacing an exact snippet (preferred for edits). "
               "To ADD code, anchor on existing lines: old_text = the exact lines the new code goes "
               "next to, new_text = those same lines plus the addition.",
               {"path": {"type": "string", "description": "Path to the file"},
                "old_text": {"type": "string", "description": "Exact text copied verbatim from the file, incl. indentation; never empty; must match exactly once"},
                "new_text": {"type": "string", "description": "Replacement text (for additions: old_text repeated plus the new code)"}}, ["path", "old_text", "new_text"]),
            fn("list_files", "List files in a directory",
               {"path": {"type": "string", "description": "Directory path (default '.')"}}, []),
            fn("find_files", "Find files matching a glob pattern (e.g. **/*.py)",
               {"pattern": {"type": "string", "description": "Glob pattern"}}, ["pattern"]),
            fn("run_command", "Run a whitelisted read-only shell command",
               {"cmd": {"type": "string", "description": "Command to run"}}, ["cmd"]),
            fn("find_symbol", "Find a function/class definition by name (AST-aware)",
               {"name": {"type": "string", "description": "Symbol name"},
                "language": {"type": "string", "description": "python|javascript (default python)"}}, ["name"]),
            fn("list_symbols", "List the functions/classes defined in a file (AST outline)",
               {"path": {"type": "string", "description": "Path to the file"}}, ["path"]),
        ]
        if self.use_rag:
            tools.append(fn("search_code", "Semantic search over the code index (RAG)",
                            {"query": {"type": "string", "description": "Natural-language query"},
                             "k": {"type": "integer", "description": "Number of hits (default 8)"}}, ["query"]))
        if self.use_web:   # off by default: the only tool that leaves the machine
            tools.append(fn("web_search", "Search the web via Exa (optional)",
                            {"query": {"type": "string", "description": "Search query"},
                             "num_results": {"type": "integer", "description": "Max results (default 5)"}}, ["query"]))
        return tools

    def get_system_prompt(self) -> str:
        rag_line = "\n- search_code: semantic search over the indexed codebase" if self.use_rag else ""
        return (
            "You are a local coding assistant that uses tools to explore and edit a repo. "
            "Give precise, concise answers grounded in files and line numbers you actually read.\n\n"
            f"Current directory: {self.current_directory}\n\n"
            "Tools:\n"
            "- read_file / edit_file / write_file / list_files / find_files\n"
            "- run_command (whitelisted, read-only)\n"
            "- find_symbol / list_symbols (AST-aware: prefer these over grep for code structure)"
            f"{rag_line}\n\n"
            "Prefer find_symbol/list_symbols for 'where is X defined' and 'what's in this file'. "
            "To MODIFY an existing file, read it first, then use edit_file with the exact "
            "old_text to replace -- never rewrite the whole file from memory, and never "
            "pass an empty old_text. To ADD code, anchor on existing lines. Example: to add "
            "a method after `    def reset(self):\\n        self.total = 0`, call edit_file "
            "with old_text exactly that snippet and new_text the same snippet plus "
            "`\\n\\n    def multiply(self, x):\\n        ...`. "
            "Use write_file only to CREATE new files; it overwrites the entire file. "
            "When the user names a file, class, or function, LOCATE it first with "
            "find_symbol or find_files (it may live in a subdirectory) -- never create "
            "a new file unless the user asked for one or the search found nothing. "
            "Cite file paths and line ranges. Respond only with tool calls or a final answer."
        )

    def _interactive_web_approval(self, query: str) -> bool:
        """Show the outbound query and require confirmation before network access."""
        self.spinner.stop()
        print(f"\n{Fore.YELLOW}The agent wants to search the web for:{Style.RESET_ALL}")
        print(query)
        try:
            answer = input(
                f"{Fore.YELLOW}Send this query? [y/N] {Style.RESET_ALL}"
            ).strip().lower()
        except EOFError:
            answer = "n"
        self.spinner.start()
        return answer in ("y", "yes")

    # --- REPL -----------------------------------------------------------------
    def run(self):
        print(f"{Fore.CYAN}Local code assistant — model: {self.model} "
              f"(router={'on' if self.use_router else 'off'}, "
              f"rag={'on' if self.use_rag else 'off'}, "
              f"web={'on' if self.use_web else 'off'})")
        print(f"{Fore.CYAN}Type 'exit' to quit, 'debug' to toggle debug.")
        # Interactive session: gate every write behind a diff + approval prompt.
        self._approver = self._interactive_approval
        if self.use_web:
            self._web_approver = self._interactive_web_approval
        self.messages.append({"role": "system", "content": self.get_system_prompt()})
        while True:
            try:
                user_input = input(f"\n{Fore.GREEN}You: {Style.RESET_ALL}")
                if user_input.lower() in ("exit", "quit"):
                    break
                if user_input.lower() == "debug":
                    self.debug = not self.debug
                    print(f"{Fore.CYAN}Debug mode: {self.debug}")
                    continue
                if not user_input.strip():
                    continue
                self.spinner.start()
                response = self.process_user_input(user_input)
                self.spinner.stop()
                print(f"\n{Fore.BLUE}Agent: {Style.RESET_ALL}{response}")
            except KeyboardInterrupt:
                self.spinner.stop()
                break
            except Exception as e:
                self.spinner.stop()
                print(f"\n{Fore.RED}Error: {e}")
        print(f"{Fore.CYAN}Exiting...")
