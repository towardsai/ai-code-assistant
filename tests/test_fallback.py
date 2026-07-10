"""Tool-call-from-content fallback tests (no Ollama required).

Some local models emit tool calls as JSON text in `content` instead of the
structured tool_calls field; the agent recovers them. This is the deterministic
parser behind that recovery.
"""
import json

from agent.core import CodingAgent


def test_recovers_single_tool_call():
    a = CodingAgent()
    content = json.dumps({"name": "find_symbol", "arguments": {"name": "process_payment"}})
    calls = a._tool_calls_from_content(content)
    assert calls == [("find_symbol", {"name": "process_payment"})]


def test_recovers_from_code_fence():
    a = CodingAgent()
    content = '```json\n{"name": "read_file", "arguments": {"path": "x.py"}}\n```'
    calls = a._tool_calls_from_content(content)
    assert calls == [("read_file", {"path": "x.py"})]


def test_parameters_alias():
    a = CodingAgent()
    content = json.dumps({"name": "list_files", "parameters": {"path": "."}})
    assert a._tool_calls_from_content(content) == [("list_files", {"path": "."})]


def test_recovers_call_embedded_in_prose_and_bash_fence():
    # The real failure from manual testing: qwen2.5-coder:7b wraps the write_file
    # call in prose + a ```bash fence instead of emitting a structured tool call.
    a = CodingAgent()
    content = (
        "Here's how you can add a `multiply` method.\n\n"
        "```python\ndef multiply(self, x):\n    return self.total\n```\n\n"
        "To apply these changes, you can run:\n\n"
        '```bash\n{"name": "write_file", "arguments": '
        '{"path": "./calculator.py", "content": "class Calculator: pass"}}\n```\n\n'
        "This will write the updated content."
    )
    calls = a._tool_calls_from_content(content)
    assert calls == [("write_file", {"path": "./calculator.py",
                                     "content": "class Calculator: pass"})]


def test_prose_without_a_tool_call_is_ignored():
    # An explanation that mentions JSON-ish text but no valid tool call stays empty.
    a = CodingAgent()
    content = "You could store it as {\"key\": \"value\"} in a config file."
    assert a._tool_calls_from_content(content) == []


def test_plain_answer_is_not_a_tool_call():
    a = CodingAgent()
    assert a._tool_calls_from_content("The function adds two numbers.") == []


def test_unknown_tool_name_ignored():
    a = CodingAgent()
    content = json.dumps({"name": "definitely_not_a_tool", "arguments": {}})
    assert a._tool_calls_from_content(content) == []


def test_none_content():
    a = CodingAgent()
    assert a._tool_calls_from_content(None) == []


def test_recovers_function_call_syntax():
    # Second real failure from manual testing: the model narrates the call in
    # Python function syntax -- write_file({...}) -- instead of a tool call.
    a = CodingAgent()
    content = (
        "Write this updated content back to `calculator.py`:\n\n"
        '```python\nwrite_file({"path": "calculator.py", '
        '"content": "class Calculator: pass"})\n```'
    )
    calls = a._tool_calls_from_content(content)
    assert calls == [("write_file", {"path": "calculator.py",
                                     "content": "class Calculator: pass"})]


def test_recovers_call_syntax_with_python_quotes_and_newline():
    # The exact shape from the manual session: a newline after "content": and a
    # single-quoted (Python-style) string value, which json.loads rejects.
    a = CodingAgent()
    content = (
        "```python\n"
        'write_file({"path": "examples/sample_project/calculator.py", "content":\n'
        "'class Calculator:\\n    def multiply(self, x):\\n        return x'})\n"
        "```"
    )
    calls = a._tool_calls_from_content(content)
    assert len(calls) == 1
    name, args = calls[0]
    assert name == "write_file"
    assert args["path"] == "examples/sample_project/calculator.py"
    assert args["content"].startswith("class Calculator:")
    assert "def multiply" in args["content"]


def test_call_syntax_with_unknown_name_ignored():
    a = CodingAgent()
    content = 'os.system({"cmd": "rm -rf /"})'
    assert a._tool_calls_from_content(content) == []


def test_call_syntax_does_not_shadow_structured_recovery():
    # When a proper {"name": ...} object is present, it wins; the call-syntax
    # scan only runs as a last resort.
    a = CodingAgent()
    content = json.dumps({"name": "read_file", "arguments": {"path": "a.py"}})
    assert a._tool_calls_from_content(content) == [("read_file", {"path": "a.py"})]


def test_final_answer_pseudo_call_unwrapped():
    # Some models end the turn with a fake tool call instead of plain text.
    a = CodingAgent()
    content = '```json\n{"name": "final_answer", "arguments": "The method was added."}\n```'
    assert a._clean_final_answer(content) == "The method was added."


def test_normal_answer_passes_through_clean():
    a = CodingAgent()
    assert a._clean_final_answer("Done: added multiply.") == "Done: added multiply."
    # JSON that is data, not a pseudo-call, is left alone.
    assert a._clean_final_answer('{"result": 42}') == '{"result": 42}'


def test_mangled_param_key_repaired(tmp_path):
    # From a real trace: the model emitted the JSON key "\new_text", which
    # parses to newline + "ew_text". The repair maps it back via the schema.
    a = CodingAgent()
    a.current_directory = tmp_path.resolve()
    a._approver = lambda path, old, new: True     # explicit library opt-in
    (tmp_path / "a.py").write_text("x = 1\n")
    out = a.execute_tool("edit_file", {"path": "a.py", "old_text": "x = 1\n",
                                       "\new_text": "x = 2\n"})
    assert "Edited" in out
    assert (tmp_path / "a.py").read_text() == "x = 2\n"


def test_ambiguous_or_unknown_keys_left_alone(tmp_path):
    a = CodingAgent()
    a.current_directory = tmp_path.resolve()
    out = a.execute_tool("read_file", {"totally_wrong": "a.py"})
    assert "Error executing read_file" in out          # TypeError still surfaces


def test_command_line_style_call_detected_as_attempt():
    # From a real session: 'edit_file /path/to/file.py {"old_text": ...}' --
    # no parens, no "name" key, placeholder path. Not recoverable (the path is
    # fake), but it must be detected so the loop nudges instead of ending.
    a = CodingAgent()
    content = 'edit_file /path/to/file.py {"old_text":"x","new_text":"y"}'
    assert a._looks_like_attempted_call(content) is True
    assert a._tool_calls_from_content(content) == []   # correctly NOT auto-run


def test_prose_mentioning_a_tool_is_not_an_attempt():
    a = CodingAgent()
    assert a._looks_like_attempted_call(
        "I used edit_file to add the method (see the diff).") is False
    assert a._looks_like_attempted_call("The method was added.") is False


def test_plain_tool_name_without_payload_is_not_an_attempt():
    a = CodingAgent()
    assert a._looks_like_attempted_call("read_file") is False
