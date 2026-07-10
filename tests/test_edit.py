"""edit_file tests (Part 3): targeted edits that preserve the rest of the file.

The tool exists because write_file's whole-file overwrite forces the model to
reproduce the entire file from memory, where a small model drifts and typos.
edit_file replaces one exact snippet mechanically; everything else is untouched
by construction. No Ollama required.
"""
import pytest

from agent.core import CodingAgent

ORIGINAL = (
    "class Calculator:\n"
    "    def add(self, x):\n"
    "        return x\n"
    "\n"
    "    def reset(self):\n"
    "        return 0\n"
)


@pytest.fixture
def agent(tmp_path):
    a = CodingAgent()
    a.current_directory = tmp_path.resolve()
    a._approver = lambda path, old, new: True     # explicit library opt-in
    (tmp_path / "calc.py").write_text(ORIGINAL)
    return a


def test_edit_replaces_snippet_and_preserves_rest(agent):
    result = agent.edit_file(
        "calc.py",
        old_text="    def reset(self):\n        return 0\n",
        new_text=("    def multiply(self, x, y):\n        return x * y\n"
                  "\n    def reset(self):\n        return 0\n"))
    assert "Edited" in result
    text = (agent.current_directory / "calc.py").read_text()
    assert "def multiply" in text
    assert "def add" in text            # untouched code survives by construction
    assert text.startswith("class Calculator:")


def test_old_text_not_found_fails_loudly(agent):
    result = agent.edit_file("calc.py", old_text="def subtract", new_text="x")
    assert "not found" in result
    assert (agent.current_directory / "calc.py").read_text() == ORIGINAL


def test_ambiguous_old_text_fails_loudly(agent):
    # "return" appears twice; the tool must refuse rather than guess.
    result = agent.edit_file("calc.py", old_text="        return", new_text="x")
    assert "occurs 2 times" in result
    assert (agent.current_directory / "calc.py").read_text() == ORIGINAL


def test_empty_old_text_rejected(agent):
    assert "empty" in agent.edit_file("calc.py", old_text="", new_text="x")


def test_missing_file_points_to_write_file(agent):
    result = agent.edit_file("nope.py", old_text="a", new_text="b")
    assert "does not exist" in result and "write_file" in result


def test_path_escape_blocked(agent):
    result = agent.edit_file("../../etc/passwd", old_text="root", new_text="x")
    assert "escapes the working directory" in result


def test_edit_goes_through_approval_gate(agent):
    seen = {}

    def approver(path, old, new):
        seen.update(path=path, old=old, new=new)
        return False                              # simulate the user typing "n"

    agent._approver = approver
    result = agent.edit_file("calc.py", old_text="return x", new_text="return x + 1")
    assert "rejected" in result
    assert (agent.current_directory / "calc.py").read_text() == ORIGINAL
    # The gate sees full before/after contents, so the diff shows the real change.
    assert seen["old"] == ORIGINAL
    assert "return x + 1" in seen["new"]


def test_edit_file_registered_as_tool(agent):
    names = {t["function"]["name"] for t in agent.get_tools_definition()}
    assert "edit_file" in names
    out = agent.execute_tool("edit_file", {"path": "calc.py",
                                           "old_text": "def add", "new_text": "def add"})
    assert "Error" not in out or "must be different" in out or "Edited" in out


def test_edit_introducing_syntax_error_is_refused(agent):
    # The manual-session failure: new_text jammed two statements on one line
    # ('self.total *= x        return self.total'). Must be refused pre-approval.
    result = agent.edit_file(
        "calc.py",
        old_text="        return x\n",
        new_text="        self.x *= 2        return x\n")
    assert "syntax error" in result
    assert (agent.current_directory / "calc.py").read_text() == ORIGINAL


def test_write_file_with_syntax_error_is_refused(agent):
    result = agent.write_file("broken.py", "def f(:\n    pass\n")
    assert "syntax error" in result
    assert not (agent.current_directory / "broken.py").exists()


def test_non_python_files_skip_the_syntax_gate(agent):
    assert "Wrote" in agent.write_file("notes.txt", "this is not python {{{")
