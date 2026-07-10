"""Diff-based write approval-gate tests (Part 3 advanced feature). No Ollama.

The gate is the fix for the manual-test failure where the model returned only a
new method and write_file's blind overwrite wiped the rest of the file. The
default DENIES writes (an unattended agent once overwrote a README while
answering a read-only question); run() installs the interactive gate for the
CLI, and library users opt in explicitly.
"""
from agent.core import CodingAgent, render_diff


def _agent(tmp_path, approve: bool = True) -> CodingAgent:
    a = CodingAgent()
    a.current_directory = tmp_path.resolve()
    if approve:
        a._approver = lambda path, old, new: True     # explicit library opt-in
    return a


def test_default_denies_writes(tmp_path):
    # No approver installed -> writes are refused; the workspace stays intact.
    a = CodingAgent()
    a.current_directory = tmp_path.resolve()
    result = a.write_file("a.py", "x = 1\n")
    assert "rejected" in result
    assert not (a.current_directory / "a.py").exists()


def test_explicit_opt_in_allows_writes(tmp_path):
    a = _agent(tmp_path)
    assert "Wrote" in a.write_file("a.py", "x = 1\n")
    assert (a.current_directory / "a.py").read_text() == "x = 1\n"


def test_rejected_write_leaves_file_untouched(tmp_path):
    a = _agent(tmp_path)
    (a.current_directory / "a.py").write_text("original\n")
    a._approver = lambda path, old, new: False        # simulate the user typing "n"
    result = a.write_file("a.py", "DESTROYED")
    assert "rejected" in result
    assert (a.current_directory / "a.py").read_text() == "original\n"


def test_approver_receives_old_and_new(tmp_path):
    a = _agent(tmp_path)
    (a.current_directory / "a.py").write_text("old = 1\n")
    seen = {}

    def approver(path, old, new):
        seen.update(path=path, old=old, new=new)
        return True

    a._approver = approver
    a.write_file("a.py", "new = 2\n")
    assert seen == {"path": "a.py", "old": "old = 1\n", "new": "new = 2\n"}


def test_new_file_shows_empty_old(tmp_path):
    a = _agent(tmp_path)
    seen = {}
    a._approver = lambda path, old, new: seen.update(old=old) or True
    a.write_file("new.py", "print('hi')\n")
    assert seen["old"] == ""                          # a create, not an overwrite


def test_render_diff_marks_added_lines():
    d = render_diff("a.py", "def add(): ...\n",
                    "def add(): ...\ndef multiply(): ...\n", color=False)
    assert "+def multiply(): ..." in d


def test_render_diff_shows_the_destructive_overwrite():
    # The exact manual-test hazard: model sends only the method, overwriting the
    # whole class. The diff SHOWS the deletions, which is what lets a human reject.
    old = "class Calculator:\n    def add(self, x):\n        return x\n"
    new = "\n    def multiply(self, x, y):\n        return x * y\n"
    d = render_diff("calculator.py", old, new, color=False)
    assert "-class Calculator:" in d
    assert "-    def add(self, x):" in d


# --- deletion warning (the reject-signal amplifier) ---------------------------
from agent.core import deletion_warning

ORIGINAL_FILE = (
    '"""Docstring."""\n\nclass Calculator:\n    def add(self, x):\n'
    '        self.total += x\n        return self.total\n\n'
    'def process_payment(amount, balance):\n    return balance - amount\n')


def test_destructive_rewrite_triggers_warning():
    # The manual-session failure: a "tidy up" write_file that dropped the
    # docstrings, the returns, and process_payment entirely.
    rewritten = ('import math\n\nclass Calculator:\n    def add(self, x):\n'
                 '        self.total += x\n')
    warning = deletion_warning(ORIGINAL_FILE, rewritten)
    assert warning is not None and "reject" in warning


def test_small_addition_does_not_warn():
    added = ORIGINAL_FILE + '\n    def multiply(self, x):\n        return x\n'
    assert deletion_warning(ORIGINAL_FILE, added) is None


def test_new_small_file_does_not_warn():
    assert deletion_warning("", "x = 1\n") is None
