"""Security guard tests (Part 3): path containment and the command whitelist.

These run without Ollama -- they exercise the deterministic guards, which are
the pieces most likely to be wrong and most expensive to get wrong.
"""
import pathlib

import pytest

from agent.core import CodingAgent


@pytest.fixture
def agent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    a = CodingAgent()
    a.current_directory = tmp_path.resolve()
    return a


# --- Path containment ---------------------------------------------------------
def test_read_inside_workspace_ok(agent):
    (agent.current_directory / "hello.txt").write_text("hi")
    assert "hi" in agent.read_file("hello.txt")


def test_read_escape_is_blocked(agent):
    result = agent.read_file("../../../../etc/passwd")
    assert "escapes the working directory" in result


def test_absolute_path_escape_blocked(agent):
    result = agent.read_file("/etc/passwd")
    assert "escapes the working directory" in result


def test_write_escape_is_blocked(agent):
    result = agent.write_file("../evil.txt", "x")
    assert "escapes the working directory" in result
    assert not (agent.current_directory.parent / "evil.txt").exists()


def test_symlinked_root_allows_inside_writes(tmp_path):
    # If the working dir itself is reached via a symlink (macOS /var -> /private/var),
    # a relative write inside it must still be allowed, not flagged as an escape.
    real = tmp_path / "real_workspace"
    real.mkdir()
    link = tmp_path / "linked_workspace"
    try:
        link.symlink_to(real)
    except OSError:
        pytest.skip("symlinks not supported here")
    a = CodingAgent()
    a.current_directory = link               # unresolved, symlinked root
    a._approver = lambda path, old, new: True
    assert "Wrote" in a.write_file("greet.py", "x = 1\n")
    assert (real / "greet.py").exists()


def test_symlink_escape_blocked(agent):
    # A symlink pointing outside the workspace must not be a read primitive.
    outside = agent.current_directory.parent / "outside_secret.txt"
    outside.write_text("SECRET")
    link = agent.current_directory / "link.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks not supported here")
    assert "escapes the working directory" in agent.read_file("link.txt")


def test_sensitive_file_read_blocked(agent):
    (agent.current_directory / ".env").write_text("SECRET=1")
    assert "sensitive" in agent.read_file(".env")


def test_sensitive_files_hidden_from_listing_and_glob(agent):
    (agent.current_directory / ".env").write_text("SECRET=1")
    assert ".env" not in agent.list_files(".")
    assert agent.find_files(".env").startswith("No files found")


def test_external_symlink_hidden_from_listing(agent):
    outside = agent.current_directory.parent / "outside.txt"
    outside.write_text("SECRET")
    try:
        (agent.current_directory / "link.txt").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks not supported here")
    assert "link.txt" not in agent.list_files(".")


# --- Command whitelist + injection -------------------------------------------
def test_allowed_command_runs(agent):
    (agent.current_directory / "a.txt").write_text("data")
    assert "data" in agent.run_command("cat a.txt")


def test_disallowed_command_blocked(agent):
    assert "not allowed" in agent.run_command("rm -rf .")


def test_shell_chaining_does_not_execute_second_command(agent):
    # The classic bypass: first token is allowed, second is destructive.
    marker = agent.current_directory / "should_not_exist.txt"
    out = agent.run_command("echo hi; touch should_not_exist.txt")
    # No shell => the whole thing is argv to echo; nothing gets created.
    assert not marker.exists()
    assert "should_not_exist.txt" in out or "hi" in out  # echoed literally


def test_pipe_chaining_blocked(agent):
    agent.run_command("cat a.txt | rm -rf .")
    assert not _destroyed(agent.current_directory)


def test_sensitive_path_blocked(agent):
    (agent.current_directory / ".env").write_text("SECRET=1")
    assert "sensitive path" in agent.run_command("cat .env")


def test_absolute_path_argument_blocked(agent):
    # An allowed command must not become a read primitive for outside files.
    assert "points outside the workspace" in agent.run_command("cat /etc/passwd")


def test_find_command_not_allowed(agent):
    # `find` is off the whitelist: its flags can act (`find . -delete`,
    # `find . -exec rm {} +`) and bypass the allow-list, so it is rejected
    # outright. File finding goes through the dedicated find_files tool.
    assert "not allowed" in agent.run_command("find /etc -maxdepth 0")


def test_find_delete_is_blocked(agent):
    # Regression: `find . -delete` must not run. Before `find` was removed from
    # the whitelist, this wiped the workspace (flags are skipped by arg containment).
    (agent.current_directory / "keep.txt").write_text("data")
    assert "not allowed" in agent.run_command("find . -delete")
    assert (agent.current_directory / "keep.txt").exists()


def test_parent_traversal_argument_blocked(agent):
    assert "points outside the workspace" in agent.run_command("cat ../../etc/hosts")


def test_flags_are_not_treated_as_paths(agent):
    (agent.current_directory / "a.txt").write_text("one\ntwo\nthree\n")
    out = agent.run_command("head -n 2 a.txt")
    assert "one" in out and "three" not in out


# --- find_files containment ---------------------------------------------------
def test_find_files_absolute_pattern_rejected(agent):
    assert "must be a relative path" in agent.find_files("/etc/*")


def test_find_files_parent_pattern_rejected(agent):
    assert "must be a relative path" in agent.find_files("../*")


def test_find_files_inside_ok(agent):
    (agent.current_directory / "x.py").write_text("# hi")
    assert "x.py" in agent.find_files("*.py")


# --- web search is off by default --------------------------------------------
def test_web_search_not_registered_by_default(agent):
    names = {t["function"]["name"] for t in agent.get_tools_definition()}
    assert "web_search" not in names
    assert "not implemented" in agent.execute_tool("web_search", {"query": "x"})


def test_web_search_registered_when_enabled(tmp_path):
    a = CodingAgent(use_web=True)
    a.current_directory = tmp_path.resolve()
    names = {t["function"]["name"] for t in a.get_tools_definition()}
    assert "web_search" in names


def test_web_search_denied_by_default(tmp_path, monkeypatch):
    a = CodingAgent(use_web=True)
    a.current_directory = tmp_path.resolve()
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    called = False

    def fake_post(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr("agent.core.requests.post", fake_post)
    assert "rejected" in a.web_search("explain this code")
    assert called is False


def test_web_search_redacts_then_sends_approved_query(tmp_path, monkeypatch):
    a = CodingAgent(use_web=True)
    a.current_directory = tmp_path.resolve()
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    approved = []
    sent = []
    a._web_approver = lambda query: approved.append(query) or True

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"results": []}

    def fake_post(*args, **kwargs):
        sent.append(kwargs["json"]["query"])
        return Response()

    monkeypatch.setattr("agent.core.requests.post", fake_post)
    a.web_search("api_key=abcdefghijklmnop")
    assert approved == ["[REDACTED]"]
    assert sent == ["[REDACTED]"]


def _destroyed(path: pathlib.Path) -> bool:
    return not path.exists()
