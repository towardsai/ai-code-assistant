"""AST symbol-tool tests (Steps 4-5). No Ollama required."""
import pathlib

from agent import symbols

SAMPLE = pathlib.Path(__file__).parent.parent / "examples" / "sample_project"


def test_find_symbol_returns_span():
    hits = symbols.find_symbol(SAMPLE, "process_payment")
    assert len(hits) == 1
    hit = hits[0]
    assert hit["start"] < hit["end"]
    assert "def process_payment" in hit["snippet"]
    assert "insufficient funds" in hit["snippet"]


def test_find_symbol_class():
    hits = symbols.find_symbol(SAMPLE, "Calculator")
    assert len(hits) == 1
    assert "class Calculator" in hits[0]["snippet"]


def test_find_symbol_missing():
    assert symbols.find_symbol(SAMPLE, "does_not_exist") == []


def test_list_symbols_outline():
    outline = symbols.list_symbols(SAMPLE / "calculator.py")
    names = {s["name"] for s in outline}
    # class, its methods, and the module-level function all surface.
    assert {"Calculator", "add", "subtract", "reset", "process_payment"} <= names
    kinds = {s["name"]: s["kind"] for s in outline}
    assert kinds["Calculator"] == "class"
    assert kinds["add"] == "function"   # python methods parse as function_definition


def test_detect_language():
    assert symbols.detect_language("foo.py") == "python"
    assert symbols.detect_language("foo.js") == "javascript"
    assert symbols.detect_language("foo.unknown") == "python"  # default


def test_find_symbol_skips_symlink_escape(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "secret.py"
    outside.write_text('def leaked_symbol():\n    return "TOP_SECRET"\n')
    try:
        (root / "link.py").symlink_to(outside)
    except OSError:
        return
    assert symbols.find_symbol(root, "leaked_symbol") == []
