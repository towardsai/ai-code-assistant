"""AST-aware symbol tools (Article 6, Part 2 / Steps 4-5).

Structure-aware lookup backed by tree-sitter. Where ``grep`` matches text,
``find_symbol``/``list_symbols`` parse the file and return whole definitions
(functions, classes, methods) with their exact line spans.
"""
from __future__ import annotations

import pathlib
from typing import Iterator, Optional

from tree_sitter import Language, Parser
import tree_sitter_python
import tree_sitter_javascript

from .security import is_sensitive_path, resolve_in_workspace

# File extension per language and the reverse map used for detection.
EXT = {"python": "py", "javascript": "js"}
LANG_BY_EXT = {".py": "python", ".js": "javascript"}

# The node types that count as a "symbol" definition in each grammar.
DEFINITION_NODES = {
    "python": ("function_definition", "class_definition"),
    "javascript": ("function_declaration", "class_declaration", "method_definition"),
}

# Directories we never want to parse.
IGNORE_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache",
               ".pytest_cache", "dist", "build", ".code_index"}

_GRAMMARS = {"python": tree_sitter_python, "javascript": tree_sitter_javascript}
_PARSERS: dict[str, Parser] = {}


def get_parser(language: str) -> Parser:
    """Return a cached tree-sitter parser for ``language``."""
    if language not in _PARSERS:
        if language not in _GRAMMARS:
            raise ValueError(f"unsupported language: {language!r}")
        _PARSERS[language] = Parser(Language(_GRAMMARS[language].language()))
    return _PARSERS[language]


def walk(node) -> Iterator:
    """Depth-first walk over a tree-sitter node and all its descendants."""
    yield node
    for child in node.children:
        yield from walk(child)


def detect_language(file_path) -> str:
    """Infer a language from a file's extension; default to python."""
    return LANG_BY_EXT.get(pathlib.Path(file_path).suffix, "python")


def is_ignored(file_path: pathlib.Path) -> bool:
    return any(part in IGNORE_DIRS for part in file_path.parts)


def find_symbol(root: pathlib.Path, name: str, language: str = "python") -> list[dict]:
    """Find every definition of ``name`` under ``root`` for ``language``.

    Returns the file path, 1-based start/end lines, and source snippet for each
    matching function or class -- not every line that merely mentions the name.
    """
    root = pathlib.Path(root).resolve()
    parser = get_parser(language)
    node_types = DEFINITION_NODES[language]
    hits: list[dict] = []
    for file_path in sorted(root.rglob(f"*.{EXT[language]}")):
        try:
            resolved_path = resolve_in_workspace(root, file_path)
        except ValueError:
            continue
        if is_ignored(file_path) or is_sensitive_path(resolved_path):
            continue
        source = resolved_path.read_bytes()
        tree = parser.parse(source)
        lines = source.decode("utf-8", "replace").splitlines()
        for node in walk(tree.root_node):
            if node.type in node_types:
                ident = node.child_by_field_name("name")
                if ident and ident.text.decode() == name:
                    start, end = node.start_point[0] + 1, node.end_point[0] + 1
                    hits.append({
                        "file": str(file_path),
                        "start": start,
                        "end": end,
                        "snippet": "\n".join(lines[start - 1:end]),
                    })
    return hits


def list_symbols(file_path, language: Optional[str] = None) -> list[dict]:
    """Return the outline of one file: its functions, classes, and line ranges."""
    language = language or detect_language(file_path)
    parser = get_parser(language)
    source = pathlib.Path(file_path).read_bytes()
    tree = parser.parse(source)
    outline: list[dict] = []
    for node in walk(tree.root_node):
        if node.type in DEFINITION_NODES[language]:
            ident = node.child_by_field_name("name")
            outline.append({
                "kind": node.type.split("_")[0],          # "function" | "class" | "method"
                "name": ident.text.decode() if ident else "?",
                "start": node.start_point[0] + 1,
                "end": node.end_point[0] + 1,
            })
    return outline
