"""Optional RAG-over-code upgrade path (Article 6, Part 2 / Step 9).

The top rung of the decision ladder: chunk code by AST node, embed the chunks,
store them in Chroma, and expose ``search_code`` as a tool. Skip this for a
single repo that exploration handles -- it is cost without benefit until grep
and symbol lookup stop scaling.

Note: indexing and querying MUST use the same embedding model. We pass
precomputed embeddings into Chroma on ``add`` and query with
``query_embeddings`` (also precomputed), so the query never falls back to
Chroma's default text embedder, which would be a different vector space.
"""
from __future__ import annotations

import pathlib
from typing import Iterator

from . import symbols
from .security import is_sensitive_path, resolve_in_workspace

DEFAULT_EMBED_MODEL = "nomic-embed-text"   # a code-tuned embedder is preferable
DEFAULT_INDEX_PATH = "./code_index"


def embed(texts: list[str], client, model: str = DEFAULT_EMBED_MODEL) -> list[list[float]]:
    """Embed a batch of texts via the Ollama-compatible embeddings endpoint."""
    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]


def ast_chunks(file_path, language: str = "python") -> Iterator[dict]:
    """Yield one chunk per function / class, with name and line span."""
    parser = symbols.get_parser(language)
    source = pathlib.Path(file_path).read_bytes()
    tree = parser.parse(source)
    lines = source.decode("utf-8", "replace").splitlines()
    for node in symbols.walk(tree.root_node):
        if node.type in symbols.DEFINITION_NODES[language]:
            ident = node.child_by_field_name("name")
            start, end = node.start_point[0] + 1, node.end_point[0] + 1
            yield {
                "source": "\n".join(lines[start - 1:end]),
                "name": ident.text.decode() if ident else "?",
                "start": start,
                "end": end,
            }


def index_repo(root, client, index_path: str = DEFAULT_INDEX_PATH,
               collection_name: str = "code"):
    """Chunk every .py file by AST node, embed the chunks, store them in Chroma."""
    import chromadb

    chunks = []
    resolved_root = pathlib.Path(root).resolve()
    for file_path in resolved_root.rglob("*.py"):
        try:
            resolved_path = resolve_in_workspace(resolved_root, file_path)
        except ValueError:
            continue
        if symbols.is_ignored(file_path) or is_sensitive_path(resolved_path):
            continue
        for sym in ast_chunks(resolved_path):
            chunks.append({"text": sym["source"], "file": str(file_path),
                           "symbol": sym["name"], "start": sym["start"], "end": sym["end"]})
    if not chunks:
        return None

    embeddings = embed([c["text"] for c in chunks], client)
    collection = chromadb.PersistentClient(path=index_path).get_or_create_collection(collection_name)
    collection.add(
        # symbol name is not unique (overloads, same-named methods), so the id
        # includes the start line to stay collision-free.
        ids=[f"{c['file']}::{c['symbol']}::{c['start']}" for c in chunks],
        embeddings=embeddings,
        documents=[c["text"] for c in chunks],
        metadatas=[{"file": c["file"], "symbol": c["symbol"],
                    "start": c["start"], "end": c["end"]} for c in chunks],
    )
    return collection


def search_code(collection, query: str, client, k: int = 8) -> list[dict]:
    """Semantic search over the code index, embedding the query with the same model."""
    query_embeddings = embed([query], client)
    hits = collection.query(query_embeddings=query_embeddings, n_results=k)
    return [
        {"file": m["file"], "symbol": m["symbol"],
         "lines": f"{m['start']}-{m['end']}", "snippet": d}
        for d, m in zip(hits["documents"][0], hits["metadatas"][0])
    ]
