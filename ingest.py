"""Ingest pipeline pour le RAG Feynman-Robinson — version bilingue (EN + FR).

Walks corpus/ pour trouver les .txt avec headers `# tag:`, `# lang:` et `# source:`,
chunke, et insère dans ChromaDB local (data/chroma/).

6 collections cibles, mappées sur (mode, langue) :
- feynman_refuse_en, feynman_refuse_fr      → mode CHEAT
- feynman_explain_en, feynman_explain_fr    → mode UNDERSTAND
- robinson_meaning_en, robinson_meaning_fr  → mode MEANING

Run :  python ingest.py
Idempotent : reset les 6 collections à chaque run.
"""
from __future__ import annotations

import os

# Doit être set AVANT l'import chromadb (cf app.py pour le contexte).
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import sys
from pathlib import Path

import chromadb

CORPUS_DIR = Path(__file__).parent / "corpus"
CHROMA_DIR = Path(__file__).parent / "data" / "chroma"

VALID_TAGS = {"feynman-refuse", "feynman-explain", "robinson-meaning"}
VALID_LANGS = {"en", "fr"}
DEFAULT_LANG = "en"  # si un fichier ne déclare pas # lang:, on assume EN (defaut projet)

# Collection naming : <tag normalisé>_<lang>
def collection_name(tag: str, lang: str) -> str:
    return f"{tag.replace('-', '_')}_{lang}"


ALL_COLLECTIONS = [collection_name(t, l) for t in VALID_TAGS for l in VALID_LANGS]

TARGET_WORDS = 280
OVERLAP_WORDS = 50


def parse_file(path: Path) -> dict | None:
    """Lit un .txt avec header `# tag:` `# lang:` `# source:` et retourne {tag, lang, source, content, file}."""
    raw = path.read_text(encoding="utf-8")
    tag: str | None = None
    lang: str | None = None
    source = "unknown"
    body_lines: list[str] = []
    for line in raw.splitlines():
        if line.startswith("# tag:"):
            tag = line.split(":", 1)[1].strip()
        elif line.startswith("# lang:"):
            lang = line.split(":", 1)[1].strip().lower()
        elif line.startswith("# source:"):
            source = line.split(":", 1)[1].strip()
        elif line.startswith("# "):
            continue
        else:
            body_lines.append(line)
    if not tag:
        print(f"  warning: {path} sans header # tag:, ignoré")
        return None
    if tag not in VALID_TAGS:
        print(f"  warning: {path} tag inconnu '{tag}', ignoré (attendu : {sorted(VALID_TAGS)})")
        return None
    if lang is None:
        lang = DEFAULT_LANG
        print(f"  info: {path} sans header # lang:, assumé '{DEFAULT_LANG}'")
    if lang not in VALID_LANGS:
        print(f"  warning: {path} langue inconnue '{lang}', ignoré (attendu : {sorted(VALID_LANGS)})")
        return None
    content = "\n".join(body_lines).strip()
    if not content:
        print(f"  warning: {path} contenu vide, ignoré")
        return None
    return {
        "tag": tag,
        "lang": lang,
        "source": source,
        "content": content,
        "file": str(path.relative_to(CORPUS_DIR.parent)),
    }


def chunk_text(text: str, target: int = TARGET_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    """Chunker simple : split par paragraphes, merge jusqu'à ~target mots, garde overlap."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for p in paragraphs:
        pw = len(p.split())
        if current_words + pw > target and current:
            chunks.append("\n\n".join(current))
            if current and len(current[-1].split()) <= overlap:
                current = [current[-1]]
                current_words = len(current[-1].split())
            else:
                current = []
                current_words = 0
        current.append(p)
        current_words += pw
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def main() -> int:
    if not CORPUS_DIR.exists():
        print(f"error: corpus directory introuvable: {CORPUS_DIR}", file=sys.stderr)
        return 1

    print(f"Loading corpus from {CORPUS_DIR}/ ...")
    docs = []
    for txt in sorted(CORPUS_DIR.glob("**/*.txt")):
        doc = parse_file(txt)
        if doc is not None:
            docs.append(doc)
    print(f"  found {len(docs)} valid documents")

    if not docs:
        print("error: aucun document valide, abort", file=sys.stderr)
        return 1

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Initializing Chroma at {CHROMA_DIR}/ ...")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Reset toutes les collections cibles (idempotent)
    for name in ALL_COLLECTIONS:
        try:
            client.delete_collection(name)
        except Exception:
            pass

    counters: dict[str, int] = {name: 0 for name in ALL_COLLECTIONS}

    for doc in docs:
        name = collection_name(doc["tag"], doc["lang"])
        collection = client.get_or_create_collection(name)
        chunks = chunk_text(doc["content"])
        if not chunks:
            continue
        offset = counters[name]
        ids = [f"{name}_{offset + i}" for i in range(len(chunks))]
        metadatas = [
            {"source": doc["source"], "file": doc["file"], "lang": doc["lang"]}
            for _ in chunks
        ]
        collection.add(ids=ids, documents=chunks, metadatas=metadatas)
        counters[name] += len(chunks)
        print(f"  + {doc['file']} → {name} ({len(chunks)} chunks)")

    print("\nIngestion terminée.")
    for name, count in counters.items():
        marker = "" if count else "  ← vide"
        print(f"  {name}: {count} chunks{marker}")

    empty = [name for name, count in counters.items() if count == 0]
    if empty:
        print(f"\nwarning: collections vides : {empty}")
        print("  la décision A2 court-circuitera les requêtes qui retombent dans ces partitions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
