#!/usr/bin/env python3
"""ChromaDB Re-indexing script for StudyMatrix RAG / Lecture Workbench.

Run this script whenever you change your embedding model (HF_EMBED_MODEL in .env)
to regenerate dense vector embeddings in ChromaDB and refresh BM25 token indexes.

Usage:
    py scripts/reindex.py [--force]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from core.indexing import (
    CHROMA_DIR,
    INDEX_DIR,
    get_embedding_model,
    index_lecture,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("reindex")


def reindex_all(force: bool = False) -> None:
    embed_model = os.getenv("HF_EMBED_MODEL") or os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    logger.info("Starting re-indexing with model: %s", embed_model)

    lectures_file = PROJECT_ROOT / "data" / "lectures.json"
    if not lectures_file.exists():
        logger.warning("No lectures.json registry found at %s. Nothing to re-index.", lectures_file)
        return

    with open(lectures_file, "r", encoding="utf-8") as f:
        lectures: dict = json.load(f)

    if not lectures:
        logger.info("Lectures registry is empty. Nothing to re-index.")
        return

    logger.info("Found %d registered lecture(s).", len(lectures))

    # Pre-load embedding model once
    model = get_embedding_model()
    logger.info("Loaded embedding model successfully.")

    import chromadb
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    success_count = 0
    fail_count = 0

    for lecture_id, record in lectures.items():
        name = record.get("original_name") or record.get("title") or lecture_id
        col_name = f"lecture_{lecture_id}"
        logger.info("Re-indexing [%s] '%s'...", lecture_id[:12], name)

        try:
            # Delete old collection to prevent vector dimension mismatch
            try:
                chroma_client.delete_collection(name=col_name)
                logger.info("  Cleared previous Chroma collection: %s", col_name)
            except Exception:
                pass  # Did not exist

            result = index_lecture(
                record,
                progress_callback=lambda p, msg: logger.info("  [%d%%] %s", int(p * 100), msg),
            )
            logger.info("  Successfully re-indexed %d chunks for %s", result["chunk_count"], lecture_id)
            success_count += 1
        except Exception as exc:
            logger.error("  Failed to re-index lecture %s: %s", lecture_id, exc)
            fail_count += 1

    logger.info("Re-indexing complete: %d succeeded, %d failed.", success_count, fail_count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-index ChromaDB with new embedding model")
    parser.add_argument("--force", action="store_true", help="Force re-indexing all lectures")
    args = parser.parse_args()
    reindex_all(force=args.force)
