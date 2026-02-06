#!/usr/bin/env python3
"""
Extract mathematical concepts from textbooks and create a vector knowledge base.

This script:
1. Extracts theorems, definitions, propositions, examples, etc. from text files
2. Chunks them intelligently
3. Creates embeddings using sentence-transformers (runs on CPU)
4. Saves the knowledge base for use on Kaggle

Usage:
    python scripts/extract_math_concepts.py --books-dir books/ --output knowledge_base/

Requirements:
    pip install sentence-transformers numpy
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

import numpy as np


# Lazy import for sentence-transformers
def _require_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is required. Install with: pip install sentence-transformers"
        )


@dataclass
class MathConcept:
    """A mathematical concept extracted from a textbook."""

    concept_type: (
        str  # theorem, definition, proposition, example, lemma, corollary, etc.
    )
    title: str | None
    content: str
    source_book: str
    chapter: str | None = None
    page: int | None = None

    def to_text(self) -> str:
        """Convert to text for embedding."""
        parts = []
        if self.concept_type:
            parts.append(f"[{self.concept_type.upper()}]")
        if self.title:
            parts.append(f"{self.title}:")
        parts.append(self.content)
        return " ".join(parts)


# Patterns to detect mathematical concepts
CONCEPT_PATTERNS = [
    # Formal structures
    (
        r"(?i)^(theorem|thm\.?)\s*(\d+[\.\d]*)?[:\.\s]*(.*?)(?=\n\n|\n(?=(?:theorem|definition|lemma|corollary|proposition|proof|example|remark))|\Z)",
        "theorem",
    ),
    (
        r"(?i)^(definition|def\.?)\s*(\d+[\.\d]*)?[:\.\s]*(.*?)(?=\n\n|\n(?=(?:theorem|definition|lemma|corollary|proposition|proof|example|remark))|\Z)",
        "definition",
    ),
    (
        r"(?i)^(lemma)\s*(\d+[\.\d]*)?[:\.\s]*(.*?)(?=\n\n|\n(?=(?:theorem|definition|lemma|corollary|proposition|proof|example|remark))|\Z)",
        "lemma",
    ),
    (
        r"(?i)^(corollary|cor\.?)\s*(\d+[\.\d]*)?[:\.\s]*(.*?)(?=\n\n|\n(?=(?:theorem|definition|lemma|corollary|proposition|proof|example|remark))|\Z)",
        "corollary",
    ),
    (
        r"(?i)^(proposition|prop\.?)\s*(\d+[\.\d]*)?[:\.\s]*(.*?)(?=\n\n|\n(?=(?:theorem|definition|lemma|corollary|proposition|proof|example|remark))|\Z)",
        "proposition",
    ),
    (
        r"(?i)^(axiom)\s*(\d+[\.\d]*)?[:\.\s]*(.*?)(?=\n\n|\n(?=(?:theorem|definition|lemma|corollary|proposition|proof|example|remark))|\Z)",
        "axiom",
    ),
    (
        r"(?i)^(example)\s*(\d+[\.\d]*)?[:\.\s]*(.*?)(?=\n\n|\n(?=(?:theorem|definition|lemma|corollary|proposition|proof|example|remark|solution))|\Z)",
        "example",
    ),
    (
        r"(?i)^(remark)\s*(\d+[\.\d]*)?[:\.\s]*(.*?)(?=\n\n|\n(?=(?:theorem|definition|lemma|corollary|proposition|proof|example|remark))|\Z)",
        "remark",
    ),
]


def extract_concepts_regex(text: str, source_book: str) -> list[MathConcept]:
    """Extract concepts using regex patterns."""
    concepts = []

    for pattern, concept_type in CONCEPT_PATTERNS:
        for match in re.finditer(pattern, text, re.MULTILINE | re.DOTALL):
            content = match.group(0).strip()
            # Clean up content
            content = re.sub(r"\s+", " ", content)
            content = content[:2000]  # Limit length

            if len(content) > 50:  # Skip very short matches
                title_match = re.match(r"(?i)^(\w+)\s*(\d+[\.\d]*)?[:\.\s]*", content)
                title = title_match.group(0).strip() if title_match else None

                concepts.append(
                    MathConcept(
                        concept_type=concept_type,
                        title=title,
                        content=content,
                        source_book=source_book,
                    )
                )

    return concepts


def chunk_text_smart(
    text: str, source_book: str, chunk_size: int = 1000, overlap: int = 200
) -> list[MathConcept]:
    """
    Smart chunking that tries to preserve paragraph boundaries.
    Falls back to sliding window for unstructured text.
    """
    chunks = []

    # Split by double newlines (paragraphs)
    paragraphs = re.split(r"\n\s*\n", text)

    current_chunk = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += "\n\n" + para if current_chunk else para
        else:
            if current_chunk:
                chunks.append(
                    MathConcept(
                        concept_type="text_chunk",
                        title=None,
                        content=current_chunk[:chunk_size],
                        source_book=source_book,
                    )
                )
            current_chunk = para

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(
            MathConcept(
                concept_type="text_chunk",
                title=None,
                content=current_chunk[:chunk_size],
                source_book=source_book,
            )
        )

    return chunks


def extract_all_concepts(text: str, source_book: str) -> list[MathConcept]:
    """Extract both structured concepts and general chunks."""
    concepts = []

    # First extract formal structures
    formal_concepts = extract_concepts_regex(text, source_book)
    concepts.extend(formal_concepts)

    # Also create general chunks for context
    chunks = chunk_text_smart(text, source_book, chunk_size=800, overlap=100)
    concepts.extend(chunks)

    # Deduplicate by content similarity
    seen_content = set()
    unique_concepts = []
    for c in concepts:
        content_key = c.content[:100]  # Use first 100 chars as key
        if content_key not in seen_content:
            seen_content.add(content_key)
            unique_concepts.append(c)

    return unique_concepts


def create_embeddings(
    concepts: list[MathConcept],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
) -> np.ndarray:
    """Create embeddings for all concepts."""
    SentenceTransformer = _require_sentence_transformers()

    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    texts = [c.to_text() for c in concepts]

    print(f"Creating embeddings for {len(texts)} concepts...")
    start = time.time()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    elapsed = time.time() - start
    print(f"Embeddings created in {elapsed:.1f}s ({len(texts)/elapsed:.1f} concepts/s)")

    return embeddings


def save_knowledge_base(
    concepts: list[MathConcept],
    embeddings: np.ndarray,
    output_dir: Path,
    metadata: dict | None = None,
):
    """Save the knowledge base to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save embeddings as numpy
    np.save(output_dir / "embeddings.npy", embeddings)

    # Save concepts as JSON (for readability) and pickle (for speed)
    concepts_data = [asdict(c) for c in concepts]

    with open(output_dir / "concepts.json", "w") as f:
        json.dump(concepts_data, f, indent=2)

    with open(output_dir / "concepts.pkl", "wb") as f:
        pickle.dump(concepts, f)

    # Save metadata
    meta = {
        "n_concepts": len(concepts),
        "embedding_dim": embeddings.shape[1],
        "embedding_model": (
            metadata.get("model_name", "unknown") if metadata else "unknown"
        ),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        **(metadata or {}),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Knowledge base saved to {output_dir}")
    print(f"  - {len(concepts)} concepts")
    print(f"  - Embedding shape: {embeddings.shape}")


def load_knowledge_base(kb_dir: Path) -> tuple[list[MathConcept], np.ndarray, dict]:
    """Load knowledge base from disk."""
    embeddings = np.load(kb_dir / "embeddings.npy")

    with open(kb_dir / "concepts.pkl", "rb") as f:
        concepts = pickle.load(f)

    with open(kb_dir / "metadata.json") as f:
        metadata = json.load(f)

    return concepts, embeddings, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Extract math concepts and create knowledge base"
    )
    parser.add_argument(
        "--books-dir",
        type=str,
        default="books",
        help="Directory containing book text files",
    )
    parser.add_argument(
        "--output", type=str, default="knowledge_base", help="Output directory"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence transformer model",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for embedding"
    )

    args = parser.parse_args()

    books_dir = Path(args.books_dir)
    output_dir = Path(args.output)

    # Find all text files
    book_files = list(books_dir.glob("*.txt"))
    if not book_files:
        print(f"No .txt files found in {books_dir}")
        return

    print(f"Found {len(book_files)} books: {[f.name for f in book_files]}")

    # Extract concepts from all books
    all_concepts = []
    for book_file in book_files:
        print(f"\nProcessing {book_file.name}...")
        text = book_file.read_text(encoding="utf-8", errors="ignore")
        concepts = extract_all_concepts(text, source_book=book_file.stem)
        print(f"  Extracted {len(concepts)} concepts")
        all_concepts.extend(concepts)

    print(f"\nTotal concepts: {len(all_concepts)}")

    # Show concept type distribution
    from collections import Counter

    type_counts = Counter(c.concept_type for c in all_concepts)
    print("\nConcept types:")
    for ctype, count in type_counts.most_common():
        print(f"  {ctype}: {count}")

    # Create embeddings
    embeddings = create_embeddings(
        all_concepts, model_name=args.model, batch_size=args.batch_size
    )

    # Save
    save_knowledge_base(
        all_concepts,
        embeddings,
        output_dir,
        metadata={
            "model_name": args.model,
            "books": [f.name for f in book_files],
        },
    )


if __name__ == "__main__":
    main()
