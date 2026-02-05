"""
Math Knowledge Retriever - retrieves relevant mathematical concepts at runtime.

This is designed to be lightweight and run on CPU during Kaggle inference.
The knowledge base must be pre-computed using extract_math_concepts.py.

Usage:
    from olympiad_llm.aimo3.math_retriever import MathRetriever
    
    retriever = MathRetriever.load("/kaggle/input/math-kb")
    relevant = retriever.retrieve(problem_text, top_k=5)
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MathConcept:
    """A mathematical concept from the knowledge base."""
    concept_type: str
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
    
    def to_prompt_format(self) -> str:
        """Format for injection into LLM prompt."""
        type_label = self.concept_type.upper() if self.concept_type else "CONCEPT"
        if self.title:
            return f"**{type_label}** ({self.title}): {self.content}"
        return f"**{type_label}**: {self.content}"


@dataclass
class RetrievalResult:
    """Result from knowledge retrieval."""
    concept: MathConcept
    score: float  # Similarity score (higher = more relevant)
    rank: int


class MathRetriever:
    """
    Retrieves relevant mathematical concepts using semantic similarity.
    
    Designed to run on CPU with minimal memory footprint.
    """
    
    def __init__(
        self,
        concepts: list[MathConcept],
        embeddings: np.ndarray,
        model_name: str = "all-MiniLM-L6-v2",
        model_path: str | None = None,
        cpu_only: bool = True,
    ):
        self.concepts = concepts
        self.embeddings = embeddings.astype(np.float32)  # Ensure float32 for efficiency
        self.model_name = model_name
        # Local path to model directory (for offline/Kaggle use)
        # If set, loads from this path instead of downloading
        self.model_path = model_path
        # Force CPU-only inference (recommended for Kaggle)
        self.cpu_only = cpu_only
        self._encoder = None
        
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings_normalized = self.embeddings / (norms + 1e-9)
    
    @classmethod
    def load(cls, kb_dir: str | Path, model_path: str | None = None, cpu_only: bool = True) -> "MathRetriever":
        """Load knowledge base from directory.
        
        Args:
            kb_dir: Path to knowledge base directory containing embeddings.npy, concepts.json, metadata.json
            model_path: Optional local path to sentence-transformer model directory.
                       If None, will download from HuggingFace (requires internet).
                       For Kaggle offline use, set to the mounted model path, e.g.:
                       /kaggle/input/sentence-transformersall-minilm-l6-v2/all-MiniLM-L6-v2
            cpu_only: If True, force CPU-only inference (recommended for Kaggle)
        """
        kb_dir = Path(kb_dir)
        
        # Load embeddings
        embeddings = np.load(kb_dir / "embeddings.npy")
        
        # Load concepts - prefer JSON for portability, fallback to pickle
        concepts_json_path = kb_dir / "concepts.json"
        concepts_pkl_path = kb_dir / "concepts.pkl"
        
        if concepts_json_path.exists():
            with open(concepts_json_path) as f:
                concepts_data = json.load(f)
            concepts = [
                MathConcept(
                    concept_type=c.get("concept_type", ""),
                    title=c.get("title"),
                    content=c.get("content", ""),
                    source_book=c.get("source_book", ""),
                    chapter=c.get("chapter"),
                    page=c.get("page"),
                )
                for c in concepts_data
            ]
        elif concepts_pkl_path.exists():
            with open(concepts_pkl_path, "rb") as f:
                concepts = pickle.load(f)
        else:
            raise FileNotFoundError(f"No concepts file found in {kb_dir}")
        
        # Load metadata
        with open(kb_dir / "metadata.json") as f:
            metadata = json.load(f)
        
        model_name = metadata.get("embedding_model", "all-MiniLM-L6-v2")
        
        return cls(concepts, embeddings, model_name, model_path=model_path, cpu_only=cpu_only)
    
    def _get_encoder(self):
        """Lazy-load the sentence transformer encoder (CPU only by default)."""
        if self._encoder is None:
            model_id = self.model_path if self.model_path else self.model_name
            device = "cpu" if self.cpu_only else "cuda"
            
            # For local paths (Kaggle offline), use direct transformers loading
            # This avoids sentence-transformers format issues
            if self.model_path and self.model_path.startswith("/"):
                self._encoder = self._load_direct_transformer(model_id, device)
            else:
                # For model names (online), use sentence-transformers
                self._encoder = self._load_sentence_transformer(model_id, device)
        
        return self._encoder
    
    def _load_direct_transformer(self, model_path: str, device: str):
        """Load transformer model directly (for Kaggle offline mode)."""
        import torch
        from transformers import AutoModel, AutoTokenizer, AutoConfig, BertConfig
        
        print(f"[Retriever] Loading transformer directly from {model_path}")
        
        # Load config and patch model_type if missing
        try:
            config = AutoConfig.from_pretrained(model_path)
        except ValueError:
            print("[Retriever] AutoConfig failed to detect model type, falling back to BertConfig")
            config = BertConfig.from_pretrained(model_path)
        
        if not hasattr(config, 'model_type') or config.model_type is None:
            # MiniLM models are BERT-based
            config.model_type = "bert"
        
        model = AutoModel.from_pretrained(model_path, config=config)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = model.to(device)
        model.eval()
        
        embedding_dim = config.hidden_size
        
        # Create wrapper with mean pooling
        class DirectTransformerEncoder:
            def __init__(self, model, tokenizer, device, dim):
                self.model = model
                self.tokenizer = tokenizer
                self.device = device
                self.embedding_dim = dim
            
            def encode(self, sentences, convert_to_numpy=True, **kwargs):
                if isinstance(sentences, str):
                    sentences = [sentences]
                inputs = self.tokenizer(
                    sentences, padding=True, truncation=True,
                    max_length=512, return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model(**inputs)
                # Mean pooling
                attention_mask = inputs["attention_mask"]
                token_embeddings = outputs.last_hidden_state
                mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                embeddings = sum_embeddings / sum_mask
                if convert_to_numpy:
                    return embeddings.cpu().numpy()
                return embeddings
        
        encoder = DirectTransformerEncoder(model, tokenizer, device, embedding_dim)
        print(f"[Retriever] ✓ Loaded transformer with {embedding_dim}d embeddings")
        return encoder
    
    def _load_sentence_transformer(self, model_name: str, device: str):
        """Load model via sentence-transformers (for online/named models)."""
        try:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer(model_name, device=device)
            print(f"[Retriever] ✓ Loaded SentenceTransformer: {model_name}")
            return encoder
        except ImportError:
            raise ImportError(
                "sentence-transformers required for query encoding. "
                "Install with: pip install sentence-transformers"
            )
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode a query string to embedding."""
        encoder = self._get_encoder()
        embedding = encoder.encode([query], convert_to_numpy=True)[0]
        return embedding.astype(np.float32)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        concept_types: list[str] | None = None,
        min_score: float = 0.0,
    ) -> tuple[list[RetrievalResult], dict[str, Any]]:
        """
        Retrieve top-k most relevant concepts for a query.
        
        Args:
            query: The problem text or search query
            top_k: Number of results to return
            concept_types: Optional filter by concept type (e.g., ["theorem", "definition"])
            min_score: Minimum similarity score threshold
            
        Returns:
            Tuple of (results, metadata) where:
            - results: List of RetrievalResult sorted by relevance (highest first)
            - metadata: Dict with timing and stats (retrieval_time_ms, query_len, results_count, etc.)
        """
        start_time = time.time()
        query_len = len(query)
        
        # Encode query
        query_emb = self.encode_query(query)
        query_emb_norm = query_emb / (np.linalg.norm(query_emb) + 1e-9)
        
        # Compute cosine similarities
        scores = np.dot(self.embeddings_normalized, query_emb_norm)
        
        # Apply concept type filter if specified
        if concept_types:
            type_set = set(t.lower() for t in concept_types)
            mask = np.array([c.concept_type.lower() in type_set for c in self.concepts])
            scores = np.where(mask, scores, -np.inf)
        
        # Get top-k indices
        if top_k >= len(scores):
            top_indices = np.argsort(scores)[::-1]
        else:
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        
        # Build results
        results = []
        for rank, idx in enumerate(top_indices):
            score = float(scores[idx])
            if score < min_score:
                break
            results.append(RetrievalResult(
                concept=self.concepts[idx],
                score=score,
                rank=rank + 1,
            ))
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Return results and metadata for tracing
        metadata = {
            "retrieval_time_ms": round(elapsed_ms, 2),
            "query_len": query_len,
            "results_count": len(results),
            "top_k_requested": top_k,
            "min_score_threshold": min_score,
            "concept_types_filtered": concept_types is not None,
            "avg_score": round(np.mean([r.score for r in results]), 3) if results else 0.0,
        }
        
        logger.debug(
            f"Retrieved {len(results)} concepts for {query_len}-char query in {elapsed_ms:.1f}ms"
        )
        
        return results, metadata
    
    def retrieve_for_problem(
        self,
        problem: str,
        top_k: int = 5,
        include_examples: bool = True,
        include_definitions: bool = True,
    ) -> tuple[str, dict[str, Any]]:
        """
        Retrieve relevant concepts and format for prompt injection.
        
        Returns tuple of (formatted_string, metadata) where:
        - formatted_string: Ready to inject into the LLM prompt
        - metadata: Retrieval stats for tracing
        """
        # Filter concept types based on preferences
        concept_types = None
        if not include_examples or not include_definitions:
            concept_types = ["theorem", "lemma", "corollary", "proposition", "axiom"]
            if include_definitions:
                concept_types.append("definition")
            if include_examples:
                concept_types.append("example")
        
        results, metadata = self.retrieve(
            query=problem,
            top_k=top_k,
            concept_types=concept_types,
            min_score=0.3,  # Only include reasonably relevant results
        )
        
        if not results:
            return "", metadata
        
        lines = ["**Potentially Relevant Mathematical Concepts:**", ""]
        for r in results:
            lines.append(f"- {r.concept.to_prompt_format()}")
            lines.append("")
        
        return "\n".join(lines), metadata


def benchmark_retrieval(kb_dir: str, test_queries: list[str] | None = None):
    """Benchmark retrieval speed."""
    
    print("Loading knowledge base...")
    start = time.time()
    retriever = MathRetriever.load(kb_dir)
    load_time = time.time() - start
    print(f"Loaded in {load_time:.2f}s ({len(retriever.concepts)} concepts)")
    
    if test_queries is None:
        test_queries = [
            "Find all positive integers n such that n divides 2^n - 1",
            "Prove that the sum of the first n odd numbers equals n squared",
            "Let f be a function satisfying f(x+y) = f(x) + f(y) for all real x,y",
            "Count the number of ways to tile a 2xn board with dominoes",
            "Prove there are infinitely many primes",
        ]
    
    print(f"\nBenchmarking {len(test_queries)} queries...")
    
    total_time = 0
    for query in test_queries:
        start = time.time()
        results = retriever.retrieve(query, top_k=5)
        elapsed = time.time() - start
        total_time += elapsed
        
        print(f"\nQuery: {query[:60]}...")
        print(f"  Time: {elapsed*1000:.1f}ms")
        print(f"  Top results:")
        for r in results[:3]:
            print(f"    [{r.score:.3f}] {r.concept.concept_type}: {r.concept.content[:60]}...")
    
    avg_time = total_time / len(test_queries)
    print(f"\nAverage retrieval time: {avg_time*1000:.1f}ms per query")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        benchmark_retrieval(sys.argv[1])
    else:
        print("Usage: python math_retriever.py <knowledge_base_dir>")
