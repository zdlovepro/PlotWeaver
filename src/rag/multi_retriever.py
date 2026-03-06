from typing import Optional

import numpy as np

from .embedder import Embedder
from .vector_store import VectorStore


class MultiRetriever:
    """Multi-path retrieval with weighted reranking across 4 retrieval strategies."""

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        top_k: int = 10,
        weights: Optional[dict] = None,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k
        # Default retrieval path weights
        self.weights = weights or {
            "semantic": 0.4,
            "metadata_filter": 0.3,
            "reverse": 0.15,
            "theme": 0.15,
        }

    def retrieve(
        self,
        query: str,
        narrative_function: Optional[str] = None,
        emotion_tone: Optional[str] = None,
        themes: Optional[list[str]] = None,
    ) -> list[dict]:
        """Run 4-path retrieval and return reranked results."""
        query_emb = self.embedder.embed_one(query)

        # Path 1: Semantic similarity
        semantic_results = self.vector_store.query_plots(query_emb, n_results=self.top_k)

        # Path 2: Metadata filter by narrative_function or emotion_tone
        metadata_results = []
        if narrative_function:
            metadata_results = self.vector_store.query_plots(
                query_emb,
                n_results=self.top_k,
                where={"narrative_function": {"$eq": narrative_function}},
            )
        elif emotion_tone:
            metadata_results = self.vector_store.query_plots(
                query_emb,
                n_results=self.top_k,
                where={"emotion_tone": {"$eq": emotion_tone}},
            )

        # Path 3: Reverse semantic (query with negated embedding - find contrasting plots)
        reverse_emb = [-v for v in query_emb]
        reverse_results = self.vector_store.query_plots(reverse_emb, n_results=self.top_k // 2)

        # Path 4: Theme-based retrieval
        theme_results = []
        if themes:
            theme_query = " ".join(themes)
            theme_emb = self.embedder.embed_one(theme_query)
            theme_results = self.vector_store.query_plots(theme_emb, n_results=self.top_k)

        # Merge and rerank
        all_results = self._rerank(
            semantic_results,
            metadata_results,
            reverse_results,
            theme_results,
        )
        return all_results[: self.top_k]

    def _rerank(
        self,
        semantic: list[dict],
        metadata: list[dict],
        reverse: list[dict],
        theme: list[dict],
    ) -> list[dict]:
        """Weighted reranking: aggregate scores across all retrieval paths."""
        scores: dict[str, float] = {}
        result_cache: dict[str, dict] = {}

        def _add(results: list[dict], weight: float) -> None:
            n = len(results)
            if n == 0:
                return
            for rank, r in enumerate(results):
                doc_id = r["id"]
                # Score inversely proportional to distance * rank penalty
                score = weight * (1.0 - r.get("distance", 0.5)) * (1.0 - rank / n)
                scores[doc_id] = scores.get(doc_id, 0.0) + score
                result_cache[doc_id] = r

        _add(semantic, self.weights["semantic"])
        _add(metadata, self.weights["metadata_filter"])
        _add(reverse, self.weights["reverse"])
        _add(theme, self.weights["theme"])

        sorted_ids = sorted(scores, key=lambda k: scores[k], reverse=True)
        reranked = []
        for doc_id in sorted_ids:
            item = dict(result_cache[doc_id])
            item["rerank_score"] = scores[doc_id]
            reranked.append(item)
        return reranked
