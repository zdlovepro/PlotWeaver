from typing import Optional

import numpy as np

from ..models.character import Character, CharacterMapping
from ..rag.embedder import Embedder


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    va = np.array(a, dtype=float)
    vb = np.array(b, dtype=float)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


def _hungarian_assignment(cost_matrix: np.ndarray) -> list[tuple[int, int]]:
    """Solve the assignment problem using the Hungarian algorithm (pure scipy)."""
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore[import]
        row_ind, col_ind = linear_sum_assignment(-cost_matrix)  # maximise similarity
        return list(zip(row_ind.tolist(), col_ind.tolist()))
    except ImportError:
        # Greedy fallback if scipy not available
        assignments = []
        used_cols = set()
        for row in range(cost_matrix.shape[0]):
            best_col = -1
            best_val = -1.0
            for col in range(cost_matrix.shape[1]):
                if col not in used_cols and cost_matrix[row, col] > best_val:
                    best_val = cost_matrix[row, col]
                    best_col = col
            if best_col >= 0:
                assignments.append((row, best_col))
                used_cols.add(best_col)
        return assignments


class CharacterMapper:
    """Maps characters between source and target novels using cosine similarity + Hungarian algorithm."""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder

    def map_characters(
        self,
        source_characters: list[Character],
        target_characters: list[Character],
    ) -> list[CharacterMapping]:
        """Find optimal one-to-one character mappings between two novels."""
        if not source_characters or not target_characters:
            return []

        # Build description strings for embedding
        source_texts = [self._character_text(c) for c in source_characters]
        target_texts = [self._character_text(c) for c in target_characters]

        source_embs = self.embedder.embed(source_texts)
        target_embs = self.embedder.embed(target_texts)

        # Build similarity matrix
        n, m = len(source_embs), len(target_embs)
        sim_matrix = np.zeros((n, m), dtype=float)
        for i, se in enumerate(source_embs):
            for j, te in enumerate(target_embs):
                sim_matrix[i, j] = _cosine_similarity(se, te)

        # Solve assignment
        assignments = _hungarian_assignment(sim_matrix)

        mappings = []
        for src_idx, tgt_idx in assignments:
            mappings.append(
                CharacterMapping(
                    source_character_id=source_characters[src_idx].id,
                    target_character_id=target_characters[tgt_idx].id,
                    similarity_score=float(sim_matrix[src_idx, tgt_idx]),
                    source_novel_id=source_characters[src_idx].novel_id,
                    target_novel_id=target_characters[tgt_idx].novel_id,
                )
            )
        return mappings

    @staticmethod
    def _character_text(c: Character) -> str:
        parts = [c.name]
        if c.role:
            parts.append(c.role)
        if c.description:
            parts.append(c.description)
        if c.traits:
            parts.append(" ".join(c.traits))
        return " ".join(parts)
