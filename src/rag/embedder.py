from typing import Optional

import numpy as np


class Embedder:
    """Generates embeddings using sentence-transformers (local) or OpenAI API."""

    def __init__(
        self,
        provider: str = "local",
        model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        api_key: str = "",
        openai_model: str = "text-embedding-3-small",
    ):
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key
        self.openai_model = openai_model
        self._local_model = None

    def _get_local_model(self):
        if self._local_model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for local embeddings: "
                    "pip install sentence-transformers>=2.2.0"
                ) from exc
            self._local_model = SentenceTransformer(self.model)
        return self._local_model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return embeddings for a list of texts."""
        if not texts:
            return []
        if self.provider == "local":
            return self._embed_local(texts)
        elif self.provider == "openai":
            return self._embed_openai(texts)
        else:
            raise ValueError(f"Unknown embedding provider: {self.provider!r}")

    def embed_one(self, text: str) -> list[float]:
        """Return embedding for a single text."""
        return self.embed([text])[0]

    def _embed_local(self, texts: list[str]) -> list[list[float]]:
        model = self._get_local_model()
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.tolist()

    def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        try:
            from openai import OpenAI  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("openai package is required: pip install openai>=1.0.0") from exc
        client = OpenAI(api_key=self.api_key)
        response = client.embeddings.create(model=self.openai_model, input=texts)
        return [item.embedding for item in response.data]
