from typing import Optional, Any

from ..models.plot_unit import AnnotatedPlotUnit
from ..models.character import Character


class VectorStore:
    """ChromaDB wrapper providing persistent storage for plot units, characters, and themes."""

    COLLECTION_PLOTS = "plot_units"
    COLLECTION_CHARACTERS = "characters"
    COLLECTION_THEMES = "themes"

    def __init__(self, persist_directory: str = "data/vectordb"):
        self.persist_directory = persist_directory
        self._client = None
        self._collections: dict[str, Any] = {}

    def _get_client(self):
        if self._client is None:
            try:
                import chromadb  # type: ignore[import]
            except ImportError as exc:
                raise ImportError("chromadb is required: pip install chromadb>=0.4.0") from exc
            self._client = chromadb.PersistentClient(path=self.persist_directory)
        return self._client

    def _get_collection(self, name: str):
        if name not in self._collections:
            client = self._get_client()
            self._collections[name] = client.get_or_create_collection(name)
        return self._collections[name]

    def add_plot_units(self, units: list[AnnotatedPlotUnit], embeddings: list[list[float]]) -> None:
        """Store annotated plot units with their embeddings."""
        collection = self._get_collection(self.COLLECTION_PLOTS)
        ids = [u.plot_unit.id for u in units]
        documents = [u.plot_unit.content for u in units]
        metadatas = [
            {
                "novel_id": u.plot_unit.novel_id,
                "chapter_id": u.plot_unit.chapter_id,
                "narrative_function": u.narrative_function,
                "emotion_tone": u.emotion_tone,
                "tension_level": str(u.tension_level),
                "conflict_type": u.conflict_type,
                "event_type": u.plot_unit.event_type,
                "themes": ",".join(u.themes),
            }
            for u in units
        ]
        collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def add_characters(self, characters: list[Character], embeddings: list[list[float]]) -> None:
        """Store characters with their embeddings."""
        collection = self._get_collection(self.COLLECTION_CHARACTERS)
        ids = [c.id for c in characters]
        documents = [f"{c.name}: {c.description}" for c in characters]
        metadatas = [
            {
                "novel_id": c.novel_id,
                "name": c.name,
                "role": c.role,
                "traits": ",".join(c.traits),
            }
            for c in characters
        ]
        collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def query_plots(
        self,
        query_embedding: list[float],
        n_results: int = 10,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """Query plot units by embedding similarity."""
        collection = self._get_collection(self.COLLECTION_PLOTS)
        kwargs: dict = {"query_embeddings": [query_embedding], "n_results": n_results}
        if where:
            kwargs["where"] = where
        results = collection.query(**kwargs)
        return self._format_results(results)

    def query_characters(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """Query characters by embedding similarity."""
        collection = self._get_collection(self.COLLECTION_CHARACTERS)
        kwargs: dict = {"query_embeddings": [query_embedding], "n_results": n_results}
        if where:
            kwargs["where"] = where
        results = collection.query(**kwargs)
        return self._format_results(results)

    def _format_results(self, results: dict) -> list[dict]:
        """Convert ChromaDB query results to a list of dicts."""
        formatted = []
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        for i, doc_id in enumerate(ids):
            formatted.append(
                {
                    "id": doc_id,
                    "document": documents[i] if i < len(documents) else "",
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                    "distance": distances[i] if i < len(distances) else 1.0,
                }
            )
        return formatted

    def count_plots(self) -> int:
        collection = self._get_collection(self.COLLECTION_PLOTS)
        return collection.count()
