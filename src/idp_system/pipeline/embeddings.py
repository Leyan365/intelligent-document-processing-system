"""Embedding generation placeholder."""


class EmbeddingService:
    """Future local BGE-M3 embedding component."""

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError("Local embedding generation will be implemented in a later phase.")
