"""Local sentence-transformers embedding generation."""

from functools import lru_cache
from typing import Any


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def generate_embedding(text: str, model_name: str = DEFAULT_EMBEDDING_MODEL) -> list[float]:
    """Generate a normalized embedding for text with a local model."""
    if not text or not text.strip():
        raise ValueError("text is required to generate an embedding")

    model = _load_model(model_name)
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


class EmbeddingService:
    """Thin wrapper around a sentence-transformers model."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self.model_name = model_name

    def embed(self, text: str) -> list[float]:
        return generate_embedding(text, self.model_name)

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = _load_model(self.model_name)
        embeddings = model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()


@lru_cache(maxsize=2)
def _load_model(model_name: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for semantic embeddings. "
            "Install package: sentence-transformers"
        ) from exc

    return SentenceTransformer(model_name)


if __name__ == "__main__":
    vector = generate_embedding("Invoice for office supplies with total amount due.")
    print(len(vector))
