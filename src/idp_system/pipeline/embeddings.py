"""Local sentence-transformers embedding generation."""

import os
from functools import lru_cache
from typing import Any


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ALLOW_REMOTE_MODEL_DOWNLOAD_ENV = "IDP_ALLOW_REMOTE_MODEL_DOWNLOAD"


class EmbeddingModelUnavailableError(RuntimeError):
    """Raised when the local semantic-search model is not available."""


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
    allow_remote_download = os.getenv(ALLOW_REMOTE_MODEL_DOWNLOAD_ENV, "").lower() in {
        "1",
        "true",
        "yes",
    }
    if not allow_remote_download:
        # These flags must be set before importing sentence-transformers so
        # Transformers does not probe for optional files over the network.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for semantic embeddings. "
            "Install package: sentence-transformers"
        ) from exc

    try:
        # Search is a local feature. Do not block the Streamlit UI while the
        # Hugging Face client retries an unavailable network connection.
        return SentenceTransformer(model_name, local_files_only=not allow_remote_download)
    except Exception as exc:
        raise EmbeddingModelUnavailableError(
            f"The semantic-search model '{model_name}' is not cached locally. "
            "Download it once while online to enable semantic search."
        ) from exc


if __name__ == "__main__":
    vector = generate_embedding("Invoice for office supplies with total amount due.")
    print(len(vector))
