import numpy as np
from functools import lru_cache
from sentence_transformers import SentenceTransformer

# MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
MODEL_NAME = "pritamdeka/SapBERT-mnli-snli-scinli-scitail-mednli-stsb"

@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def embed(texts: list[str]) -> np.ndarray:
    """Embed a list of strings, returning a normalized matrix of shape (n, d)."""
    model = get_model()
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

def pairwise_similarity(texts_a: list[str], texts_b: list[str]) -> np.ndarray:
    """Compute similarity between corresponding pairs, returns array of shape (n,)."""
    emb_a = embed(texts_a)
    emb_b = embed(texts_b)
    return np.sum(emb_a * emb_b, axis=1)
