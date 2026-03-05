import logging
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingService:

    def __init__(self):
        self._model = None  # Lazy-loaded on first use

    @property
    def model(self):
        if self._model is None:
            logger.info("Loading sentence-transformer model: %s (first use)", MODEL_NAME)
            self._model = SentenceTransformer(MODEL_NAME)
            logger.info("Sentence-transformer model loaded successfully")
        return self._model

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string. Returns normalised vector."""
        return self.model.encode(text, normalize_embeddings=True)

    def embed_batch(self, texts: list) -> np.ndarray:
        """Embed a batch of texts. Returns normalised vectors."""
        return self.model.encode(texts, normalize_embeddings=True)

    def similarity(self, query_vec: np.ndarray, candidate_vecs: np.ndarray) -> np.ndarray:
        """Cosine similarity — vectors are already normalised so this is a dot product."""
        return np.dot(candidate_vecs, query_vec)
