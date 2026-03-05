import logging
import numpy as np
from services import data_service
from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)


class RAGService:

    def __init__(self, embedding_service, hf_api_key=""):
        self.embedding_service = embedding_service

        # Use HF Inference API with a model strictly supported on the free Serverless Chat endpoint
        self.client = InferenceClient(
            model="meta-llama/Llama-3.2-3B-Instruct",
            token=hf_api_key if hf_api_key else None
        )
        logger.info("RAG service initialised with HF Inference API")

    def retrieve(self, query: str, user_id: int, top_k: int = 5) -> list:
        """Embed query and retrieve the most similar journal entries for the user."""
        try:
            query_vec = self.embedding_service.embed(query)

            entries = data_service.get_embeddings_for_user(user_id)

            if not entries:
                logger.info("No embeddings found for user %s", user_id)
                return []

            # Deserialise stored embeddings
            candidate_vecs = np.array([
                np.frombuffer(e["embedding"], dtype=np.float32)
                for e in entries
            ])

            # Score and rank
            scores = self.embedding_service.similarity(query_vec, candidate_vecs)
            top_indices = np.argsort(scores)[::-1][:top_k]

            return [entries[i] for i in top_indices]

        except Exception:
            logger.exception("Retrieval failed for user %s", user_id)
            return []

    def generate(self, query: str, retrieved_entries: list, analytics: dict, forecast: dict) -> str:
        """Build context from retrieved entries + analytics + forecast, then generate response via LLM."""

        if not retrieved_entries:
            entries_text = "No past journal entries available."
        else:
            entries_text = "\n\n".join([
                f"[{e.get('date', 'N/A')}] Mood: {e.get('mood_score', 'N/A')}/10\n{e.get('journal_entry', '')}"
                for e in retrieved_entries
            ])

        # Build analytics context
        analytics_section = "No analytics data available."
        if analytics:
            analytics_section = (
                f"- 7-day avg mood: {analytics.get('avg_mood_7d', 'N/A')}\n"
                f"- Trend: {analytics.get('trend_label', 'N/A')}\n"
                f"- Avg sleep: {analytics.get('avg_sleep', 'N/A')} hrs\n"
                f"- Mood volatility: {analytics.get('volatility_label', 'N/A')}"
            )

        # Build forecast context
        forecast_section = "No forecast data available."
        if forecast:
            forecast_section = "\n".join([
                f"- {k}-day forecast: {v:.2f}" if isinstance(v, float) else f"- {k}-day: {v}"
                for k, v in forecast.items()
            ])

        context = f"""User's relevant past journal entries:
{entries_text}

Current analytics:
{analytics_section}

Mood forecast:
{forecast_section}"""

        system_prompt = (
            "You are a personal mood analytics assistant. "
            "You have access to a user's journal entries and mood data. "
            "Give specific, grounded insights based only on the data provided. "
            "Do not give therapy or clinical advice. "
            "Do not make things up. If the data doesn't support a conclusion, say so. "
            "Keep responses concise and helpful."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]

        try:
            response = self.client.chat_completion(
                messages,
                max_tokens=500,
                temperature=0.7,
                seed=42
            )
            return response.choices[0].message.content.strip()

        except Exception:
            logger.exception("LLM generation failed")
            return "I'm sorry, I couldn't generate a response right now. Please try again later."
