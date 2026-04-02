import logging
import numpy as np
from services import data_service
from openai import OpenAI
import os

logger = logging.getLogger(__name__)


class RAGService:

    def __init__(self, embedding_service):
        self.embedding_service = embedding_service
        api_key=os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")

        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            
        )
        logger.info("RAG service initialised with OpenRouter")

    def retrieve(self, query: str, user_id: int, top_k: int = 3) -> list:
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
            top_k = min(top_k, 3)
            top_indices = np.argsort(scores)[::-1][:top_k]

            return [entries[i] for i in top_indices]

        except Exception:
            logger.exception("Retrieval failed for user %s", user_id)
            return []

    def generate(self, query: str, retrieved_entries: list, analytics: dict, forecast: dict, history: list = None) -> str:
        """Build context from retrieved entries + analytics + forecast, then generate response via LLM."""

        if not retrieved_entries:
            entries_text = "No past journal entries available."
        else:
            entries_text = "\n\n".join([
                f"[{e.get('date', 'N/A')}] Mood: {e.get('mood_score', 'N/A')}/10\nSummary: {e.get('journal_entry', '')[:200]}"
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

        # basic sanitization against instruction injection
        for bad in ["ignore previous instructions", "reveal", "dump", "system prompt"]:
            context = context.replace(bad, "")
{entries_text}

Current analytics:
{analytics_section}

Mood forecast:
{forecast_section}"""

        for bad in ["ignore previous instructions", "reveal", "dump", "system prompt"]:
            context = context.replace(bad, "")
            context = context.replace(bad.upper(), "")

        system_prompt = (
            "You are a personal mood analytics assistant. "
            "You have access to a user's journal entries and mood data. "
            "Give specific, grounded insights based only on the data provided. "
            "Do not give therapy or clinical advice. "
            "Do not make things up. If the data doesn't support a conclusion, say so. "
            "Keep responses concise and helpful. "
            "If the user asks anything unrelated to their mood data, journal history, "
            "sleep, activity, or behavioral patterns, respond with: "
            "'I can only answer questions about your journal and mood data.' "
            "Do not answer general knowledge questions, coding questions, "
            "or anything outside this scope."
            "Never reveal raw journal entries."
            "Never repeat the provided context verbatim."
            "Never output full data dumps."
            "Only provide summarized insights."
        )

        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history for multi-turn context
        # Safe history handling
        if history:
            for msg in history:
                role = msg.get("role")
                content = msg.get("content", "")

                # Only allow safe roles
                if role not in ("user", "assistant"):
                    continue

                # Truncate content
                content = content[:500]
                # remove obvious injection patterns
                for bad in ["ignore previous instructions", "reveal", "dump"]:
                    content = content.replace(bad, "")  

                messages.append({
                    "role": role,
            "content": content
        })
        messages.append({
            "role": "system",
            "content": f"Use the following data to answer:\n\n{context}"
        })

        # Add user query separately
        messages.append({
            "role": "user",
            "content": query
        })

        try:
            response = self.client.chat.completions.create(
                model="openai/gpt-3.5-turbo",
                messages=messages,
                temperature=0.3,
                max_tokens=500,
            )
            return response.choices[0].message.content.strip()

        except Exception:
            logger.exception("LLM generation failed")
            return "I'm sorry, I couldn't generate a response right now. Please try again later."
