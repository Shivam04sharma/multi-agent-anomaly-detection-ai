"""LLM client — OpenAI (GPT-4o).

Used for:
  - Step 5: Intent parsing (prompt_builder)
  - Step 10: Narrative generation (explanation_engine)
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod

import structlog
from config import settings

logger = structlog.get_logger()


def _strip_fences(text: str) -> str:
    return text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()


class BaseLLMClient(ABC):
    @abstractmethod
    async def complete(self, system: str, user: str, temperature: float, max_tokens: int) -> str:
        """Return raw text from the LLM."""

    async def complete_json(
        self, system: str, user: str, temperature: float, max_tokens: int
    ) -> dict:
        raw = await self.complete(system, user, temperature, max_tokens)
        try:
            return json.loads(_strip_fences(raw))
        except json.JSONDecodeError as exc:
            logger.warning("llm_json_parse_failed", error=str(exc), raw=raw[:300])
            raise ValueError(f"LLM returned non-JSON response: {raw[:200]}") from exc

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier used in API responses."""


class OpenAIClient(BaseLLMClient):
    @property
    def model_name(self) -> str:
        return f"openai/{settings.openai_model}"

    async def complete(self, system: str, user: str, temperature: float, max_tokens: int) -> str:
        import openai

        client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        resp = await client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = resp.choices[0].message.content or ""
        logger.info("llm_complete", model=settings.openai_model, tokens=resp.usage.total_tokens)  # type: ignore[union-attr]
        return text


# Singleton factory
_client: OpenAIClient | None = None


def get_llm_client() -> OpenAIClient:
    global _client
    if _client is None:
        _client = OpenAIClient()
    return _client
