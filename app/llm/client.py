from __future__ import annotations

import json
import re
from typing import Optional

from app.config import settings


class LLMError(RuntimeError):
    pass


class LLMClient:
    """Thin wrapper around the Google Gemini Generate Content API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = (settings.gemini_api_key if api_key is None else api_key).strip()
        self.model = (model or settings.gemini_model).strip() or settings.gemini_model
        self._client = None

    def _get_client(self):
        if self._client is None:
            if not self.api_key:
                raise LLMError(
                    "GEMINI_API_KEY is not set. "
                    "Set it in your environment or .env file."
                )

            try:
                from google import genai

                self._client = genai.Client(api_key=self.api_key)
            except ImportError:
                raise LLMError(
                    "The google-genai SDK is not installed. Install project requirements."
                ) from None
            except Exception:
                raise LLMError(
                    "Could not initialize Gemini. Check GEMINI_API_KEY and try again."
                ) from None

        return self._client

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 1500,
        temperature: float = 0.2,
    ) -> str:
        """Return raw text from Gemini."""

        client = self._get_client()

        try:
            response = client.models.generate_content(
                model=self.model,
                contents=user,
                config={
                    "system_instruction": system,
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                },
            )
        except Exception:
            raise LLMError(
                "Gemini request failed. Check GEMINI_MODEL, GEMINI_API_KEY, "
                "and network connectivity."
            ) from None

        text = getattr(response, "text", None)

        if text:
            return text.strip()

        raise LLMError("Gemini returned no text output.")

    def complete_json(
        self,
        system: str,
        user: str,
        max_tokens: int = 1500,
        temperature: float = 0.2,
    ) -> dict:
        """Return a parsed JSON object from Gemini."""

        json_system = f"""
{system}

IMPORTANT:
Return ONLY valid JSON.
Do not use markdown.
Do not wrap the JSON in ``` or any other code fence.
""".strip()

        raw = self.complete(
            system=json_system,
            user=user,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        cleaned = _strip_code_fences(raw)

        try:
            return json.loads(cleaned)

        except json.JSONDecodeError:
            # Try to extract a JSON object from surrounding text
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)

            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass

            raise LLMError("Gemini returned a response that was not valid JSON.") from None


_default_client: Optional[LLMClient] = None


def _strip_code_fences(text: str) -> str:
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(
            r"^```[a-zA-Z]*\n?",
            "",
            text,
        )

        text = re.sub(
            r"```$",
            "",
            text.strip(),
        )

    return text.strip()


def get_llm_client() -> LLMClient:
    global _default_client

    if _default_client is None:
        _default_client = LLMClient()

    return _default_client