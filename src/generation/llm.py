import requests
from openai import OpenAI
from src.config import (
    LLM_PROVIDER, OLLAMA_BASE_URL, OLLAMA_MODEL,
    OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL,
    GEMINI_API_KEY, GEMINI_MODEL,
)

try:
    from google import genai
    from google.genai import types
except ImportError:  # pragma: no cover - exercised only when dependency is missing.
    class _GeminiTypes:
        class GenerateContentConfig:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

    class _GeminiModule:
        class Client:
            def __init__(self, *args, **kwargs):
                raise RuntimeError(
                    "google-genai is not installed. Install it with 'python -m pip install google-genai'."
                )

    genai = _GeminiModule()
    types = _GeminiTypes()


def generate_answer(prompt: str) -> str:
    if LLM_PROVIDER == "ollama":
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=180,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()

    if LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is missing. Set it in your .env file.")
        kwargs = {"api_key": OPENAI_API_KEY}
        if OPENAI_BASE_URL:
            kwargs["base_url"] = OPENAI_BASE_URL
        client = OpenAI(**kwargs)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return (response.choices[0].message.content or "").strip()

    if LLM_PROVIDER == "gemini":
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is missing. Set it in your .env file.")
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.1),
        )
        return (response.text or "").strip()

    raise ValueError("LLM_PROVIDER must be either 'ollama', 'openai', or 'gemini'.")
