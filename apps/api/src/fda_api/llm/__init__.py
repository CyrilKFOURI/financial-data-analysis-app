"""Provider-neutral LLM access for the analyst agent (M30 §3, J25 slice 251).

One small interface, three wire formats spoken over plain HTTPS (no vendor SDK):
- `anthropic`: Claude Messages API;
- `gemini`: Gemini Interactions API;
- `openai`: OpenAI Chat Completions, also spoken by Mistral, Groq, DeepSeek, OpenRouter, xAI,
  Ollama and LM Studio, and any compatible server.

Keys are read from the environment (.env) on the server and never leave it.
"""

from fda_api.llm.base import LLMError, Reply, Session, ToolCall, ToolSpec
from fda_api.llm.providers import PROVIDERS, Provider, get_provider, provider_status

__all__ = [
    "PROVIDERS",
    "LLMError",
    "Provider",
    "Reply",
    "Session",
    "ToolCall",
    "ToolSpec",
    "get_provider",
    "provider_status",
]
