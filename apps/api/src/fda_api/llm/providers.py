"""Provider registry and the three wire-format adapters.

Each provider is configured from the environment:
- `<NAME>_API_KEY` holds the key (none for local servers);
- `FDA_LLM_MODEL_<NAME>` overrides the default model;
- `FDA_LLM_BASE_URL_<NAME>` overrides the endpoint (self-hosted gateways, proxies).

`FDA_LLM_PROVIDER` picks the default provider. When it is unset, the first configured one in
registry order is used.
"""

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Literal

import httpx2 as httpx

from fda_api.llm.base import (
    LLMError,
    Reply,
    Session,
    ToolCall,
    ToolSpec,
    inline_refs,
    parse_arguments,
    post_json,
)

Wire = Literal["anthropic", "gemini", "openai"]
ANTHROPIC_VERSION = "2026-10-21"
MAX_TOKENS = 16000


@dataclass(frozen=True, slots=True)
class Provider:
    name: str
    label: str
    wire: Wire
    base_url: str
    key_env: str | None
    default_model: str | None
    local: bool = False

    @property
    def key(self) -> str | None:
        return os.getenv(self.key_env) if self.key_env else None

    @property
    def url(self) -> str:
        return os.getenv(f"FDA_LLM_BASE_URL_{self.name.upper()}", self.base_url).rstrip("/")

    def model(self, requested: str | None = None) -> str | None:
        return (
            requested
            or os.getenv(f"FDA_LLM_MODEL_{self.name.upper()}")
            or (os.getenv("FDA_LLM_MODEL") if os.getenv("FDA_LLM_PROVIDER") == self.name else None)
            or self.default_model
        )

    @property
    def configured(self) -> bool:
        return self.local or bool(self.key)

    def session(
        self, model: str, system: str, tools: list[ToolSpec], history: list[tuple[str, str]]
    ) -> Session:
        if not self.configured:
            raise LLMError(f"{self.label}: set {self.key_env} in .env")
        if self.wire == "anthropic":
            return AnthropicSession(self, model, system, tools, history)
        if self.wire == "gemini":
            return GeminiSession(self, model, system, tools, history)
        return OpenAISession(self, model, system, tools, history)


PROVIDERS: dict[str, Provider] = {
    p.name: p
    for p in (
        Provider(
            "anthropic",
            "Claude (Anthropic)",
            "anthropic",
            "https://api.anthropic.com/v1",
            "ANTHROPIC_API_KEY",
            "claude-opus-5-5",
        ),
        Provider(
            "gemini",
            "Gemini (Google)",
            "gemini",
            "https://generativelanguage.googleapis.com/v1beta",
            "GEMINI_API_KEY",
            "gemini-3.5-flash",
        ),
        Provider("openai", "OpenAI", "openai", "https://api.openai.com/v1", "OPENAI_API_KEY", None),
        Provider(
            "mistral",
            "Mistral",
            "openai",
            "https://api.mistral.ai/v1",
            "MISTRAL_API_KEY",
            "mistral-large-latest",
        ),
        Provider(
            "deepseek",
            "DeepSeek",
            "openai",
            "https://api.deepseek.com/v1",
            "DEEPSEEK_API_KEY",
            "deepseek-chat",
        ),
        Provider("groq", "Groq", "openai", "https://api.groq.com/openai/v1", "GROQ_API_KEY", None),
        Provider("xai", "xAI (Grok)", "openai", "https://api.x.ai/v1", "XAI_API_KEY", None),
        Provider(
            "openrouter",
            "OpenRouter (any model)",
            "openai",
            "https://openrouter.ai/api/v1",
            "OPENROUTER_API_KEY",
            None,
        ),
        Provider(
            "ollama",
            "Ollama (local)",
            "openai",
            "http://127.0.0.1:11434/v1",
            None,
            None,
            local=True,
        ),
        Provider(
            "lmstudio",
            "LM Studio (local)",
            "openai",
            "http://127.0.0.1:1234/v1",
            None,
            None,
            local=True,
        ),
        Provider(
            "custom",
            "Custom OpenAI-compatible",
            "openai",
            os.getenv("FDA_LLM_BASE_URL_CUSTOM", "http://127.0.0.1:8000/v1"),
            "CUSTOM_API_KEY",
            None,
        ),
    )
}


_MODEL_CACHE: dict[str, tuple[float, list[str]]] = {}
MODEL_CACHE_SECONDS = 600


def local_models(provider: Provider) -> list[str]:
    """Models a local server has loaded (OpenAI-compatible `/models`); [] when unreachable."""
    try:
        response = httpx.get(f"{provider.url}/models", timeout=2.0)
        data = response.json().get("data", []) if response.status_code == 200 else []
    except (httpx.HTTPError, ValueError):
        return []
    return [str(m["id"]) for m in data if isinstance(m, dict) and m.get("id")]


def remote_models(provider: Provider) -> list[str]:
    """Models the key can use (each API's model list), cached 10 min; [] on any failure."""
    if not provider.key:
        return []
    cached = _MODEL_CACHE.get(provider.name)
    if cached and time.monotonic() - cached[0] < MODEL_CACHE_SECONDS:
        return cached[1]
    headers: dict[str, str] = {}
    url = f"{provider.url}/models"
    if provider.wire == "anthropic":
        headers = {"x-api-key": provider.key, "anthropic-version": ANTHROPIC_VERSION}
    elif provider.wire == "gemini":
        headers, url = {"x-goog-api-key": provider.key}, f"{url}?pageSize=200"
    else:
        headers = {"authorization": f"Bearer {provider.key}"}
    try:
        response = httpx.get(url, headers=headers, timeout=5.0)
        body = response.json() if response.status_code == 200 else {}
    except (httpx.HTTPError, ValueError):
        body = {}
    rows = body.get("models") or body.get("data") or []
    names = sorted(
        str(m.get("id") or m.get("name", "")).split("/")[-1]
        for m in rows
        if isinstance(m, dict) and (m.get("id") or m.get("name"))
    )
    if provider.wire == "gemini":
        names = [n for n in names if n.startswith("gemini") and "embedding" not in n]
    _MODEL_CACHE[provider.name] = (time.monotonic(), names)
    return names


def model_list(provider: Provider) -> list[str]:
    """The models a provider says it offers (empty when the list cannot be fetched)."""
    return local_models(provider) if provider.local else remote_models(provider)


def get_provider(name: str | None = None) -> Provider:
    chosen = name or os.getenv("FDA_LLM_PROVIDER")
    if chosen:
        if chosen not in PROVIDERS:
            raise LLMError(f"Unknown provider {chosen}; one of {', '.join(PROVIDERS)}")
        return PROVIDERS[chosen]
    for provider in PROVIDERS.values():
        if provider.configured and not provider.local:
            return provider
    for provider in PROVIDERS.values():
        if provider.local and local_models(provider):
            return provider
    raise LLMError("No LLM configured: add an API key to .env or start Ollama / LM Studio")


def provider_status() -> list[dict[str, Any]]:
    out = []
    for p in PROVIDERS.values():
        models = local_models(p) if p.local else remote_models(p)
        out.append(
            {
                "name": p.name,
                "label": p.label,
                "wire": p.wire,
                "configured": bool(models) if p.local else p.configured,
                "local": p.local,
                "key_env": p.key_env,
                "model": p.model() or (models[0] if models else None),
                "models": models,
            }
        )
    return out


# ── Anthropic Messages API ───────────────────────────────────────────────────


class AnthropicSession:
    def __init__(
        self,
        provider: Provider,
        model: str,
        system: str,
        tools: list[ToolSpec],
        history: list[tuple[str, str]],
    ) -> None:
        self.provider, self.model, self.system = provider, model, system
        self.tools = [
            {"name": t.name, "description": t.description, "input_schema": t.schema} for t in tools
        ]
        self.messages: list[dict[str, Any]] = [
            {"role": "assistant" if role == "assistant" else "user", "content": text}
            for role, text in history
        ]

    def send(
        self, user_text: str | None = None, results: list[tuple[ToolCall, str]] | None = None
    ) -> Reply:
        if results:
            self.messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": call.id, "content": text}
                        for call, text in results
                    ],
                }
            )
        if user_text is not None:
            self.messages.append({"role": "user", "content": user_text})
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "system": [
                {"type": "text", "text": self.system, "cache_control": {"type": "ephemeral"}}
            ],
            "messages": self.messages,
        }
        if self.tools:
            body["tools"] = self.tools
        headers = {
            "x-api-key": self.provider.key or "",
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        data = post_json(f"{self.provider.url}/messages", headers, body)
        content = data.get("content") or []
        # the assistant turn goes back verbatim (thinking blocks included)
        self.messages.append({"role": "assistant", "content": content})
        text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
        calls = [
            ToolCall(str(b.get("id")), str(b.get("name")), parse_arguments(b.get("input")))
            for b in content
            if b.get("type") == "tool_use"
        ]
        usage = data.get("usage") or {}
        return Reply(
            text=text,
            tool_calls=calls,
            usage={
                "input": int(usage.get("input_tokens") or 0),
                "output": int(usage.get("output_tokens") or 0),
                "cache_read": int(usage.get("cache_read_input_tokens") or 0),
            },
            stop=str(data.get("stop_reason") or ""),
        )


# ── Gemini Interactions API ──────────────────────────────────────────────────


class GeminiSession:
    def __init__(
        self,
        provider: Provider,
        model: str,
        system: str,
        tools: list[ToolSpec],
        history: list[tuple[str, str]],
    ) -> None:
        self.provider, self.model, self.system = provider, model, system
        self.tools = [
            {
                "type": "function",
                "name": t.name,
                "description": t.description,
                "parameters": inline_refs(t.schema),
            }
            for t in tools
        ]
        self.steps: list[dict[str, Any]] = []
        for role, text in history:
            if role == "assistant":
                self.steps.append(
                    {"type": "model_output", "content": [{"type": "text", "text": text}]}
                )
            else:
                self.steps.append(self._user(text))

    @staticmethod
    def _user(text: str) -> dict[str, Any]:
        return {"type": "user_input", "content": [{"type": "text", "text": text}]}

    def send(
        self, user_text: str | None = None, results: list[tuple[ToolCall, str]] | None = None
    ) -> Reply:
        for call, text in results or []:
            self.steps.append(
                {
                    "type": "function_result",
                    "name": call.name,
                    "call_id": call.id,
                    "result": [{"type": "text", "text": text}],
                }
            )
        if user_text is not None:
            self.steps.append(self._user(user_text))
        body: dict[str, Any] = {
            "model": self.model,
            "store": False,
            "input": self.steps,
            "system_instruction": self.system,
        }
        if self.tools:
            body["tools"] = self.tools
        headers = {"x-goog-api-key": self.provider.key or "", "content-type": "application/json"}
        data = post_json(f"{self.provider.url}/interactions", headers, body)
        steps = [s for s in data.get("steps") or data.get("outputs") or [] if isinstance(s, dict)]
        # model steps (thoughts with their signatures, calls, outputs) go back verbatim
        self.steps.extend(s for s in steps if s.get("type") != "user_input")
        calls = [
            ToolCall(
                str(s.get("id") or s.get("call_id") or s.get("name")),
                str(s.get("name")),
                parse_arguments(s.get("arguments", s.get("args"))),
            )
            for s in steps
            if s.get("type") == "function_call"
        ]
        text = data.get("output_text")
        if not isinstance(text, str):
            parts = []
            for s in steps:
                if s.get("type") != "model_output":
                    continue
                content = s.get("content")
                if isinstance(content, str):
                    parts.append(content)
                for item in content if isinstance(content, list) else []:
                    if isinstance(item, dict) and item.get("type", "text") == "text":
                        parts.append(str(item.get("text", "")))
            text = "".join(parts)
        usage = data.get("usage") or {}
        return Reply(
            text=text,
            tool_calls=calls,
            usage={
                "input": int(usage.get("total_input_tokens") or usage.get("input_tokens") or 0),
                "output": int(usage.get("total_output_tokens") or usage.get("output_tokens") or 0),
            },
            stop=str(data.get("status") or ""),
        )


# ── OpenAI Chat Completions (and every compatible server) ────────────────────


class OpenAISession:
    def __init__(
        self,
        provider: Provider,
        model: str,
        system: str,
        tools: list[ToolSpec],
        history: list[tuple[str, str]],
    ) -> None:
        self.provider, self.model = provider, model
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": inline_refs(t.schema),
                },
            }
            for t in tools
        ]
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        self.messages += [
            {"role": "assistant" if role == "assistant" else "user", "content": text}
            for role, text in history
        ]

    def send(
        self, user_text: str | None = None, results: list[tuple[ToolCall, str]] | None = None
    ) -> Reply:
        for call, text in results or []:
            self.messages.append({"role": "tool", "tool_call_id": call.id, "content": text})
        if user_text is not None:
            self.messages.append({"role": "user", "content": user_text})
        body: dict[str, Any] = {"model": self.model, "messages": self.messages}
        if self.tools:
            body["tools"] = self.tools
        headers = {"content-type": "application/json"}
        if self.provider.key:
            headers["authorization"] = f"Bearer {self.provider.key}"
        timeout = 900.0 if self.provider.local else 180.0  # local: first call loads the model
        data = post_json(f"{self.provider.url}/chat/completions", headers, body, timeout)
        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"{self.provider.label}: empty answer {json.dumps(data)[:200]}")
        message = dict(choices[0].get("message") or {})
        message["role"] = "assistant"
        message.setdefault("content", "")
        self.messages.append(message)
        calls = [
            ToolCall(
                str(c.get("id") or f"call_{i}"),
                str((c.get("function") or {}).get("name")),
                parse_arguments((c.get("function") or {}).get("arguments")),
            )
            for i, c in enumerate(message.get("tool_calls") or [])
        ]
        usage = data.get("usage") or {}
        return Reply(
            text=str(message.get("content") or ""),
            tool_calls=calls,
            usage={
                "input": int(usage.get("prompt_tokens") or 0),
                "output": int(usage.get("completion_tokens") or 0),
            },
            stop=str(choices[0].get("finish_reason") or ""),
        )
