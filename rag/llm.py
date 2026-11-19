from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str:
        """Genera una respuesta a partir del prompt recibido."""


@dataclass(frozen=True)
class OllamaConfig:
    provider: str = "ollama"
    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5:7b-instruct"
    temperature: float = 0.1
    timeout_seconds: float = 300.0
    max_tokens: int = 512
    api_key: str = ""

    @classmethod
    def from_env(cls) -> "OllamaConfig":
        temperature = float(os.getenv("RAG_LLM_TEMPERATURE", "0.1"))
        if not 0 <= temperature <= 2:
            raise ValueError("RAG_LLM_TEMPERATURE debe estar entre 0 y 2")
        max_tokens = int(os.getenv("RAG_LLM_MAX_TOKENS", "512"))
        if max_tokens <= 0:
            raise ValueError("RAG_LLM_MAX_TOKENS debe ser positivo")
        return cls(
            provider=os.getenv("RAG_LLM_PROVIDER", "ollama"),
            base_url=(
                os.getenv("RAG_LLM_BASE_URL")
                or os.getenv("OLLAMA_BASE_URL", cls.base_url)
            ).rstrip("/"),
            model=os.getenv("RAG_LLM_MODEL", cls.model),
            temperature=temperature,
            timeout_seconds=float(os.getenv("RAG_LLM_TIMEOUT", "300")),
            max_tokens=max_tokens,
            api_key=os.getenv("RAG_LLM_API_KEY", ""),
        )


class OllamaClient:
    def __init__(self, config: OllamaConfig | None = None) -> None:
        self.config = config or OllamaConfig.from_env()

    def generate(self, prompt: str) -> str:
        return _post_generation(
            f"{self.config.base_url}/api/generate",
            {
                "model": self.config.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                },
            },
            self.config,
            lambda body: body.get("response"),
        )

    def check_connection(self) -> None:
        request = Request(f"{self.config.base_url}/api/tags", method="GET")
        try:
            with urlopen(request, timeout=5) as response:
                if response.status != 200:
                    raise RuntimeError(
                        f"Ollama respondió con estado HTTP {response.status}"
                    )
        except HTTPError as error:
            raise RuntimeError(f"Ollama respondió con estado HTTP {error.code}") from error
        except URLError as error:
            raise RuntimeError(
                f"No se pudo conectar con Ollama en {self.config.base_url}: {error.reason}"
            ) from error
        except TimeoutError as error:
            raise RuntimeError("Ollama agotó el tiempo de espera") from error


class OpenAICompatibleClient:
    def __init__(self, config: OllamaConfig | None = None) -> None:
        self.config = config or OllamaConfig.from_env()

    def generate(self, prompt: str) -> str:
        if not self.config.api_key:
            raise RuntimeError("RAG_LLM_API_KEY es obligatorio para este proveedor")
        return _post_generation(
            f"{self.config.base_url}/chat/completions",
            {
                "model": self.config.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            },
            self.config,
            lambda body: (
                body.get("choices", [{}])[0]
                .get("message", {})
                .get("content")
            ),
            headers={"Authorization": f"Bearer {self.config.api_key}"},
        )

    def check_connection(self) -> None:
        if not self.config.api_key:
            raise RuntimeError("RAG_LLM_API_KEY es obligatorio para este proveedor")


def build_llm_client() -> LLMClient:
    config = OllamaConfig.from_env()
    if config.provider in {"openai", "openai_compatible", "foundry", "groq", "openrouter"}:
        return OpenAICompatibleClient(config)
    if config.provider == "ollama":
        return OllamaClient(config)
    raise ValueError(f"RAG_LLM_PROVIDER no soportado: {config.provider}")


def _post_generation(
    url: str,
    payload: dict[str, object],
    config: OllamaConfig,
    response_parser,
    *,
    headers: dict[str, str] | None = None,
) -> str:
    if not payload.get("prompt") and not payload.get("messages"):
        raise ValueError("el prompt no puede estar vacío")
    request_headers = {"Content-Type": "application/json", **(headers or {})}
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=request_headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=config.timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"El proveedor LLM rechazó la solicitud ({error.code}): {detail}"
        ) from error
    except URLError as error:
        raise RuntimeError(
            f"No se pudo conectar con el proveedor LLM en {config.base_url}: {error.reason}"
        ) from error
    except TimeoutError as error:
        raise RuntimeError(
            f"El proveedor LLM agotó el tiempo de espera ({config.timeout_seconds:.0f} s); "
            "prueba un modelo más pequeño o aumenta RAG_LLM_TIMEOUT"
        ) from error

    answer = response_parser(body)
    if not isinstance(answer, str):
        raise RuntimeError("El proveedor LLM devolvió una respuesta sin contenido")
    return answer.strip()


def build_rag_prompt(question: str, context: str) -> str:
    if not question.strip():
        raise ValueError("la pregunta no puede estar vacía")
    if not context.strip():
        raise ValueError("el contexto no puede estar vacío")
    return f"""Responde usando únicamente el contexto proporcionado.
Si el contexto no contiene evidencia suficiente, indícalo explícitamente.
No inventes autores, años ni referencias.
Responde en el idioma de la pregunta.
Cita las afirmaciones con el formato (Autor, Año).

Contexto:
{context}

Pregunta:
{question}
"""
