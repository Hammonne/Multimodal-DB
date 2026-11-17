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
    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5:7b-instruct"
    temperature: float = 0.1
    timeout_seconds: float = 180.0

    @classmethod
    def from_env(cls) -> "OllamaConfig":
        temperature = float(os.getenv("RAG_LLM_TEMPERATURE", "0.1"))
        if not 0 <= temperature <= 2:
            raise ValueError("RAG_LLM_TEMPERATURE debe estar entre 0 y 2")
        return cls(
            base_url=os.getenv("OLLAMA_BASE_URL", cls.base_url).rstrip("/"),
            model=os.getenv("RAG_LLM_MODEL", cls.model),
            temperature=temperature,
            timeout_seconds=float(os.getenv("RAG_LLM_TIMEOUT", "180")),
        )


class OllamaClient:
    def __init__(self, config: OllamaConfig | None = None) -> None:
        self.config = config or OllamaConfig.from_env()

    def generate(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("el prompt no puede estar vacío")
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.config.temperature},
        }
        request = Request(
            f"{self.config.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Ollama rechazó la solicitud ({error.code}): {detail}"
            ) from error
        except URLError as error:
            raise RuntimeError(
                f"No se pudo conectar con Ollama en {self.config.base_url}: {error.reason}"
            ) from error
        except TimeoutError as error:
            raise RuntimeError("Ollama agotó el tiempo de espera") from error

        answer = body.get("response")
        if not isinstance(answer, str):
            raise RuntimeError("Ollama devolvió una respuesta sin el campo 'response'")
        return answer.strip()

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
