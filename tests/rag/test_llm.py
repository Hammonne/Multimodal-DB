from __future__ import annotations

from rag.llm import OllamaConfig, build_rag_prompt


def test_build_rag_prompt_requires_context_and_includes_guardrails():
    prompt = build_rag_prompt("¿Qué es HNSW?", "HNSW es un grafo navegable.")

    assert "Responde usando únicamente el contexto proporcionado." in prompt
    assert "¿Qué es HNSW?" in prompt
    assert "HNSW es un grafo navegable." in prompt


def test_ollama_config_reads_model_and_url(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama:11434/")
    monkeypatch.setenv("RAG_LLM_MODEL", "qwen2.5:7b-instruct")

    config = OllamaConfig.from_env()

    assert config.base_url == "http://ollama:11434"
    assert config.model == "qwen2.5:7b-instruct"
    assert config.max_tokens == 512
