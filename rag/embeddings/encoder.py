from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import json
import os

import numpy as np


class EmbeddingEncoder:
    """Codifica texto con ``intfloat/multilingual-e5-small`` y guarda vectores."""

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        *,
        device: str | None = None,
        max_length: int | None = None,
    ) -> None:
        self.max_length = max_length or int(os.getenv("RAG_EMBEDDING_MAX_LENGTH", "256"))
        if self.max_length <= 0:
            raise ValueError("max_length debe ser positivo")
        self.model_name = model_name
        self.device = device
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._device: Any | None = None

    def _load(self) -> tuple[Any, Any, Any]:
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model, self._device
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "EmbeddingEncoder requiere torch y transformers; "
                "instálelos antes de codificar embeddings"
            ) from exc

        selected_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = torch.device(selected_device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name).to(self._device)
        self._model.eval()
        return self._tokenizer, self._model, self._device

    def encode(
        self,
        texts: Iterable[str],
        *,
        prefix: str = "passage: ",
        batch_size: int = 16,
    ) -> np.ndarray:
        if batch_size <= 0:
            raise ValueError("batch_size debe ser positivo")
        values = [prefix + str(text) for text in texts]
        if not values:
            return np.empty((0, 0), dtype=np.float32)

        tokenizer, model, device = self._load()
        import torch

        batches: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(values), batch_size):
                encoded = tokenizer(
                    values[start : start + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                output = model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).expand(output.size()).float()
                pooled = (output * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
                batches.append(normalized.cpu().numpy().astype(np.float32))
        return np.vstack(batches)

    def encode_passages(self, texts: Iterable[str], *, batch_size: int = 16) -> np.ndarray:
        return self.encode(texts, prefix="passage: ", batch_size=batch_size)

    def encode_queries(self, texts: Iterable[str], *, batch_size: int = 16) -> np.ndarray:
        return self.encode(texts, prefix="query: ", batch_size=batch_size)

    @staticmethod
    def save(path: str | Path, embeddings: np.ndarray) -> None:
        values = np.asarray(embeddings, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError("embeddings debe ser una matriz 2D")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, embeddings=values)

    @staticmethod
    def load(path: str | Path) -> np.ndarray:
        with np.load(path) as data:
            if "embeddings" not in data:
                raise ValueError("el archivo NPZ no contiene 'embeddings'")
            return np.asarray(data["embeddings"], dtype=np.float32)

    def build_hnsw(
        self,
        texts: Iterable[str],
        chunk_ids: Iterable[int],
        *,
        embeddings_path: str | Path = "data/rag/embeddings.npz",
        hnsw_path: str | Path = "data/rag/hnsw.npz",
        mapping_path: str | Path = "data/rag/chunk_mapping.json",
        batch_size: int = 16,
    ) -> np.ndarray:
        """Codifica pasajes y genera los tres artefactos del índice vectorial."""
        values = list(texts)
        ids = [int(chunk_id) for chunk_id in chunk_ids]
        if len(values) != len(ids):
            raise ValueError("texts y chunk_ids deben tener la misma longitud")
        if not values:
            raise ValueError("no se puede construir un índice vectorial vacío")

        embeddings = self.encode_passages(values, batch_size=batch_size)
        self.save(embeddings_path, embeddings)

        from indexes.hnsw import HNSW

        index = HNSW(M=16, ef_construction=200, metrica="coseno")
        index.agregar(embeddings, ids=np.asarray(ids, dtype=np.int64))
        Path(hnsw_path).parent.mkdir(parents=True, exist_ok=True)
        index.guardar(hnsw_path)

        mapping_file = Path(mapping_path)
        mapping_file.parent.mkdir(parents=True, exist_ok=True)
        mapping_file.write_text(
            json.dumps(
                {str(position): chunk_id for position, chunk_id in enumerate(ids)},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return embeddings
