"""Persisted dense retrieval using multilingual E5 embeddings."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .corpus import CorpusDocument, sha256_file

DEFAULT_MODEL_ID = "intfloat/multilingual-e5-small"
DEFAULT_MODEL_REVISION = "fd1525a9fd15316a2d503bf26ab031a61d056e98"
INDEX_MANIFEST = "manifest.json"
INDEX_EMBEDDINGS = "embeddings.npy"


class TextEncoder(Protocol):
    model_id: str
    model_revision: str
    dimension: int
    max_length: int

    def encode(
        self,
        texts: Sequence[str],
        *,
        input_type: str,
        batch_size: int = 8,
    ) -> np.ndarray: ...


class E5Encoder:
    """Transformers implementation of the multilingual-E5 model card recipe."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        revision: str,
        device: str = "cpu",
        max_length: int = 512,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "vector retrieval requires the 'retrieval' optional dependencies"
            ) from exc
        self._torch = torch
        self.model_id = model_id
        self.max_length = max_length
        self.device = device
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
        )
        self._model = AutoModel.from_pretrained(model_id, revision=revision)
        self._model.to(device)
        self._model.eval()
        self.model_revision = (
            getattr(self._model.config, "_commit_hash", None) or revision
        )
        self.dimension = int(self._model.config.hidden_size)

    def encode(
        self,
        texts: Sequence[str],
        *,
        input_type: str,
        batch_size: int = 8,
    ) -> np.ndarray:
        if input_type not in {"query", "passage"}:
            raise ValueError("input_type must be query or passage")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        prefixed = [f"{input_type}: {text}" for text in texts]
        batches: list[np.ndarray] = []
        for start in range(0, len(prefixed), batch_size):
            encoded = self._tokenizer(
                prefixed[start : start + batch_size],
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self._torch.no_grad():
                hidden = self._model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).bool()
            hidden = hidden.masked_fill(~mask, 0.0)
            pooled = hidden.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            normalized = self._torch.nn.functional.normalize(pooled, p=2, dim=1)
            batches.append(normalized.cpu().numpy().astype(np.float32))
        if not batches:
            return np.empty((0, self.dimension), dtype=np.float32)
        return np.concatenate(batches, axis=0)


@dataclass(frozen=True)
class VectorSearchResult:
    rank: int
    pmid: str
    score: float
    document_sha256: str


class VectorIndex:
    """Cosine-similarity index with byte-verifiable persisted embeddings."""

    def __init__(
        self,
        *,
        documents: Sequence[CorpusDocument],
        embeddings: np.ndarray,
        model_id: str,
        model_revision: str,
        max_length: int,
    ) -> None:
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(documents):
            raise ValueError("embedding rows must match corpus documents")
        if not len(documents):
            raise ValueError("vector index requires at least one document")
        norms = np.linalg.norm(matrix, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-4):
            raise ValueError("document embeddings must be L2-normalized")
        self.documents = tuple(documents)
        self.embeddings = matrix
        self.model_id = model_id
        self.model_revision = model_revision
        self.max_length = max_length

    @classmethod
    def build(
        cls,
        documents: Sequence[CorpusDocument],
        *,
        encoder: TextEncoder,
        batch_size: int = 8,
    ) -> VectorIndex:
        passages = [
            f"Title: {document.title}\nAbstract: {document.abstract}"
            for document in documents
        ]
        embeddings = encoder.encode(
            passages,
            input_type="passage",
            batch_size=batch_size,
        )
        return cls(
            documents=documents,
            embeddings=embeddings,
            model_id=encoder.model_id,
            model_revision=encoder.model_revision,
            max_length=encoder.max_length,
        )

    def search(
        self,
        query: str,
        *,
        encoder: TextEncoder,
        top_k: int = 10,
    ) -> list[VectorSearchResult]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if (
            encoder.model_id != self.model_id
            or encoder.model_revision != self.model_revision
        ):
            raise ValueError("query encoder does not match persisted vector index")
        query_vector = encoder.encode([query], input_type="query")
        if query_vector.shape != (1, self.embeddings.shape[1]):
            raise ValueError("query embedding dimension does not match index")
        scores = self.embeddings @ query_vector[0]
        order = sorted(
            range(len(self.documents)),
            key=lambda index: (
                -float(scores[index]),
                int(self.documents[index].pmid),
                self.documents[index].pmid,
            ),
        )
        return [
            VectorSearchResult(
                rank=rank,
                pmid=self.documents[index].pmid,
                score=round(float(scores[index]), 8),
                document_sha256=self.documents[index].content_sha256,
            )
            for rank, index in enumerate(order[:top_k], start=1)
        ]

    def save(self, index_dir: Path, *, corpus_path: Path) -> dict[str, object]:
        if index_dir.exists():
            raise ValueError(f"vector index directory already exists: {index_dir}")
        index_dir.mkdir(parents=True)
        embeddings_path = index_dir / INDEX_EMBEDDINGS
        np.save(embeddings_path, self.embeddings, allow_pickle=False)
        manifest = {
            "manifest_version": "1.0.0",
            "index_id": "bioevidence-vector-e5-v1",
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "max_length": self.max_length,
            "embedding_dimension": self.embeddings.shape[1],
            "normalized": True,
            "query_prefix": "query: ",
            "passage_prefix": "passage: ",
            "pooling": "attention-mask mean pooling",
            "similarity": "cosine via normalized dot product",
            "corpus_sha256": sha256_file(corpus_path),
            "embeddings_sha256": sha256_file(embeddings_path),
            "document_order": [document.pmid for document in self.documents],
            "document_sha256": {
                document.pmid: document.content_sha256
                for document in self.documents
            },
        }
        (index_dir / INDEX_MANIFEST).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return manifest

    @classmethod
    def load(
        cls,
        index_dir: Path,
        *,
        documents: Sequence[CorpusDocument],
        corpus_path: Path,
    ) -> VectorIndex:
        manifest = read_index_manifest(index_dir)
        embeddings_path = index_dir / INDEX_EMBEDDINGS
        if sha256_file(corpus_path) != manifest.get("corpus_sha256"):
            raise ValueError("vector index corpus hash mismatch")
        if sha256_file(embeddings_path) != manifest.get("embeddings_sha256"):
            raise ValueError("vector embeddings hash mismatch")
        expected_order = [document.pmid for document in documents]
        if expected_order != manifest.get("document_order"):
            raise ValueError("vector index document order mismatch")
        expected_hashes = {
            document.pmid: document.content_sha256 for document in documents
        }
        if expected_hashes != manifest.get("document_sha256"):
            raise ValueError("vector index document content hash mismatch")
        embeddings = np.load(embeddings_path, allow_pickle=False)
        if embeddings.shape[1] != manifest.get("embedding_dimension"):
            raise ValueError("vector index embedding dimension mismatch")
        return cls(
            documents=documents,
            embeddings=embeddings,
            model_id=_manifest_text(manifest, "model_id"),
            model_revision=_manifest_text(manifest, "model_revision"),
            max_length=int(manifest.get("max_length", 0)),
        )


def read_index_manifest(index_dir: Path) -> dict[str, object]:
    value = json.loads((index_dir / INDEX_MANIFEST).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("vector index manifest must be an object")
    return value


def _manifest_text(manifest: dict[str, object], key: str) -> str:
    value = manifest.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"vector index manifest missing {key}")
    return value
