"""
FAISS-backed offers database.

This module replaces the previous Postgres/pgvector implementation.
Offer vectors are stored in a local FAISS index file and offer metadata is stored as JSON.

Both `mcp-server` and `topwr-api` containers must mount the same `FAISS_DIR` volume.
"""

from __future__ import annotations

import asyncio
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np
from langchain_openai import OpenAIEmbeddings

from src.mcp_server.tools.karierownik.settings import Settings


def _to_json_serializable(value: Any) -> Any:
    """Convert date/datetime objects to ISO strings for JSON persistence."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _to_json_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_json_serializable(v) for v in value]
    return value


def _parse_iso_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if not isinstance(value, str) or not value:
        return None
    try:
        # Scrapers usually produce ISO strings; handle a trailing 'Z' if present.
        v = value.replace("Z", "+00:00")
        return datetime.fromisoformat(v).date()
    except Exception:
        return None


class OffersDBAsync:
    """
    Async wrapper with a FAISS backend.

    FAISS itself is sync; embeddings are executed in a thread pool to avoid blocking the event loop.
    """

    settings = Settings()
    _embeddings: OpenAIEmbeddings | None = None

    _lock = asyncio.Lock()
    _loaded: bool = False

    _index: faiss.Index | None = None
    _links: List[str] = []  # index id -> offer link
    _meta: Dict[str, Dict[str, Any]] = {}  # offer link -> offer dict (JSON-safe)
    _dim: int | None = None

    @classmethod
    def _paths(cls) -> dict[str, Path]:
        base = Path(cls.settings.FAISS_DIR)
        return {
            "base": base,
            "index": base / cls.settings.FAISS_INDEX_FILENAME,
            "id_map": base / cls.settings.FAISS_IDMAP_FILENAME,
            "meta": base / cls.settings.FAISS_METADATA_FILENAME,
        }

    @classmethod
    def _normalize(cls, vec: np.ndarray) -> np.ndarray:
        vec = vec.astype("float32", copy=False)
        norm = np.linalg.norm(vec)
        if norm == 0.0:
            return vec
        return vec / norm

    @classmethod
    def _get_embeddings(cls) -> OpenAIEmbeddings:
        if cls._embeddings is not None:
            return cls._embeddings
        if not cls.settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required for offers embeddings (FAISS backend).")
        cls._embeddings = OpenAIEmbeddings(api_key=cls.settings.OPENAI_API_KEY.get_secret_value())
        return cls._embeddings

    @classmethod
    async def _embed_query(cls, text: str) -> np.ndarray:
        loop = asyncio.get_running_loop()
        vec = await loop.run_in_executor(None, cls._get_embeddings().embed_query, text)
        return cls._normalize(np.asarray(vec, dtype="float32"))

    @classmethod
    async def _embed_documents(cls, texts: List[str]) -> np.ndarray:
        loop = asyncio.get_running_loop()
        vecs = await loop.run_in_executor(None, cls._get_embeddings().embed_documents, texts)
        arr = np.asarray(vecs, dtype="float32")
        # Normalize row-wise for cosine similarity (IndexFlatIP expects normalized vectors).
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return arr / norms

    @classmethod
    async def _ensure_loaded(cls) -> None:
        if cls._loaded:
            return
        async with cls._lock:
            if cls._loaded:
                return

            paths = cls._paths()
            base = paths["base"]
            base.mkdir(parents=True, exist_ok=True)

            if paths["index"].exists() and paths["id_map"].exists() and paths["meta"].exists():
                cls._index = faiss.read_index(str(paths["index"]))
                with paths["id_map"].open("r", encoding="utf-8") as f:
                    cls._links = json.load(f)
                with paths["meta"].open("r", encoding="utf-8") as f:
                    cls._meta = json.load(f)
                cls._dim = int(cls._index.d)
            else:
                # Start empty; dimension will be discovered on first embed.
                cls._index = None
                cls._links = []
                cls._meta = {}
                cls._dim = None

            cls._loaded = True

    @classmethod
    def _maybe_init_index(cls, dim: int) -> None:
        if cls._index is not None:
            return
        cls._dim = dim
        # Cosine similarity via dot-product on normalized vectors.
        cls._index = faiss.IndexFlatIP(dim)

    @classmethod
    def _save_to_disk(cls) -> None:
        paths = cls._paths()
        base = paths["base"]
        base.mkdir(parents=True, exist_ok=True)

        if cls._index is not None:
            faiss.write_index(cls._index, str(paths["index"]))
        with paths["id_map"].open("w", encoding="utf-8") as f:
            json.dump(cls._links, f, ensure_ascii=False)
        with paths["meta"].open("w", encoding="utf-8") as f:
            json.dump(cls._meta, f, ensure_ascii=False)

    @classmethod
    def _prepare_offer(cls, offer: Dict[str, Any]) -> Dict[str, Any]:
        # Ensure required keys exist (keep whatever scrapers provide).
        # Make it JSON-safe, especially date/datetime.
        return _to_json_serializable(dict(offer))

    @classmethod
    async def create_vector_index(cls) -> None:
        """Ensure FAISS store directory exists and metadata/index are loadable."""
        await cls._ensure_loaded()
        # If we loaded an index, nothing else is needed.
        if cls._index is not None:
            return
        # If metadata exists but index is missing, rebuild the index.
        if cls._meta:
            await cls._rebuild_index_from_meta()

    @classmethod
    async def _rebuild_index_from_meta(cls) -> None:
        paths = cls._paths()
        base = paths["base"]
        base.mkdir(parents=True, exist_ok=True)

        if not cls._meta:
            cls._index = None
            cls._links = []
            cls._dim = None
            cls._save_to_disk()
            return

        # Embed all offers and rebuild.
        # Use _links order as source of truth if present; otherwise derive from meta.
        links = cls._links or list(cls._meta.keys())
        texts = [cls._meta[l].get("description", "") for l in links]
        vectors = await cls._embed_documents(texts)
        dim = int(vectors.shape[1])
        cls._dim = dim
        cls._index = faiss.IndexFlatIP(dim)
        cls._index.add(vectors)
        cls._links = links
        cls._save_to_disk()

    @classmethod
    async def get_current_offers_links(cls, source: str | None = None) -> List[str]:
        await cls._ensure_loaded()
        if source is None:
            return list(cls._links)
        return [l for l in cls._links if cls._meta.get(l, {}).get("source") == source]

    @classmethod
    async def add_offer(cls, offer: Dict[str, Any]) -> None:
        await cls._ensure_loaded()

        link = offer.get("link")
        if not link:
            return

        async with cls._lock:
            prepared = cls._prepare_offer(offer)
            exists = link in cls._meta

            # Replacement: remove existing vector by rebuilding without the link.
            if exists:
                await cls.remove_offers([link])

            # Re-load state after remove_offers rebuild.
            await cls._ensure_loaded()

            description = prepared.get("description", "")
            vec = await cls._embed_query(description)
            cls._maybe_init_index(int(vec.shape[0]))
            assert cls._index is not None

            # Append vector + metadata.
            cls._index.add(vec.reshape(1, -1).astype("float32"))
            cls._links.append(link)
            cls._meta[link] = prepared
            cls._save_to_disk()

    @classmethod
    async def add_offers(cls, offers: List[Dict[str, Any]]) -> None:
        """Add multiple offers; handles replacements."""
        await cls._ensure_loaded()
        if not offers:
            return

        # Process sequentially to keep the logic simple and robust.
        for offer in offers:
            await cls.add_offer(offer)

    @classmethod
    async def remove_offer(cls, offer_link: str) -> None:
        await cls.remove_offers([offer_link])

    @classmethod
    async def remove_offers(cls, offers_links: List[str]) -> None:
        await cls._ensure_loaded()
        if not offers_links:
            return

        remove_set = set(offers_links)

        async with cls._lock:
            if not cls._meta:
                return

            # Fast path: if index is missing, just drop metadata.
            if cls._index is None:
                cls._links = [l for l in cls._links if l not in remove_set]
                for l in list(cls._meta.keys()):
                    if l in remove_set:
                        cls._meta.pop(l, None)
                cls._save_to_disk()
                return

            keep_links = [l for l in cls._links if l not in remove_set]
            # Reconstruct vectors for IndexFlatIP (supports reconstruct).
            keep_ids = [i for i, l in enumerate(cls._links) if l in set(keep_links)]

            dim = int(cls._index.d)
            new_index = faiss.IndexFlatIP(dim)
            if keep_ids:
                vectors = np.vstack([cls._index.reconstruct(i) for i in keep_ids]).astype("float32")
                # Vectors in the old index should already be normalized, but normalize again to be safe.
                vectors = cls._normalize(vectors) if len(vectors.shape) == 1 else vectors
                new_index.add(vectors)

            cls._index = new_index
            cls._links = keep_links
            for link in list(cls._meta.keys()):
                if link in remove_set:
                    cls._meta.pop(link, None)

            cls._save_to_disk()

    @classmethod
    async def get_outdated_offers(cls) -> List[str]:
        """Return links with date_closing earlier than today."""
        await cls._ensure_loaded()
        today = date.today()
        outdated: List[str] = []
        for link in cls._links:
            meta = cls._meta.get(link, {})
            closing = _parse_iso_date(meta.get("date_closing"))
            if closing is not None and closing < today:
                outdated.append(link)
        return outdated

    @classmethod
    def _matches_filters(
        cls,
        offer: Dict[str, Any],
        include_filters: Dict[str, List] | None,
        exclude_filters: Dict[str, List] | None,
    ) -> bool:
        if include_filters:
            for key, values in include_filters.items():
                if values:
                    if offer.get(key) not in values:
                        return False
        if exclude_filters:
            for key, values in exclude_filters.items():
                if values:
                    if offer.get(key) in values:
                        return False
        return True

    @classmethod
    async def similarity_search_cosine(
        cls,
        query: str,
        k: int = 5,
        offset: int = 0,
        include_filters: Dict[str, List] | None = None,
        exclude_filters: Dict[str, List] | None = None,
    ) -> List[Dict[str, Any]]:
        await cls._ensure_loaded()

        if cls._index is None or not cls._links:
            return []

        # Search more than needed to satisfy filters + offset.
        top_n = max(k + offset + 50, 100)
        top_n = min(top_n, len(cls._links))

        query_vec = await cls._embed_query(query)
        query_vec = query_vec.reshape(1, -1).astype("float32")

        scores, ids = cls._index.search(query_vec, top_n)
        _ = scores  # scores are not currently returned to the caller

        results: List[Dict[str, Any]] = []
        matched = 0

        id_list = ids[0]
        for idx in id_list:
            if idx < 0:
                continue
            link = cls._links[int(idx)]
            offer = cls._meta.get(link, {})
            if not cls._matches_filters(offer, include_filters, exclude_filters):
                continue

            if matched < offset:
                matched += 1
                continue

            results.append(offer)
            matched += 1
            if len(results) >= k:
                break

        return results

