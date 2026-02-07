from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import litellm
import numpy as np
from sentence_transformers import SentenceTransformer

from trendx.utils.logging import get_logger
from trendx.utils.json_extract import extract_json
from trendx.utils.llm_debug import log_llm_event

logger = get_logger(__name__)


@dataclass
class TaxonomyConfig:
    llm_provider: str
    llm_model: str
    api_key_env: str | None
    base_url: str | None
    embedding_model: str


def _model_name(provider: str, model: str) -> str:
    if "/" in model:
        return model
    return f"{provider}/{model}" if provider else model


def _seed_from_week(week: str, n_items: int) -> int:
    digest = hashlib.sha256(f"{week}:{n_items}".encode()).hexdigest()
    return int(digest[:8], 16)


def _item_text(item: Dict[str, Any]) -> str:
    return (
        item.get("title")
        or item.get("cleaned_text")
        or item.get("raw_text")
        or item.get("url")
        or ""
    )


def _ensure_embeddings(
    items: List[Dict[str, Any]], model_name: str
) -> List[Dict[str, Any]]:
    missing = [i for i in items if not i.get("embedding")]
    if not missing:
        return items

    model = SentenceTransformer(model_name)
    texts = [_item_text(i) for i in missing]
    vectors = model.encode(texts, normalize_embeddings=True).tolist()

    for item, vec in zip(missing, vectors):
        item["embedding"] = vec

    return items


async def _llm_buckets(titles: Sequence[str], cfg: TaxonomyConfig) -> List[str]:
    system = (
        "You are a taxonomy generator. Return JSON only with a list of 6-8 short topic labels."
    )
    user = {
        "instruction": "Create 6-8 topic buckets covering these titles.",
        "titles": list(titles),
    }

    api_key = os.getenv(cfg.api_key_env) if cfg.api_key_env else None
    if not api_key and cfg.llm_provider != "ollama":
        logger.warning("LLM buckets skipped: missing API key")
        return ["General"]

    request_payload = {"system": system, "user": user}
    resp = await litellm.acompletion(
        model=_model_name(cfg.llm_provider, cfg.llm_model),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user)},
        ],
        temperature=0.2,
        api_key=api_key,
        api_base=cfg.base_url if cfg.base_url else None,
    )
    content = resp["choices"][0]["message"]["content"]
    log_llm_event(
        stage="taxonomy_buckets",
        model=_model_name(cfg.llm_provider, cfg.llm_model),
        request=request_payload,
        response={"content": content},
    )
    try:
        data = extract_json(content)
        if data is None:
            raise ValueError("no json")
        buckets = data.get("buckets") or data.get("topics") or data
        buckets = [b for b in buckets if isinstance(b, str)]
        return buckets[:8]
    except Exception:
        logger.warning("LLM bucket parse failed, fallback to Generic")
        return ["General"]


async def _llm_subtopics(bucket_label: str, titles: Sequence[str], cfg: TaxonomyConfig) -> List[str]:
    system = (
        "You are a taxonomy refiner. Return JSON only with 3-5 subtopic labels for the bucket."
    )
    user = {
        "bucket": bucket_label,
        "titles": list(titles),
    }

    api_key = os.getenv(cfg.api_key_env) if cfg.api_key_env else None
    if not api_key and cfg.llm_provider != "ollama":
        logger.warning("LLM subtopics skipped: missing API key")
        return [bucket_label]

    request_payload = {"system": system, "user": user}
    resp = await litellm.acompletion(
        model=_model_name(cfg.llm_provider, cfg.llm_model),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user)},
        ],
        temperature=0.2,
        api_key=api_key,
        api_base=cfg.base_url if cfg.base_url else None,
    )
    content = resp["choices"][0]["message"]["content"]
    log_llm_event(
        stage="taxonomy_subtopics",
        model=_model_name(cfg.llm_provider, cfg.llm_model),
        request=request_payload,
        response={"content": content},
    )
    try:
        data = extract_json(content)
        if data is None:
            raise ValueError("no json")
        subs = data.get("subtopics") or data.get("topics") or data
        subs = [s for s in subs if isinstance(s, str)]
        return subs[:5]
    except Exception:
        logger.warning("LLM subtopic parse failed, fallback to bucket label")
        return [bucket_label]


async def cluster_with_taxonomy(
    week: str, items: List[Dict[str, Any]], cfg: TaxonomyConfig
) -> List[Dict[str, Any]]:
    if not items:
        return []

    items = _ensure_embeddings(items, cfg.embedding_model)

    seed = _seed_from_week(week, len(items))
    rng = random.Random(seed)
    sample_size = min(50, len(items))
    sample = rng.sample(items, sample_size)
    sample_titles = [_item_text(i) for i in sample]

    buckets = await _llm_buckets(sample_titles, cfg)
    if not buckets:
        buckets = ["General"]

    model = SentenceTransformer(cfg.embedding_model)
    bucket_vecs = model.encode(buckets, normalize_embeddings=True)

    item_vecs = np.array([i["embedding"] for i in items])
    bucket_vecs = np.array(bucket_vecs)

    sims = item_vecs @ bucket_vecs.T
    bucket_idx = sims.argmax(axis=1)

    bucket_map: Dict[str, List[Dict[str, Any]]] = {b: [] for b in buckets}
    for item, idx in zip(items, bucket_idx):
        bucket_map[buckets[int(idx)]].append(item)

    clusters: List[Dict[str, Any]] = []
    misc_items: List[Dict[str, Any]] = []

    for bucket_label, bucket_items in bucket_map.items():
        if len(bucket_items) >= 10:
            titles = [_item_text(i) for i in bucket_items]
            subtopics = await _llm_subtopics(bucket_label, titles[:50], cfg)
            if subtopics and subtopics != [bucket_label]:
                sub_vecs = model.encode(subtopics, normalize_embeddings=True)
                sub_vecs = np.array(sub_vecs)
                bucket_item_vecs = np.array([i["embedding"] for i in bucket_items])
                sub_sims = bucket_item_vecs @ sub_vecs.T
                sub_idx = sub_sims.argmax(axis=1)

                sub_map: Dict[str, List[Dict[str, Any]]] = {s: [] for s in subtopics}
                for item, idx in zip(bucket_items, sub_idx):
                    sub_map[subtopics[int(idx)]].append(item)

                for label, sub_items in sub_map.items():
                    if len(sub_items) < 3:
                        misc_items.extend(sub_items)
                    else:
                        clusters.append(_cluster_payload(label, sub_items))
                continue

        if len(bucket_items) < 3:
            misc_items.extend(bucket_items)
        else:
            clusters.append(_cluster_payload(bucket_label, bucket_items))

    if misc_items:
        clusters.append(_cluster_payload("Miscellaneous", misc_items))

    return clusters


def _cluster_payload(label: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "label": label,
        "item_ids": [i.get("id") or i.get("url") for i in items],
        "items": items,
    }
