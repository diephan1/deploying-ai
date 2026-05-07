"""Service 2: hybrid recipe search over a persistent ChromaDB collection.

Combines an optional **lexical pre-filter** (Chroma ``where`` on ``area`` and
``category`` metadata) with **semantic ranking** by sentence-transformer
embeddings. Structured fields are joined back from ``recipes.csv`` (pandas)
per the assignment's "no SQLite" rule.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

import chromadb
import pandas as pd
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain.tools import tool
from pydantic import BaseModel, Field

from utils.logger import get_logger

_logs = get_logger(__name__)

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "recipes.csv"
CHROMA_PATH = HERE / "chroma_store"
COLLECTION_NAME = "recipes"
EMBED_MODEL = "all-MiniLM-L6-v2"

KNOWN_AREAS = {
    "algerian", "argentina", "argentinian", "australian", "british", "canadian",
    "chinese", "croatian", "egyptian", "filipino", "france", "french", "greek",
    "india", "indian", "irish", "italian", "jamaican", "japanese", "kenyan",
    "malaysian", "mexican", "moroccan", "netherlands", "dutch", "norway",
    "norwegian", "polish", "portuguese", "russian", "saudi arabian", "slovakia",
    "slovak", "spanish", "syrian", "thai", "tunisian", "turkish", "ukrainian",
    "united states", "american", "uruguayan", "venezuela", "venezuelan",
    "vietnamese",
}

AREA_ALIASES = {
    "argentinian": "Argentina",
    "french": "France",
    "indian": "India",
    "dutch": "Netherlands",
    "norwegian": "Norway",
    "slovak": "Slovakia",
    "american": "United States",
    "venezuelan": "Venezuela",
}

KNOWN_CATEGORIES = {
    "beef", "breakfast", "chicken", "dessert", "goat", "lamb", "miscellaneous",
    "pasta", "pork", "seafood", "side", "starter", "vegan", "vegetarian",
}


class RecipeHit(BaseModel):
    title: str
    area: str = ""
    category: str = ""
    ingredients: str = ""
    snippet: str = Field("", description="First lines of the cooking instructions.")


@lru_cache(maxsize=1)
def _get_collection():
    """Open the persistent ChromaDB collection. Cached so we open it once."""
    if not CHROMA_PATH.exists():
        raise FileNotFoundError(
            f"chroma_store not found at {CHROMA_PATH}. "
            "Run `uv run python -m assignment_chat.build_index` first."
        )
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)


@lru_cache(maxsize=1)
def _get_recipes_df() -> pd.DataFrame:
    return pd.read_csv(CSV_PATH, dtype={"id": str}).fillna("")


def _autodetect(query: str) -> tuple[Optional[str], Optional[str]]:
    """Sniff out cuisine/category words in the query for a lexical pre-filter."""
    q = query.lower()
    detected_area: Optional[str] = None
    for word in KNOWN_AREAS:
        if word in q:
            detected_area = AREA_ALIASES.get(word, word.title())
            break

    detected_cat: Optional[str] = None
    for word in KNOWN_CATEGORIES:
        if word in q:
            detected_cat = word.capitalize()
            break

    return detected_area, detected_cat


def _normalize_area(area: Optional[str]) -> Optional[str]:
    """Normalize an LLM-passed area string against AREA_ALIASES."""
    if not area:
        return None
    a = area.strip().lower()
    if a in AREA_ALIASES:
        return AREA_ALIASES[a]
    if a in KNOWN_AREAS:
        return area.strip().title()
    return area.strip()


def _normalize_category(category: Optional[str]) -> Optional[str]:
    if not category:
        return None
    c = category.strip().lower()
    if c in KNOWN_CATEGORIES:
        return c.capitalize()
    return category.strip()


def _build_where(area: Optional[str], category: Optional[str]) -> Optional[dict]:
    clauses: list[dict] = []
    if area:
        clauses.append({"area": area})
    if category:
        clauses.append({"category": category})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


@tool
def recommend_recipes(
    query: str,
    area: Optional[str] = None,
    category: Optional[str] = None,
    n: int = 3,
) -> str:
    """Recommend recipes from a curated cookbook using semantic + lexical search.

    Use this when the user asks open-ended things like "what should I cook
    tonight?", "give me ideas with eggplant", "something quick", "an indian
    curry", or "a vegetarian dessert".

    Args:
        query: Free-text description of what the user wants.
        area: Optional cuisine filter (Italian, Indian, Mexican, Thai, etc.).
            If omitted, the tool will auto-detect from the query.
        category: Optional category filter (Vegetarian, Vegan, Seafood,
            Dessert, Pasta, Chicken, Beef, etc.).
        n: Number of recipes to return (default 3, max 5).

    Returns a JSON list of up to ``n`` matching recipes, each with title,
    cuisine, category, ingredient list, and a snippet of the instructions.
    """
    _logs.info(f"recommend_recipes(query={query!r}, area={area}, category={category}, n={n})")
    n = max(1, min(n, 5))

    auto_area, auto_cat = _autodetect(query)
    eff_area = _normalize_area(area) or auto_area
    eff_cat = _normalize_category(category) or auto_cat
    where = _build_where(eff_area, eff_cat)

    try:
        collection = _get_collection()
    except FileNotFoundError as exc:
        _logs.error(str(exc))
        return f"My cookbook isn't loaded right now: {exc}"

    n_in_db = collection.count()
    n_results = min(n, max(1, n_in_db))

    raw = collection.query(query_texts=[query], n_results=n_results, where=where)

    if where and not (raw["ids"] and raw["ids"][0]):
        _logs.info("hybrid where returned 0 hits; falling back to pure semantic")
        raw = collection.query(query_texts=[query], n_results=n_results)

    df = _get_recipes_df()
    df_indexed = df.set_index("id")
    hit_ids = raw["ids"][0] if raw["ids"] else []

    hits: list[RecipeHit] = []
    for rid in hit_ids:
        if rid not in df_indexed.index:
            continue
        row = df_indexed.loc[rid]
        instructions = str(row.get("instructions", ""))
        snippet = " ".join(instructions.split())[:400]
        hits.append(
            RecipeHit(
                title=str(row["title"]),
                area=str(row.get("area", "")),
                category=str(row.get("category", "")),
                ingredients=str(row.get("ingredients", "")),
                snippet=snippet,
            )
        )

    if not hits:
        return "I rummaged through my cookbook but found nothing that fits, tesoro. Try different words?"

    return json.dumps([h.model_dump() for h in hits], ensure_ascii=False)
