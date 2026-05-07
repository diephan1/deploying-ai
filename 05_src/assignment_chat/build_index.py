"""One-time builder for the recipe corpus + persistent ChromaDB index.

Run once with::

    cd 05_src
    uv run python -m assignment_chat.build_index

Outputs:
- ``05_src/assignment_chat/recipes.csv`` (~0.6 MB)
- ``05_src/assignment_chat/chroma_store/`` (persistent Chroma collection ``recipes``)

The grader is **not** expected to run this; both outputs are committed to the
repo. See the readme for the embedding process.
"""

from __future__ import annotations

import json
import string
import time
from pathlib import Path

import chromadb
import pandas as pd
import requests
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from tqdm import tqdm

from utils.logger import get_logger

_logs = get_logger(__name__)

THEMEALDB_BASE = "https://www.themealdb.com/api/json/v1/1"
HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "recipes.csv"
CHROMA_PATH = HERE / "chroma_store"
COLLECTION_NAME = "recipes"
EMBED_MODEL = "all-MiniLM-L6-v2"


def fetch_meals_for_letter(letter: str) -> list[dict]:
    """Return TheMealDB meals whose name starts with ``letter`` (full details)."""
    try:
        resp = requests.get(f"{THEMEALDB_BASE}/search.php", params={"f": letter}, timeout=10)
        resp.raise_for_status()
        meals = resp.json().get("meals") or []
        return meals
    except (requests.RequestException, json.JSONDecodeError) as exc:
        _logs.warning(f"failed to fetch letter {letter}: {exc}")
        return []


def flatten_meal(meal: dict) -> dict:
    """Flatten the awkward strIngredient1..20 fields into clean csv columns."""
    ingredients: list[str] = []
    for i in range(1, 21):
        item = (meal.get(f"strIngredient{i}") or "").strip()
        measure = (meal.get(f"strMeasure{i}") or "").strip()
        if item:
            ingredients.append(f"{measure} {item}".strip())
    return {
        "id": meal.get("idMeal", ""),
        "title": (meal.get("strMeal") or "").strip(),
        "area": (meal.get("strArea") or "").strip(),
        "category": (meal.get("strCategory") or "").strip(),
        "ingredients": " | ".join(ingredients),
        "instructions": (meal.get("strInstructions") or "").strip(),
        "thumb": (meal.get("strMealThumb") or "").strip(),
        "source": (meal.get("strSource") or "").strip(),
    }


def scrape_all_meals() -> pd.DataFrame:
    """Walk a-z on TheMealDB and return a deduplicated DataFrame."""
    rows: list[dict] = []
    for letter in tqdm(list(string.ascii_lowercase), desc="scraping letters"):
        meals = fetch_meals_for_letter(letter)
        for meal in meals:
            rows.append(flatten_meal(meal))
        time.sleep(0.1)
    df = pd.DataFrame(rows).drop_duplicates(subset=["id"]).reset_index(drop=True)
    df = df[df["title"].astype(bool) & df["instructions"].astype(bool)]
    return df


def make_doc(row: pd.Series) -> str:
    """Build the embedding text for a recipe.

    Includes title, cuisine, category, ingredient list, and a Tags line so
    queries like "vegetarian indian curry" or "what can I make with eggplant"
    have strong recall. Truncates instructions to the first 600 chars to keep
    docs tight.
    """
    instructions = (row["instructions"] or "")[:600]
    return (
        f"{row['title']}\n"
        f"Cuisine: {row['area']}\n"
        f"Category: {row['category']}\n"
        f"Ingredients: {row['ingredients']}\n"
        f"Tags: {row['area']}, {row['category']}\n"
        f"Instructions: {instructions}"
    )


def build_chroma(df: pd.DataFrame) -> None:
    """Embed every row and persist into a Chroma collection at ``chroma_store/``."""
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    ids = df["id"].astype(str).tolist()
    docs = [make_doc(r) for _, r in df.iterrows()]
    metas = [
        {
            "title": r["title"],
            "area": r["area"] or "Unknown",
            "category": r["category"] or "Unknown",
            "id": str(r["id"]),
        }
        for _, r in df.iterrows()
    ]

    batch = 64
    for i in tqdm(range(0, len(ids), batch), desc="embedding"):
        collection.add(
            ids=ids[i : i + batch],
            documents=docs[i : i + batch],
            metadatas=metas[i : i + batch],
        )

    _logs.info(f"persisted {collection.count()} recipes into {CHROMA_PATH}")


def main() -> None:
    _logs.info("scraping TheMealDB...")
    df = scrape_all_meals()
    _logs.info(f"scraped {len(df)} unique recipes")

    df.to_csv(CSV_PATH, index=False)
    _logs.info(f"wrote {CSV_PATH} ({CSV_PATH.stat().st_size / 1024:.1f} KB)")

    _logs.info("building chroma collection (downloads MiniLM ~80 MB on first run)...")
    build_chroma(df)
    _logs.info("done.")


if __name__ == "__main__":
    main()
