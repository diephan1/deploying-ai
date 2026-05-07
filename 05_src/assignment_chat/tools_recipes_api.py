"""Service 1: TheMealDB API tools.

Returns *condensed* :class:`RecipeBrief` Pydantic models -- never raw API JSON --
so even before the LLM rewrites the response in Sophia's voice, no source URLs,
video links, or internal IDs leak through.
"""

from __future__ import annotations

import json
from typing import Optional

import requests
from langchain.tools import tool
from pydantic import BaseModel, Field

from utils.logger import get_logger

_logs = get_logger(__name__)

THEMEALDB_BASE = "https://www.themealdb.com/api/json/v1/1"
REQUEST_TIMEOUT = 8


class IngredientLine(BaseModel):
    item: str
    measure: str = ""


class RecipeBrief(BaseModel):
    """Condensed view of a meal -- no IDs, no URLs, no video links."""

    title: str = Field(..., description="Name of the dish.")
    area: str = Field("", description="Cuisine of origin (e.g. Italian, Indian).")
    category: str = Field("", description="Dish category (e.g. Vegetarian, Seafood).")
    ingredients: list[IngredientLine] = Field(default_factory=list)
    instructions: str = Field("", description="Cooking steps as plain text.")


def _flatten_meal(meal: dict) -> RecipeBrief:
    """Turn TheMealDB's awkward strIngredient1..20 dict into a clean RecipeBrief."""
    ingredients: list[IngredientLine] = []
    for i in range(1, 21):
        item = (meal.get(f"strIngredient{i}") or "").strip()
        measure = (meal.get(f"strMeasure{i}") or "").strip()
        if item:
            ingredients.append(IngredientLine(item=item, measure=measure))

    return RecipeBrief(
        title=(meal.get("strMeal") or "").strip(),
        area=(meal.get("strArea") or "").strip(),
        category=(meal.get("strCategory") or "").strip(),
        ingredients=ingredients,
        instructions=(meal.get("strInstructions") or "").strip(),
    )


def _get(endpoint: str, params: dict) -> Optional[dict]:
    """GET an endpoint and return the parsed JSON, or None on any failure."""
    try:
        resp = requests.get(f"{THEMEALDB_BASE}/{endpoint}", params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, json.JSONDecodeError) as exc:
        _logs.warning(f"TheMealDB call to {endpoint} {params} failed: {exc}")
        return None


@tool
def search_recipe_by_name(name: str) -> str:
    """Look up a recipe on TheMealDB by its *name* (or partial name).

    Use this when the user asks for a *specific* dish like "lasagna",
    "tiramisu", or "pad thai". Returns the first matching recipe with full
    ingredients and instructions, or a friendly error message if nothing
    matches or the API is unreachable.
    """
    _logs.info(f"search_recipe_by_name(name={name!r})")
    data = _get("search.php", {"s": name})
    if data is None:
        return "The recipe book is closed right now (API unreachable). Try again in a moment, tesoro."
    meals = data.get("meals") or []
    if not meals:
        return f"Mamma mia, I have no recipe for '{name}' in my book. Try a different name?"
    return _flatten_meal(meals[0]).model_dump_json()


@tool
def lookup_recipe_by_ingredient(ingredient: str) -> str:
    """Find a recipe that uses the given primary ingredient.

    Use this when the user asks "what can I make with chicken?", "give me a
    recipe with eggplant", etc. Internally does a two-step lookup: first
    ``filter.php?i=`` (returns stubs) then ``lookup.php?i=`` for the full
    details of the top hit. Returns a single recipe with full ingredients and
    instructions, or a friendly error if nothing matches.
    """
    _logs.info(f"lookup_recipe_by_ingredient(ingredient={ingredient!r})")
    stub_data = _get("filter.php", {"i": ingredient})
    if stub_data is None:
        return "The pantry index is offline (API unreachable). Try again in a moment, tesoro."
    stubs = stub_data.get("meals") or []
    if not stubs:
        return f"No recipes in my book use '{ingredient}' as the star ingredient. Try a different one?"

    top_id = stubs[0].get("idMeal")
    if not top_id:
        return f"Found something with '{ingredient}' but couldn't fetch the details, tesoro."

    detail_data = _get("lookup.php", {"i": top_id})
    if detail_data is None:
        return "I see a recipe but can't open it right now (API unreachable). Try again in a moment."
    meals = detail_data.get("meals") or []
    if not meals:
        return f"Found a hint of a recipe with '{ingredient}' but the details slipped away, tesoro."
    return _flatten_meal(meals[0]).model_dump_json()


@tool
def lookup_recipe_details(meal_id: str) -> str:
    """Fetch full ingredients and instructions for a specific TheMealDB meal id.

    Useful when a previous tool call gave you an id and you need the full
    recipe. Returns a friendly error if the id is unknown.
    """
    _logs.info(f"lookup_recipe_details(meal_id={meal_id!r})")
    data = _get("lookup.php", {"i": meal_id})
    if data is None:
        return "Recipe lookup failed (API unreachable). Try again in a moment, tesoro."
    meals = data.get("meals") or []
    if not meals:
        return f"No recipe with id {meal_id} in my book."
    return _flatten_meal(meals[0]).model_dump_json()


@tool
def random_recipe() -> str:
    """Get a single random recipe from TheMealDB.

    Use this when the user says "surprise me", "give me something random",
    "I don't know what to cook tonight", etc.
    """
    _logs.info("random_recipe()")
    data = _get("random.php", {})
    if data is None:
        return "Even the dice are tired (API unreachable). Try again in a moment, tesoro."
    meals = data.get("meals") or []
    if not meals:
        return "No random recipe came back. Strange. Try again?"
    return _flatten_meal(meals[0]).model_dump_json()
