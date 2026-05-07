"""Service 3: deterministic kitchen-math tools (function calling).

These tools never call an external service. They demonstrate function calling
as the assignment requires for the open-ended Service 3 -- the LLM extracts
arguments from natural language, the tools do the arithmetic exactly, and the
LLM rephrases the answer in Sophia's voice.

Why function calling and not web search / MCP for Service 3:
unit conversion and recipe scaling are exactly the kind of brittle arithmetic
where letting the LLM "do it in its head" leads to silent errors. Function
calling guarantees correct numbers without external dependencies or API keys.
"""

from __future__ import annotations

from typing import Optional

from langchain.tools import tool
from pydantic import BaseModel, Field

from utils.logger import get_logger

_logs = get_logger(__name__)


VOLUME_TO_ML = {
    "ml": 1.0,
    "milliliter": 1.0,
    "milliliters": 1.0,
    "millilitre": 1.0,
    "millilitres": 1.0,
    "l": 1000.0,
    "liter": 1000.0,
    "liters": 1000.0,
    "litre": 1000.0,
    "litres": 1000.0,
    "tsp": 4.92892,
    "teaspoon": 4.92892,
    "teaspoons": 4.92892,
    "tbsp": 14.7868,
    "tablespoon": 14.7868,
    "tablespoons": 14.7868,
    "cup": 236.588,
    "cups": 236.588,
    "fl oz": 29.5735,
    "floz": 29.5735,
    "fluid ounce": 29.5735,
    "fluid ounces": 29.5735,
    "pint": 473.176,
    "pints": 473.176,
    "quart": 946.353,
    "quarts": 946.353,
}

WEIGHT_TO_G = {
    "g": 1.0,
    "gram": 1.0,
    "grams": 1.0,
    "kg": 1000.0,
    "kilogram": 1000.0,
    "kilograms": 1000.0,
    "oz": 28.3495,
    "ounce": 28.3495,
    "ounces": 28.3495,
    "lb": 453.592,
    "lbs": 453.592,
    "pound": 453.592,
    "pounds": 453.592,
    "mg": 0.001,
    "milligram": 0.001,
    "milligrams": 0.001,
}

GAS_MARK_TO_C = {
    "1/4": 110, "1/2": 130, "1": 140, "2": 150, "3": 170, "4": 180,
    "5": 190, "6": 200, "7": 220, "8": 230, "9": 240, "10": 260,
}


class IngredientLine(BaseModel):
    name: str = Field(..., description="Ingredient name, e.g. 'flour'.")
    qty: float = Field(..., description="Numeric quantity, e.g. 2.5.")
    unit: str = Field("", description="Unit, e.g. 'cup' or 'g'. Empty for 'eggs', 'pinch'.")


def _normalize_unit(u: str) -> str:
    return u.strip().lower().rstrip(".")


@tool
def convert_volume(amount: float, from_unit: str, to_unit: str) -> str:
    """Convert a volume between cooking units.

    Supported units (case-insensitive): ml, l, tsp, tbsp, cup, fl oz, pint, quart.
    Example: ``convert_volume(2, "cup", "ml")`` -> "473.18 ml".
    """
    fu, tu = _normalize_unit(from_unit), _normalize_unit(to_unit)
    if fu not in VOLUME_TO_ML:
        return f"I don't know the volume unit '{from_unit}'. Try ml, l, tsp, tbsp, cup, fl oz, pint, or quart."
    if tu not in VOLUME_TO_ML:
        return f"I don't know the volume unit '{to_unit}'. Try ml, l, tsp, tbsp, cup, fl oz, pint, or quart."
    ml = amount * VOLUME_TO_ML[fu]
    out = ml / VOLUME_TO_ML[tu]
    _logs.info(f"convert_volume({amount} {fu} -> {tu}) = {out:.4f}")
    return f"{out:.2f} {to_unit}"


@tool
def convert_weight(amount: float, from_unit: str, to_unit: str) -> str:
    """Convert a weight between cooking units.

    Supported units (case-insensitive): mg, g, kg, oz, lb.
    Example: ``convert_weight(8, "oz", "g")`` -> "226.80 g".
    """
    fu, tu = _normalize_unit(from_unit), _normalize_unit(to_unit)
    if fu not in WEIGHT_TO_G:
        return f"I don't know the weight unit '{from_unit}'. Try mg, g, kg, oz, or lb."
    if tu not in WEIGHT_TO_G:
        return f"I don't know the weight unit '{to_unit}'. Try mg, g, kg, oz, or lb."
    g = amount * WEIGHT_TO_G[fu]
    out = g / WEIGHT_TO_G[tu]
    _logs.info(f"convert_weight({amount} {fu} -> {tu}) = {out:.4f}")
    return f"{out:.2f} {to_unit}"


@tool
def convert_temperature(value: str, from_unit: str, to_unit: str) -> str:
    """Convert oven temperature between Celsius (C), Fahrenheit (F), and gas mark.

    For gas mark, ``value`` may be a fraction like '1/4' or an integer 1-10.
    Example: ``convert_temperature("180", "C", "F")`` -> "356 F".
    Example: ``convert_temperature("4", "gas", "C")`` -> "180 C".
    """
    fu, tu = _normalize_unit(from_unit), _normalize_unit(to_unit)
    aliases = {
        "celsius": "c", "centigrade": "c", "c": "c",
        "fahrenheit": "f", "f": "f",
        "gas mark": "gas", "gas": "gas", "gm": "gas",
    }
    fu = aliases.get(fu, fu)
    tu = aliases.get(tu, tu)

    if fu == "gas":
        if value not in GAS_MARK_TO_C:
            return f"Gas mark '{value}' is not standard. Use 1/4, 1/2, or 1 through 10."
        celsius: float = float(GAS_MARK_TO_C[value])
    else:
        try:
            v = float(value)
        except ValueError:
            return f"Couldn't read '{value}' as a number, tesoro."
        if fu == "c":
            celsius = v
        elif fu == "f":
            celsius = (v - 32) * 5.0 / 9.0
        else:
            return f"Unknown temperature unit '{from_unit}'. Try C, F, or gas mark."

    if tu == "c":
        return f"{round(celsius)} C"
    if tu == "f":
        return f"{round(celsius * 9.0 / 5.0 + 32)} F"
    if tu == "gas":
        nearest = min(GAS_MARK_TO_C.items(), key=lambda kv: abs(kv[1] - celsius))
        return f"gas mark {nearest[0]}"
    return f"Unknown temperature unit '{to_unit}'. Try C, F, or gas mark."


@tool
def scale_recipe(
    servings_from: float,
    servings_to: float,
    ingredients: list[IngredientLine],
) -> str:
    """Scale ingredient quantities from one serving count to another.

    Example: ``scale_recipe(4, 6, [{"name": "flour", "qty": 2, "unit": "cup"}])``
    returns the same ingredient list with ``qty`` multiplied by ``6/4 = 1.5``.
    """
    if servings_from <= 0:
        return "Servings_from must be greater than zero, tesoro."
    if servings_to <= 0:
        return "Servings_to must be greater than zero, tesoro."

    factor = servings_to / servings_from
    lines = []
    for ing in ingredients:
        new_qty = round(ing.qty * factor, 2)
        unit = f" {ing.unit}" if ing.unit else ""
        lines.append(f"- {new_qty}{unit} {ing.name}")
    header = f"Scaled from {servings_from} to {servings_to} servings (x{factor:.2f}):\n"
    return header + "\n".join(lines)
