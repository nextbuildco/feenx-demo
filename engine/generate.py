"""
FeenX meal-plan generator teaser (engine A).

Hybrid: retrieval over curated recipes + LLM for rationale.

Production path: becomes a Convex Agent with tool access to the Convex RAG
component (vector search on a larger recipe corpus). Prompt + filtering logic
stay identical.
"""

from __future__ import annotations

import json
import os
import random
from typing import Literal

from anthropic import Anthropic


Slot = Literal["Breakfast", "Lunch", "Snack", "Dinner"]


RATIONALE_SYSTEM_PROMPT = """You are the FeenX meal-rationale writer. Given a \
client profile and a proposed meal, write 3 short "why we picked this" insight \
chips that will display on the mobile meal-detail screen.

Voice:
- Warm, respectful of the client's goal and constraints.
- Each chip is 8-14 words.
- No emoji. No exclamation marks.
- One chip should reference their goal. One should reference a constraint \
(allergy, aversion, cooking time, equipment, or chronic condition). One should \
reference training context or variety.

Output STRICT JSON:
{"insights": ["<chip 1>", "<chip 2>", "<chip 3>"]}
No other text.
"""


def filter_recipes_for_client(
    recipes: list[dict],
    client: dict,
    slot: Slot,
) -> list[dict]:
    """
    Apply hard constraints to the recipe library:
    - slot match
    - allergy safety (never serve an allergen)
    - aversion avoidance (never serve a hated ingredient keyword in name)
    - cooking time within budget
    - equipment available (if recipe requires specific equipment)
    """
    result: list[dict] = []
    cooking_budget = client.get("cookingMinutesWeeknight", 30)
    equipment = set(client.get("equipment", []))
    allergies = {a.lower() for a in client.get("allergies", [])}
    aversions = {a.lower() for a in client.get("aversions", [])}

    for r in recipes:
        if r["slot"] != slot:
            continue
        # Cooking time budget.
        if r["prepMinutes"] > cooking_budget + 5:
            continue
        # Equipment: every required piece must be available, OR recipe lists none.
        req = set(r.get("equipment", []))
        if req and not req.issubset(equipment):
            continue
        # Allergies: never serve allergens.
        contains = {c.lower() for c in r.get("contains", [])}
        if contains.intersection(allergies):
            continue
        # Aversions: exclude if any aversion keyword appears in the recipe name.
        lower_name = r["name"].lower()
        if any(av in lower_name for av in aversions):
            continue
        result.append(r)
    return result


def score_recipe_for_client(recipe: dict, client: dict, intensity: str) -> float:
    """
    Prefer recipes that fit the client's goal and training intensity.
    Higher score = better fit.
    """
    score = 1.0
    goal = client.get("goal", "")
    conditions = [c.lower() for c in client.get("chronicConditions", [])]
    tags = {t.lower() for t in recipe.get("tags", [])}
    glycemic = recipe.get("glycemicLoad", "medium")

    # Goal-based boosts.
    if goal == "Fat loss":
        if "low-carb" in tags or "high-protein" in tags:
            score += 0.3
        if recipe["kcal"] < 500:
            score += 0.1
    elif goal == "Muscle gain":
        if "high-protein" in tags:
            score += 0.3
        if recipe["proteinG"] >= 35:
            score += 0.2
    elif goal == "Performance":
        if intensity in ("High", "Moderate") and recipe["carbsG"] >= 40:
            score += 0.25
    elif goal == "Maintenance":
        score += 0.05  # neutral

    # Condition-based boosts.
    if any("diabetes" in c for c in conditions):
        if glycemic == "low":
            score += 0.35
        if "t2d-friendly" in tags or "high-fiber" in tags:
            score += 0.2
    if any("pcos" in c for c in conditions):
        if "pcos-friendly" in tags or "low-carb" in tags:
            score += 0.3
    if any("hypertension" in c for c in conditions):
        if "heart-healthy" in tags:
            score += 0.3

    # Training-day context.
    if intensity == "High" and "post-workout" in tags:
        score += 0.15

    return score


def pick_meal(recipes: list[dict], client: dict, slot: Slot, intensity: str) -> dict | None:
    """Filter + score + return the top-scoring recipe for a slot."""
    candidates = filter_recipes_for_client(recipes, client, slot)
    if not candidates:
        return None
    # Score each and pick the top. Break ties with a stable randomizer.
    scored = [(score_recipe_for_client(r, client, intensity), random.random(), r) for r in candidates]
    scored.sort(key=lambda t: (-t[0], t[1]))
    return scored[0][2]


def generate_day_plan(
    recipes: list[dict],
    client: dict,
    intensity: str,
) -> list[dict]:
    """Pick one recipe per slot for a single day."""
    meals: list[dict] = []
    for slot in ["Breakfast", "Lunch", "Snack", "Dinner"]:
        meal = pick_meal(recipes, client, slot, intensity)  # type: ignore[arg-type]
        if meal:
            meals.append(meal)
    return meals


def generate_insights_for_meal(client: dict, meal: dict, intensity: str) -> list[str]:
    """
    Use the LLM to produce the 3 "why we picked this" chips.
    Falls back to deterministic canned insights if the API is unreachable.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return _fallback_insights(client, meal, intensity)

    client_api = Anthropic(api_key=api_key)
    context = {
        "client": {
            "firstName": client["firstName"],
            "goal": client["goal"],
            "chronicConditions": client["chronicConditions"],
            "allergies": client["allergies"],
            "aversions": client["aversions"],
            "cookingMinutesWeeknight": client["cookingMinutesWeeknight"],
            "equipment": client["equipment"],
        },
        "meal": meal,
        "trainingIntensity": intensity,
    }
    user_prompt = (
        "Generate the 3 insight chips for this meal. Return only the JSON.\n\n"
        f"Context:\n{json.dumps(context, indent=2)}"
    )
    try:
        msg = client_api.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=400,
            system=RATIONALE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw = msg.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        parsed = json.loads(raw)
        if "insights" in parsed and isinstance(parsed["insights"], list) and len(parsed["insights"]) == 3:
            return parsed["insights"]
    except Exception:
        pass
    return _fallback_insights(client, meal, intensity)


def _fallback_insights(client: dict, meal: dict, intensity: str) -> list[str]:
    out: list[str] = []
    goal = client.get("goal", "")
    conditions = [c.lower() for c in client.get("chronicConditions", [])]
    if goal == "Fat loss":
        out.append(f"Keeps you full at {meal['kcal']} kcal -- on track for your fat-loss target.")
    elif goal == "Muscle gain":
        out.append(f"Packs {meal['proteinG']}g protein -- supports your muscle-gain window.")
    elif goal == "Performance":
        out.append(f"{meal['carbsG']}g carbs -- fuels your {intensity.lower()}-intensity training day.")
    else:
        out.append(f"Balanced macros at {meal['kcal']} kcal -- matches your maintenance target.")
    if meal["prepMinutes"] <= client.get("cookingMinutesWeeknight", 30):
        out.append(f"{meal['prepMinutes']} min prep -- fits your weeknight cooking budget.")
    if any("diabetes" in c for c in conditions) and meal.get("glycemicLoad") == "low":
        out.append("Low-glycemic -- supports steady blood sugar through the afternoon.")
    elif any("pcos" in c for c in conditions):
        out.append("Protein-forward pairing -- aligned with PCOS insulin sensitivity.")
    else:
        out.append("Fresh ingredients on this week's grocery list -- nothing extra to buy.")
    return out[:3]
