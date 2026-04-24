"""
FeenX rationale generator -- the LLM wrapper for smart-swap copy.

The deterministic math lives in recalc.py. This module takes that structured
result and generates the emotional + cascade copy shown to the user.

Production path: becomes a Convex Agent component call. Prompt + schema stay
identical; only the transport changes.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict

from anthropic import Anthropic

from .recalc import RecalcResult


SYSTEM_PROMPT = """You are the copywriter for FeenX, an adaptive nutrition \
operating system for personal trainers and their clients. Your job is to write \
short, calm, editorial-quality copy that accompanies a recalculated meal plan \
after a client swaps a meal.

Voice constraints (non-negotiable):
- Warm, confident, never punitive.
- Second-person ("your plan", "your week"). Never "the user."
- BANNED words: cheat, blow it, burn off, bounce back, slipped up, guilt, sin, \
catch up, make up for, off the rails, damage, wreck.
- Never shame the client for eating what they ate.
- Frame the swap as an adjustment, not a failure.
- No emoji. No exclamation marks. No hedging ("maybe", "sort of").
- Editorial tone: think wellness magazine, not startup app.

Output format: return STRICT JSON matching this schema:
{
  "variants": [
    "<one-sentence emotional line, 8-14 words>",
    "<one-sentence emotional line, 8-14 words>",
    "<one-sentence emotional line, 8-14 words>"
  ],
  "cascade": "<one sentence explaining the specific cascade, 12-22 words>",
  "whyItWorks": "<one sentence explaining why this adjustment respects the client's goal or training schedule, 12-22 words>"
}

All three variants must be distinct. One can reference data (the kcal number), \
one can reference emotion (no guilt), one can reference system capability \
(adaptive nutrition). No greeting. No sign-off. No meta commentary.
"""


def generate_rationale(
    client: dict,
    logged_meal: dict,
    replaced_meal: dict,
    result: RecalcResult,
) -> dict:
    """
    Generate the 3-variant emotional copy + cascade + why-it-works explanation
    for a smart-swap event.

    Returns a dict with keys: variants (list[str]), cascade (str), whyItWorks (str).
    Falls back to canned copy if the API call fails.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return _fallback_copy(result)

    client_api = Anthropic(api_key=api_key)

    # Build the context the model needs to write good copy.
    context = {
        "client": {
            "firstName": client["firstName"],
            "goal": client["goal"],
            "chronicConditions": client["chronicConditions"],
            "trainingWeek": client["trainingWeek"],
        },
        "logged_meal": {
            "name": logged_meal.get("label") or logged_meal.get("name"),
            "kcal": logged_meal["kcal"],
            "proteinG": logged_meal["proteinG"],
        },
        "replaced_meal": {
            "name": replaced_meal["name"],
            "kcal": replaced_meal["kcal"],
            "proteinG": replaced_meal["proteinG"],
        },
        "recalc": {
            "deltaKcal": result.deltaKcal,
            "strategy": result.strategy,
            "remainingDays": [
                {
                    "dayOfWeek": d.dayOfWeek,
                    "intensity": d.intensity,
                    "originalKcal": d.originalKcal,
                    "newKcal": d.newKcal,
                    "kcalDelta": d.kcalDelta,
                }
                for d in result.remainingDays
            ],
        },
    }

    user_prompt = (
        "Generate the smart-swap rationale copy for this event. "
        "Return only the JSON object specified in the system prompt.\n\n"
        f"Context:\n{json.dumps(context, indent=2)}"
    )

    try:
        message = client_api.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=800,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw = message.content[0].text.strip()
        # Strip accidental markdown fences if the model wraps JSON.
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        parsed = json.loads(raw)
        # Validate shape.
        if (
            "variants" in parsed
            and isinstance(parsed["variants"], list)
            and len(parsed["variants"]) == 3
            and "cascade" in parsed
            and "whyItWorks" in parsed
        ):
            return parsed
        return _fallback_copy(result)
    except Exception:
        return _fallback_copy(result)


def _fallback_copy(result: RecalcResult) -> dict:
    """Canned copy if the LLM is unreachable. Matches the voice."""
    delta = result.deltaKcal
    if delta > 0:
        variants = [
            "No guilt, no catch-up. Just adjusted.",
            "Your week is still on target. This is what adaptive nutrition means.",
            f"{delta} kcal over today -- redistributed across the rest of the week, barely a ripple.",
        ]
    else:
        variants = [
            f"{abs(delta)} kcal under today -- we'll add it back smartly across the week.",
            "Your plan is living, not fixed. This is what adaptive means.",
            "Today's light. The rest of the week absorbs it without forcing catch-up.",
        ]
    return {
        "variants": variants,
        "cascade": result.cascadeNote,
        "whyItWorks": (
            "The adjustment protects your training days and leans into rest days, "
            "so your week-level totals stay aligned with your goal."
            if result.strategy == "smart"
            else "The adjustment spreads evenly across your remaining days so no one day carries the whole shift."
        ),
    }
