"""
FeenX adaptive recalculator.

Deterministic, LLM-free math. Given a plan, a log event, and a redistribution
strategy, this module computes the new per-day targets for the rest of the week.

The math is provable and auditable. The LLM (rationale.py) never touches the
numbers -- it only writes the explanation copy on top of this output.

Production path: this file becomes a Convex action. Inputs stay identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Strategy = Literal["even", "smart"]

# Training-intensity weights for the "smart by training day" strategy.
# Higher weight = trim MORE from this day (rest days absorb more).
# Lower weight = trim LESS (training days stay fueled).
INTENSITY_WEIGHT: dict[str, float] = {
    "Rest": 2.0,
    "Low": 1.5,
    "Moderate": 1.0,
    "High": 0.5,
}

# Absolute safety floors -- the solver must never push a day below these.
# These are the lower bound; the per-call `min_kcal_floor` is computed relative
# to the client's plan (default: 80% of avg day target, clamped to >= 1200).
ABSOLUTE_MIN_KCAL_PER_DAY = 1200
MIN_PROTEIN_PER_DAY = 120  # g


@dataclass
class Macros:
    kcal: int
    proteinG: int
    carbsG: int
    fatG: int

    def __add__(self, other: "Macros") -> "Macros":
        return Macros(
            kcal=self.kcal + other.kcal,
            proteinG=self.proteinG + other.proteinG,
            carbsG=self.carbsG + other.carbsG,
            fatG=self.fatG + other.fatG,
        )

    def __sub__(self, other: "Macros") -> "Macros":
        return Macros(
            kcal=self.kcal - other.kcal,
            proteinG=self.proteinG - other.proteinG,
            carbsG=self.carbsG - other.carbsG,
            fatG=self.fatG - other.fatG,
        )


@dataclass
class DayTotal:
    dayOfWeek: str
    intensity: str
    originalKcal: int
    newKcal: int
    kcalDelta: int = field(init=False)

    def __post_init__(self) -> None:
        self.kcalDelta = self.newKcal - self.originalKcal


@dataclass
class RecalcResult:
    deltaKcal: int           # excess (+) or deficit (-) from the log event
    todayLoggedKcal: int     # actual kcal consumed today
    todayTargetKcal: int     # original daily target
    remainingDays: list[DayTotal]
    strategy: Strategy
    cascadeNote: str         # human-readable specific cascade (e.g. "tomorrow's breakfast...")


def compute_day_total(meals: list[dict]) -> Macros:
    """Sum macros across a day's 4 meals."""
    totals = Macros(0, 0, 0, 0)
    for m in meals:
        totals = totals + Macros(
            kcal=m["kcal"],
            proteinG=m["proteinG"],
            carbsG=m["carbsG"],
            fatG=m["fatG"],
        )
    return totals


def recalc_rest_of_week(
    plan_days: list[dict],
    today_index: int,
    logged_meal: dict,
    replaced_meal: dict,
    strategy: Strategy = "even",
    min_kcal_floor: int | None = None,
) -> RecalcResult:
    """
    Core recalc function.

    `plan_days`     full 7-day plan (list of day dicts from plans.json)
    `today_index`   which day in the list is today (0..6)
    `logged_meal`   what the client actually ate (from swap_options.json or custom)
    `replaced_meal` the planned meal that got replaced (from the day's meals array)
    `strategy`      "even" or "smart"
    `min_kcal_floor` per-client minimum kcal/day. If None, defaults to 80% of the
                     average daily target across the plan (client-relative floor).
    """
    today = plan_days[today_index]
    today_target = compute_day_total(today["meals"])

    # Client-relative floor: 80% of average day target unless caller overrides.
    if min_kcal_floor is None:
        avg_target = sum(compute_day_total(d["meals"]).kcal for d in plan_days) / max(len(plan_days), 1)
        min_kcal_floor = max(1200, int(avg_target * 0.80))

    # Compute delta: actual - planned for the meal that was swapped.
    delta = Macros(
        kcal=logged_meal["kcal"] - replaced_meal["kcal"],
        proteinG=logged_meal["proteinG"] - replaced_meal["proteinG"],
        carbsG=logged_meal["carbsG"] - replaced_meal["carbsG"],
        fatG=logged_meal["fatG"] - replaced_meal["fatG"],
    )

    # Today's final actual kcal: every meal today except the replaced one,
    # plus the logged substitute.
    today_actual_kcal = sum(
        (logged_meal["kcal"] if m["slot"] == replaced_meal["slot"] else m["kcal"])
        for m in today["meals"]
    )

    # Remaining days to redistribute across.
    remaining = plan_days[today_index + 1:]
    if not remaining:
        # No days left to redistribute -- today absorbs everything.
        return RecalcResult(
            deltaKcal=delta.kcal,
            todayLoggedKcal=today_actual_kcal,
            todayTargetKcal=today_target.kcal,
            remainingDays=[],
            strategy=strategy,
            cascadeNote="End of week -- no remaining days to redistribute into. Fresh start Monday.",
        )

    # Weights per day based on strategy.
    if strategy == "even":
        weights = [1.0] * len(remaining)
    else:  # "smart"
        weights = [INTENSITY_WEIGHT.get(d["intensity"], 1.0) for d in remaining]

    weight_sum = sum(weights)

    # Distribute the delta inversely. If delta is +275 kcal (user ate over),
    # we trim the next days. If delta is -200 (under-ate), we add back.
    #
    # We trim each day proportional to its weight / weight_sum.
    # Sign: subtract delta from remaining days.
    day_targets: list[DayTotal] = []
    remaining_delta = delta.kcal  # the amount we still need to distribute

    for i, day in enumerate(remaining):
        day_total = compute_day_total(day["meals"])

        # Allocate a share of the total delta.
        share = (weights[i] / weight_sum) * delta.kcal
        share_int = round(share)

        new_kcal = day_total.kcal - share_int

        # Enforce safety floors. If trimming would push below MIN_KCAL,
        # clamp and push the unclaimed portion to the next days.
        if new_kcal < min_kcal_floor:
            unclaimed = min_kcal_floor - new_kcal  # positive -- amount we couldn't cut
            new_kcal = min_kcal_floor
            # Defer the unclaimed portion to the remaining days.
            # (Simple heuristic -- split evenly across later days.)
            later = len(remaining) - i - 1
            if later > 0:
                per_later = unclaimed / later
                # Mutate the weights so future shares pick up the slack.
                for j in range(i + 1, len(remaining)):
                    weights[j] = weights[j] + (per_later / max(day_total.kcal, 1))

        day_targets.append(DayTotal(
            dayOfWeek=day["dayOfWeek"],
            intensity=day["intensity"],
            originalKcal=day_total.kcal,
            newKcal=new_kcal,
        ))

    # Cascade note: which day absorbed the biggest adjustment, and which meal.
    biggest = max(day_targets, key=lambda d: abs(d.kcalDelta))
    if abs(biggest.kcalDelta) > 0:
        direction = "lower" if biggest.kcalDelta < 0 else "higher"
        cascade = (
            f"{biggest.dayOfWeek}'s plan will run ~{abs(biggest.kcalDelta)} kcal "
            f"{direction} to absorb today's adjustment."
        )
    else:
        cascade = "Today's shift was small enough that your remaining days stay on target."

    return RecalcResult(
        deltaKcal=delta.kcal,
        todayLoggedKcal=today_actual_kcal,
        todayTargetKcal=today_target.kcal,
        remainingDays=day_targets,
        strategy=strategy,
        cascadeNote=cascade,
    )
