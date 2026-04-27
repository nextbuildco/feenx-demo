"""
FeenX AI demo -- Streamlit app.

Three tabs:
  1. Client Profile     -- inspect seeded client + their 7-day plan
  2. Smart Swap         -- deterministic recalc + LLM rationale
  3. Generate Meal      -- curated library + scoring + LLM rationale

Run locally:
    ANTHROPIC_API_KEY=sk-... streamlit run app.py

Deploy: Streamlit Community Cloud -- set ANTHROPIC_API_KEY in App Secrets.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import pandas as pd
import streamlit as st

# Streamlit Cloud secrets -> env. Must run before engine imports use os.getenv.
try:
    if "ANTHROPIC_API_KEY" in st.secrets:
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
except Exception:
    pass

from engine.generate import generate_day_plan, generate_insights_for_meal
from engine.rationale import generate_rationale
from engine.recalc import compute_day_total, recalc_rest_of_week


# ---------- Page config + theme ----------

st.set_page_config(
    page_title="FeenX -- Adaptive Nutrition Demo",
    page_icon="●",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Brand-adjacent CSS. Keeps the demo editorial, not default Streamlit.
st.markdown(
    """
    <style>
    .stApp { background-color: #f9f6ec; color: #1a1a1a; }
    header[data-testid="stHeader"] { background-color: #f9f6ec; }
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e1e1e1;
    }
    section[data-testid="stSidebar"] *,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div { color: #1a1a1a; }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 { color: #172f29; }
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    section[data-testid="stSidebar"] small { color: #3a3a3a; }
    h1, h2, h3, h4 {
        font-family: Georgia, "Playfair Display", serif;
        color: #172f29;
    }
    p, span, label, li { color: #1a1a1a; }
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li { color: #1a1a1a; }
    .stTabs { margin-top: 28px; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; padding-bottom: 4px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 999px;
        padding: 10px 20px;
        border: 1px solid #e1e1e1;
        color: #172f29;
    }
    .stTabs [data-baseweb="tab"] p,
    .stTabs [data-baseweb="tab"] span { color: #172f29 !important; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #172f29;
        border-color: #172f29;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] p,
    .stTabs [data-baseweb="tab"][aria-selected="true"] span,
    .stTabs [data-baseweb="tab"][aria-selected="true"] div { color: #f9f6ec !important; }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }
    .stTabs [data-baseweb="tab-border"] { display: none; }
    .stButton > button {
        background-color: #d63d00 !important;
        color: #ffffff !important;
        border-radius: 999px !important;
        border: none !important;
        padding: 10px 24px !important;
        font-weight: 500 !important;
    }
    .stButton > button p,
    .stButton > button span,
    .stButton > button div { color: #ffffff !important; }
    .stButton > button:hover {
        background-color: #b83300 !important;
        color: #ffffff !important;
        border: none !important;
    }
    .stButton > button:hover p,
    .stButton > button:hover span,
    .stButton > button:hover div { color: #ffffff !important; }
    .stButton > button:focus,
    .stButton > button:active {
        background-color: #b83300 !important;
        color: #ffffff !important;
        box-shadow: none !important;
        outline: 2px solid #172f29 !important;
        outline-offset: 2px !important;
    }
    .stButton > button:focus p,
    .stButton > button:focus span,
    .stButton > button:focus div,
    .stButton > button:active p,
    .stButton > button:active span,
    .stButton > button:active div { color: #ffffff !important; }
    .feenx-pill {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 13px;
        margin-right: 8px;
        margin-bottom: 6px;
    }
    .feenx-pill-sage    { background-color: #dcf2eb; color: #2a7188; }
    .feenx-pill-amber   { background-color: #fac62c; color: #1a1a1a; }
    .feenx-pill-crimson { background-color: #e0293b; color: #ffffff; }
    .feenx-card {
        background: #ffffff;
        border: 1px solid #e1e1e1;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 12px;
    }
    .feenx-variant {
        background: #dcf2eb;
        border-left: 4px solid #1ca885;
        padding: 14px 18px;
        border-radius: 10px;
        margin-bottom: 10px;
        font-style: italic;
        font-family: Georgia, serif;
        font-size: 18px;
        color: #172f29;
    }
    .feenx-stat { font-family: Georgia, serif; font-size: 36px; color: #172f29; font-weight: 900; }
    .feenx-stat-label { font-size: 12px; color: #676767; text-transform: uppercase; letter-spacing: 0.12em; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Data loading ----------

DATA_DIR = Path(__file__).parent / "data"


@st.cache_data
def load_clients() -> dict:
    return json.loads((DATA_DIR / "clients.json").read_text())


@st.cache_data
def load_plans() -> dict:
    return json.loads((DATA_DIR / "plans.json").read_text())


@st.cache_data
def load_swap_options() -> list[dict]:
    return json.loads((DATA_DIR / "swap_options.json").read_text())


@st.cache_data
def load_recipes() -> list[dict]:
    return json.loads((DATA_DIR / "recipes.json").read_text())


clients = load_clients()
plans = load_plans()
swap_options = load_swap_options()
recipes = load_recipes()


# ---------- Sidebar: client picker + about ----------

with st.sidebar:
    st.markdown("### FeenX")
    st.caption("Adaptive nutrition engine demo")
    st.divider()

    client_labels = {
        "madison-r": "Madison R. • Performance / Perimenopause",
        "diego-l":   "Diego L. • Type 2 Diabetes",
        "rachel-t":  "Rachel T. • GLP-1 (Wegovy)",
    }
    chosen_id = st.radio(
        "Client",
        options=list(client_labels.keys()),
        format_func=lambda k: client_labels[k],
        index=0,
    )
    client = clients[chosen_id]
    plan = plans[chosen_id]

    st.divider()
    st.caption(
        "This demo shows how FeenX adapts a nutrition plan to real life. "
        "Swap a meal and watch the week rebalance. Generate a new meal and "
        "see why it was picked for this client."
    )
    api_ok = bool(os.getenv("ANTHROPIC_API_KEY"))
    if api_ok:
        st.success("LLM connected ✓")
    else:
        st.warning("No API key detected — using canned fallback copy.")


# ---------- Header ----------

st.markdown(f"# {client['firstName']} {client['lastName']}")
st.markdown(
    f"<div class='feenx-stat-label'>"
    f"Goal • {client['goal']}"
    f" • {client['age']} • {client['sexAtBirth']}"
    f" • {client['heightCm']}cm • {client['currentWeightKg']}kg"
    + (f" • {', '.join(client['chronicConditions'])}" if client['chronicConditions'] else "")
    + "</div>",
    unsafe_allow_html=True,
)


# ---------- Tabs ----------

tab_profile, tab_swap, tab_generate = st.tabs([
    "Client profile",
    "Smart swap",
    "Generate a meal",
])


# =================================================================
# Tab 1: Client Profile
# =================================================================

with tab_profile:
    col1, col2, col3, col4 = st.columns(4)
    targets = client["targets"]
    for col, label, value in [
        (col1, "Daily kcal target",       f"{targets['kcal']:,}"),
        (col2, "Protein (g)",             f"{targets['proteinG']}"),
        (col3, "Carbs (g)",               f"{targets['carbsG']}"),
        (col4, "Fat (g)",                 f"{targets['fatG']}"),
    ]:
        with col:
            st.markdown(
                f"<div class='feenx-card'>"
                f"<div class='feenx-stat-label'>{label}</div>"
                f"<div class='feenx-stat'>{value}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("### Constraints FeenX respects")
    allergies = client["allergies"] or ["None"]
    aversions = client["aversions"] or ["None"]
    equipment = client["equipment"] or ["None"]
    for label, vals, css_class in [
        ("Allergies",       allergies, "feenx-pill-crimson"),
        ("Dislikes",        aversions, "feenx-pill-amber"),
        ("Equipment",       equipment, "feenx-pill-sage"),
    ]:
        pills = " ".join(
            f"<span class='feenx-pill {css_class}'>{v}</span>" for v in vals
        )
        st.markdown(
            f"<div style='margin-bottom: 10px;'>"
            f"<span class='feenx-stat-label'>{label}</span>"
            f"<div style='margin-top: 6px;'>{pills}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("### Training week")
    training_df = pd.DataFrame([
        {"Day": d["day"], "Intensity": d["intensity"]}
        for d in client["trainingWeek"]
    ])
    st.dataframe(training_df, hide_index=True, use_container_width=True)

    st.markdown("### This week's plan")
    for day in plan["days"]:
        day_total = compute_day_total(day["meals"])
        with st.expander(
            f"**{day['dayOfWeek']}** • {day['intensity']} "
            f"• {day_total.kcal:,} kcal • {day_total.proteinG}P / {day_total.carbsG}C / {day_total.fatG}F",
            expanded=(day["dayOfWeek"] == "Mon"),
        ):
            meals_df = pd.DataFrame([
                {
                    "Slot": m["slot"],
                    "Meal": m["name"],
                    "kcal": m["kcal"],
                    "P": m["proteinG"],
                    "C": m["carbsG"],
                    "F": m["fatG"],
                    "Prep (min)": m["prepMinutes"],
                }
                for m in day["meals"]
            ])
            st.dataframe(meals_df, hide_index=True, use_container_width=True)


# =================================================================
# Tab 2: Smart Swap
# =================================================================

with tab_swap:
    st.markdown(
        "### The moment life gets messy"
    )
    st.markdown(
        "Most nutrition apps ask the user to catch up. FeenX eliminates catch-up. "
        "Pick what actually happened, pick how you want the week to rebalance, and "
        "watch the plan adapt."
    )

    # Step 1: pick which meal got replaced.
    today = plan["days"][0]  # treat Mon as "today"
    meal_labels = {m["slot"]: m["name"] for m in today["meals"] if m.get("status") == "Pending" or m["slot"] != "Breakfast"}
    # Default to Lunch if present.
    default_slot = "Lunch" if "Lunch" in meal_labels else list(meal_labels.keys())[0]

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**Which meal got swapped?**")
        slot = st.selectbox(
            "Slot",
            options=list(meal_labels.keys()),
            index=list(meal_labels.keys()).index(default_slot),
            label_visibility="collapsed",
        )
        replaced_meal = next(m for m in today["meals"] if m["slot"] == slot)
        st.markdown(
            f"<div class='feenx-card'>"
            f"<div class='feenx-stat-label'>Planned • {replaced_meal['slot']}</div>"
            f"<div style='font-family: Georgia, serif; font-size: 22px; color: #172f29;'>{replaced_meal['name']}</div>"
            f"<div style='margin-top: 8px; color: #3a3a3a;'>"
            f"{replaced_meal['kcal']} kcal • {replaced_meal['proteinG']}g P • "
            f"{replaced_meal['carbsG']}g C • {replaced_meal['fatG']}g F"
            f"</div></div>",
            unsafe_allow_html=True,
        )

    with col_r:
        st.markdown("**What did they actually eat?**")
        option_labels = [o["label"] for o in swap_options]
        picked_label = st.selectbox("Substitute", options=option_labels, index=0, label_visibility="collapsed")
        logged_meal = next(o for o in swap_options if o["label"] == picked_label)
        # Allow editable macros for custom path.
        use_custom = st.checkbox("Edit macros manually", value=False)
        if use_custom:
            c1, c2, c3, c4 = st.columns(4)
            logged_meal = dict(logged_meal)  # copy
            logged_meal["kcal"]     = c1.number_input("kcal",     value=logged_meal["kcal"],     min_value=50, max_value=2500, step=10)
            logged_meal["proteinG"] = c2.number_input("Protein",  value=logged_meal["proteinG"], min_value=0,  max_value=200,  step=1)
            logged_meal["carbsG"]   = c3.number_input("Carbs",    value=logged_meal["carbsG"],   min_value=0,  max_value=300,  step=1)
            logged_meal["fatG"]     = c4.number_input("Fat",      value=logged_meal["fatG"],     min_value=0,  max_value=150,  step=1)
        else:
            st.markdown(
                f"<div class='feenx-card'>"
                f"<div class='feenx-stat-label'>Logged • {logged_meal['label']}</div>"
                f"<div style='margin-top: 8px; color: #3a3a3a;'>"
                f"{logged_meal['kcal']} kcal • {logged_meal['proteinG']}g P • "
                f"{logged_meal['carbsG']}g C • {logged_meal['fatG']}g F"
                f"</div></div>",
                unsafe_allow_html=True,
            )

    # Redistribution strategy toggle.
    st.markdown("**How should FeenX rebalance the rest of the week?**")
    strategy_label = st.radio(
        "Strategy",
        options=["Even split", "Smart by training day"],
        horizontal=True,
        label_visibility="collapsed",
        help=(
            "Even split spreads the adjustment evenly across remaining days. "
            "Smart by training day trims more on rest days and protects fuel on training days."
        ),
    )
    strategy = "even" if strategy_label == "Even split" else "smart"

    st.divider()

    if st.button("Recalculate the week", type="primary", use_container_width=False):
        result = recalc_rest_of_week(
            plan_days=plan["days"],
            today_index=0,
            logged_meal=logged_meal,
            replaced_meal=replaced_meal,
            strategy=strategy,
        )

        # Headline stat.
        delta = result.deltaKcal
        verb = "over" if delta > 0 else "under"
        st.markdown(
            f"<div class='feenx-card' style='border-left: 4px solid #d63d00;'>"
            f"<div class='feenx-stat-label'>Today's shift</div>"
            f"<div class='feenx-stat'>{abs(delta):+,} kcal {verb}</div>"
            f"<div style='color: #3a3a3a; margin-top: 6px;'>"
            f"Replaced <b>{replaced_meal['name']}</b> ({replaced_meal['kcal']} kcal) with "
            f"<b>{logged_meal.get('label') or logged_meal.get('name')}</b> ({logged_meal['kcal']} kcal)."
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Before/after bar chart.
        st.markdown("### The rest of your week, adjusted")
        chart_df = pd.DataFrame([
            {"Day": d.dayOfWeek, "State": "Before", "kcal": d.originalKcal}
            for d in result.remainingDays
        ] + [
            {"Day": d.dayOfWeek, "State": "After", "kcal": d.newKcal}
            for d in result.remainingDays
        ])
        chart_pivot = chart_df.pivot(index="Day", columns="State", values="kcal").reindex(
            [d.dayOfWeek for d in result.remainingDays]
        )
        st.bar_chart(chart_pivot, color=["#1ca885", "#d63d00"], use_container_width=True)

        # Per-day table.
        table_df = pd.DataFrame([
            {
                "Day": d.dayOfWeek,
                "Training": d.intensity,
                "Before (kcal)": d.originalKcal,
                "After (kcal)": d.newKcal,
                "Delta": f"{d.kcalDelta:+}",
            }
            for d in result.remainingDays
        ])
        st.dataframe(table_df, hide_index=True, use_container_width=True)

        # LLM-generated copy.
        st.markdown("### How FeenX talks to the client about it")
        with st.spinner("Generating rationale copy..."):
            copy = generate_rationale(
                client=client,
                logged_meal=logged_meal,
                replaced_meal=replaced_meal,
                result=result,
            )
        for v in copy["variants"]:
            st.markdown(f"<div class='feenx-variant'>{v}</div>", unsafe_allow_html=True)

        st.markdown(
            f"<div class='feenx-card'>"
            f"<div class='feenx-stat-label'>Cascade</div>"
            f"<div style='margin-top: 6px;'>{copy['cascade']}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='feenx-card'>"
            f"<div class='feenx-stat-label'>Why it works</div>"
            f"<div style='margin-top: 6px;'>{copy['whyItWorks']}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        with st.expander("See the underlying structured output (this is what the backend returns)"):
            st.json({
                "deltaKcal": result.deltaKcal,
                "todayLoggedKcal": result.todayLoggedKcal,
                "todayTargetKcal": result.todayTargetKcal,
                "strategy": result.strategy,
                "cascadeNote": result.cascadeNote,
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
                "copy": copy,
            })


# =================================================================
# Tab 3: Generate Meal
# =================================================================

with tab_generate:
    st.markdown("### Generate a meal for this client")
    st.markdown(
        "FeenX filters a curated recipe library by the client's hard constraints "
        "(allergies, aversions, cooking time, equipment), then scores by goal, "
        "training intensity, and chronic condition. The LLM writes the \"why we "
        "picked this\" chips shown on the mobile meal-detail screen."
    )

    c1, c2 = st.columns(2)
    with c1:
        target_slot = st.selectbox("Meal slot", options=["Breakfast", "Lunch", "Snack", "Dinner"], index=1)
    with c2:
        intensity = st.selectbox(
            "Today's training intensity",
            options=["Rest", "Low", "Moderate", "High"],
            index=2,
        )

    if st.button("Pick a meal", type="primary"):
        from engine.generate import pick_meal  # local import avoids circulars
        chosen = pick_meal(recipes, client, target_slot, intensity)
        if chosen is None:
            st.error(
                "No recipe in the curated library matches this client's hard "
                "constraints for this slot. In production, the engine would fall "
                "back to LLM generation within the constraint space."
            )
        else:
            st.markdown(
                f"<div class='feenx-card'>"
                f"<div class='feenx-stat-label'>{target_slot} • {intensity} training day</div>"
                f"<div style='font-family: Georgia, serif; font-size: 28px; color: #172f29; margin-top: 4px;'>"
                f"{chosen['name']}</div>"
                f"<div style='color: #3a3a3a; margin-top: 6px;'>"
                f"{chosen['kcal']} kcal • {chosen['proteinG']}g P • "
                f"{chosen['carbsG']}g C • {chosen['fatG']}g F • "
                f"{chosen['prepMinutes']} min prep"
                f"</div></div>",
                unsafe_allow_html=True,
            )

            st.markdown("#### Why we picked this")
            with st.spinner("Writing rationale..."):
                insights = generate_insights_for_meal(client, chosen, intensity)
            for chip in insights:
                st.markdown(
                    f"<span class='feenx-pill feenx-pill-sage'>{chip}</span>",
                    unsafe_allow_html=True,
                )

            with st.expander("See why this recipe scored highest (constraint + score breakdown)"):
                from engine.generate import filter_recipes_for_client, score_recipe_for_client
                candidates = filter_recipes_for_client(recipes, client, target_slot)  # type: ignore[arg-type]
                score_df = pd.DataFrame([
                    {
                        "Recipe": r["name"],
                        "kcal": r["kcal"],
                        "Protein": r["proteinG"],
                        "Prep": r["prepMinutes"],
                        "Tags": ", ".join(r.get("tags", [])),
                        "Score": round(score_recipe_for_client(r, client, intensity), 2),
                    }
                    for r in candidates
                ]).sort_values("Score", ascending=False)
                st.dataframe(score_df, hide_index=True, use_container_width=True)

    st.divider()
    st.markdown(
        "<div class='feenx-stat-label'>How this becomes production</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "In production, the recipe library lives in the backend with vector "
        "search over thousands of recipes. The filter + score logic becomes a "
        "server action. The LLM rationale call becomes an agent with tool "
        "access to the client's full history. The mobile app calls one "
        "function and receives the same structured result you see here."
    )
