# FeenX — Adaptive Nutrition Demo

Interactive Streamlit demo of the AI behind FeenX:

- **Smart Swap:** deterministic math redistributes macros across the rest of the week when a client eats off-plan. LLM writes the rationale copy on top.
- **Generate a meal:** filters a curated recipe library by the client's hard constraints (allergies, aversions, cooking time, equipment), scores by goal + training intensity + chronic condition, then uses the LLM to write the "why we picked this" chips.

Three seeded clients to demo variety: Sam K. (fat loss), Diego L. (Type 2 Diabetes), Alyssa P. (PCOS).

## Run locally

```bash
cd feenx-demo
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env: add your ANTHROPIC_API_KEY
export $(grep -v '^#' .env | xargs)
streamlit run app.py
```

Browser opens at http://localhost:8501.

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to https://share.streamlit.io → New app → point at this repo, branch, `app.py`.
3. In **Settings → Secrets**, paste:
   ```
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
4. Deploy.

## What the demo shows

**Tab 1 — Client profile**
Inspect the seeded client: goal, body metrics, training week, allergies, aversions, equipment, and the full 7-day plan.

**Tab 2 — Smart swap**
1. Pick which meal got swapped (default: lunch).
2. Pick what they actually ate (Chipotle bowl, pizza slice, Cava bowl, etc.) or edit macros manually.
3. Pick redistribution strategy: **Even split** or **Smart by training day**.
4. Click **Recalculate the week**.

The app computes the delta, redistributes it across remaining days respecting safety floors (min 1500 kcal/day, min 120g protein/day), renders a before/after bar chart, shows a per-day delta table, and generates three variants of emotional copy + a cascade explanation via the LLM.

The structured output at the bottom of the tab is exactly the shape the mobile app would receive from a real backend call.

**Tab 3 — Generate a meal**
Pick a slot and training intensity. FeenX filters the curated library by hard constraints, scores every candidate by goal + condition + training context, picks the top, and the LLM writes the 3 insight chips. Expand the breakdown to see the full candidate scoring.

## Architecture

```
app.py                         Streamlit UI
engine/
  recalc.py                    Deterministic recalc math
  rationale.py                 LLM wrapper for smart-swap copy
  generate.py                  Filter + score + LLM wrapper for meal generation
data/
  clients.json                 3 seeded client profiles
  plans.json                   7-day plans per client
  swap_options.json            9 restaurant / off-plan options
  recipes.json                 30 curated meals with tags + constraints
```

**Boundary between math and LLM:** the LLM never touches the numbers. It only writes the explanation copy on top of a structured result from `recalc.py`. The math must be provable and auditable.

**Path to production:** `recalc.py` becomes a backend action. `generate.py` becomes an agent with tool access to a full recipe corpus via vector search. `rationale.py` becomes a second agent writing voice-consistent copy. The mobile app calls the backend, receives the same structured result shape, and renders identically.

## Copy voice rules (enforced in prompts)

- Warm, confident, never punitive
- Banned words: cheat, blow it, burn off, bounce back, slipped up, guilt, sin, catch up, make up for, damage, wreck
- No emoji, no exclamation marks, no hedging
- Editorial tone — wellness magazine, not startup app
