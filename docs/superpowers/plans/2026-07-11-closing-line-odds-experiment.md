# Closing-Line Odds Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a throwaway comparison script that trains two variants of the model in the same run — one with the pre-match average market odds already in production, one with football-data.co.uk's true closing-line odds — to decide whether the live-odds pipeline in Phase 2 should specifically chase "freshest possible price" or whether the pre-match-average approach already wired in is good enough.

**Architecture:** A single new script, `src/goles/experiment_closing_lines.py`, reuses the existing `fetch_odds`, `build_dataset`, `train_gbt`/`fit_platt_scaling`/`raw_predictions`/`apply_platt_scaling`, `split_by_season`, and `rows_to_arrays` building blocks unchanged. It adds one small locally-scoped matching function that deliberately duplicates `persist_odds`'s match/team-name/date-matching logic (rather than modifying production code) to resolve closing-odds columns to `match_id`s, then overwrites the 4 market-probability features on a deep copy of the dataset rows before training a second model. No schema changes, no persistence, no `FEATURE_NAMES` changes, no automated tests — this matches the repo's existing precedent for I/O-heavy one-off scripts (`ingest_odds.py`, `ingest_cards.py`).

**Tech Stack:** Unchanged — Python 3.11+, `requests`, `pandas`, `lightgbm`, `sqlite3`. No new dependencies.

## Global Constraints

- No schema changes, no new columns on `matches`, no changes to `FEATURE_NAMES` or `build_dataset` in `src/goles/dataset.py`, no persistence of closing odds anywhere.
- The script must never call `save_model` — it must not overwrite the persisted production model (`data/model/booster.txt` / `platt.json`).
- Use the same `TEST_SEASON = "2324"` / `VALIDATION_SEASON = "2223"` split as `train_gbt.py`, so results are directly comparable to the recorded baseline (BSS 0.0335 main / 0.0237 replica — see `docs/superpowers/plans/2026-07-11-goal-predictor-market-rest-features.md`).
- Closing-line columns (`AvgCH`/`AvgCD`/`AvgCA`/`AvgC>2.5`/`AvgC<2.5`) are present in football-data.co.uk CSVs for seasons `1920`–`2324` but NOT `1819` (verified) — rows/matches without them default all 4 closing-derived market features to `0.0`, the same "no market data available" convention `build_dataset` already uses.
- Decision rule: variant B (closing) beats variant A (pre-match average) by more than +0.002 BSS → real signal favoring closing-line freshness for the live pipeline. Within ±0.002 → flat, pre-match-average timing is good enough. Worse than -0.002 → favors pre-match-average timing.
- No automated tests for this script (I/O-heavy, one-off comparison — same precedent as `ingest_odds.py`/`ingest_cards.py`).

---

### Task 1: Write and run the closing-line odds comparison script

**Files:**
- Create: `src/goles/experiment_closing_lines.py`

**Interfaces:**
- Consumes: `goles.loaders.football_data.{LEAGUE_CODES, fetch_odds, compute_no_vig_probabilities, compute_no_vig_two_way, normalize_team_name}`, `goles.dataset.{FEATURE_NAMES, build_dataset, rows_to_arrays, split_by_season}`, `goles.db.{get_connection, init_db}`, `goles.gbt_model.{train_gbt, fit_platt_scaling, raw_predictions, apply_platt_scaling}`, `goles.backtest.BacktestResult`.
- Produces: nothing consumed by later tasks — this script's console output (BSS for both variants, market-feature importances, coverage %) is the deliverable itself, appended to the design spec by Task 2.

No automated tests for this task — manual verification only (Step 2 below), per the Global Constraints.

- [ ] **Step 1: Write the script**

`src/goles/experiment_closing_lines.py`:
```python
from __future__ import annotations

import copy
import sqlite3

import pandas as pd

from goles.backtest import BacktestResult
from goles.dataset import DatasetRow, FEATURE_NAMES, build_dataset, rows_to_arrays, split_by_season
from goles.db import get_connection, init_db
from goles.gbt_model import apply_platt_scaling, fit_platt_scaling, raw_predictions, train_gbt
from goles.loaders.football_data import (
    LEAGUE_CODES,
    compute_no_vig_probabilities,
    compute_no_vig_two_way,
    fetch_odds,
    normalize_team_name,
)

SEASONS = ["1819", "1920", "2021", "2122", "2223", "2324"]
TEST_SEASON = "2324"
VALIDATION_SEASON = "2223"
MARKET_FEATURE_NAMES = ["own_market_wp", "opp_market_wp", "market_draw_wp", "market_over25_wp"]


def _to_iso_date(football_data_date: str) -> str:
    """Converts football-data.co.uk's DD/MM/YYYY to our ISO YYYY-MM-DD.
    Duplicated from football_data.py's private helper of the same name --
    this script is deliberately isolated from production code, see the
    design spec at docs/superpowers/specs/2026-07-11-closing-line-odds-experiment-design.md."""
    day, month, year = football_data_date.split("/")
    if len(year) == 2:
        year = "20" + year
    return f"{year}-{month.zfill(2)}-{day.zfill(2)}"


def match_closing_odds(
    conn: sqlite3.Connection, odds_df: pd.DataFrame
) -> dict[int, tuple[float, float, float, float]]:
    """Matches football-data.co.uk rows to match_ids the same way
    persist_odds does (league + season + date + normalized home/away team
    name), but reads the CLOSING odds columns (AvgCH/AvgCD/AvgCA/
    AvgC>2.5/AvgC<2.5) instead of the pre-match average columns, and
    returns no-vig probabilities keyed by match_id instead of writing them
    to the database. Rows missing any closing column (all of season 1819,
    verified), or with no matching match_id, are simply absent from the
    returned dict -- callers must treat a missing match_id as "no closing
    data available", not an error. Deliberately duplicates persist_odds's
    matching logic instead of importing/modifying it, to keep this
    experiment fully isolated from production code."""
    closing_by_match_id: dict[int, tuple[float, float, float, float]] = {}
    required_columns = ["AvgCH", "AvgCD", "AvgCA", "AvgC>2.5", "AvgC<2.5"]
    for row_dict in odds_df.to_dict("records"):
        values = [row_dict.get(col) for col in required_columns]
        if any(v is None or (isinstance(v, float) and v != v) for v in values):
            continue
        close_home, close_draw, close_away, close_over, close_under = values

        home_name = normalize_team_name(row_dict["HomeTeam"])
        away_name = normalize_team_name(row_dict["AwayTeam"])
        date_iso = _to_iso_date(row_dict["Date"])

        home_row = conn.execute("SELECT team_id FROM teams WHERE name = ?", (home_name,)).fetchone()
        away_row = conn.execute("SELECT team_id FROM teams WHERE name = ?", (away_name,)).fetchone()
        if home_row is None or away_row is None:
            continue
        home_id, away_id = home_row[0], away_row[0]

        match_row = conn.execute(
            """SELECT match_id FROM matches
               WHERE league = ? AND season = ? AND date = ?
                 AND home_team_id = ? AND away_team_id = ?""",
            (row_dict["understat_league"], row_dict["understat_season"], date_iso, home_id, away_id),
        ).fetchone()
        if match_row is None:
            continue

        home_wp, draw_wp, away_wp = compute_no_vig_probabilities(close_home, close_draw, close_away)
        over_wp, _ = compute_no_vig_two_way(close_over, close_under)
        closing_by_match_id[match_row[0]] = (home_wp, draw_wp, away_wp, over_wp)
    return closing_by_match_id


def build_closing_variant_rows(
    rows: list[DatasetRow], closing_by_match_id: dict[int, tuple[float, float, float, float]]
) -> list[DatasetRow]:
    """Returns a deep copy of `rows` with the 4 market-probability features
    overwritten using closing-line odds where available for that match_id,
    defaulting to 0.0 (matches build_dataset's existing "no market data
    available" convention) where a match_id has no closing-odds match."""
    closing_rows = copy.deepcopy(rows)
    for row in closing_rows:
        closing = closing_by_match_id.get(row.match_id)
        home_wp, draw_wp, away_wp, over_wp = closing if closing is not None else (0.0, 0.0, 0.0, 0.0)
        row.features["own_market_wp"] = home_wp if row.team == "home" else away_wp
        row.features["opp_market_wp"] = away_wp if row.team == "home" else home_wp
        row.features["market_draw_wp"] = draw_wp
        row.features["market_over25_wp"] = over_wp
    return closing_rows


def train_and_evaluate(rows: list[DatasetRow], label: str) -> float:
    """Trains and Platt-calibrates a LightGBM model on `rows` using the
    project's standard TEST_SEASON/VALIDATION_SEASON split, prints its BSS
    and market-feature importances, and returns the test BSS for the
    caller's final comparison."""
    train_rows, valid_rows, test_rows = split_by_season(rows, TEST_SEASON, VALIDATION_SEASON)
    X_train, y_train = rows_to_arrays(train_rows)
    X_valid, y_valid = rows_to_arrays(valid_rows)
    X_test, y_test = rows_to_arrays(test_rows)

    booster = train_gbt(X_train, y_train, X_valid, y_valid)
    valid_raw = raw_predictions(booster, X_valid)
    a, b = fit_platt_scaling(valid_raw, y_valid)
    test_raw = raw_predictions(booster, X_test)
    test_calibrated = apply_platt_scaling(test_raw, a, b)

    result = BacktestResult(
        predicted_probs=test_calibrated,
        actual_outcomes=[bool(y) for y in y_test],
    )

    print(f"\n=== Variante: {label} ===")
    print(f"Brier score: {result.brier_score:.4f}")
    print(f"Brier score (base ingenua): {result.no_skill_brier_score:.4f}")
    print(f"Brier Skill Score: {result.brier_skill_score:.4f}")

    importances = dict(zip(FEATURE_NAMES, booster.feature_importance(importance_type="gain")))
    print("Importancia de features de mercado:")
    for name in MARKET_FEATURE_NAMES:
        print(f"  {name}: {importances[name]:.1f}")

    return result.brier_skill_score


def main() -> None:
    conn = get_connection()
    init_db(conn)

    print("Construyendo el dataset (variante A: cuotas pre-partido, ya en produccion)...")
    rows_pre_match = build_dataset(conn)
    print(f"{len(rows_pre_match)} filas construidas.")

    print("Descargando cuotas de football-data.co.uk (para extraer las columnas de cierre)...")
    odds_df = fetch_odds(LEAGUE_CODES, SEASONS)
    closing_by_match_id = match_closing_odds(conn, odds_df)

    distinct_match_ids = {r.match_id for r in rows_pre_match}
    matched_count = sum(1 for mid in distinct_match_ids if mid in closing_by_match_id)
    coverage = matched_count / len(distinct_match_ids) if distinct_match_ids else 0.0
    print(f"Cobertura de cuotas de cierre: {matched_count}/{len(distinct_match_ids)} partidos ({coverage:.1%}).")

    print("Construyendo variante B (cuotas de cierre)...")
    rows_closing = build_closing_variant_rows(rows_pre_match, closing_by_match_id)

    bss_pre_match = train_and_evaluate(rows_pre_match, "A - cuotas pre-partido (baseline)")
    bss_closing = train_and_evaluate(rows_closing, "B - cuotas de cierre")

    delta = bss_closing - bss_pre_match
    print("\n=== Comparacion ===")
    print(f"BSS variante A (pre-partido): {bss_pre_match:.4f}")
    print(f"BSS variante B (cierre): {bss_closing:.4f}")
    print(f"Delta (B - A): {delta:+.4f}")
    if delta > 0.002:
        print("Senal real a favor de cuotas de cierre (fuera de la banda +-0.002).")
    elif delta < -0.002:
        print("Senal real a favor de cuotas pre-partido (cierre empeora, fuera de la banda +-0.002).")
    else:
        print("Resultado plano (dentro de +-0.002): sin senal clara para priorizar cuotas de cierre en el pipeline en vivo.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it for real**

Run (PowerShell, venv activated): `python -m goles.experiment_closing_lines`

Expected: completes in well under a minute (small CSVs, two LightGBM fits on a low-thousands-of-rows dataset — comparable in cost to a single `train_gbt.py` run). Read the coverage percentage first — it should be roughly 5/6 of matches (all seasons except `1819`), so expect somewhere around 80-85% coverage. If coverage is far lower than that, do not trust the comparison — stop and investigate (check whether `fetch_odds` actually retrieved the closing columns for the affected seasons) before drawing a conclusion from the BSS delta.

- [ ] **Step 3: Sanity-check the output**

Confirm both variants printed a Brier Skill Score and that variant A's BSS is close to the recorded baseline (0.0335, main run) — if variant A's BSS differs substantially from 0.0335, something is off in how `rows_pre_match` was built (it should be numerically identical to what `train_gbt.py` produces, since it calls the same `build_dataset(conn)` with no argument changes) and the comparison should not be trusted until that's resolved.

- [ ] **Step 4: Commit**

```powershell
git add src/goles/experiment_closing_lines.py
git commit -m "feat: add closing-line vs pre-match odds comparison experiment script"
```

---

### Task 2: Record the result

**Files:**
- Modify: `docs/superpowers/specs/2026-07-11-closing-line-odds-experiment-design.md`

**Interfaces:**
- Consumes: the console output from Task 1, Step 2.

- [ ] **Step 1: Append the result**

Add a `## Resultado` section at the end of `docs/superpowers/specs/2026-07-11-closing-line-odds-experiment-design.md` with: the coverage percentage, both BSS numbers, the delta, which side of the ±0.002 band it falls on, and the market-feature-importance numbers for both variants (so a future reader can see whether closing-derived `own_market_wp` out- or under-ranks the pre-match-average version). State the concrete implication for Phase 2's live-odds pipeline design (chase closing-line freshness specifically, or pre-match-average timing is sufficient).

- [ ] **Step 2: Commit**

```powershell
git add docs/superpowers/specs/2026-07-11-closing-line-odds-experiment-design.md
git commit -m "docs: record closing-line vs pre-match odds experiment result"
```
