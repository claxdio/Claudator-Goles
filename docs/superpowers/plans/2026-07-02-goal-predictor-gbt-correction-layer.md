# LightGBM Correction Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a properly validated LightGBM model, trained on richer engineered features across multiple seasons, that is honestly evaluated against a held-out season and against the existing Poisson baseline — to find out whether a real, repeatable predictive edge exists before any live infrastructure gets built on top of it.

**Architecture:** Expand the historical dataset from one season to six (2018-19 through 2023-24, Premier League + Bundesliga) so there's enough data for a tree model to generalize rather than memorize. Add a richer feature-engineering function (cumulative match xG, shot quality, game state, momentum-as-trend-ratio) alongside the existing Poisson-model features — research (see Global Constraints) found these are the signals actually documented to carry predictive power, unlike the raw recent-shot-count blend that made calibration worse in the previous plan. Train a LightGBM classifier on the resulting feature set (which includes the existing Poisson prediction and trailing-xG prior as two of its inputs, so the tree can use or ignore them), calibrate it with Platt scaling fit on a validation season, and evaluate it on a completely held-out final season using the same Brier Skill Score framework already built — reporting the result honestly, whichever way it goes.

**Tech Stack:** Same as before (Python 3.11+, `sqlite3`, `pandas`, `scipy`, `pytest`) plus two new free, open-source dependencies: `lightgbm` (gradient boosted trees) and `numpy` (used directly for the Platt-scaling math).

## Global Constraints

- No paid services or API keys — `lightgbm` and `numpy` are free, open-source PyPI packages; all data continues to come from the existing free Understat/ClubElo sources.
- **Temporal discipline is non-negotiable**: the test season (`2324`, the most recent) must never be used for training or for fitting the Platt-scaling calibration — only for final evaluation. The validation season (`2223`) is used only for calibration fitting, never for training the tree model itself. This mirrors real production use (train on the past, predict the future) and avoids the kind of leakage that inflated the old `_pre_match_xg_per90` prior removed in the previous plan.
- Reuse existing infrastructure — do not duplicate logic: shot-dict shape (`{"minute", "team", "xg", "is_goal"}`) from `features.py`/`understat.py`, `BacktestResult`/`CUTOFF_MINUTES`/`DEFAULT_BLEND`/`HORIZON_MINUTES`/`RECENT_WINDOW_MINUTES` from `backtest.py`, `trailing_xg_per90` from `priors.py`, `dynamic_lambda`/`prob_goal_in_window` from `model.py`.
- Research findings that inform this design (from a literature review of in-play soccer goal-prediction and small-sample gradient-boosted-tree practice), each with a specific consequence for this plan:
  - Cumulative match-level xG, shot quality (max single-shot xG, count of shots above an xG threshold), and game-state (score differential × time remaining) are the features with the strongest real backing — the feature set in Task 2 is built around these, not around the recent-15-minute-window signal that already failed in the previous plan.
  - A team-quality/pre-match-strength signal is a well-established anchor in in-game win-probability modeling — Task 3 includes the existing `trailing_xg_per90` prior as a feature for exactly this reason (ClubElo integration is deferred; see Próximos pasos).
  - Tree models on datasets below roughly 1,500 matches tend to memorize team/season idiosyncrasies rather than generalize — Task 1 expands the historical pull to 6 seasons (≈4,100 total matches, ≈2,700+ in the training split alone) specifically to clear this threshold.
  - Recommended regularization for a low-thousands-of-matches dataset: shallow trees (`max_depth` 3-4, `num_leaves` ~15), `min_data_in_leaf` 100-200, low learning rate (0.01-0.05), moderate L2 — Task 4's `train_gbt` uses exactly these ranges.
  - Platt/sigmoid calibration is the safer default over isotonic regression at this sample size (isotonic overfits readily with only a few hundred to low-thousand calibration rows) — Task 4 implements Platt scaling, not isotonic.
  - A realistic definition of success: a Brier Skill Score in the 0.01-0.05 range against the naive baseline would be a genuine, meaningful result at this scale; anything above ~0.08-0.10 should be treated as a possible leakage bug and re-checked before being believed, not celebrated outright.
- Must run correctly on Windows/PowerShell — use `pathlib.Path`, no POSIX-only assumptions (matches prior plans' constraint).
- No network calls in any unit test (Tasks 2-4). Task 1 (multi-season ingestion) and Task 5 (end-to-end training run) are explicitly network-touching/manual-verification tasks, following the same precedent as the original `cli.py` in the foundations plan.

---

### Task 1: Multi-season historical data ingestion

**Files:**
- Create: `src/goles/ingest_history.py`

**Interfaces:**
- Consumes: `goles.db.get_connection/init_db` (existing), `goles.loaders.understat.fetch_understat_shots/persist_shots/shots_to_records` (existing, unchanged).
- Produces: a runnable script (`python -m goles.ingest_history`) that populates `data/goles.db` with 6 seasons of Premier League + Bundesliga data. No new importable functions are needed by later tasks — later tasks read from the database, not from this script.

This task has no automated tests, matching the precedent set by `src/goles/cli.py` in the foundations plan (Task 8) — it is a thin script whose only real verification is running it against the live network/cache, which the other tasks in that plan already established as the appropriate pattern for this kind of I/O-heavy, hard-to-mock script.

- [ ] **Step 1: Write the script**

`src/goles/ingest_history.py`:
```python
from __future__ import annotations

from goles.db import get_connection, init_db
from goles.loaders.understat import fetch_understat_shots, persist_shots, shots_to_records

LEAGUES = ["ENG-Premier League", "GER-Bundesliga"]
# Six seasons gives ~4,100 total matches across both leagues -- enough that
# a held-out test season and a held-out validation season each still leave
# a training split comfortably above the ~1,500-match threshold below which
# gradient-boosted trees tend to memorize rather than generalize (see this
# plan's Global Constraints).
SEASONS = ["1819", "1920", "2021", "2122", "2223", "2324"]


def main() -> None:
    conn = get_connection()
    init_db(conn)

    print(f"Descargando datos de Understat para {LEAGUES} temporadas {SEASONS}...")
    print("La primera corrida sin cache puede tardar bastante (~1 partido/seg).")
    shots_df = fetch_understat_shots(LEAGUES, SEASONS)
    records = shots_to_records(shots_df)
    print(f"{len(records)} eventos de tiro descargados. Guardando en la base de datos...")
    persist_shots(conn, records)
    print("Listo.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it for real**

Run (PowerShell, venv activated): `python -m goles.ingest_history`

Expected: this will make real, sustained network calls to Understat for seasons the local cache doesn't already have (2018-19 through 2022-23 are new; 2023-24 is already cached from the foundations phase). Budget up to an hour of wall-clock time for the uncached seasons — this is normal (Understat has no bulk endpoint, `soccerdata` fetches one match at a time). `persist_shots` is idempotent, so it's safe to re-run this script if it's interrupted; already-persisted matches are skipped. When it finishes, the database should hold roughly 4,000+ matches across the 6 seasons combined.

- [ ] **Step 3: Commit**

```powershell
git add src/goles/ingest_history.py
git commit -m "feat: add multi-season historical data ingestion script"
```

---

### Task 2: Richer feature engineering (`compute_ml_features`)

**Files:**
- Modify: `src/goles/features.py` (add a new function; do not touch `MatchState`/`compute_state_at_minute`/`goal_in_window`, which the existing Poisson baseline still depends on)
- Test: `tests/test_features.py` (append)

**Interfaces:**
- Consumes: shot dicts in the existing shape `{"minute": int, "team": "home"/"away", "xg": float, "is_goal": bool}`.
- Produces: `compute_ml_features(shots: list[dict], cutoff_minute: int, team: str) -> dict[str, float]`, returning exactly these 19 keys: `is_home, minute, minutes_remaining, score_diff, score_diff_x_minutes_remaining, own_xg_total, opp_xg_total, xg_diff, own_xg_rate, opp_xg_rate, own_max_shot_xg, opp_max_shot_xg, own_big_chances, opp_big_chances, own_recent_xg, opp_recent_xg, own_trend, own_time_since_shot, time_since_goal`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_features.py`:
```python
ML_SAMPLE_SHOTS = [
    {"minute": 5, "team": "home", "xg": 0.05, "is_goal": False},
    {"minute": 12, "team": "home", "xg": 0.30, "is_goal": False},
    {"minute": 20, "team": "away", "xg": 0.10, "is_goal": False},
    {"minute": 34, "team": "home", "xg": 0.40, "is_goal": True},
    {"minute": 50, "team": "away", "xg": 0.15, "is_goal": False},
    {"minute": 63, "team": "away", "xg": 0.55, "is_goal": True},
    {"minute": 70, "team": "home", "xg": 0.18, "is_goal": False},
]


def test_compute_ml_features_home_perspective_basic_totals():
    features = compute_ml_features(ML_SAMPLE_SHOTS, cutoff_minute=65, team="home")
    assert features["is_home"] == 1.0
    assert features["score_diff"] == 0.0  # 1-1 at minute 65
    assert abs(features["own_xg_total"] - 0.75) < 1e-9  # 0.05+0.30+0.40
    assert abs(features["opp_xg_total"] - 0.80) < 1e-9  # 0.10+0.15+0.55
    assert abs(features["xg_diff"] - (0.75 - 0.80)) < 1e-9


def test_compute_ml_features_away_perspective_mirrors_home():
    home_features = compute_ml_features(ML_SAMPLE_SHOTS, cutoff_minute=65, team="home")
    away_features = compute_ml_features(ML_SAMPLE_SHOTS, cutoff_minute=65, team="away")
    assert away_features["is_home"] == 0.0
    assert abs(away_features["own_xg_total"] - home_features["opp_xg_total"]) < 1e-9
    assert abs(away_features["opp_xg_total"] - home_features["own_xg_total"]) < 1e-9
    assert away_features["score_diff"] == -home_features["score_diff"]


def test_compute_ml_features_big_chances_uses_xg_threshold():
    features = compute_ml_features(ML_SAMPLE_SHOTS, cutoff_minute=65, team="home")
    assert features["own_big_chances"] == 2.0  # the 0.30 and 0.40 xg shots
    assert abs(features["own_max_shot_xg"] - 0.40) < 1e-9


def test_compute_ml_features_time_since_shot_and_goal():
    features = compute_ml_features(ML_SAMPLE_SHOTS, cutoff_minute=65, team="away")
    assert abs(features["own_time_since_shot"] - 2.0) < 1e-9  # away's last shot: minute 63
    assert abs(features["time_since_goal"] - 2.0) < 1e-9  # last goal overall: minute 63


def test_compute_ml_features_never_uses_shots_after_cutoff():
    features = compute_ml_features(ML_SAMPLE_SHOTS, cutoff_minute=65, team="home")
    assert abs(features["own_xg_total"] - 0.75) < 1e-9  # excludes the 0.18 shot at minute 70


def test_compute_ml_features_minutes_remaining_and_interaction_when_tied():
    features = compute_ml_features(ML_SAMPLE_SHOTS, cutoff_minute=65, team="home")
    assert features["minutes_remaining"] == 25.0  # 90 - 65
    assert features["score_diff_x_minutes_remaining"] == 0.0


def test_compute_ml_features_score_diff_interaction_when_leading():
    features = compute_ml_features(ML_SAMPLE_SHOTS, cutoff_minute=40, team="home")
    assert features["score_diff"] == 1.0  # 1-0 at minute 40
    assert features["minutes_remaining"] == 50.0
    assert features["score_diff_x_minutes_remaining"] == 50.0


def test_compute_ml_features_trend_ratio_reflects_recent_burst():
    features = compute_ml_features(ML_SAMPLE_SHOTS, cutoff_minute=40, team="home")
    own_xg_rate = 0.75 / 40  # 0.05+0.30+0.40 over 40 minutes elapsed
    own_recent_rate = 0.40 / 15  # only the minute-34 shot falls in (25,40]
    expected_trend = own_recent_rate / own_xg_rate
    assert abs(features["own_trend"] - expected_trend) < 1e-6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_features.py -v -k compute_ml_features`
Expected: FAIL with `ImportError` or `NameError: name 'compute_ml_features' is not defined`

- [ ] **Step 3: Write the implementation**

Add this function to `src/goles/features.py` (after `goal_in_window`; do not modify anything above it):

```python
def compute_ml_features(shots: list[dict], cutoff_minute: int, team: str) -> dict[str, float]:
    """Computes an engineered feature set for predicting whether `team`
    ("home" or "away") scores in the next 15 minutes, from `team`'s own
    perspective (own_* vs opp_*), using only shots with minute <=
    cutoff_minute (no look-ahead) -- the same discipline as
    `compute_state_at_minute`.

    Deliberately asymmetric by design: only `team`'s own recent-form trend
    and time-since-last-shot are included, not the opponent's -- keeping
    the feature count modest relative to the available training data (see
    this plan's Global Constraints on overfitting risk at this data scale).
    The opponent's own trend/recency is captured when this function is
    called again with `team` set to the opponent for that team's own
    prediction row.
    """
    opponent = "away" if team == "home" else "home"
    past_shots = [s for s in shots if s["minute"] <= cutoff_minute]
    own_shots = [s for s in past_shots if s["team"] == team]
    opp_shots = [s for s in past_shots if s["team"] == opponent]

    own_goals = sum(1 for s in own_shots if s["is_goal"])
    opp_goals = sum(1 for s in opp_shots if s["is_goal"])

    own_xg_total = sum(s["xg"] for s in own_shots)
    opp_xg_total = sum(s["xg"] for s in opp_shots)

    minutes_elapsed = max(cutoff_minute, 1)
    minutes_remaining = float(max(90 - cutoff_minute, 0))

    own_xg_rate = own_xg_total / minutes_elapsed
    opp_xg_rate = opp_xg_total / minutes_elapsed

    own_max_shot_xg = max((s["xg"] for s in own_shots), default=0.0)
    opp_max_shot_xg = max((s["xg"] for s in opp_shots), default=0.0)

    own_big_chances = float(sum(1 for s in own_shots if s["xg"] > 0.2))
    opp_big_chances = float(sum(1 for s in opp_shots if s["xg"] > 0.2))

    recent_window_start = cutoff_minute - 15
    own_recent_xg = sum(s["xg"] for s in own_shots if s["minute"] > recent_window_start)
    opp_recent_xg = sum(s["xg"] for s in opp_shots if s["minute"] > recent_window_start)
    own_recent_rate = own_recent_xg / 15.0
    own_trend = own_recent_rate / own_xg_rate if own_xg_rate > 0 else 0.0

    own_last_shot_minute = max((s["minute"] for s in own_shots), default=0)
    own_time_since_shot = float(cutoff_minute - own_last_shot_minute)

    goal_minutes = [s["minute"] for s in past_shots if s["is_goal"]]
    time_since_goal = float(cutoff_minute - max(goal_minutes)) if goal_minutes else float(cutoff_minute)

    score_diff = float(own_goals - opp_goals)

    return {
        "is_home": 1.0 if team == "home" else 0.0,
        "minute": float(cutoff_minute),
        "minutes_remaining": minutes_remaining,
        "score_diff": score_diff,
        "score_diff_x_minutes_remaining": score_diff * minutes_remaining,
        "own_xg_total": own_xg_total,
        "opp_xg_total": opp_xg_total,
        "xg_diff": own_xg_total - opp_xg_total,
        "own_xg_rate": own_xg_rate,
        "opp_xg_rate": opp_xg_rate,
        "own_max_shot_xg": own_max_shot_xg,
        "opp_max_shot_xg": opp_max_shot_xg,
        "own_big_chances": own_big_chances,
        "opp_big_chances": opp_big_chances,
        "own_recent_xg": own_recent_xg,
        "opp_recent_xg": opp_recent_xg,
        "own_trend": own_trend,
        "own_time_since_shot": own_time_since_shot,
        "time_since_goal": time_since_goal,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_features.py -v`
Expected: all tests pass (Phase 1's existing 4 + this task's 8 new ones)

- [ ] **Step 5: Commit**

```powershell
git add src/goles/features.py tests/test_features.py
git commit -m "feat: add richer ML feature engineering (cumulative xG, shot quality, game state)"
```

---

### Task 3: Dataset builder with season-based splitting

**Files:**
- Modify: `src/goles/backtest.py` (rename `_load_match_shots` to `load_match_shots` — drop the leading underscore since it's now used cross-module; update its one call site inside `run_backtest`; no behavior change)
- Create: `src/goles/dataset.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `goles.backtest.load_match_shots/CUTOFF_MINUTES/DEFAULT_BLEND/HORIZON_MINUTES/RECENT_WINDOW_MINUTES` (the renamed function plus existing constants), `goles.features.compute_ml_features/compute_state_at_minute/goal_in_window`, `goles.model.dynamic_lambda/prob_goal_in_window`, `goles.priors.trailing_xg_per90`.
- Produces: `FEATURE_NAMES: list[str]` (21 entries: the 19 from `compute_ml_features` plus `trailing_prior_xg` and `poisson_prob`), `DatasetRow` dataclass (`match_id: int, season: str, team: str, cutoff: int, features: dict[str, float], label: bool`), `build_dataset(conn, cutoff_minutes: list[int] = CUTOFF_MINUTES, blend: float = DEFAULT_BLEND) -> list[DatasetRow]`, `split_by_season(rows: list[DatasetRow], test_season: str, validation_season: str) -> tuple[list[DatasetRow], list[DatasetRow], list[DatasetRow]]` (returns `train, validation, test`), `rows_to_arrays(rows: list[DatasetRow]) -> tuple[list[list[float]], list[int]]`.

- [ ] **Step 1: Rename `_load_match_shots` in `backtest.py`**

In `src/goles/backtest.py`: rename the function `_load_match_shots` to `load_match_shots` (drop the underscore), and update the single call site inside `run_backtest` (`shots = _load_match_shots(conn, match_id, home_team_id, away_team_id)` becomes `shots = load_match_shots(conn, match_id, home_team_id, away_team_id)`). No other change to this function or to `run_backtest`.

- [ ] **Step 2: Run the existing suite to confirm the rename didn't break anything**

Run: `pytest tests/test_backtest.py -v`
Expected: all existing tests still pass (this is a pure rename, no behavior change)

- [ ] **Step 3: Commit the rename separately**

```powershell
git add src/goles/backtest.py
git commit -m "refactor: make load_match_shots public for reuse in dataset.py"
```

- [ ] **Step 4: Write the failing tests for the new dataset module**

`tests/test_dataset.py`:
```python
import pytest

from goles.dataset import FEATURE_NAMES, build_dataset, rows_to_arrays, split_by_season
from goles.db import get_connection, init_db
from goles.loaders.understat import persist_shots


def _seed_multi_season_matches(conn):
    """One match per season, three seasons, all featuring Team A at home,
    so build_dataset/split_by_season have something real to work with."""
    records_a = [
        {
            "match_id": 1, "league": "TEST", "season": "SeasonA", "date": "2021-08-01",
            "home_team": "Team A", "away_team": "Team B",
            "minute": 20, "team": "home", "xg": 0.3, "is_goal": False,
        },
    ]
    persist_shots(conn, records_a)

    records_b = [
        {
            "match_id": 2, "league": "TEST", "season": "SeasonB", "date": "2022-08-01",
            "home_team": "Team A", "away_team": "Team C",
            "minute": 25, "team": "home", "xg": 0.5, "is_goal": True,
        },
    ]
    persist_shots(conn, records_b)

    records_c = [
        {
            "match_id": 3, "league": "TEST", "season": "SeasonC", "date": "2023-08-01",
            "home_team": "Team A", "away_team": "Team D",
            "minute": 30, "team": "away", "xg": 0.2, "is_goal": False,
        },
    ]
    persist_shots(conn, records_c)
    conn.commit()


def test_build_dataset_produces_one_row_per_match_team_cutoff():
    conn = get_connection(":memory:")
    init_db(conn)
    _seed_multi_season_matches(conn)

    rows = build_dataset(conn, cutoff_minutes=[20, 25])
    # 3 matches * 2 teams * 2 cutoffs = 12 rows
    assert len(rows) == 12
    assert all(set(FEATURE_NAMES) == set(r.features.keys()) for r in rows)


def test_split_by_season_separates_correctly():
    conn = get_connection(":memory:")
    init_db(conn)
    _seed_multi_season_matches(conn)
    rows = build_dataset(conn, cutoff_minutes=[20])

    train, validation, test = split_by_season(rows, test_season="SeasonC", validation_season="SeasonB")
    assert all(r.season == "SeasonC" for r in test)
    assert all(r.season == "SeasonB" for r in validation)
    assert all(r.season == "SeasonA" for r in train)
    assert len(train) + len(validation) + len(test) == len(rows)


def test_split_by_season_rejects_same_test_and_validation_season():
    with pytest.raises(ValueError):
        split_by_season([], test_season="X", validation_season="X")


def test_rows_to_arrays_matches_feature_order_and_label():
    conn = get_connection(":memory:")
    init_db(conn)
    _seed_multi_season_matches(conn)
    rows = build_dataset(conn, cutoff_minutes=[20])

    X, y = rows_to_arrays(rows)
    assert len(X) == len(rows) == len(y)
    assert len(X[0]) == len(FEATURE_NAMES)
    for row, x_vec in zip(rows, X):
        assert x_vec == [row.features[name] for name in FEATURE_NAMES]
    assert set(y) <= {0, 1}
```

- [ ] **Step 5: Run tests to verify they fail**

Run: `pytest tests/test_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'goles.dataset'`

- [ ] **Step 6: Write the implementation**

`src/goles/dataset.py`:
```python
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from goles.backtest import (
    CUTOFF_MINUTES,
    DEFAULT_BLEND,
    HORIZON_MINUTES,
    RECENT_WINDOW_MINUTES,
    load_match_shots,
)
from goles.features import compute_ml_features, compute_state_at_minute, goal_in_window
from goles.model import dynamic_lambda, prob_goal_in_window
from goles.priors import trailing_xg_per90

FEATURE_NAMES = [
    "is_home",
    "minute",
    "minutes_remaining",
    "score_diff",
    "score_diff_x_minutes_remaining",
    "own_xg_total",
    "opp_xg_total",
    "xg_diff",
    "own_xg_rate",
    "opp_xg_rate",
    "own_max_shot_xg",
    "opp_max_shot_xg",
    "own_big_chances",
    "opp_big_chances",
    "own_recent_xg",
    "opp_recent_xg",
    "own_trend",
    "own_time_since_shot",
    "time_since_goal",
    "trailing_prior_xg",
    "poisson_prob",
]


@dataclass
class DatasetRow:
    match_id: int
    season: str
    team: str
    cutoff: int
    features: dict[str, float]
    label: bool


def build_dataset(
    conn: sqlite3.Connection,
    cutoff_minutes: list[int] = CUTOFF_MINUTES,
    blend: float = DEFAULT_BLEND,
) -> list[DatasetRow]:
    """Builds one row per (match, team, cutoff) across every match stored
    in the database, computing the full ML feature set (Task 2) plus the
    existing trailing-xG prior and Poisson prediction as two additional
    features, and the goal-in-next-15-minutes label."""
    rows: list[DatasetRow] = []
    matches = conn.execute(
        "SELECT match_id, home_team_id, away_team_id, league, season FROM matches"
    ).fetchall()

    for match_id, home_team_id, away_team_id, league, season in matches:
        shots = load_match_shots(conn, match_id, home_team_id, away_team_id)
        if not shots:
            continue
        for team, team_id in (("home", home_team_id), ("away", away_team_id)):
            prior = trailing_xg_per90(conn, team_id, league, season, match_id)
            for cutoff in cutoff_minutes:
                ml_features = compute_ml_features(shots, cutoff, team)

                state = compute_state_at_minute(shots, cutoff, window=RECENT_WINDOW_MINUTES)
                recent_xg = state.home_xg_last15 if team == "home" else state.away_xg_last15
                lam = dynamic_lambda(
                    pre_match_xg_per90=prior,
                    in_match_xg_recent=recent_xg,
                    recent_window_minutes=RECENT_WINDOW_MINUTES,
                    horizon_minutes=HORIZON_MINUTES,
                    blend=blend,
                )
                poisson_prob = prob_goal_in_window(lam)

                full_features = dict(ml_features)
                full_features["trailing_prior_xg"] = prior
                full_features["poisson_prob"] = poisson_prob

                label = goal_in_window(shots, cutoff, HORIZON_MINUTES, team)
                rows.append(
                    DatasetRow(
                        match_id=match_id,
                        season=season,
                        team=team,
                        cutoff=cutoff,
                        features=full_features,
                        label=label,
                    )
                )
    return rows


def split_by_season(
    rows: list[DatasetRow], test_season: str, validation_season: str
) -> tuple[list[DatasetRow], list[DatasetRow], list[DatasetRow]]:
    """Splits rows into (train, validation, test) by season: `test_season`
    is held out entirely for final evaluation, `validation_season` is used
    only for calibration fitting, and every other season is training data.
    Raises ValueError if `test_season == validation_season`."""
    if test_season == validation_season:
        raise ValueError("test_season and validation_season must differ")
    train = [r for r in rows if r.season not in (test_season, validation_season)]
    validation = [r for r in rows if r.season == validation_season]
    test = [r for r in rows if r.season == test_season]
    return train, validation, test


def rows_to_arrays(rows: list[DatasetRow]) -> tuple[list[list[float]], list[int]]:
    """Converts DatasetRows into a feature matrix (columns ordered per
    FEATURE_NAMES) and a label vector, ready for LightGBM."""
    X = [[r.features[name] for name in FEATURE_NAMES] for r in rows]
    y = [int(r.label) for r in rows]
    return X, y
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/test_dataset.py -v`
Expected: 4 passed

- [ ] **Step 8: Run the full suite**

Run: `pytest -v`
Expected: all tests across every file pass, no regressions

- [ ] **Step 9: Commit**

```powershell
git add src/goles/dataset.py tests/test_dataset.py
git commit -m "feat: add dataset builder with season-based train/validation/test splitting"
```

---

### Task 4: LightGBM training and Platt-scaling calibration

**Files:**
- Modify: `pyproject.toml` (add `lightgbm` and `numpy` dependencies)
- Create: `src/goles/gbt_model.py`
- Test: `tests/test_gbt_model.py`

**Interfaces:**
- Consumes: nothing from other tasks directly (operates on plain `list[list[float]]`/`list[int]` arrays, as produced by `dataset.rows_to_arrays` from Task 3, but doesn't import `dataset.py` itself — keeping this module reusable independent of how the arrays were built).
- Produces: `train_gbt(X_train, y_train, X_valid, y_valid) -> lgb.Booster`, `raw_predictions(booster, X) -> list[float]`, `fit_platt_scaling(raw_probs: list[float], y_true: list[int]) -> tuple[float, float]`, `apply_platt_scaling(raw_probs: list[float], a: float, b: float) -> list[float]`.

- [ ] **Step 1: Add the new dependencies**

In `pyproject.toml`, add `"lightgbm>=4.0"` and `"numpy>=1.26"` to the `dependencies` list (alongside the existing `pandas`, `requests`, `soccerdata`, `scipy`).

Run (PowerShell, venv activated): `pip install -e ".[dev]"`
Expected: installs `lightgbm` and `numpy` with no errors.

- [ ] **Step 2: Write the failing tests**

`tests/test_gbt_model.py`:
```python
import random

from goles.gbt_model import apply_platt_scaling, fit_platt_scaling, raw_predictions, train_gbt


def test_train_gbt_separates_a_simple_deterministic_pattern():
    random.seed(42)
    X_train, y_train = [], []
    for _ in range(400):
        x0 = random.random()
        X_train.append([x0, random.random()])
        y_train.append(1 if x0 > 0.5 else 0)
    X_valid, y_valid = [], []
    for _ in range(100):
        x0 = random.random()
        X_valid.append([x0, random.random()])
        y_valid.append(1 if x0 > 0.5 else 0)

    booster = train_gbt(X_train, y_train, X_valid, y_valid)
    preds = raw_predictions(booster, X_valid)
    assert all(0.0 <= p <= 1.0 for p in preds)
    correct = sum(1 for p, y in zip(preds, y_valid) if (p >= 0.5) == bool(y))
    assert correct / len(y_valid) > 0.85


def test_fit_platt_scaling_recovers_near_identity_for_already_calibrated_probs():
    random.seed(0)
    raw_probs = []
    y_true = []
    for _ in range(500):
        p = random.random()
        raw_probs.append(p)
        y_true.append(1 if random.random() < p else 0)

    a, b = fit_platt_scaling(raw_probs, y_true)
    calibrated = apply_platt_scaling(raw_probs, a, b)
    mean_abs_diff = sum(abs(c - p) for c, p in zip(calibrated, raw_probs)) / len(raw_probs)
    assert mean_abs_diff < 0.1


def test_platt_scaling_corrects_a_systematically_overconfident_model():
    random.seed(1)
    true_rate = 0.3
    n = 500
    y_true = [1 if random.random() < true_rate else 0 for _ in range(n)]
    raw_probs = [0.8] * n  # a badly overconfident constant prediction

    a, b = fit_platt_scaling(raw_probs, y_true)
    calibrated = apply_platt_scaling(raw_probs, a, b)
    assert abs(calibrated[0] - true_rate) < abs(0.8 - true_rate)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_gbt_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'goles.gbt_model'`

- [ ] **Step 4: Write the implementation**

`src/goles/gbt_model.py`:
```python
from __future__ import annotations

import lightgbm as lgb
import numpy as np
from scipy.optimize import minimize


def train_gbt(
    X_train: list[list[float]],
    y_train: list[int],
    X_valid: list[list[float]],
    y_valid: list[int],
) -> lgb.Booster:
    """Trains a LightGBM binary classifier with regularization appropriate
    for a low-thousands-of-matches dataset. Reports validation loss via
    `valid_sets` so training progress is visible, but does not rely on an
    early-stopping callback (its API differs across lightgbm versions) --
    the shallow depth, min_data_in_leaf, and L2 regularization below are
    the primary overfitting controls, with a fixed, conservative
    `num_boost_round`."""
    train_set = lgb.Dataset(X_train, label=y_train)
    valid_set = lgb.Dataset(X_valid, label=y_valid, reference=train_set)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": 15,
        "max_depth": 4,
        "min_data_in_leaf": 100,
        "learning_rate": 0.03,
        "lambda_l2": 5.0,
        "verbosity": -1,
        "seed": 42,
        "deterministic": True,
    }

    booster = lgb.train(
        params,
        train_set,
        num_boost_round=300,
        valid_sets=[valid_set],
    )
    return booster


def raw_predictions(booster: lgb.Booster, X: list[list[float]]) -> list[float]:
    """Raw (uncalibrated) predicted probabilities from the trained model,
    using all trained trees (no early stopping is used, so there is no
    'best iteration' to select)."""
    return list(booster.predict(X))


def fit_platt_scaling(raw_probs: list[float], y_true: list[int]) -> tuple[float, float]:
    """Fits a 2-parameter Platt/sigmoid calibration: calibrated_prob =
    sigmoid(a * logit(raw_prob) + b), by minimizing negative log-likelihood
    against y_true. Returns (a, b)."""
    eps = 1e-6
    clipped = [min(max(p, eps), 1 - eps) for p in raw_probs]
    logits = np.array([np.log(p / (1 - p)) for p in clipped])
    y = np.array(y_true, dtype=float)

    def neg_log_likelihood(params: np.ndarray) -> float:
        a, b = params
        z = a * logits + b
        log_sig = -np.logaddexp(0.0, -z)
        log_one_minus_sig = -np.logaddexp(0.0, z)
        return -np.sum(y * log_sig + (1 - y) * log_one_minus_sig)

    result = minimize(neg_log_likelihood, x0=np.array([1.0, 0.0]), method="Nelder-Mead")
    a, b = result.x
    return float(a), float(b)


def apply_platt_scaling(raw_probs: list[float], a: float, b: float) -> list[float]:
    """Applies a fitted Platt scaling (a, b) to raw predicted probabilities."""
    eps = 1e-6
    calibrated = []
    for p in raw_probs:
        p_clipped = min(max(p, eps), 1 - eps)
        logit = np.log(p_clipped / (1 - p_clipped))
        z = a * logit + b
        calibrated.append(float(1.0 / (1.0 + np.exp(-z))))
    return calibrated
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_gbt_model.py -v`
Expected: 3 passed

- [ ] **Step 6: Run the full suite**

Run: `pytest -v`
Expected: all tests across every file pass, no regressions

- [ ] **Step 7: Commit**

```powershell
git add pyproject.toml tests/test_gbt_model.py src/goles/gbt_model.py
git commit -m "feat: add LightGBM training and Platt-scaling calibration"
```

---

### Task 5: End-to-end training run against real multi-season data

**Files:**
- Create: `src/goles/train_gbt.py`

**Interfaces:**
- Consumes: `goles.dataset.build_dataset/split_by_season/rows_to_arrays/FEATURE_NAMES` (Task 3), `goles.gbt_model.train_gbt/raw_predictions/fit_platt_scaling/apply_platt_scaling` (Task 4), `goles.backtest.BacktestResult` (existing).

This task requires Task 1's multi-season ingestion to be complete before it can produce a meaningful result (it needs the `1819`-`2223` seasons in the database as training/validation data, in addition to the `2324` test season already present from the foundations phase).

- [ ] **Step 1: Write the script**

`src/goles/train_gbt.py`:
```python
from __future__ import annotations

from goles.backtest import BacktestResult
from goles.dataset import FEATURE_NAMES, build_dataset, rows_to_arrays, split_by_season
from goles.db import get_connection, init_db
from goles.gbt_model import apply_platt_scaling, fit_platt_scaling, raw_predictions, train_gbt

TEST_SEASON = "2324"
VALIDATION_SEASON = "2223"
# blend=0.1 was the best-performing (least-bad) Poisson configuration found
# in the calibration-improvements plan's real backtest -- used here so the
# Poisson comparison in this report reflects the strongest baseline found
# so far, not the arbitrary original default of 0.5.
POISSON_COMPARISON_BLEND = 0.1


def main() -> None:
    conn = get_connection()
    init_db(conn)

    print("Construyendo el dataset completo desde la base de datos...")
    rows = build_dataset(conn, blend=POISSON_COMPARISON_BLEND)
    print(f"{len(rows)} filas construidas.")

    train_rows, valid_rows, test_rows = split_by_season(rows, TEST_SEASON, VALIDATION_SEASON)
    print(f"Train: {len(train_rows)}  Validation: {len(valid_rows)}  Test: {len(test_rows)}")

    X_train, y_train = rows_to_arrays(train_rows)
    X_valid, y_valid = rows_to_arrays(valid_rows)
    X_test, y_test = rows_to_arrays(test_rows)

    print("Entrenando LightGBM...")
    booster = train_gbt(X_train, y_train, X_valid, y_valid)

    print("Calibrando con Platt scaling sobre el set de validacion...")
    valid_raw = raw_predictions(booster, X_valid)
    a, b = fit_platt_scaling(valid_raw, y_valid)

    print("Evaluando en la temporada de test (nunca vista durante entrenamiento ni calibracion)...")
    test_raw = raw_predictions(booster, X_test)
    test_calibrated = apply_platt_scaling(test_raw, a, b)

    gbt_result = BacktestResult(
        predicted_probs=test_calibrated,
        actual_outcomes=[bool(y) for y in y_test],
    )

    print("\n=== LightGBM (calibrado) en la temporada de test ===")
    print(f"Muestras evaluadas: {len(gbt_result.predicted_probs)}")
    print(f"Brier score: {gbt_result.brier_score:.4f}")
    print(f"Brier score (base ingenua): {gbt_result.no_skill_brier_score:.4f}")
    print(f"Brier Skill Score: {gbt_result.brier_skill_score:.4f}  (>0 = mejor que la base ingenua)")
    print("Calibracion (bin_low, prob. media predicha, frecuencia real, n):")
    for bin_low, mean_pred, mean_actual, count in gbt_result.calibration_bins():
        print(f"  [{bin_low:.1f}-{bin_low + 0.2:.1f}) pred={mean_pred:.3f} real={mean_actual:.3f} n={count}")

    poisson_test_probs = [r.features["poisson_prob"] for r in test_rows]
    poisson_result = BacktestResult(
        predicted_probs=poisson_test_probs,
        actual_outcomes=[bool(y) for y in y_test],
    )
    print(f"\n=== Poisson baseline (blend={POISSON_COMPARISON_BLEND}), misma temporada de test ===")
    print(f"Brier Skill Score (Poisson): {poisson_result.brier_skill_score:.4f}")

    print("\n=== Importancia de features (LightGBM, ganancia total) ===")
    importances = booster.feature_importance(importance_type="gain")
    for name, importance in sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1]):
        print(f"  {name}: {importance:.1f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it against the real, multi-season data**

Run (PowerShell, venv activated): `python -m goles.train_gbt`

Expected: no new network calls (all data was ingested in Task 1 and is read from `data/goles.db`). Prints train/validation/test row counts (test and validation should each be roughly 1/6th of the total, per the 6-season split), then trains, calibrates, and prints the LightGBM result's Brier Skill Score side by side with the Poisson baseline's, both evaluated on the identical `2324` test-season rows. Read the Brier Skill Score honestly: if it's meaningfully positive (per this plan's Global Constraints, roughly 0.01-0.05 is a real, expected-sized win; much higher should be treated with suspicion, not celebrated) and clearly beats the Poisson comparison, this is the model to carry into Phase 2. If it's still not positive, that is the honest result to report back, and the feature-importance printout is the next diagnostic to look at (which engineered signals did the tree actually lean on, if any).

- [ ] **Step 3: Commit**

```powershell
git add src/goles/train_gbt.py
git commit -m "feat: add end-to-end LightGBM training/evaluation script against real multi-season data"
```

## Próximos pasos (fuera de alcance de este plan)

If Task 5's real run shows a genuine, positive Brier Skill Score, the trained model needs a persistence mechanism (save/load the booster + Platt-scaling parameters, e.g. via `booster.save_model()` and a small JSON sidecar for `(a, b)`) before it can be used by Phase 2's live inference — not built here, since it depends on knowing the model is worth persisting. If the result is still flat or negative, the next lever (not attempted in this plan) is wiring ClubElo in as a genuinely independent pre-match signal alongside the trailing-xG prior (deferred from the foundations plan's own Próximos pasos, since it requires team-name matching between ClubElo and Understat, which is a real, separate piece of engineering risk not worth taking on speculatively before knowing whether the richer feature set alone already closes the gap). Also deferred: a guard against empty/non-ISO dates in `trailing_xg_per90` and a documented decision for how a prior match with zero recorded shots should be handled (both flagged in the calibration-improvements plan's final review as items to address "before `priors.py` touches live data").
