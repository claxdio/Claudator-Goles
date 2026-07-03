# Calibration Improvements (Phase 1.5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the two known issues behind the Phase 1 backtest's overconfidence at higher predicted probabilities — the look-ahead-biased pre-match prior and the untested effect of `blend`/late cutoffs — and add a no-skill baseline comparison so future reports can say whether the model has real skill, not just a Brier score in isolation.

**Architecture:** A new `src/goles/priors.py` module computes each team's trailing season-to-date average xG (using only matches strictly before the one being predicted, by date) as the pre-match prior, replacing `backtest.py`'s old same-match-xG prior. `backtest.py` gains a no-skill baseline Brier score and Brier Skill Score on `BacktestResult`, a `cutoff_minutes` parameter on `run_backtest` (so late cutoffs can be excluded from a run), and a `compare_blends`/`print_comparison` pair for sweeping `blend` values. `cli.py` is updated to print all of this against the real, already-downloaded Understat data.

**Tech Stack:** Same as Phase 1 — Python 3.11+, `sqlite3` (stdlib), `pandas`, `pytest`. No new dependencies.

## Global Constraints

- No paid services or API keys — this phase is entirely offline, reusing the existing free ClubElo/Understat data already in `data/goles.db`.
- No network calls in any test — all new tests use `:memory:` SQLite databases and directly-constructed fixtures, exactly like Phase 1's tests.
- This plan replaces `src/goles/backtest.py`'s `_pre_match_xg_per90` function (documented in the Phase 1 plan as an intentional, temporary look-ahead-biased simplification) with `src/goles/priors.py`'s `trailing_xg_per90`. Delete `_pre_match_xg_per90` entirely — do not leave it as dead code.
- Must run correctly on Windows/PowerShell (the developer's environment) — use `pathlib.Path`/parametrized SQL, no POSIX-only assumptions (matches Phase 1's constraint).
- Every new record dict used in tests that flows through `persist_shots` must include a `"date"` key (format `"YYYY-MM-DD"`), even though `persist_shots` defaults a missing one to `""` — the chronological-ordering logic this plan adds is meaningless without real, distinct dates in test fixtures.

---

### Task 1: Trailing season-to-date prior (`priors.py`)

**Files:**
- Create: `src/goles/priors.py`
- Test: `tests/test_priors.py`

**Interfaces:**
- Consumes: `goles.db.get_connection/init_db/get_or_create_team` (existing, from Phase 1).
- Produces: `team_matches_chronological(conn, team_id: int, league: str, season: str) -> list[tuple[int, str]]`, `team_match_xg(conn, match_id: int, team_id: int) -> float`, `trailing_xg_per90(conn, team_id: int, league: str, season: str, before_match_id: int) -> float`.

- [ ] **Step 1: Write the failing tests**

`tests/test_priors.py`:
```python
import pytest

from goles.db import get_connection, get_or_create_team, init_db
from goles.priors import team_matches_chronological, trailing_xg_per90


def _insert_match(conn, understat_id, league, season, date, home_id, away_id):
    conn.execute(
        """INSERT INTO matches
           (understat_id, league, season, date, home_team_id, away_team_id)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (understat_id, league, season, date, home_id, away_id),
    )
    return conn.execute(
        "SELECT match_id FROM matches WHERE understat_id = ?", (understat_id,)
    ).fetchone()[0]


def _insert_shot(conn, match_id, team_id, xg):
    conn.execute(
        "INSERT INTO shots (match_id, minute, team_id, xg, is_goal) VALUES (?, 10, ?, ?, 0)",
        (match_id, team_id, xg),
    )


def test_team_matches_chronological_orders_by_date():
    conn = get_connection(":memory:")
    init_db(conn)
    arsenal = get_or_create_team(conn, "Arsenal")
    chelsea = get_or_create_team(conn, "Chelsea")
    fulham = get_or_create_team(conn, "Fulham")
    m2 = _insert_match(conn, 2, "ENG-Premier League", "2324", "2023-09-01", arsenal, fulham)
    m1 = _insert_match(conn, 1, "ENG-Premier League", "2324", "2023-08-11", arsenal, chelsea)
    conn.commit()

    matches = team_matches_chronological(conn, arsenal, "ENG-Premier League", "2324")
    assert matches == [(m1, "2023-08-11"), (m2, "2023-09-01")]


def test_trailing_xg_per90_returns_zero_for_teams_first_match_of_season():
    conn = get_connection(":memory:")
    init_db(conn)
    arsenal = get_or_create_team(conn, "Arsenal")
    chelsea = get_or_create_team(conn, "Chelsea")
    m1 = _insert_match(conn, 1, "ENG-Premier League", "2324", "2023-08-11", arsenal, chelsea)
    conn.commit()

    assert trailing_xg_per90(conn, arsenal, "ENG-Premier League", "2324", m1) == 0.0


def test_trailing_xg_per90_averages_only_strictly_earlier_matches():
    conn = get_connection(":memory:")
    init_db(conn)
    arsenal = get_or_create_team(conn, "Arsenal")
    chelsea = get_or_create_team(conn, "Chelsea")
    fulham = get_or_create_team(conn, "Fulham")
    everton = get_or_create_team(conn, "Everton")

    m1 = _insert_match(conn, 1, "ENG-Premier League", "2324", "2023-08-11", arsenal, chelsea)
    m2 = _insert_match(conn, 2, "ENG-Premier League", "2324", "2023-08-19", arsenal, fulham)
    m3 = _insert_match(conn, 3, "ENG-Premier League", "2324", "2023-08-26", arsenal, everton)

    _insert_shot(conn, m1, arsenal, 1.0)
    _insert_shot(conn, m1, arsenal, 0.5)  # match 1 total xg for arsenal = 1.5
    _insert_shot(conn, m2, arsenal, 2.5)  # match 2 total xg for arsenal = 2.5
    _insert_shot(conn, m3, arsenal, 9.0)  # match 3 is the one being predicted -- must be excluded
    conn.commit()

    # before match 3: average of matches 1 and 2 = (1.5 + 2.5) / 2 = 2.0
    result = trailing_xg_per90(conn, arsenal, "ENG-Premier League", "2324", m3)
    assert abs(result - 2.0) < 1e-9

    # before match 2: only match 1 counts = 1.5
    result2 = trailing_xg_per90(conn, arsenal, "ENG-Premier League", "2324", m2)
    assert abs(result2 - 1.5) < 1e-9


def test_trailing_xg_per90_raises_for_unknown_match():
    conn = get_connection(":memory:")
    init_db(conn)
    arsenal = get_or_create_team(conn, "Arsenal")
    chelsea = get_or_create_team(conn, "Chelsea")
    _insert_match(conn, 1, "ENG-Premier League", "2324", "2023-08-11", arsenal, chelsea)
    conn.commit()

    with pytest.raises(ValueError):
        trailing_xg_per90(conn, arsenal, "ENG-Premier League", "2324", before_match_id=999)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_priors.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'goles.priors'`

- [ ] **Step 3: Write the implementation**

`src/goles/priors.py`:
```python
from __future__ import annotations

import sqlite3


def team_match_xg(conn: sqlite3.Connection, match_id: int, team_id: int) -> float:
    """Total shot xG recorded for `team_id` in a single match."""
    row = conn.execute(
        "SELECT COALESCE(SUM(xg), 0.0) FROM shots WHERE match_id = ? AND team_id = ?",
        (match_id, team_id),
    ).fetchone()
    return row[0]


def team_matches_chronological(
    conn: sqlite3.Connection, team_id: int, league: str, season: str
) -> list[tuple[int, str]]:
    """Returns (match_id, date) for every match `team_id` played in
    (league, season), ordered by date ascending."""
    rows = conn.execute(
        """SELECT match_id, date FROM matches
           WHERE league = ? AND season = ? AND (home_team_id = ? OR away_team_id = ?)
           ORDER BY date ASC""",
        (league, season, team_id, team_id),
    ).fetchall()
    return [(row[0], row[1]) for row in rows]


def trailing_xg_per90(
    conn: sqlite3.Connection,
    team_id: int,
    league: str,
    season: str,
    before_match_id: int,
) -> float:
    """Average xG scored by `team_id` per match, across all of its matches
    in (league, season) that occurred strictly before `before_match_id` by
    date. This is a proper pre-match prior: it never looks at
    `before_match_id` itself or any later match, unlike a same-match xG
    total.

    Returns 0.0 if there is no strictly-earlier match for this team this
    season (e.g. matchday 1) — a neutral prior rather than a crash.

    Raises ValueError if `before_match_id` is not among `team_id`'s matches
    in (league, season).
    """
    matches = team_matches_chronological(conn, team_id, league, season)
    match_dates = {mid: date for mid, date in matches}
    if before_match_id not in match_dates:
        raise ValueError(
            f"match_id={before_match_id} not found for team_id={team_id} "
            f"in league={league!r} season={season!r}"
        )
    before_date = match_dates[before_match_id]
    prior_match_ids = [mid for mid, date in matches if date < before_date]
    if not prior_match_ids:
        return 0.0
    total_xg = sum(team_match_xg(conn, mid, team_id) for mid in prior_match_ids)
    return total_xg / len(prior_match_ids)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_priors.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```powershell
git add src/goles/priors.py tests/test_priors.py
git commit -m "feat: add trailing season-to-date xG prior to replace look-ahead-biased prior"
```

---

### Task 2: No-skill baseline and Brier Skill Score

**Files:**
- Modify: `src/goles/backtest.py` (add two properties to `BacktestResult`; extend `print_report`)
- Test: `tests/test_backtest.py` (add tests; append to existing file)

**Interfaces:**
- Consumes: existing `BacktestResult` dataclass (`predicted_probs: list[float]`, `actual_outcomes: list[bool]`, existing `brier_score` property).
- Produces: `BacktestResult.no_skill_brier_score` (property, `float`), `BacktestResult.brier_skill_score` (property, `float`). `print_report`'s existing `n_bins: int = 5` parameter is unchanged; it gains two new printed lines.

- [ ] **Step 1: Write the failing tests**

Read the current top of `tests/test_backtest.py` first to see its existing imports (it already imports `BacktestResult`, `get_connection`, `init_db`, `persist_shots`, `run_backtest`, `CUTOFF_MINUTES` from earlier Phase 1 work) — add `import math` if not already present, and append these test functions to the end of the file (do not remove any existing tests):

```python
def test_no_skill_brier_score_matches_base_rate_formula():
    result = BacktestResult(
        predicted_probs=[0.1, 0.9, 0.1, 0.9],
        actual_outcomes=[False, True, True, False],
    )
    # base rate = 2/4 = 0.5 -> no-skill brier = 0.5 * 0.5 = 0.25
    assert abs(result.no_skill_brier_score - 0.25) < 1e-9


def test_no_skill_brier_score_handles_empty_data():
    result = BacktestResult(predicted_probs=[], actual_outcomes=[])
    assert math.isnan(result.no_skill_brier_score)


def test_brier_skill_score_is_one_for_perfect_predictions():
    result = BacktestResult(
        predicted_probs=[0.0, 1.0, 0.0, 1.0],
        actual_outcomes=[False, True, False, True],
    )
    assert abs(result.brier_skill_score - 1.0) < 1e-9


def test_brier_skill_score_is_zero_when_model_matches_naive_baseline():
    # always predicting exactly the base rate (0.5) makes model_brier == no_skill_brier
    result = BacktestResult(
        predicted_probs=[0.5, 0.5, 0.5, 0.5],
        actual_outcomes=[False, True, True, False],
    )
    assert abs(result.brier_skill_score - 0.0) < 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_backtest.py -v -k "no_skill or skill_score"`
Expected: FAIL with `AttributeError: 'BacktestResult' object has no attribute 'no_skill_brier_score'`

- [ ] **Step 3: Write the implementation**

Add `import math` to the top of `src/goles/backtest.py` if it isn't already imported (it currently is not — only `sqlite3` and `dataclasses.dataclass` are imported).

Add these two properties to the `BacktestResult` dataclass, immediately after the existing `brier_score` property:

```python
    @property
    def no_skill_brier_score(self) -> float:
        """Brier score of the naive baseline that always predicts the
        empirical base rate (mean of actual_outcomes) instead of using any
        live signal. This is the reference score a model must beat to have
        any real skill."""
        n = len(self.actual_outcomes)
        if n == 0:
            return float("nan")
        base_rate = sum(float(o) for o in self.actual_outcomes) / n
        return base_rate * (1.0 - base_rate)

    @property
    def brier_skill_score(self) -> float:
        """Brier Skill Score: 1 - (model_brier / no_skill_brier). Positive
        means the model beats the naive base-rate baseline; zero means it's
        exactly as good; negative means it's worse than just guessing the
        base rate for every prediction."""
        ref = self.no_skill_brier_score
        if ref == 0.0 or math.isnan(ref):
            return float("nan")
        return 1.0 - (self.brier_score / ref)
```

Update `print_report` (which currently takes `result: BacktestResult, n_bins: int = 5`) to print the two new numbers right after the existing Brier score line and before the calibration table:

```python
    print(f"Brier score (base ingenua): {result.no_skill_brier_score:.4f}")
    print(f"Brier Skill Score: {result.brier_skill_score:.4f}  (>0 = mejor que la base ingenua)")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_backtest.py -v`
Expected: all tests pass (existing Phase 1 tests + 4 new ones)

- [ ] **Step 5: Commit**

```powershell
git add src/goles/backtest.py tests/test_backtest.py
git commit -m "feat: add no-skill baseline and Brier Skill Score to backtest reporting"
```

---

### Task 3: Wire the trailing prior into `run_backtest`, add cutoff/blend sweeping

**Files:**
- Modify: `src/goles/backtest.py` (replace `_pre_match_xg_per90` usage, add `cutoff_minutes` parameter, add `compare_blends`/`print_comparison`)
- Test: `tests/test_backtest.py` (append)

**Interfaces:**
- Consumes: `goles.priors.trailing_xg_per90(conn, team_id, league, season, before_match_id) -> float` (Task 1), existing `BacktestResult` (Task 2).
- Produces: `run_backtest(conn, team: str = "home", blend: float = DEFAULT_BLEND, cutoff_minutes: list[int] = CUTOFF_MINUTES) -> BacktestResult` (gains the `cutoff_minutes` parameter; the `_pre_match_xg_per90` look-ahead-biased helper is deleted), `compare_blends(conn, team: str, blends: list[float], cutoff_minutes: list[int] = CUTOFF_MINUTES) -> dict[float, BacktestResult]`, `print_comparison(results: dict[float, BacktestResult]) -> None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_backtest.py` (add `import math` at the top if Task 2 didn't already add it, and add `from goles.priors import trailing_xg_per90` is NOT needed here — the test only calls `run_backtest`/`compare_blends`, which internally use `priors`):

```python
def _seed_two_chronological_matches(conn):
    """Two matches for the same home team (Team A) in the same
    league/season, on different dates, so the trailing prior for the
    second match must come from the first match's xG -- never from the
    second match's own shots."""
    records_m1 = [
        {
            "match_id": 501, "league": "TEST", "season": "2025-26", "date": "2025-08-01",
            "home_team": "Team A", "away_team": "Team B",
            "minute": 20, "team": "home", "xg": 3.0, "is_goal": False,
        },
    ]
    persist_shots(conn, records_m1)

    records_m2 = [
        {
            "match_id": 502, "league": "TEST", "season": "2025-26", "date": "2025-08-10",
            "home_team": "Team A", "away_team": "Team C",
            "minute": 30, "team": "home", "xg": 9.0, "is_goal": True,
        },
        {
            "match_id": 502, "league": "TEST", "season": "2025-26", "date": "2025-08-10",
            "home_team": "Team A", "away_team": "Team C",
            "minute": 78, "team": "away", "xg": 0.4, "is_goal": False,
        },
    ]
    persist_shots(conn, records_m2)
    conn.commit()


def test_run_backtest_uses_trailing_prior_not_same_match_xg():
    conn = get_connection(":memory:")
    init_db(conn)
    _seed_two_chronological_matches(conn)

    result = run_backtest(conn, team="home", cutoff_minutes=[20])
    assert len(result.predicted_probs) == 2

    # Match 501 (Team A's first match this season): trailing prior = 0.0
    # (no earlier match), but its own shot at minute 20 falls inside the
    # (5,20] rolling window, so in_match_xg_recent=3.0 drives the
    # prediction: lambda = 0.5*(3.0/15)*15 = 1.5
    expected_m501 = 1.0 - math.exp(-1.5)

    # Match 502 (Team A's second match): trailing prior = 3.0, taken from
    # match 501 -- NOT match 502's own 9.0 xG shot, which the fix must
    # never use as this match's own prior. No home shots fall inside match
    # 502's (5,20] window (its only home shot is at minute 30), so
    # in_match_xg_recent=0: lambda = 0.5*(3.0/90)*15 = 0.25
    expected_m502 = 1.0 - math.exp(-0.25)

    assert sorted(round(p, 6) for p in result.predicted_probs) == sorted(
        round(p, 6) for p in [expected_m501, expected_m502]
    )


def test_run_backtest_accepts_custom_cutoff_minutes():
    conn = get_connection(":memory:")
    init_db(conn)
    _seed_two_chronological_matches(conn)

    result = run_backtest(conn, team="home", cutoff_minutes=[10, 20, 30])
    assert len(result.predicted_probs) == 2 * 3


def test_compare_blends_returns_one_result_per_blend():
    conn = get_connection(":memory:")
    init_db(conn)
    _seed_two_chronological_matches(conn)

    results = compare_blends(conn, team="home", blends=[0.0, 0.5, 1.0], cutoff_minutes=[20])
    assert set(results.keys()) == {0.0, 0.5, 1.0}
    for result in results.values():
        assert len(result.predicted_probs) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_backtest.py -v -k "trailing_prior or custom_cutoff or compare_blends"`
Expected: FAIL — `run_backtest()` raises `TypeError: run_backtest() got an unexpected keyword argument 'cutoff_minutes'`, and `compare_blends`/`print_comparison` are undefined.

- [ ] **Step 3: Write the implementation**

In `src/goles/backtest.py`:

1. Add this import at the top: `from goles.priors import trailing_xg_per90`
2. Delete the entire `_pre_match_xg_per90` function and its docstring.
3. Replace `run_backtest` with:

```python
def run_backtest(
    conn: sqlite3.Connection,
    team: str = "home",
    blend: float = DEFAULT_BLEND,
    cutoff_minutes: list[int] = CUTOFF_MINUTES,
) -> BacktestResult:
    """Replays every stored match at each cutoff minute in `cutoff_minutes`,
    predicting P(goal in the next HORIZON_MINUTES for `team`) with the
    Poisson baseline -- using each team's trailing season-to-date average
    xG as the pre-match prior (never the match's own xG) -- and comparing
    it against what actually happened."""
    predicted_probs: list[float] = []
    actual_outcomes: list[bool] = []

    matches = conn.execute(
        "SELECT match_id, home_team_id, away_team_id, league, season FROM matches"
    ).fetchall()

    for match_id, home_team_id, away_team_id, league, season in matches:
        shots = _load_match_shots(conn, match_id, home_team_id, away_team_id)
        if not shots:
            continue
        team_id = home_team_id if team == "home" else away_team_id
        pre_match_xg = trailing_xg_per90(conn, team_id, league, season, match_id)

        for cutoff in cutoff_minutes:
            state = compute_state_at_minute(shots, cutoff, window=RECENT_WINDOW_MINUTES)
            recent_xg = state.home_xg_last15 if team == "home" else state.away_xg_last15
            lam = dynamic_lambda(
                pre_match_xg_per90=pre_match_xg,
                in_match_xg_recent=recent_xg,
                recent_window_minutes=RECENT_WINDOW_MINUTES,
                horizon_minutes=HORIZON_MINUTES,
                blend=blend,
            )
            predicted_probs.append(prob_goal_in_window(lam))
            actual_outcomes.append(goal_in_window(shots, cutoff, HORIZON_MINUTES, team))

    return BacktestResult(predicted_probs=predicted_probs, actual_outcomes=actual_outcomes)


def compare_blends(
    conn: sqlite3.Connection,
    team: str,
    blends: list[float],
    cutoff_minutes: list[int] = CUTOFF_MINUTES,
) -> dict[float, BacktestResult]:
    """Runs run_backtest once per blend value and returns each result keyed
    by the blend used, so different blend settings can be compared on the
    same data."""
    return {
        blend: run_backtest(conn, team=team, blend=blend, cutoff_minutes=cutoff_minutes)
        for blend in blends
    }


def print_comparison(results: dict[float, BacktestResult]) -> None:
    print(f"{'blend':>6} | {'brier':>8} | {'no-skill':>8} | {'BSS':>8} | {'n':>6}")
    for blend, result in sorted(results.items()):
        print(
            f"{blend:6.2f} | {result.brier_score:8.4f} | "
            f"{result.no_skill_brier_score:8.4f} | {result.brier_skill_score:8.4f} | "
            f"{len(result.predicted_probs):6d}"
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_backtest.py -v`
Expected: all tests pass (Phase 1 tests + Task 2's tests + Task 3's new tests)

- [ ] **Step 5: Run the full suite**

Run: `pytest -v`
Expected: all tests across every file pass, no regressions

- [ ] **Step 6: Commit**

```powershell
git add src/goles/backtest.py tests/test_backtest.py
git commit -m "feat: wire trailing prior into run_backtest, add cutoff/blend sweeping"
```

---

### Task 4: Report the real before/after numbers via the CLI

**Files:**
- Modify: `src/goles/cli.py`

**Interfaces:**
- Consumes: `run_backtest`, `compare_blends`, `print_comparison`, `print_report` (all from Task 2/3).

- [ ] **Step 1: Update the CLI**

Replace `src/goles/cli.py`'s contents with:

```python
from __future__ import annotations

from goles.backtest import compare_blends, print_comparison, print_report, run_backtest
from goles.db import get_connection, init_db
from goles.loaders.understat import fetch_understat_shots, persist_shots, shots_to_records

LEAGUES = ["ENG-Premier League", "GER-Bundesliga"]
SEASONS = ["2324"]
# Cutoffs 75 and 80 project a 15-minute goal window past most matches' real
# playing time (regulation ends at 90 + stoppage), which structurally lowers
# the true positive rate for those two cutoffs specifically -- excluded here
# to see whether that alone explains some of the high-probability overconfidence.
EARLY_CUTOFFS = list(range(20, 71, 5))


def main() -> None:
    conn = get_connection()
    init_db(conn)

    print(f"Descargando datos de Understat para {LEAGUES} temporada {SEASONS}...")
    shots_df = fetch_understat_shots(LEAGUES, SEASONS)
    records = shots_to_records(shots_df)
    print(f"{len(records)} eventos de tiro descargados. Guardando en la base de datos...")
    persist_shots(conn, records)

    print("\n=== Backtest con blend=0.5 (default), todos los cortes ===")
    result = run_backtest(conn, team="home")
    print_report(result)

    print("\n=== Backtest con blend=0.5, excluyendo cortes tardios (75, 80) ===")
    result_early = run_backtest(conn, team="home", cutoff_minutes=EARLY_CUTOFFS)
    print_report(result_early)

    print("\n=== Comparacion de valores de blend (cortes tempranos) ===")
    comparison = compare_blends(
        conn, team="home", blends=[0.1, 0.3, 0.5, 0.7, 0.9], cutoff_minutes=EARLY_CUTOFFS
    )
    print_comparison(comparison)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it against the real, already-cached data**

Run (PowerShell, venv activated): `python -m goles.cli`

Expected: this reuses the local Understat cache (no new network calls, should complete in well under a minute) and `persist_shots` is a no-op for all 686 already-persisted matches (per Phase 1's idempotency fix). It prints three sections: the default full-range backtest (compare its Brier Skill Score against Phase 1's un-fixed run — it should improve now that the prior no longer leaks future information), the early-cutoffs-only backtest, and the blend comparison table. Read the Brier Skill Score in each: if it's still ≤ 0 anywhere, that combination is no better than guessing the historical base rate and should not be used for live alerting later. Note which blend value (if any) gets closest to a positive, stable Brier Skill Score across both cutoff sets — that is the value to carry into Phase 2, not the current default of 0.5.

- [ ] **Step 3: Commit**

```powershell
git add src/goles/cli.py
git commit -m "feat: report no-skill baseline and blend/cutoff comparison from the real backtest"
```

## Próximos pasos (fuera de alcance de este plan)

If the Brier Skill Score comparison in Task 4 shows the Poisson baseline (with its best blend/cutoff setting) reliably beats the no-skill baseline, that combination becomes the default for Phase 2 and this offline validation loop is done. If it does NOT clearly beat the no-skill baseline even after these fixes, the honest next step is not more Poisson tuning — it's the LightGBM correction layer from the original architecture plan (trained via walk-forward validation on the residual between the Poisson prediction and the actual outcome), and/or wiring ClubElo (built in Phase 1's Task 3, still unused) in as a second, independent pre-match signal alongside the trailing xG average. Either way, Phase 2 (live ingestion + Telegram) should use whichever cutoff range and blend value this plan's real run shows has genuine skill — not the untested Phase 1 defaults.
