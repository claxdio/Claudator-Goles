# Fundamentos del Predictor de Goles (histórico + modelo base + backtest) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the offline, zero-cost foundation of the live goal-predictor: a SQLite schema, free historical data loaders (ClubElo ratings, Understat shot-level events), a rolling-window feature engine, a calibrated Poisson baseline goal-probability model, and a backtest harness that reports Brier score and calibration — all runnable and testable without any live scraping, Betfair, or Telegram involved.

**Architecture:** A small Python package (`src/goles`) with one file per responsibility (db, loaders/elo, loaders/understat, features, model, backtest). Historical shot-level events (minute, team, xG, goal/no-goal) are the core data unit; the feature engine reconstructs "what the model would have seen" at any minute of a historical match, and the backtest harness replays every stored match at multiple cutoff minutes to check whether the Poisson baseline's predicted probability of "goal in the next 15 minutes" matches what actually happened.

**Tech Stack:** Python 3.11+, `sqlite3` (stdlib), `pandas`, `requests`, `soccerdata` (wraps ClubElo/Understat scraping so we don't reinvent it), `scipy` (available for later phases), `pytest` for tests.

## Global Constraints

- No paid services or API keys anywhere in this phase — every data source used here (ClubElo, Understat via `soccerdata`) is free and requires no authentication.
- All network-touching functions (`requests.get`, `soccerdata.Understat(...)`) must be unit-tested via mocking — no test in this plan may require real network access.
- Storage is SQLite under `data/goles.db` (the `data/` directory is gitignored — this is generated state, not source).
- Must run correctly on Windows/PowerShell (the developer's environment) — avoid POSIX-only path assumptions; use `pathlib.Path` everywhere.
- Free Understat shot-level data only covers 6 leagues (ENG-Premier League, ESP-La Liga, GER-Bundesliga, ITA-Serie A, FRA-Ligue 1, RUS-Premier League). Of the Tier 1/2 goal-friendly leagues identified in the architecture plan, only **ENG-Premier League** and **GER-Bundesliga** are covered — this phase's historical training/backtest data is scoped to those two leagues only. The other target leagues (Eredivisie, Eliteserien, Danish Superliga, Belgian Pro League, Swiss Super League, Scottish Premiership) have no free minute-level historical shot data and will be bootstrapped later from live production logging (documented as follow-up, not built in this plan).
- `run_backtest`'s pre-match xG prior (Task 7) uses each match's own total shot xG as a stand-in for a "pre-match expectation" — this is a **known simplification with look-ahead bias** (it technically uses full-match information as if it were known beforehand). It is acceptable for this foundations phase, whose purpose is to validate the pipeline and the Poisson math end-to-end, but it must be replaced before any live/production use with a genuine pre-match prior (e.g., each team's trailing season-to-date average xG per 90, computed only from strictly earlier matches). This replacement is out of scope for this plan and is listed under "Próximos pasos" at the end.

---

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/goles/__init__.py`
- Create: `src/goles/loaders/__init__.py`
- Create: `tests/__init__.py`

**Interfaces:**
- Produces: an installable `goles` package (`pip install -e ".[dev]"`) and a working `pytest` command that later tasks add tests to.

- [ ] **Step 1: Create the directory structure and config files**

`pyproject.toml`:
```toml
[project]
name = "goles"
version = "0.1.0"
description = "Live football goal-probability predictor: free historical data, baseline model, backtesting."
requires-python = ">=3.11"
dependencies = [
    "pandas>=2.2",
    "requests>=2.31",
    "soccerdata>=1.9",
    "scipy>=1.13",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]

[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
```

`.gitignore`:
```
data/
__pycache__/
*.pyc
.venv/
*.egg-info/
.pytest_cache/
```

`src/goles/__init__.py`: empty file.
`src/goles/loaders/__init__.py`: empty file.
`tests/__init__.py`: empty file.

- [ ] **Step 2: Create a virtual environment and install the package in editable/dev mode**

Run (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```
Expected: installs `pandas`, `requests`, `soccerdata`, `scipy`, `pytest` and the `goles` package with no errors.

- [ ] **Step 3: Verify pytest runs with zero tests**

Run: `pytest -v`
Expected: `no tests ran` (exit code 0 or 5, not an error/crash) — confirms the package and pytest config are wired correctly before any test files exist.

- [ ] **Step 4: Initialize git and commit**

```powershell
git init
git add pyproject.toml .gitignore src tests
git commit -m "chore: scaffold goles package"
```

---

### Task 2: Database schema

**Files:**
- Create: `src/goles/db.py`
- Test: `tests/test_db.py`

**Interfaces:**
- Produces: `get_connection(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection`, `init_db(conn: sqlite3.Connection) -> None`, `get_or_create_team(conn: sqlite3.Connection, name: str) -> int`.
- Consumes: nothing from other tasks.

- [ ] **Step 1: Write the failing tests**

`tests/test_db.py`:
```python
from goles.db import get_connection, init_db, get_or_create_team


def test_init_db_creates_expected_tables():
    conn = get_connection(":memory:")
    init_db(conn)
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cur.fetchall()}
    assert {"teams", "matches", "shots", "elo_ratings"} <= tables


def test_get_or_create_team_is_idempotent():
    conn = get_connection(":memory:")
    init_db(conn)
    id1 = get_or_create_team(conn, "Arsenal")
    id2 = get_or_create_team(conn, "Arsenal")
    assert id1 == id2
    id3 = get_or_create_team(conn, "Chelsea")
    assert id3 != id1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_db.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'goles.db'`

- [ ] **Step 3: Write the implementation**

`src/goles/db.py`:
```python
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Union

DEFAULT_DB_PATH = Path("data/goles.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS teams (
    team_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS matches (
    match_id INTEGER PRIMARY KEY AUTOINCREMENT,
    understat_id INTEGER UNIQUE,
    league TEXT NOT NULL,
    season TEXT NOT NULL,
    date TEXT NOT NULL,
    home_team_id INTEGER NOT NULL REFERENCES teams(team_id),
    away_team_id INTEGER NOT NULL REFERENCES teams(team_id),
    home_goals INTEGER NOT NULL DEFAULT 0,
    away_goals INTEGER NOT NULL DEFAULT 0,
    home_elo_wp REAL,
    away_elo_wp REAL,
    draw_elo_wp REAL
);

CREATE TABLE IF NOT EXISTS shots (
    shot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL REFERENCES matches(match_id),
    minute INTEGER NOT NULL,
    team_id INTEGER NOT NULL REFERENCES teams(team_id),
    xg REAL NOT NULL,
    is_goal INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS elo_ratings (
    team_name TEXT NOT NULL,
    date TEXT NOT NULL,
    elo REAL NOT NULL,
    PRIMARY KEY (team_name, date)
);
"""


def get_connection(db_path: Union[str, Path] = DEFAULT_DB_PATH) -> sqlite3.Connection:
    if db_path != ":memory:":
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def get_or_create_team(conn: sqlite3.Connection, name: str) -> int:
    cur = conn.execute("SELECT team_id FROM teams WHERE name = ?", (name,))
    row = cur.fetchone()
    if row:
        return row[0]
    cur = conn.execute("INSERT INTO teams (name) VALUES (?)", (name,))
    conn.commit()
    return cur.lastrowid
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_db.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```powershell
git add src/goles/db.py tests/test_db.py
git commit -m "feat: add sqlite schema and team upsert helper"
```

---

### Task 3: ClubElo loader

**Files:**
- Create: `src/goles/loaders/elo.py`
- Test: `tests/test_elo_loader.py`

**Interfaces:**
- Produces: `fetch_elo_ratings(date: str) -> pd.DataFrame`, `elo_win_draw_probabilities(home_elo: float, away_elo: float, home_advantage: float = 100.0) -> tuple[float, float, float]`.
- Consumes: nothing from other tasks (standalone HTTP + math).

- [ ] **Step 1: Write the failing tests**

`tests/test_elo_loader.py`:
```python
from unittest.mock import Mock, patch

from goles.loaders.elo import elo_win_draw_probabilities, fetch_elo_ratings

SAMPLE_CSV = (
    "Rank,Club,Country,Level,Elo,From,To\n"
    "1,Man City,ENG,1,2010.5,2026-06-25,2026-07-02\n"
    "2,Bayern,GER,1,1980.2,2026-06-25,2026-07-02\n"
)


def test_fetch_elo_ratings_parses_csv():
    mock_response = Mock()
    mock_response.text = SAMPLE_CSV
    mock_response.raise_for_status = Mock()
    with patch("goles.loaders.elo.requests.get", return_value=mock_response) as mock_get:
        df = fetch_elo_ratings("2026-07-02")
    mock_get.assert_called_once_with("http://api.clubelo.com/2026-07-02", timeout=30)
    assert list(df["Club"]) == ["Man City", "Bayern"]
    assert df.loc[df["Club"] == "Man City", "Elo"].iloc[0] == 2010.5


def test_elo_win_draw_probabilities_favor_stronger_team():
    home_win, draw, away_win = elo_win_draw_probabilities(home_elo=2000, away_elo=1700)
    assert home_win > away_win
    assert abs((home_win + draw + away_win) - 1.0) < 1e-9


def test_elo_win_draw_probabilities_close_match_has_home_edge_and_nonzero_draw():
    home_win, draw, away_win = elo_win_draw_probabilities(home_elo=1800, away_elo=1800)
    assert home_win > away_win
    assert draw > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_elo_loader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'goles.loaders.elo'`

- [ ] **Step 3: Write the implementation**

`src/goles/loaders/elo.py`:
```python
from __future__ import annotations

import io

import pandas as pd
import requests

CLUBELO_BASE_URL = "http://api.clubelo.com"


def fetch_elo_ratings(date: str) -> pd.DataFrame:
    """Fetch all clubs' Elo ratings as of `date` (format YYYY-MM-DD) from
    ClubElo's free, unauthenticated CSV endpoint."""
    response = requests.get(f"{CLUBELO_BASE_URL}/{date}", timeout=30)
    response.raise_for_status()
    return pd.read_csv(io.StringIO(response.text))


def elo_win_draw_probabilities(
    home_elo: float, away_elo: float, home_advantage: float = 100.0
) -> tuple[float, float, float]:
    """Convert two Elo ratings into (home_win_prob, draw_prob, away_win_prob)
    using the standard Elo logistic expectation, adjusted for a fixed
    home-advantage offset (in Elo points) and a fixed draw band."""
    diff = (home_elo + home_advantage) - away_elo
    home_or_draw_edge = 1 / (1 + 10 ** (-diff / 400))
    draw_band = 0.25
    home_win = max(home_or_draw_edge - draw_band / 2, 0.0)
    away_win = max((1 - home_or_draw_edge) - draw_band / 2, 0.0)
    draw = max(1.0 - home_win - away_win, 0.0)
    total = home_win + draw + away_win
    return home_win / total, draw / total, away_win / total
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_elo_loader.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```powershell
git add src/goles/loaders/elo.py tests/test_elo_loader.py
git commit -m "feat: add free ClubElo loader and win/draw probability conversion"
```

---

### Task 4: Understat shots loader and persistence

**Files:**
- Create: `src/goles/loaders/understat.py`
- Test: `tests/test_understat_loader.py`

**Interfaces:**
- Consumes: `goles.db.get_or_create_team(conn, name) -> int` (Task 2).
- Produces: `fetch_understat_shots(leagues: list[str], seasons: list[str]) -> pd.DataFrame`, `shots_to_records(shots_df: pd.DataFrame) -> list[dict]` (each record: `match_id, league, season, home_team, away_team, minute, team ("home"/"away"), xg, is_goal`), `persist_shots(conn: sqlite3.Connection, records: list[dict]) -> None`.

- [ ] **Step 1: Write the failing tests**

`tests/test_understat_loader.py`:
```python
from unittest.mock import MagicMock, patch

import pandas as pd

from goles.db import get_connection, init_db
from goles.loaders.understat import (
    fetch_understat_shots,
    persist_shots,
    shots_to_records,
)


def test_fetch_understat_shots_delegates_to_soccerdata_reader():
    fake_df = pd.DataFrame([{"game_id": 1}])
    mock_reader = MagicMock()
    mock_reader.read_shot_events.return_value = fake_df
    with patch("goles.loaders.understat.sd.Understat", return_value=mock_reader) as mock_cls:
        result = fetch_understat_shots(["ENG-Premier League"], ["2023-24"])
    mock_cls.assert_called_once_with(leagues=["ENG-Premier League"], seasons=["2023-24"])
    assert result is fake_df


def test_shots_to_records_normalizes_team_side_and_goal_flag():
    df = pd.DataFrame(
        [
            {
                "game_id": 101, "league": "ENG-Premier League", "season": "2023-24",
                "home_team": "Arsenal", "away_team": "Chelsea", "team": "Arsenal",
                "minute": 23, "xG": 0.15, "result": "MissedShots",
            },
            {
                "game_id": 101, "league": "ENG-Premier League", "season": "2023-24",
                "home_team": "Arsenal", "away_team": "Chelsea", "team": "Chelsea",
                "minute": 41, "xG": 0.42, "result": "Goal",
            },
        ]
    )
    records = shots_to_records(df)
    assert records[0]["team"] == "home"
    assert records[0]["is_goal"] is False
    assert records[1]["team"] == "away"
    assert records[1]["is_goal"] is True
    assert records[1]["xg"] == 0.42


def test_persist_shots_creates_match_and_shots_and_updates_score():
    conn = get_connection(":memory:")
    init_db(conn)
    records = [
        {
            "match_id": 101, "league": "ENG-Premier League", "season": "2023-24",
            "home_team": "Arsenal", "away_team": "Chelsea",
            "minute": 23, "team": "home", "xg": 0.15, "is_goal": False,
        },
        {
            "match_id": 101, "league": "ENG-Premier League", "season": "2023-24",
            "home_team": "Arsenal", "away_team": "Chelsea",
            "minute": 41, "team": "away", "xg": 0.42, "is_goal": True,
        },
    ]
    persist_shots(conn, records)
    row = conn.execute(
        "SELECT home_goals, away_goals FROM matches WHERE understat_id = 101"
    ).fetchone()
    assert row == (0, 1)
    shot_count = conn.execute("SELECT COUNT(*) FROM shots").fetchone()[0]
    assert shot_count == 2


def test_persist_shots_is_idempotent_for_the_same_match_id():
    conn = get_connection(":memory:")
    init_db(conn)
    record = {
        "match_id": 202, "league": "ENG-Premier League", "season": "2023-24",
        "home_team": "Arsenal", "away_team": "Chelsea",
        "minute": 10, "team": "home", "xg": 0.2, "is_goal": False,
    }
    persist_shots(conn, [record])
    persist_shots(conn, [record])
    match_count = conn.execute(
        "SELECT COUNT(*) FROM matches WHERE understat_id = 202"
    ).fetchone()[0]
    assert match_count == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_understat_loader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'goles.loaders.understat'`

- [ ] **Step 3: Write the implementation**

`src/goles/loaders/understat.py`:
```python
from __future__ import annotations

import sqlite3

import pandas as pd
import soccerdata as sd

from goles.db import get_or_create_team


def fetch_understat_shots(leagues: list[str], seasons: list[str]) -> pd.DataFrame:
    """Fetch shot-level events for the given leagues/seasons from Understat
    via the `soccerdata` wrapper. Returns whatever DataFrame the reader
    produces (at least: game_id, minute, team, home_team, away_team, xG,
    result)."""
    reader = sd.Understat(leagues=leagues, seasons=seasons)
    return reader.read_shot_events()


def shots_to_records(shots_df: pd.DataFrame) -> list[dict]:
    """Normalize an Understat shot-events DataFrame into plain dict records
    with keys: match_id, league, season, home_team, away_team, minute,
    team ("home"/"away"), xg, is_goal."""
    records = []
    for row in shots_df.itertuples(index=False):
        row_dict = row._asdict()
        team_side = "home" if row_dict["team"] == row_dict["home_team"] else "away"
        records.append(
            {
                "match_id": row_dict["game_id"],
                "league": row_dict["league"],
                "season": row_dict["season"],
                "home_team": row_dict["home_team"],
                "away_team": row_dict["away_team"],
                "minute": int(row_dict["minute"]),
                "team": team_side,
                "xg": float(row_dict["xG"]),
                "is_goal": row_dict["result"] == "Goal",
            }
        )
    return records


def persist_shots(conn: sqlite3.Connection, records: list[dict]) -> None:
    """Insert normalized shot records into the database, creating the
    parent match row (and teams) on first sight of a given `match_id`, and
    incrementing the match's goal tally for every shot with is_goal=True.
    Safe to call repeatedly with overlapping records for the same match_id
    (existing matches are reused, not duplicated), but re-persisting the
    same shot rows will duplicate rows in `shots` — callers should persist
    each match's shots exactly once."""
    for rec in records:
        home_id = get_or_create_team(conn, rec["home_team"])
        away_id = get_or_create_team(conn, rec["away_team"])

        row = conn.execute(
            "SELECT match_id FROM matches WHERE understat_id = ?", (rec["match_id"],)
        ).fetchone()
        if row is None:
            conn.execute(
                """INSERT INTO matches
                   (understat_id, league, season, date, home_team_id, away_team_id)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (rec["match_id"], rec["league"], rec["season"], "", home_id, away_id),
            )
            row = conn.execute(
                "SELECT match_id FROM matches WHERE understat_id = ?", (rec["match_id"],)
            ).fetchone()
        match_id = row[0]

        team_id = home_id if rec["team"] == "home" else away_id
        conn.execute(
            "INSERT INTO shots (match_id, minute, team_id, xg, is_goal) VALUES (?, ?, ?, ?, ?)",
            (match_id, rec["minute"], team_id, rec["xg"], int(rec["is_goal"])),
        )
        if rec["is_goal"]:
            goal_col = "home_goals" if rec["team"] == "home" else "away_goals"
            conn.execute(
                f"UPDATE matches SET {goal_col} = {goal_col} + 1 WHERE match_id = ?",
                (match_id,),
            )
    conn.commit()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_understat_loader.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```powershell
git add src/goles/loaders/understat.py tests/test_understat_loader.py
git commit -m "feat: add Understat shot-event loader and DB persistence"
```

---

### Task 5: Feature engine (rolling-window match state)

**Files:**
- Create: `src/goles/features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Produces: `MatchState` dataclass (`minute, home_score, away_score, home_xg_last15, away_xg_last15, home_shots_last15, away_shots_last15`), `compute_state_at_minute(shots: list[dict], cutoff_minute: int, window: int = 15) -> MatchState`, `goal_in_window(shots: list[dict], cutoff_minute: int, horizon: int, team: str) -> bool`.
- Consumes: shot dicts in the same shape produced by `shots_to_records` (Task 4): `{"minute": int, "team": "home"/"away", "xg": float, "is_goal": bool}`.

- [ ] **Step 1: Write the failing tests**

`tests/test_features.py`:
```python
from goles.features import compute_state_at_minute, goal_in_window

SAMPLE_SHOTS = [
    {"minute": 10, "team": "home", "xg": 0.10, "is_goal": False},
    {"minute": 34, "team": "home", "xg": 0.35, "is_goal": True},
    {"minute": 50, "team": "away", "xg": 0.20, "is_goal": False},
    {"minute": 63, "team": "away", "xg": 0.55, "is_goal": True},
    {"minute": 70, "team": "home", "xg": 0.18, "is_goal": False},
    {"minute": 74, "team": "home", "xg": 0.60, "is_goal": True},
]


def test_compute_state_at_minute_counts_goals_scored_so_far():
    state = compute_state_at_minute(SAMPLE_SHOTS, cutoff_minute=65, window=15)
    assert state.home_score == 1
    assert state.away_score == 1


def test_compute_state_at_minute_rolling_window_only_includes_recent_shots():
    state = compute_state_at_minute(SAMPLE_SHOTS, cutoff_minute=65, window=15)
    # window is (50, 65]; only the away shot at minute 63 falls inside it
    assert state.away_shots_last15 == 1
    assert abs(state.away_xg_last15 - 0.55) < 1e-9
    assert state.home_shots_last15 == 0
    assert abs(state.home_xg_last15 - 0.0) < 1e-9


def test_goal_in_window_detects_future_goal_for_team():
    assert goal_in_window(SAMPLE_SHOTS, cutoff_minute=65, horizon=10, team="home") is True
    assert goal_in_window(SAMPLE_SHOTS, cutoff_minute=65, horizon=10, team="away") is False


def test_goal_in_window_excludes_goals_outside_horizon():
    # the home goal at minute 34 is outside the (20, 30] window
    assert goal_in_window(SAMPLE_SHOTS, cutoff_minute=20, horizon=10, team="home") is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'goles.features'`

- [ ] **Step 3: Write the implementation**

`src/goles/features.py`:
```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MatchState:
    minute: int
    home_score: int
    away_score: int
    home_xg_last15: float
    away_xg_last15: float
    home_shots_last15: int
    away_shots_last15: int


def compute_state_at_minute(
    shots: list[dict], cutoff_minute: int, window: int = 15
) -> MatchState:
    """Reconstruct match state as of `cutoff_minute`, using only shots with
    minute <= cutoff_minute (so a backtest never sees the future)."""
    home_score = sum(
        1 for s in shots if s["minute"] <= cutoff_minute and s["team"] == "home" and s["is_goal"]
    )
    away_score = sum(
        1 for s in shots if s["minute"] <= cutoff_minute and s["team"] == "away" and s["is_goal"]
    )

    window_start = cutoff_minute - window
    home_xg = sum(
        s["xg"] for s in shots if window_start < s["minute"] <= cutoff_minute and s["team"] == "home"
    )
    away_xg = sum(
        s["xg"] for s in shots if window_start < s["minute"] <= cutoff_minute and s["team"] == "away"
    )
    home_shots = sum(
        1 for s in shots if window_start < s["minute"] <= cutoff_minute and s["team"] == "home"
    )
    away_shots = sum(
        1 for s in shots if window_start < s["minute"] <= cutoff_minute and s["team"] == "away"
    )

    return MatchState(
        minute=cutoff_minute,
        home_score=home_score,
        away_score=away_score,
        home_xg_last15=home_xg,
        away_xg_last15=away_xg,
        home_shots_last15=home_shots,
        away_shots_last15=away_shots,
    )


def goal_in_window(shots: list[dict], cutoff_minute: int, horizon: int, team: str) -> bool:
    """Did `team` score a goal in (cutoff_minute, cutoff_minute + horizon]?
    Used only to *label* historical data for backtesting/training — never
    call this with information a live model wouldn't have yet."""
    return any(
        s["team"] == team and s["is_goal"] and cutoff_minute < s["minute"] <= cutoff_minute + horizon
        for s in shots
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_features.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```powershell
git add src/goles/features.py tests/test_features.py
git commit -m "feat: add rolling-window match state and goal-in-window labeling"
```

---

### Task 6: Poisson baseline model

**Files:**
- Create: `src/goles/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Produces: `dynamic_lambda(pre_match_xg_per90: float, in_match_xg_recent: float, recent_window_minutes: int, horizon_minutes: int, blend: float = 0.5) -> float`, `prob_goal_in_window(expected_goals: float) -> float`.
- Consumes: nothing from other tasks (pure math).

- [ ] **Step 1: Write the failing tests**

`tests/test_model.py`:
```python
import math

import pytest

from goles.model import dynamic_lambda, prob_goal_in_window


def test_dynamic_lambda_pure_prior_when_blend_zero():
    lam = dynamic_lambda(
        pre_match_xg_per90=1.8, in_match_xg_recent=5.0,
        recent_window_minutes=15, horizon_minutes=15, blend=0.0,
    )
    assert abs(lam - (1.8 / 90 * 15)) < 1e-9


def test_dynamic_lambda_pure_observed_when_blend_one():
    lam = dynamic_lambda(
        pre_match_xg_per90=1.8, in_match_xg_recent=0.6,
        recent_window_minutes=15, horizon_minutes=15, blend=1.0,
    )
    assert abs(lam - 0.6) < 1e-9


def test_dynamic_lambda_rejects_non_positive_window():
    with pytest.raises(ValueError):
        dynamic_lambda(1.8, 0.5, recent_window_minutes=0, horizon_minutes=15)


def test_prob_goal_in_window_matches_poisson_formula():
    assert abs(prob_goal_in_window(0.0) - 0.0) < 1e-9
    lam = 0.4
    assert abs(prob_goal_in_window(lam) - (1 - math.exp(-lam))) < 1e-9


def test_prob_goal_in_window_rejects_negative_expected_goals():
    with pytest.raises(ValueError):
        prob_goal_in_window(-0.1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'goles.model'`

- [ ] **Step 3: Write the implementation**

`src/goles/model.py`:
```python
from __future__ import annotations

import math


def dynamic_lambda(
    pre_match_xg_per90: float,
    in_match_xg_recent: float,
    recent_window_minutes: int,
    horizon_minutes: int,
    blend: float = 0.5,
) -> float:
    """Estimate expected goals for a team over the next `horizon_minutes`,
    blending a pre-match prior rate (from full-match expected xG per 90)
    with the observed in-match rate over the last `recent_window_minutes`.

    `blend` is the weight given to the in-match observed rate: 0 ignores
    the live signal entirely, 1 ignores the pre-match prior entirely.
    """
    if recent_window_minutes <= 0:
        raise ValueError("recent_window_minutes must be positive")
    prior_rate_per_minute = pre_match_xg_per90 / 90.0
    observed_rate_per_minute = in_match_xg_recent / recent_window_minutes
    blended_rate_per_minute = (
        blend * observed_rate_per_minute + (1 - blend) * prior_rate_per_minute
    )
    return blended_rate_per_minute * horizon_minutes


def prob_goal_in_window(expected_goals: float) -> float:
    """Convert an expected-goals value (a Poisson lambda) into the
    probability of at least one goal occurring: 1 - P(zero goals)."""
    if expected_goals < 0:
        raise ValueError("expected_goals cannot be negative")
    return 1.0 - math.exp(-expected_goals)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_model.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```powershell
git add src/goles/model.py tests/test_model.py
git commit -m "feat: add Poisson baseline goal-probability model"
```

---

### Task 7: Backtest harness

**Files:**
- Create: `src/goles/backtest.py`
- Test: `tests/test_backtest.py`

**Interfaces:**
- Consumes: `goles.db.get_connection/init_db` (Task 2), `goles.loaders.understat.persist_shots` (Task 4, used only in tests to seed data), `goles.features.compute_state_at_minute/goal_in_window` (Task 5), `goles.model.dynamic_lambda/prob_goal_in_window` (Task 6).
- Produces: `CUTOFF_MINUTES: list[int]`, `BacktestResult` dataclass (`predicted_probs: list[float]`, `actual_outcomes: list[bool]`, properties `brier_score` and method `calibration_bins(n_bins=5) -> list[tuple[float, float, float, int]]`), `run_backtest(conn, team: str = "home", blend: float = 0.5) -> BacktestResult`, `print_report(result: BacktestResult) -> None`.

- [ ] **Step 1: Write the failing tests**

`tests/test_backtest.py`:
```python
import math

from goles.backtest import CUTOFF_MINUTES, BacktestResult, run_backtest
from goles.db import get_connection, init_db
from goles.loaders.understat import persist_shots


def _seed_one_match(conn):
    records = []
    for m, xg, goal in [(10, 0.1, False), (30, 0.4, True), (55, 0.3, False), (78, 0.5, True)]:
        records.append(
            {
                "match_id": 1, "league": "TEST", "season": "2025-26",
                "home_team": "Team A", "away_team": "Team B",
                "minute": m, "team": "home", "xg": xg, "is_goal": goal,
            }
        )
    for m, xg, goal in [(20, 0.2, False), (65, 0.35, False)]:
        records.append(
            {
                "match_id": 1, "league": "TEST", "season": "2025-26",
                "home_team": "Team A", "away_team": "Team B",
                "minute": m, "team": "away", "xg": xg, "is_goal": goal,
            }
        )
    persist_shots(conn, records)


def test_run_backtest_returns_one_prediction_per_cutoff():
    conn = get_connection(":memory:")
    init_db(conn)
    _seed_one_match(conn)
    result = run_backtest(conn, team="home")
    assert len(result.predicted_probs) == len(CUTOFF_MINUTES)
    assert len(result.actual_outcomes) == len(CUTOFF_MINUTES)
    assert all(0.0 <= p <= 1.0 for p in result.predicted_probs)


def test_brier_score_is_between_zero_and_one():
    conn = get_connection(":memory:")
    init_db(conn)
    _seed_one_match(conn)
    result = run_backtest(conn, team="home")
    assert 0.0 <= result.brier_score <= 1.0


def test_backtest_result_handles_empty_data():
    empty = BacktestResult(predicted_probs=[], actual_outcomes=[])
    assert math.isnan(empty.brier_score)
    assert empty.calibration_bins() == []


def test_calibration_bins_groups_by_predicted_probability():
    result = BacktestResult(
        predicted_probs=[0.05, 0.12, 0.55, 0.60],
        actual_outcomes=[False, True, True, False],
    )
    bins = result.calibration_bins(n_bins=5)
    # two predictions land in bin [0.0, 0.2), two land in bin [0.4, 0.6)
    bin_counts = {round(b[0], 1): b[3] for b in bins}
    assert bin_counts[0.0] == 2
    assert bin_counts[0.4] == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_backtest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'goles.backtest'`

- [ ] **Step 3: Write the implementation**

`src/goles/backtest.py`:
```python
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from goles.features import compute_state_at_minute, goal_in_window
from goles.model import dynamic_lambda, prob_goal_in_window

CUTOFF_MINUTES = list(range(20, 81, 5))
HORIZON_MINUTES = 15
RECENT_WINDOW_MINUTES = 15
DEFAULT_BLEND = 0.5


@dataclass
class BacktestResult:
    predicted_probs: list[float]
    actual_outcomes: list[bool]

    @property
    def brier_score(self) -> float:
        n = len(self.predicted_probs)
        if n == 0:
            return float("nan")
        return sum(
            (p - float(o)) ** 2 for p, o in zip(self.predicted_probs, self.actual_outcomes)
        ) / n

    def calibration_bins(self, n_bins: int = 5) -> list[tuple[float, float, float, int]]:
        """Groups predictions into `n_bins` equal-width probability buckets
        and returns (bin_low, mean_predicted, mean_actual, count) for each
        non-empty bucket."""
        bucketed_preds: list[list[float]] = [[] for _ in range(n_bins)]
        bucketed_actuals: list[list[bool]] = [[] for _ in range(n_bins)]
        for p, o in zip(self.predicted_probs, self.actual_outcomes):
            idx = min(int(p * n_bins), n_bins - 1)
            bucketed_preds[idx].append(p)
            bucketed_actuals[idx].append(o)

        report = []
        for i in range(n_bins):
            if not bucketed_preds[i]:
                continue
            mean_pred = sum(bucketed_preds[i]) / len(bucketed_preds[i])
            mean_actual = sum(bucketed_actuals[i]) / len(bucketed_actuals[i])
            report.append((i / n_bins, mean_pred, mean_actual, len(bucketed_preds[i])))
        return report


def _load_match_shots(
    conn: sqlite3.Connection, match_id: int, home_team_id: int, away_team_id: int
) -> list[dict]:
    rows = conn.execute(
        "SELECT minute, team_id, xg, is_goal FROM shots WHERE match_id = ? ORDER BY minute",
        (match_id,),
    ).fetchall()
    shots = []
    for minute, team_id, xg, is_goal in rows:
        team = "home" if team_id == home_team_id else "away"
        shots.append({"minute": minute, "team": team, "xg": xg, "is_goal": bool(is_goal)})
    return shots


def _pre_match_xg_per90(shots: list[dict], team: str) -> float:
    """Simplified prior: sums the team's shot xG across the whole match.
    See the 'look-ahead bias' note in this plan's Global Constraints — this
    is a placeholder prior for validating the pipeline, not a production
    pre-match estimate."""
    return sum(s["xg"] for s in shots if s["team"] == team)


def run_backtest(
    conn: sqlite3.Connection, team: str = "home", blend: float = DEFAULT_BLEND
) -> BacktestResult:
    """Replays every stored match at each cutoff minute in CUTOFF_MINUTES,
    predicting P(goal in the next HORIZON_MINUTES for `team`) with the
    Poisson baseline, and comparing it against what actually happened."""
    predicted_probs: list[float] = []
    actual_outcomes: list[bool] = []

    matches = conn.execute("SELECT match_id, home_team_id, away_team_id FROM matches").fetchall()

    for match_id, home_team_id, away_team_id in matches:
        shots = _load_match_shots(conn, match_id, home_team_id, away_team_id)
        if not shots:
            continue
        pre_match_xg = _pre_match_xg_per90(shots, team)

        for cutoff in CUTOFF_MINUTES:
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


def print_report(result: BacktestResult) -> None:
    print(f"Muestras evaluadas: {len(result.predicted_probs)}")
    print(f"Brier score: {result.brier_score:.4f}")
    print("Calibracion (bin_low, prob. media predicha, frecuencia real, n):")
    for bin_low, mean_pred, mean_actual, count in result.calibration_bins():
        print(f"  [{bin_low:.1f}-{bin_low + 0.2:.1f}) pred={mean_pred:.3f} real={mean_actual:.3f} n={count}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_backtest.py -v`
Expected: 4 passed

- [ ] **Step 5: Run the full test suite**

Run: `pytest -v`
Expected: all tests from Tasks 2-7 pass (18 tests total)

- [ ] **Step 6: Commit**

```powershell
git add src/goles/backtest.py tests/test_backtest.py
git commit -m "feat: add backtest harness with Brier score and calibration bins"
```

---

### Task 8: End-to-end historical pull CLI (manual verification, not unit-tested)

**Files:**
- Create: `src/goles/cli.py`

**Interfaces:**
- Consumes: everything from Tasks 2-7.
- Produces: a runnable script `python -m goles.cli` for manual verification against real Understat data (deliberately outside the automated test suite, since it needs network access).

- [ ] **Step 1: Write the CLI script**

`src/goles/cli.py`:
```python
from __future__ import annotations

from goles.backtest import print_report, run_backtest
from goles.db import get_connection, init_db
from goles.loaders.understat import fetch_understat_shots, persist_shots, shots_to_records

LEAGUES = ["ENG-Premier League", "GER-Bundesliga"]
SEASONS = ["2324"]


def main() -> None:
    conn = get_connection()
    init_db(conn)

    print(f"Descargando datos de Understat para {LEAGUES} temporada {SEASONS}...")
    shots_df = fetch_understat_shots(LEAGUES, SEASONS)
    records = shots_to_records(shots_df)
    print(f"{len(records)} eventos de tiro descargados. Guardando en la base de datos...")
    persist_shots(conn, records)

    print("Corriendo backtest para el equipo local...")
    result = run_backtest(conn, team="home")
    print_report(result)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it manually to confirm the pipeline works end-to-end against real data**

Run (PowerShell, with the venv activated):
```powershell
python -m goles.cli
```
Expected: downloads real Understat shot events for Premier League + Bundesliga 2023-24, persists them into `data/goles.db`, and prints a Brier score plus a calibration table with a non-trivial sample count (hundreds of matches × 13 cutoff minutes each). Read the calibration table: `pred` and `real` columns should be in the same ballpark for each bucket — if they're wildly off (e.g., `pred=0.10` but `real=0.60` consistently), the blend/model needs revisiting before moving on to the ML correction layer (Plan B).

- [ ] **Step 3: Commit**

```powershell
git add src/goles/cli.py
git commit -m "feat: add CLI to pull real Understat data and print a backtest report"
```

---

## Próximos pasos (fuera de alcance de este plan)

Este plan entrega la base offline validable. Un segundo plan ("Ingesta en vivo + Telegram + orquestación") se construirá después de correr Task 8 contra datos reales y confirmar que la calibración del baseline es razonable, y cubrirá: reemplazar el prior "look-ahead" de `_pre_match_xg_per90` por un promedio de temporada estrictamente anterior por equipo; la capa de corrección LightGBM; los scrapers en vivo de Sofascore/FotMob; el cliente de Betfair Exchange; la calculadora de valor con los filtros anti-falla (blackout, frescura de cuota, tamaño de muestra mínimo); el bot de Telegram (incluida la guía de configuración con BotFather); el registrador de resultados en producción; y el job semanal de recalibración.

**Hallazgo de la revisión final de esta fase (agregado post-ejecución):** la Tarea 3 (cargador de ClubElo) quedó construida pero sin ningún consumidor — la tabla `elo_ratings` y las columnas `home_elo_wp`/`away_elo_wp`/`draw_elo_wp` de `matches` existen pero nada las llena ni las lee todavía. Esto es correcto para el alcance de esta fase (ninguna tarea lo pedía), pero quedó sin rastrear en este documento. Se agrega explícitamente al próximo plan: conectar ClubElo a la ingesta (poblar `elo_ratings` y las probabilidades pre-partido por equipo) y usar esas probabilidades como el prior real de `dynamic_lambda` en vez del prior con look-ahead bias — esto depende de que `matches.date` esté poblado (corregido en la ronda de fixes de la revisión final) para poder casar la fecha del partido con la fecha de rating de Elo correspondiente.
