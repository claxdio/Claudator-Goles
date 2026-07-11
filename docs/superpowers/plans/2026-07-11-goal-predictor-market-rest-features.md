# Market Odds + Rest-Days + Red-Card Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three data categories identified as high-value/low-effort and *orthogonal* to what we already capture: pre-match market-implied probabilities (from football-data.co.uk odds, already free and already identified as a source in Phase 1 but never actually ingested), fixture-rest-days (100% derivable from data already in the database, zero new collection), and minute-stamped red cards (also zero new collection — found inside the raw Understat cache we already have on disk, see below). Retrain and honestly compare against the current baseline (BSS 0.0190 / 0.0160 post-enrichment).

**Why these three, not the ones we already tried:** the shot-detail enrichment plan (box entries, set-piece split) came back flat (Δ within noise) because xG already encodes shot-location/situation information — those features were redundant with what the model already had. Market odds, rest-days, and red cards are different in kind: xG has zero information about betting markets, pre-match fatigue, or a numerical-advantage shock, so there's no redundancy mechanism working against them the way there was for shot detail. Red cards specifically have the largest documented single-event effect size of anything considered in this project's research so far (~30% swing in the disadvantaged team's scoring rate) — bigger than weather or referee tendencies combined.

**Architecture:** `soccerdata`'s `MatchHistory` reader is blocked by football-data.co.uk right now (verified: consistent 503 from its TLS-impersonating client across two seasons, while a plain `requests.get()` with a normal browser User-Agent gets 200 — same class of issue as Understat's occasional blocks, different specific cause). So this plan fetches the CSVs directly with `requests`, bypassing `soccerdata` for this one source. Team names differ between football-data.co.uk and our Understat-sourced `teams` table (e.g. "Man City" vs "Manchester City", "Dortmund" vs "Borussia Dortmund") — resolved with a hardcoded alias table built by empirically diffing the two real team-name lists across all 6 seasons/2 leagues we track (not guessed), so coverage is known and complete, not approximate. Rest-days reuses `priors.py`'s existing `team_matches_chronological` (no new query logic). Red cards were found empirically inside the raw Understat cache JSON's `rosters` section (not the `shots` section already mined by the enrichment plan): each player entry has `red_card` ("0"/"1") and `time` (minutes played), and for a sent-off player `time` is verified (against real matches, e.g. a Nastasic red card with `time: "65"`) to be the dismissal minute — this is minute-level in-play data, zero new network calls, sourced entirely from cache already on disk.

**Tech Stack:** Unchanged — Python 3.11+, `requests`, `pandas`, `sqlite3`, `pytest`. No new dependencies (deliberately NOT using `soccerdata.MatchHistory`, which is blocked here).

## Global Constraints

- No paid services. football-data.co.uk odds are free, no auth, no rate limit encountered in verification.
- **Do not use `soccerdata.MatchHistory`** for this plan — verified blocked (503) via its TLS-fingerprinting client; use direct `requests.get(url, headers={"User-Agent": "Mozilla/5.0"})` instead (verified working, HTTP 200).
- CSV URL pattern (verified): `https://www.football-data.co.uk/mmz4281/{season}/{code}.csv` where season is our existing `"1819"`..`"2324"` format (matches exactly) and code is `"E0"` for `ENG-Premier League`, `"D1"` for `GER-Bundesliga`.
- **Team name mapping must be a complete, hardcoded, verified table — never fuzzy-matched.** The alias dict in Task 1 was built by fetching all 6 seasons of both leagues from football-data.co.uk and diffing against our real `teams` table; every name that differs is listed explicitly. If a future season introduces a team not in this table, matching must fail loudly (raise / clearly log as unmatched), never silently guess — this mirrors the fail-loud discipline already used for zero-shot-match and own-goal edge cases in `understat.py`.
- Date format conversion required: football-data.co.uk uses `DD/MM/YYYY`; our `matches.date` is `YYYY-MM-DD` (ISO, from Understat's game string). Convert explicitly, don't string-compare across formats.
- No-vig (de-margined) probability, not raw inverse-odds: raw `1/odds` values sum to >1 (the bookmaker's overround/margin) — normalize by dividing each by the sum so the three outcomes (or two, for over/under) sum to exactly 1.0.
- Rest-days feature reuses `goles.priors.team_matches_chronological` — do not write a second query for the same data.
- **Red cards only (not yellow)** — yellow cards have a much weaker/noisier documented relationship to goal-timing and are out of scope here; keep the feature set focused on the one shock event with real evidence behind it. Only the *first* red card per team per match matters for the "numerical disadvantage" signal a second red card doesn't add new information beyond "still down a player."
- Red-card extraction reads the `rosters` section of the same raw cache files (`match_*.json`) the enrichment plan's `load_shot_details_from_cache` already reads the `shots` section of — same file, same resilience discipline (skip missing/malformed files, never raise), same `cache_dir` resolution via `soccerdata._config.DATA_DIR`.
- Feature-count discipline: exactly 8 new features total this plan — 6 in Tasks 1-4 (`own_rest_days`, `opp_rest_days`, `own_market_wp`, `opp_market_wp`, `market_draw_wp`, `market_over25_wp`) plus 2 in Tasks 5-7 (`own_red_cards`, `opp_red_cards`), bringing `FEATURE_NAMES` from 28 → 36.
- Temporal discipline unchanged: market odds are *pre-match* closing lines (fixed before kickoff, same information-timing as `trailing_prior_xg`/ClubElo would be) — using them as a feature is not look-ahead, since a live system would have this same closing-line information available before/at kickoff. Red-card counts are computed with the same `minute <= cutoff_minute` discipline as shots — a card at minute 70 must never affect a cutoff-60 prediction. Test-season rows still never touch training or calibration.
- All 67 existing tests must keep passing unmodified.

---

### Task 1: Direct football-data.co.uk loader + team-name alias table

**Files:**
- Create: `src/goles/loaders/football_data.py`
- Test: `tests/test_football_data_loader.py`

**Interfaces:**
- Produces: `LEAGUE_CODES: dict[str, str]` (`{"ENG-Premier League": "E0", "GER-Bundesliga": "D1"}`), `TEAM_NAME_ALIASES: dict[str, str]` (football-data name → our name, only for names that differ), `normalize_team_name(name: str) -> str`, `fetch_odds(leagues: dict[str, str], seasons: list[str]) -> pd.DataFrame` (concatenated raw CSV rows across leagues/seasons, with an added `understat_league` column).

- [ ] **Step 1: Write the failing tests**

`tests/test_football_data_loader.py`:
```python
from unittest.mock import Mock, patch

import pandas as pd

from goles.loaders.football_data import (
    TEAM_NAME_ALIASES,
    fetch_odds,
    normalize_team_name,
)

SAMPLE_CSV = (
    "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,AvgH,AvgD,AvgA,Avg>2.5,Avg<2.5\n"
    "E0,11/08/2023,Burnley,Man City,0,3,9.02,5.35,1.35,1.90,1.95\n"
    "E0,12/08/2023,Arsenal,Nott'm Forest,2,1,1.18,7.64,15.67,1.75,2.10\n"
)


def test_normalize_team_name_maps_known_aliases():
    assert normalize_team_name("Man City") == "Manchester City"
    assert normalize_team_name("Nott'm Forest") == "Nottingham Forest"
    assert normalize_team_name("Dortmund") == "Borussia Dortmund"


def test_normalize_team_name_passes_through_unmapped_names():
    assert normalize_team_name("Arsenal") == "Arsenal"
    assert normalize_team_name("Burnley") == "Burnley"


def test_team_name_aliases_has_no_identity_entries():
    # every value should differ from its key -- identity mappings belong in
    # "pass through unchanged", not cluttering the alias table
    for fd_name, our_name in TEAM_NAME_ALIASES.items():
        assert fd_name != our_name


def test_fetch_odds_concatenates_leagues_and_labels_them():
    mock_response = Mock()
    mock_response.text = SAMPLE_CSV
    mock_response.raise_for_status = Mock()
    with patch("goles.loaders.football_data.requests.get", return_value=mock_response) as mock_get:
        df = fetch_odds({"ENG-Premier League": "E0"}, ["2324"])
    mock_get.assert_called_once_with(
        "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
    )
    assert len(df) == 2
    assert (df["understat_league"] == "ENG-Premier League").all()
    assert list(df["HomeTeam"]) == ["Burnley", "Arsenal"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\pytest.exe tests/test_football_data_loader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'goles.loaders.football_data'`

- [ ] **Step 3: Write the implementation**

`src/goles/loaders/football_data.py`:
```python
from __future__ import annotations

import io

import pandas as pd
import requests

LEAGUE_CODES = {
    "ENG-Premier League": "E0",
    "GER-Bundesliga": "D1",
}

# Built by fetching all 6 seasons (1819-2324) of both leagues from
# football-data.co.uk and diffing team names against our real `teams`
# table (populated from Understat). Every name that differs is listed
# here explicitly -- this is a complete, verified table for the
# leagues/seasons this project currently tracks, not a guess. A team not
# in this table and not identical to our own naming will fail to match
# in Task 3's persistence step, loudly, rather than being silently
# fuzzy-matched.
TEAM_NAME_ALIASES = {
    # Premier League
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Newcastle": "Newcastle United",
    "Nott'm Forest": "Nottingham Forest",
    "West Brom": "West Bromwich Albion",
    "Wolves": "Wolverhampton Wanderers",
    # Bundesliga
    "Bielefeld": "Arminia Bielefeld",
    "Dortmund": "Borussia Dortmund",
    "Ein Frankfurt": "Eintracht Frankfurt",
    "FC Koln": "FC Cologne",
    "Fortuna Dusseldorf": "Fortuna Duesseldorf",
    "Greuther Furth": "Greuther Fuerth",
    "Hannover": "Hannover 96",
    "Heidenheim": "FC Heidenheim",
    "Hertha": "Hertha Berlin",
    "Leverkusen": "Bayer Leverkusen",
    "M'gladbach": "Borussia M.Gladbach",
    "Mainz": "Mainz 05",
    "Nurnberg": "Nuernberg",
    "RB Leipzig": "RasenBallsport Leipzig",
    "Stuttgart": "VfB Stuttgart",
}


def normalize_team_name(name: str) -> str:
    """Maps a football-data.co.uk team name to our Understat-sourced team
    name. Names not in TEAM_NAME_ALIASES are assumed identical between the
    two sources and returned unchanged."""
    return TEAM_NAME_ALIASES.get(name, name)


def fetch_odds(leagues: dict[str, str], seasons: list[str]) -> pd.DataFrame:
    """Fetches football-data.co.uk match/odds CSVs directly (bypassing
    soccerdata's MatchHistory reader, whose TLS-fingerprinting client is
    currently blocked -- verified 503 -- by this specific site; a plain
    requests.get with a standard User-Agent works). `leagues` maps our
    league name (e.g. "ENG-Premier League") to football-data.co.uk's
    short code (e.g. "E0"). Returns the concatenated raw CSV rows across
    every league/season with an added `understat_league` column."""
    frames = []
    for league_name, code in leagues.items():
        for season in seasons:
            url = f"https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
            response.raise_for_status()
            df = pd.read_csv(io.StringIO(response.text))
            df["understat_league"] = league_name
            df["understat_season"] = season
            frames.append(df)
    return pd.concat(frames, ignore_index=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\pytest.exe tests/test_football_data_loader.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```powershell
git add src/goles/loaders/football_data.py tests/test_football_data_loader.py
git commit -m "feat: add direct football-data.co.uk odds loader with verified team-name aliases"
```

---

### Task 2: No-vig probabilities + schema + persistence

**Files:**
- Modify: `src/goles/db.py` (add 4 columns to `matches`)
- Modify: `src/goles/loaders/football_data.py` (add `compute_no_vig_probabilities`, `persist_odds`)
- Test: `tests/test_football_data_loader.py` (append)

**Interfaces:**
- Produces: `compute_no_vig_probabilities(odds_home: float, odds_draw: float, odds_away: float) -> tuple[float, float, float]`, `compute_no_vig_two_way(odds_a: float, odds_b: float) -> tuple[float, float]`, `persist_odds(conn, odds_df: pd.DataFrame) -> tuple[int, int]` (returns `(matched, unmatched)` counts).

- [ ] **Step 1: Add the 4 columns**

In `src/goles/db.py`, add to the `matches` table inside `SCHEMA` (after `draw_elo_wp REAL,`):
```sql
    market_home_wp REAL,
    market_draw_wp REAL,
    market_away_wp REAL,
    market_over25_wp REAL
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_football_data_loader.py`:
```python
import pandas as pd
import pytest

from goles.db import get_connection, get_or_create_team, init_db
from goles.loaders.football_data import (
    compute_no_vig_probabilities,
    compute_no_vig_two_way,
    persist_odds,
)


def test_compute_no_vig_probabilities_sums_to_one_and_favors_favorite():
    home_wp, draw_wp, away_wp = compute_no_vig_probabilities(1.35, 5.35, 9.02)
    assert abs((home_wp + draw_wp + away_wp) - 1.0) < 1e-9
    assert home_wp > away_wp  # 1.35 is the shortest (favorite) price


def test_compute_no_vig_two_way_sums_to_one():
    over_wp, under_wp = compute_no_vig_two_way(1.90, 1.95)
    assert abs((over_wp + under_wp) - 1.0) < 1e-9


def test_persist_odds_matches_by_date_and_normalized_team_names():
    conn = get_connection(":memory:")
    init_db(conn)
    home_id = get_or_create_team(conn, "Manchester City")
    away_id = get_or_create_team(conn, "Burnley")
    conn.execute(
        """INSERT INTO matches (understat_id, league, season, date, home_team_id, away_team_id)
           VALUES (1, 'ENG-Premier League', '2324', '2023-08-11', ?, ?)""",
        (away_id, home_id),  # Burnley home, Man City away -- matches SAMPLE_CSV row 1
    )
    conn.commit()

    odds_df = pd.DataFrame(
        [
            {
                "Date": "11/08/2023", "HomeTeam": "Burnley", "AwayTeam": "Man City",
                "AvgH": 9.02, "AvgD": 5.35, "AvgA": 1.35, "Avg>2.5": 1.90, "Avg<2.5": 1.95,
                "understat_league": "ENG-Premier League", "understat_season": "2324",
            },
        ]
    )
    matched, unmatched = persist_odds(conn, odds_df)
    assert matched == 1
    assert unmatched == 0

    row = conn.execute(
        "SELECT market_home_wp, market_draw_wp, market_away_wp, market_over25_wp FROM matches"
    ).fetchone()
    assert all(v is not None for v in row)
    assert abs(row[0] + row[1] + row[2] - 1.0) < 1e-9


def test_persist_odds_counts_unmatched_rows_without_raising():
    conn = get_connection(":memory:")
    init_db(conn)
    # no matches inserted at all -- every odds row should be unmatched
    odds_df = pd.DataFrame(
        [
            {
                "Date": "11/08/2023", "HomeTeam": "Burnley", "AwayTeam": "Man City",
                "AvgH": 9.02, "AvgD": 5.35, "AvgA": 1.35, "Avg>2.5": 1.90, "Avg<2.5": 1.95,
                "understat_league": "ENG-Premier League", "understat_season": "2324",
            },
        ]
    )
    matched, unmatched = persist_odds(conn, odds_df)
    assert matched == 0
    assert unmatched == 1
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv\Scripts\pytest.exe tests/test_football_data_loader.py -v -k "no_vig or persist_odds"`
Expected: FAIL with `ImportError`

- [ ] **Step 4: Write the implementation**

Append to `src/goles/loaders/football_data.py` (add `import sqlite3` and `from goles.db import get_or_create_team` to the imports):

```python
def compute_no_vig_probabilities(
    odds_home: float, odds_draw: float, odds_away: float
) -> tuple[float, float, float]:
    """Converts three decimal odds into de-margined (no-vig) probabilities
    that sum to exactly 1.0, by normalizing the raw 1/odds values (whose
    sum exceeds 1.0 by the bookmaker's overround)."""
    raw = [1.0 / odds_home, 1.0 / odds_draw, 1.0 / odds_away]
    total = sum(raw)
    return raw[0] / total, raw[1] / total, raw[2] / total


def compute_no_vig_two_way(odds_a: float, odds_b: float) -> tuple[float, float]:
    """Same de-margining for a two-outcome market (e.g. over/under 2.5)."""
    raw = [1.0 / odds_a, 1.0 / odds_b]
    total = sum(raw)
    return raw[0] / total, raw[1] / total


def _to_iso_date(football_data_date: str) -> str:
    """Converts football-data.co.uk's DD/MM/YYYY to our ISO YYYY-MM-DD."""
    day, month, year = football_data_date.split("/")
    if len(year) == 2:
        year = "20" + year
    return f"{year}-{month.zfill(2)}-{day.zfill(2)}"


def persist_odds(conn: sqlite3.Connection, odds_df: pd.DataFrame) -> tuple[int, int]:
    """Normalizes team names, converts dates, computes no-vig probabilities,
    and updates matching rows in `matches` (joined by league + season +
    date + home/away team name, since football-data.co.uk has no id
    compatible with our understat_id). Rows with no matching database row
    are counted as unmatched but do not raise -- the caller decides what
    coverage is acceptable. Returns (matched_count, unmatched_count)."""
    matched = 0
    unmatched = 0
    for row in odds_df.itertuples(index=False):
        row_dict = row._asdict()
        home_name = normalize_team_name(row_dict["HomeTeam"])
        away_name = normalize_team_name(row_dict["AwayTeam"])
        date_iso = _to_iso_date(row_dict["Date"])

        home_row = conn.execute("SELECT team_id FROM teams WHERE name = ?", (home_name,)).fetchone()
        away_row = conn.execute("SELECT team_id FROM teams WHERE name = ?", (away_name,)).fetchone()
        if home_row is None or away_row is None:
            unmatched += 1
            continue
        home_id, away_id = home_row[0], away_row[0]

        match_row = conn.execute(
            """SELECT match_id FROM matches
               WHERE league = ? AND season = ? AND date = ?
                 AND home_team_id = ? AND away_team_id = ?""",
            (row_dict["understat_league"], row_dict["understat_season"], date_iso, home_id, away_id),
        ).fetchone()
        if match_row is None:
            unmatched += 1
            continue

        home_wp, draw_wp, away_wp = compute_no_vig_probabilities(
            row_dict["AvgH"], row_dict["AvgD"], row_dict["AvgA"]
        )
        over_wp, _ = compute_no_vig_two_way(row_dict["Avg>2.5"], row_dict["Avg<2.5"])
        conn.execute(
            """UPDATE matches
               SET market_home_wp = ?, market_draw_wp = ?, market_away_wp = ?, market_over25_wp = ?
               WHERE match_id = ?""",
            (home_wp, draw_wp, away_wp, over_wp, match_row[0]),
        )
        matched += 1
    conn.commit()
    return matched, unmatched
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv\Scripts\pytest.exe tests/test_football_data_loader.py -v`
Expected: 9 passed

- [ ] **Step 6: Run the full suite**

Run: `.venv\Scripts\pytest.exe -q`
Expected: 76 passed (67 existing + 9 new)

- [ ] **Step 7: Commit**

```powershell
git add src/goles/db.py src/goles/loaders/football_data.py tests/test_football_data_loader.py
git commit -m "feat: compute no-vig market probabilities and persist them onto matches"
```

---

### Task 3: Ingest real odds data (manual verification)

**Files:**
- Create: `src/goles/ingest_odds.py`

No automated tests (I/O-heavy script, same precedent as `ingest_history.py`).

- [ ] **Step 1: Write the script**

`src/goles/ingest_odds.py`:
```python
from __future__ import annotations

from goles.db import get_connection, init_db
from goles.loaders.football_data import LEAGUE_CODES, fetch_odds, persist_odds

SEASONS = ["1819", "1920", "2021", "2122", "2223", "2324"]


def main() -> None:
    conn = get_connection()
    init_db(conn)

    print(f"Descargando cuotas de football-data.co.uk para {list(LEAGUE_CODES.keys())} temporadas {SEASONS}...")
    odds_df = fetch_odds(LEAGUE_CODES, SEASONS)
    print(f"{len(odds_df)} filas de cuotas descargadas. Emparejando contra partidos existentes...")

    matched, unmatched = persist_odds(conn, odds_df)
    total = matched + unmatched
    coverage = matched / total if total else 0.0
    print(f"Emparejados: {matched}/{total} ({coverage:.1%}). Sin emparejar: {unmatched}.")
    if coverage < 0.95:
        print(
            "ADVERTENCIA: cobertura por debajo del 95%. Revisar TEAM_NAME_ALIASES antes de "
            "usar estas features -- puede haber un equipo nuevo sin mapear."
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it for real**

Run (PowerShell, venv activated): `python -m goles.ingest_odds`

Expected: fast (small CSVs, ~12 files total, well under a minute). Read the coverage percentage carefully — it should be very high (>95%, ideally >99%) given the alias table was built by diffing the complete real team lists. If it's meaningfully lower, do NOT proceed to Task 4/5 — stop and report the specific unmatched team names (add a temporary print of unmatched `HomeTeam`/`AwayTeam` values if needed to diagnose) so the alias table can be corrected first.

- [ ] **Step 3: Commit**

```powershell
git add src/goles/ingest_odds.py
git commit -m "feat: add odds ingestion script and populate market probabilities"
```

---

### Task 4: Rest-days prior

**Files:**
- Modify: `src/goles/priors.py` (add `days_since_last_match`)
- Test: `tests/test_priors.py` (append)

**Interfaces:**
- Consumes: existing `team_matches_chronological(conn, team_id, league, season) -> list[tuple[int, str]]`.
- Produces: `days_since_last_match(conn, team_id: int, league: str, season: str, before_match_id: int) -> float | None` (`None` for a team's first match of the season -- no prior fixture to measure from).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_priors.py`:
```python
def test_days_since_last_match_computes_the_gap():
    conn = get_connection(":memory:")
    init_db(conn)
    arsenal = get_or_create_team(conn, "Arsenal")
    chelsea = get_or_create_team(conn, "Chelsea")
    fulham = get_or_create_team(conn, "Fulham")

    m1 = _insert_match(conn, 1, "ENG-Premier League", "2324", "2023-08-11", arsenal, chelsea)
    m2 = _insert_match(conn, 2, "ENG-Premier League", "2324", "2023-08-19", arsenal, fulham)
    conn.commit()

    gap = days_since_last_match(conn, arsenal, "ENG-Premier League", "2324", m2)
    assert gap == 8.0


def test_days_since_last_match_returns_none_for_first_match_of_season():
    conn = get_connection(":memory:")
    init_db(conn)
    arsenal = get_or_create_team(conn, "Arsenal")
    chelsea = get_or_create_team(conn, "Chelsea")
    m1 = _insert_match(conn, 1, "ENG-Premier League", "2324", "2023-08-11", arsenal, chelsea)
    conn.commit()

    assert days_since_last_match(conn, arsenal, "ENG-Premier League", "2324", m1) is None
```

Add `days_since_last_match` to the existing `from goles.priors import ...` line at the top of the file.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\pytest.exe tests/test_priors.py -v -k rest_or_days_since`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write the implementation**

Add to `src/goles/priors.py` (after `trailing_xg_per90`):

```python
def days_since_last_match(
    conn: sqlite3.Connection,
    team_id: int,
    league: str,
    season: str,
    before_match_id: int,
) -> float | None:
    """Days between `team_id`'s previous match in (league, season) and the
    match identified by `before_match_id`. Returns None for a team's first
    match of the season (no prior fixture to measure a gap from) -- callers
    should treat None as "no rest-day signal available", not zero."""
    from datetime import date as _date

    matches = team_matches_chronological(conn, team_id, league, season)
    match_dates = {mid: date for mid, date in matches}
    if before_match_id not in match_dates:
        raise ValueError(
            f"match_id={before_match_id} not found for team_id={team_id} "
            f"in league={league!r} season={season!r}"
        )
    before_date = match_dates[before_match_id]
    prior_dates = sorted(date for mid, date in matches if date < before_date)
    if not prior_dates:
        return None
    last_date = prior_dates[-1]
    d1 = _date.fromisoformat(last_date)
    d2 = _date.fromisoformat(before_date)
    return float((d2 - d1).days)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\pytest.exe tests/test_priors.py -v`
Expected: all pass (existing + 2 new)

- [ ] **Step 5: Run the full suite, then commit**

Run: `.venv\Scripts\pytest.exe -q` → expected 78 passed.

```powershell
git add src/goles/priors.py tests/test_priors.py
git commit -m "feat: add days-since-last-match rest prior"
```

---

### Task 5: Raw-cache red-card extraction + `cards` table + persistence

**Files:**
- Modify: `src/goles/db.py` (add a `cards` table)
- Modify: `src/goles/loaders/understat.py` (new `load_red_cards_from_cache`, `persist_red_cards`)
- Test: `tests/test_understat_loader.py` (append)

**Interfaces:**
- Produces: `load_red_cards_from_cache(cache_dir: Path) -> dict[int, list[dict]]` (game_id → list of `{"team_h_a": "h"/"a", "minute": int}`), `persist_red_cards(conn, red_cards_by_game_id: dict[int, list[dict]]) -> tuple[int, int]` (returns `(matches_processed, matches_not_found)`).
- Consumes: raw cache JSON's `rosters` section, verified shape: `{"rosters": {"h": {"<player_id>": {"time": "65", "red_card": "1", ...}, ...}, "a": {...}}}`.

`src/goles/loaders/understat.py` already has `import json` and `from pathlib import Path` from the earlier enrichment plan's `load_shot_details_from_cache` — do not re-add them, just add the two new functions below it.

- [ ] **Step 1: Add the `cards` table**

In `src/goles/db.py`, add to `SCHEMA` (after the `elo_ratings` table definition):
```sql
CREATE TABLE IF NOT EXISTS cards (
    card_id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL REFERENCES matches(match_id),
    team_id INTEGER NOT NULL REFERENCES teams(team_id),
    minute INTEGER NOT NULL
);
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_understat_loader.py`:
```python
def test_load_red_cards_from_cache_reads_rosters(tmp_path):
    import json

    (tmp_path / "match_501.json").write_text(
        json.dumps(
            {
                "rosters": {
                    "h": {
                        "1": {"player": "Player A", "time": "65", "red_card": "1"},
                        "2": {"player": "Player B", "time": "90", "red_card": "0"},
                    },
                    "a": {
                        "3": {"player": "Player C", "time": "78", "red_card": "1"},
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    red_cards = load_red_cards_from_cache(tmp_path)

    assert red_cards[501] == [
        {"team_h_a": "h", "minute": 65},
        {"team_h_a": "a", "minute": 78},
    ]


def test_load_red_cards_from_cache_skips_matches_with_no_red_cards(tmp_path):
    import json

    (tmp_path / "match_502.json").write_text(
        json.dumps({"rosters": {"h": {"1": {"player": "Player A", "time": "90", "red_card": "0"}}, "a": {}}}),
        encoding="utf-8",
    )

    red_cards = load_red_cards_from_cache(tmp_path)

    assert 502 not in red_cards


def test_persist_red_cards_matches_by_understat_id_and_side():
    conn = get_connection(":memory:")
    init_db(conn)
    home_id = get_or_create_team(conn, "Team A")
    away_id = get_or_create_team(conn, "Team B")
    conn.execute(
        """INSERT INTO matches (understat_id, league, season, date, home_team_id, away_team_id)
           VALUES (501, 'TEST', '2324', '2023-08-11', ?, ?)""",
        (home_id, away_id),
    )
    conn.commit()

    red_cards_by_game_id = {501: [{"team_h_a": "h", "minute": 65}, {"team_h_a": "a", "minute": 78}]}
    processed, not_found = persist_red_cards(conn, red_cards_by_game_id)

    assert processed == 1
    assert not_found == 0
    rows = conn.execute("SELECT team_id, minute FROM cards ORDER BY minute").fetchall()
    assert rows == [(home_id, 65), (away_id, 78)]


def test_persist_red_cards_counts_unmatched_game_ids():
    conn = get_connection(":memory:")
    init_db(conn)
    red_cards_by_game_id = {999: [{"team_h_a": "h", "minute": 30}]}
    processed, not_found = persist_red_cards(conn, red_cards_by_game_id)
    assert processed == 0
    assert not_found == 1


def test_persist_red_cards_is_idempotent():
    conn = get_connection(":memory:")
    init_db(conn)
    home_id = get_or_create_team(conn, "Team A")
    away_id = get_or_create_team(conn, "Team B")
    conn.execute(
        """INSERT INTO matches (understat_id, league, season, date, home_team_id, away_team_id)
           VALUES (501, 'TEST', '2324', '2023-08-11', ?, ?)""",
        (home_id, away_id),
    )
    conn.commit()
    red_cards_by_game_id = {501: [{"team_h_a": "h", "minute": 65}]}
    persist_red_cards(conn, red_cards_by_game_id)
    persist_red_cards(conn, red_cards_by_game_id)  # second call: no-op for this match

    count = conn.execute("SELECT COUNT(*) FROM cards").fetchone()[0]
    assert count == 1
```

Add `load_red_cards_from_cache, persist_red_cards` to the existing `from goles.loaders.understat import ...` line at the top of the file.

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv\Scripts\pytest.exe tests/test_understat_loader.py -v -k "red_cards"`
Expected: FAIL with `ImportError`

- [ ] **Step 4: Write the implementation**

Append to `src/goles/loaders/understat.py`:
```python
def load_red_cards_from_cache(cache_dir: Path) -> dict[int, list[dict]]:
    """Reads the raw cached Understat match JSONs and returns a mapping
    match_id (game_id) -> list of {"team_h_a": "h"/"a", "minute": int} for
    every red-carded player in that match, sourced from the `rosters`
    section (NOT the `shots` section `load_shot_details_from_cache`
    reads). A red-carded player's `time` field is verified (against real
    matches) to be the minute they were dismissed.

    Resilient to missing/malformed files (skipped, not raised) -- best
    effort, like `load_shot_details_from_cache`. The match_id is taken
    from the filename (match_{id}.json), the same reliable, always-present
    naming convention the enrichment plan already verified for this cache."""
    red_cards: dict[int, list[dict]] = {}
    for match_file in Path(cache_dir).glob("match_*.json"):
        try:
            with open(match_file, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        rosters = data.get("rosters", {})
        if not isinstance(rosters, dict):
            continue
        try:
            match_id = int(match_file.stem.split("_", 1)[1])
        except (IndexError, ValueError):
            continue

        events = []
        for side in ("h", "a"):
            for player in rosters.get(side, {}).values():
                if player.get("red_card") != "1":
                    continue
                try:
                    minute = int(player["time"])
                except (KeyError, ValueError, TypeError):
                    continue
                events.append({"team_h_a": side, "minute": minute})
        if events:
            red_cards[match_id] = events
    return red_cards


def persist_red_cards(
    conn: sqlite3.Connection, red_cards_by_game_id: dict[int, list[dict]]
) -> tuple[int, int]:
    """Persists red-card events into the `cards` table, matching each
    game_id to its internal match_id via `matches.understat_id` and each
    team_h_a to home_team_id/away_team_id. Skips (as a no-op, not an
    error) any match that already has rows in `cards`, so repeated calls
    are safe. Returns (matches_processed, matches_not_found)."""
    processed = 0
    not_found = 0
    for game_id, events in red_cards_by_game_id.items():
        row = conn.execute(
            "SELECT match_id, home_team_id, away_team_id FROM matches WHERE understat_id = ?",
            (game_id,),
        ).fetchone()
        if row is None:
            not_found += 1
            continue
        match_id, home_id, away_id = row

        existing = conn.execute(
            "SELECT COUNT(*) FROM cards WHERE match_id = ?", (match_id,)
        ).fetchone()[0]
        if existing > 0:
            continue

        for event in events:
            team_id = home_id if event["team_h_a"] == "h" else away_id
            conn.execute(
                "INSERT INTO cards (match_id, team_id, minute) VALUES (?, ?, ?)",
                (match_id, team_id, event["minute"]),
            )
        processed += 1
    conn.commit()
    return processed, not_found
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv\Scripts\pytest.exe tests/test_understat_loader.py -v`
Expected: all pass (existing + 5 new)

- [ ] **Step 6: Run the full suite, then commit**

Run: `.venv\Scripts\pytest.exe -q` → expected 78 passed (67 + 6 from Tasks 1-2 + 5 here, matching whatever count Tasks 1-4 landed on plus 5).

```powershell
git add src/goles/db.py src/goles/loaders/understat.py tests/test_understat_loader.py
git commit -m "feat: extract minute-stamped red cards from the raw Understat cache"
```

---

### Task 6: Ingest red cards (manual verification)

**Files:**
- Create: `src/goles/ingest_cards.py`

No automated tests (I/O-free but still an operational script, same precedent as `ingest_odds.py` — note this one makes ZERO network calls, it only reads the already-downloaded local cache).

- [ ] **Step 1: Write the script**

`src/goles/ingest_cards.py`:
```python
from __future__ import annotations

from soccerdata._config import DATA_DIR

from goles.db import get_connection, init_db
from goles.loaders.understat import load_red_cards_from_cache, persist_red_cards


def main() -> None:
    conn = get_connection()
    init_db(conn)

    cache_dir = DATA_DIR / "Understat"
    print(f"Leyendo tarjetas rojas del cache crudo en {cache_dir}...")
    red_cards_by_game_id = load_red_cards_from_cache(cache_dir)
    total_cards = sum(len(events) for events in red_cards_by_game_id.values())
    print(f"{len(red_cards_by_game_id)} partidos con al menos una tarjeta roja ({total_cards} tarjetas en total).")

    processed, not_found = persist_red_cards(conn, red_cards_by_game_id)
    print(f"Procesados: {processed}. Partidos no encontrados en la base: {not_found}.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it for real**

Run (PowerShell, venv activated): `python -m goles.ingest_cards`

Expected: instant (no network, reads local cache only). `not_found` should be 0 (every game_id in the cache corresponds to a match already in the database, since the cache IS how the database was populated). Sanity-check the total: red cards are rare (roughly 1 per 30-40 matches in top-flight football), so expect on the order of 100-150 matches with at least one red card out of 4,116.

- [ ] **Step 3: Commit**

```powershell
git add src/goles/ingest_cards.py
git commit -m "feat: add red-card ingestion script from the local raw cache"
```

---

### Task 7: `load_match_cards` + red-card features in `compute_ml_features`

**Files:**
- Modify: `src/goles/backtest.py` (add `load_match_cards`)
- Modify: `src/goles/features.py` (`compute_ml_features` gains an optional `cards` parameter and 2 new keys)
- Test: `tests/test_backtest.py`, `tests/test_features.py` (append)

**Interfaces:**
- Produces: `load_match_cards(conn, match_id, home_team_id, away_team_id) -> list[dict]` (each `{"team": "home"/"away", "minute": int}`), `compute_ml_features(shots, cutoff_minute, team, cards: list[dict] | None = None) -> dict[str, float]` (gains `own_red_cards`, `opp_red_cards`; 28 keys total now).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_backtest.py`:
```python
def test_load_match_cards_returns_red_card_events():
    from goles.backtest import load_match_cards

    conn = get_connection(":memory:")
    init_db(conn)
    persist_shots(conn, [
        {
            "match_id": 701, "league": "TEST", "season": "2526", "date": "2025-08-01",
            "home_team": "Team A", "away_team": "Team B",
            "minute": 10, "team": "home", "xg": 0.1, "is_goal": False,
        },
    ])
    match_id, home_id, away_id = conn.execute(
        "SELECT match_id, home_team_id, away_team_id FROM matches"
    ).fetchone()
    conn.execute("INSERT INTO cards (match_id, team_id, minute) VALUES (?, ?, ?)", (match_id, away_id, 55))
    conn.commit()

    cards = load_match_cards(conn, match_id, home_id, away_id)
    assert cards == [{"team": "away", "minute": 55}]
```

Append to `tests/test_features.py`:
```python
def test_compute_ml_features_counts_red_cards_before_cutoff():
    cards = [{"team": "away", "minute": 55}, {"team": "home", "minute": 80}]
    f = compute_ml_features(ML_SAMPLE_SHOTS, cutoff_minute=65, team="home", cards=cards)
    assert f["opp_red_cards"] == 1.0  # the away card at minute 55 counts
    assert f["own_red_cards"] == 0.0  # the home card at minute 80 is after the cutoff


def test_compute_ml_features_defaults_red_cards_to_zero_without_cards_argument():
    f = compute_ml_features(ML_SAMPLE_SHOTS, cutoff_minute=65, team="home")
    assert f["own_red_cards"] == 0.0
    assert f["opp_red_cards"] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\pytest.exe tests/test_backtest.py tests/test_features.py -v -k "cards or red_card"`
Expected: FAIL — `load_match_cards` not found / `TypeError: compute_ml_features() got an unexpected keyword argument 'cards'`

- [ ] **Step 3: Implement**

In `src/goles/backtest.py`, add after `load_match_shots`:
```python
def load_match_cards(
    conn: sqlite3.Connection, match_id: int, home_team_id: int, away_team_id: int
) -> list[dict]:
    rows = conn.execute(
        "SELECT team_id, minute FROM cards WHERE match_id = ? ORDER BY minute",
        (match_id,),
    ).fetchall()
    cards = []
    for team_id, minute in rows:
        team = "home" if team_id == home_team_id else "away"
        cards.append({"team": team, "minute": minute})
    return cards
```

In `src/goles/features.py`, change `compute_ml_features`'s signature to:
```python
def compute_ml_features(
    shots: list[dict], cutoff_minute: int, team: str, cards: list[dict] | None = None
) -> dict[str, float]:
```
add this near the end of the function body, before the `return` statement:
```python
    cards = cards or []
    own_red_cards = float(sum(1 for c in cards if c["team"] == team and c["minute"] <= cutoff_minute))
    opp_red_cards = float(sum(1 for c in cards if c["team"] == opponent and c["minute"] <= cutoff_minute))
```
and add to the returned dict:
```python
        "own_red_cards": own_red_cards,
        "opp_red_cards": opp_red_cards,
```

- [ ] **Step 4: Run the full suite**

Run: `.venv\Scripts\pytest.exe -q`
Expected: all pass (prior count + 3 new)

- [ ] **Step 5: Commit**

```powershell
git add src/goles/backtest.py src/goles/features.py tests/test_backtest.py tests/test_features.py
git commit -m "feat: add minute-aware red-card features to compute_ml_features"
```

---

### Task 8: Wire all 8 new features into the dataset, retrain, report

**Files:**
- Modify: `src/goles/dataset.py` (`build_dataset` computes and attaches all 8 new features; `FEATURE_NAMES` 28 → 36)
- Test: `tests/test_dataset.py` (append)

**Interfaces:**
- Consumes: `goles.priors.days_since_last_match` (Task 4), the 4 `market_*_wp` columns on `matches` (Task 2/3), `goles.backtest.load_match_cards` (Task 7).
- Produces: `FEATURE_NAMES` gains `own_rest_days`, `opp_rest_days`, `own_market_wp`, `opp_market_wp`, `market_draw_wp`, `market_over25_wp`, `own_red_cards`, `opp_red_cards` (36 total).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dataset.py`:
```python
def test_build_dataset_includes_market_and_rest_features(tmp_path):
    conn = get_connection(":memory:")
    init_db(conn)
    _seed_multi_season_matches(conn)
    # give the SeasonA match's home team (Team A) a market row directly,
    # bypassing the odds loader (out of scope for this test)
    conn.execute(
        """UPDATE matches SET market_home_wp = 0.6, market_draw_wp = 0.25,
           market_away_wp = 0.15, market_over25_wp = 0.55
           WHERE understat_id = 1"""
    )
    conn.commit()

    rows = build_dataset(conn, cutoff_minutes=[20])
    row_a_home = next(r for r in rows if r.match_id == rows[0].match_id and r.team == "home")

    assert "own_rest_days" in row_a_home.features
    assert "opp_rest_days" in row_a_home.features
    assert row_a_home.features["own_market_wp"] == 0.6
    assert row_a_home.features["market_draw_wp"] == 0.25
    assert row_a_home.features["market_over25_wp"] == 0.55
    # Team A's first match of the season -> no prior fixture -> defaulted, not crashed
    assert row_a_home.features["own_rest_days"] == 7.0
    # no cards seeded for this match -> both default to 0
    assert row_a_home.features["own_red_cards"] == 0.0
    assert row_a_home.features["opp_red_cards"] == 0.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv\Scripts\pytest.exe tests/test_dataset.py -v -k market_and_rest`
Expected: FAIL with `KeyError: 'own_rest_days'`

- [ ] **Step 3: Implement**

In `src/goles/dataset.py`:

(a) Add to `FEATURE_NAMES` (before `"trailing_prior_xg"`): `"own_rest_days", "opp_rest_days", "own_market_wp", "opp_market_wp", "market_draw_wp", "market_over25_wp", "own_red_cards", "opp_red_cards",`.

(b) Add `from goles.priors import days_since_last_match, trailing_xg_per90` (extend the existing priors import) and add `load_match_cards` to the existing `from goles.backtest import (...)` block.

(c) Inside `build_dataset`'s match loop, change the matches query to also select the 4 market columns:
```python
    matches = conn.execute(
        """SELECT match_id, home_team_id, away_team_id, league, season,
                  market_home_wp, market_draw_wp, market_away_wp, market_over25_wp
           FROM matches"""
    ).fetchall()
```
and unpack the extra 4 values in the `for` loop header (`for match_id, home_team_id, away_team_id, league, season, market_home_wp, market_draw_wp, market_away_wp, market_over25_wp in matches:`).

(d) Still inside the per-match loop but before the `for team, team_id in (...)` loop (i.e. computed once per match, like `shots`), add:
```python
        cards = load_match_cards(conn, match_id, home_team_id, away_team_id)
```

(e) Inside the `for team, team_id in (("home", home_team_id), ("away", away_team_id)):` loop, right after computing `prior`, add:
```python
                rest_days = days_since_last_match(conn, team_id, league, season, match_id)
                rest_days = rest_days if rest_days is not None else 7.0  # default: typical off-season/international-break gap
                own_market_wp = market_home_wp if team == "home" else market_away_wp
```

(f) Change the existing `ml_features = compute_ml_features(shots, cutoff, team)` call to pass cards through: `ml_features = compute_ml_features(shots, cutoff, team, cards=cards)`.

(g) Extend the `full_features` dict assembly with:
```python
                full_features["own_rest_days"] = rest_days
                full_features["opp_rest_days"] = (
                    days_since_last_match(conn, away_team_id if team == "home" else home_team_id, league, season, match_id)
                    or 7.0
                )
                full_features["own_market_wp"] = own_market_wp if own_market_wp is not None else 0.0
                full_features["market_draw_wp"] = market_draw_wp if market_draw_wp is not None else 0.0
                full_features["market_over25_wp"] = market_over25_wp if market_over25_wp is not None else 0.0
```

Note the missing-market default is `0.0`, not a "neutral" value like 0.33 — this is intentional: it lets the tree distinguish "no market data available" (all three wp features simultaneously near 0, an unusual joint pattern) from a genuine long-shot (~0.0 alone on one side with the other two summing near 1.0). Document this reasoning as a code comment. (`own_red_cards`/`opp_red_cards` need no explicit assembly here — they already come back inside `ml_features`/`full_features` from `compute_ml_features` itself, per Task 7.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\pytest.exe tests/test_dataset.py -v`
Expected: all pass

- [ ] **Step 5: Run the full suite**

Run: `.venv\Scripts\pytest.exe -q`
Expected: all pass, no regressions

- [ ] **Step 6: Commit the code**

```powershell
git add src/goles/dataset.py tests/test_dataset.py
git commit -m "feat: wire market-odds, rest-days and red-card features into the training dataset"
```

- [ ] **Step 7: Retrain and compare (manual verification)**

Run:
```powershell
python -m goles.train_gbt
python -m goles.train_gbt_replication
```

Baseline to beat (post shot-enrichment, recorded in the enrichment plan): **BSS 0.0190** (test 2324), **BSS 0.0160** (réplica test 2223). Apply the same decision rule as the prior plan: improvement beyond ±0.002 in the SAME direction on both runs is a real signal; flat means keep-but-don't-celebrate; a meaningful regression on both runs means revert this task's `FEATURE_NAMES` additions (keep Tasks 1-7's data plumbing regardless — market odds, rest-days, and red cards are useful to have stored even if a particular featurization doesn't help).

Check feature importance specifically for `own_market_wp`/`market_draw_wp`/`market_over25_wp` (near-zero would be a surprise given they encode information no other feature has — double-check odds ingestion coverage from Task 3 wasn't silently low) and for `opp_red_cards` (per the research behind this plan, this one has the single largest documented individual effect size of anything added in this plan — a near-zero importance here is the most surprising possible outcome and worth a specific sanity check: query `SELECT COUNT(*) FROM cards` and cross-check against Task 6's "matches with a red card" printout before concluding the feature is genuinely unhelpful rather than under-populated).

- [ ] **Step 8: Record the result and commit**

Append a "## Resultado" section to this plan file with the real numbers and feature-importance findings (same format as the previous plan), then:

```powershell
git add docs/superpowers/plans/2026-07-11-goal-predictor-market-rest-features.md
git commit -m "docs: record market-odds, rest-days and red-card retrain results"
```

## Próximos pasos (fuera de alcance de este plan)

Remaining from the GBT plan's backlog: bootstrap confidence interval (optional hardening), ClubElo wiring (now lower priority given market odds cover similar ground — pre-match team-strength signal — with less engineering risk, since we already solved the team-name-matching problem here and could reuse the same alias-table pattern for ClubElo later if still wanted). Model persistence already exists (`src/goles/persistence.py`) — if this plan's features improve the model, the persisted model must be regenerated from the new training run before Phase 2 uses it. If red cards prove valuable, Phase 2's live pipeline needs a live source for card events with minute timestamps (Sofascore/FotMob both surface cards live, typically with minute — needs verification when Phase 2 is scoped) since Understat's raw cache obviously has no live equivalent.
