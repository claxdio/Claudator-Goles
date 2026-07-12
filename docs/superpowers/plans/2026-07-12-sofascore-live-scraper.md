# Sofascore Live Match-State Scraper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a service that continuously mirrors live shot and red-card events for the two tracked leagues (ENG-Premier League, GER-Bundesliga) from Sofascore into a local SQLite database, running on the developer's home Windows PC (not the VPS — Sofascore blocks datacenter IPs, verified during design) and syncing to the VPS via `scp` after each poll cycle.

**Architecture:** A new subpackage `src/goles/sofascore/` (`client.py`, `team_aliases.py`, `store.py`, `poller.py`) using `tls_requests.Client()` — the same TLS-impersonation HTTP client `soccerdata` already depends on and uses internally — against Sofascore's real (undocumented but verified-working) endpoints for live events, shot maps, and incidents. Deployed via Windows Task Scheduler rather than Docker/Dokploy.

**Tech Stack:** Python 3.11+, `tls_requests` (via the `wrapper-tls-requests` PyPI package, already an indirect dependency through `soccerdata` — added directly to `pyproject.toml` since this plan imports it directly), `sqlite3` (stdlib), `pytest`. No new paid services.

## Global Constraints

- Every Sofascore HTTP call must be mocked in tests via a duck-typed stub client (an object with a `.get()` method) — no test may require real network access, matching the rest of this project's discipline.
- `data/live_match_state.db` is a new, separate SQLite database — distinct from `data/goles.db` (historical training data) and the VPS-side `live_odds.db`. It lives under the existing gitignored `data/` directory.
- Only **red-card** incidents are persisted (yellow cards excluded at ingestion) — same rationale already documented in the market-rest-features plan: yellow cards have a much weaker/noisier documented relationship to goal-timing.
- The exact Sofascore `incidentClass` value(s) meaning "red card" are **assumed** (`"red"` for a straight red, `"yellowRed"` for a second-yellow send-off) based on community documentation of this API, not yet confirmed against a real observed red card during design (none occurred in the live match sampled). Task 5's manual verification step must confirm this against a real occurrence and the filter must be corrected if wrong — never silently miss real red cards.
- This service intentionally does **not** run on the VPS: Sofascore returns HTTP 403 from both the VPS's own IP and the UK proxy VM built for the Betfair pipeline (confirmed not a geo-block, not a UDP-buffer/QUIC artifact, and not a transient rate limit — see the design spec's "Actualización" section). It runs on the developer's home Windows PC instead, which is confirmed to work.
- No new paid services (same constraint as every prior plan in this project).
- Translating raw Sofascore fields into `compute_ml_features`'s exact input shape, running the model, and Telegram delivery are explicitly out of scope for this plan.

---

### Task 1: Sofascore HTTP client wrappers

**Files:**
- Create: `src/goles/sofascore/__init__.py` (empty)
- Create: `src/goles/sofascore/client.py`
- Test: `tests/test_sofascore_client.py`
- Modify: `pyproject.toml` (add `wrapper-tls-requests` dependency)

**Interfaces:**
- Consumes: nothing from other tasks — this task's functions take any object with a `.get(url) -> response` method (duck-typed; tests pass a stub, production passes a real `tls_requests.Client()`).
- Produces: `BASE_URL: str`, `list_live_events(client) -> list[dict]`, `get_shotmap(client, event_id: int) -> list[dict]`, `get_incidents(client, event_id: int) -> list[dict]`.

- [ ] **Step 1: Add the dependency**

In `pyproject.toml`, add to the `dependencies` list (after `"PySocks>=1.7.1",`):
```toml
    "wrapper-tls-requests>=1.2.5",
```

- [ ] **Step 2: Write the failing tests**

`tests/test_sofascore_client.py`:
```python
from unittest.mock import Mock

from goles.sofascore.client import get_incidents, get_shotmap, list_live_events


def _stub_client(json_body):
    response = Mock()
    response.json.return_value = json_body
    response.raise_for_status = Mock()
    client = Mock()
    client.get = Mock(return_value=response)
    return client


def test_list_live_events_returns_events_list():
    client = _stub_client({"events": [{"id": 1, "tournament": {"name": "Premier League"}}]})
    events = list_live_events(client)
    assert events == [{"id": 1, "tournament": {"name": "Premier League"}}]
    client.get.assert_called_once_with("https://api.sofascore.com/api/v1/sport/football/events/live")


def test_list_live_events_returns_empty_list_when_missing_key():
    client = _stub_client({})
    assert list_live_events(client) == []


def test_get_shotmap_returns_shotmap_list():
    client = _stub_client({"shotmap": [{"id": 123, "time": 10, "xg": 0.2}]})
    shots = get_shotmap(client, 999)
    assert shots == [{"id": 123, "time": 10, "xg": 0.2}]
    client.get.assert_called_once_with("https://api.sofascore.com/api/v1/event/999/shotmap")


def test_get_incidents_returns_incidents_list():
    client = _stub_client({"incidents": [{"time": 45, "incidentType": "period"}]})
    incidents = get_incidents(client, 999)
    assert incidents == [{"time": 45, "incidentType": "period"}]
    client.get.assert_called_once_with("https://api.sofascore.com/api/v1/event/999/incidents")
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv\Scripts\pytest.exe tests/test_sofascore_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'goles.sofascore'`

- [ ] **Step 4: Write the implementation**

`src/goles/sofascore/__init__.py`: empty file.

`src/goles/sofascore/client.py`:
```python
from __future__ import annotations

BASE_URL = "https://api.sofascore.com/api/v1"


def list_live_events(client) -> list[dict]:
    """Returns all currently live football events worldwide (unfiltered by
    league -- callers filter by tournament name, see poller.py)."""
    response = client.get(f"{BASE_URL}/sport/football/events/live")
    response.raise_for_status()
    return response.json().get("events", [])


def get_shotmap(client, event_id: int) -> list[dict]:
    """Returns the raw shot list for a live or completed event."""
    response = client.get(f"{BASE_URL}/event/{event_id}/shotmap")
    response.raise_for_status()
    return response.json().get("shotmap", [])


def get_incidents(client, event_id: int) -> list[dict]:
    """Returns the raw incident list (goals, cards, periods, ...) for a
    live or completed event."""
    response = client.get(f"{BASE_URL}/event/{event_id}/incidents")
    response.raise_for_status()
    return response.json().get("incidents", [])
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv\Scripts\pytest.exe tests/test_sofascore_client.py -v`
Expected: 4 passed

- [ ] **Step 6: Install the new dependency locally and run the full suite**

Run: `.venv\Scripts\pip.exe install "wrapper-tls-requests>=1.2.5" -q`
Run: `.venv\Scripts\pytest.exe -q`
Expected: all pass, no regressions

- [ ] **Step 7: Commit**

```powershell
git add pyproject.toml src/goles/sofascore/__init__.py src/goles/sofascore/client.py tests/test_sofascore_client.py
git commit -m "feat: add Sofascore HTTP client wrappers"
```

---

### Task 2: Sofascore team-name alias table

**Files:**
- Create: `src/goles/sofascore/team_aliases.py`
- Test: `tests/test_sofascore_team_aliases.py`

**Interfaces:**
- Produces: `SOFASCORE_TEAM_NAME_ALIASES: dict[str, str]`, `normalize_sofascore_team_name(name: str) -> str`.

- [ ] **Step 1: Write the failing tests**

`tests/test_sofascore_team_aliases.py`:
```python
from goles.sofascore.team_aliases import SOFASCORE_TEAM_NAME_ALIASES, normalize_sofascore_team_name


def test_normalize_sofascore_team_name_passes_through_unmapped_names():
    assert normalize_sofascore_team_name("Arsenal") == "Arsenal"
    assert normalize_sofascore_team_name("Some Unmapped Team") == "Some Unmapped Team"


def test_sofascore_team_name_aliases_has_no_identity_entries():
    for sofascore_name, our_name in SOFASCORE_TEAM_NAME_ALIASES.items():
        assert sofascore_name != our_name
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\pytest.exe tests/test_sofascore_team_aliases.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'goles.sofascore.team_aliases'`

- [ ] **Step 3: Write the implementation**

`src/goles/sofascore/team_aliases.py`:
```python
from __future__ import annotations

# Starter table -- Sofascore's exact team-name strings for our tracked
# teams haven't been observed yet (same honest-starter-table precedent as
# goles/betfair/team_aliases.py). Extend this table with real observed
# aliases once the poller runs against production and a fixture's team
# name doesn't match our Understat-sourced `teams` table -- never guess an
# entry without having seen the real Sofascore name it maps from.
SOFASCORE_TEAM_NAME_ALIASES: dict[str, str] = {}


def normalize_sofascore_team_name(name: str) -> str:
    """Maps a Sofascore team name to our Understat-sourced team name.
    Names not in SOFASCORE_TEAM_NAME_ALIASES are assumed identical and
    returned unchanged."""
    return SOFASCORE_TEAM_NAME_ALIASES.get(name, name)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\pytest.exe tests/test_sofascore_team_aliases.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```powershell
git add src/goles/sofascore/team_aliases.py tests/test_sofascore_team_aliases.py
git commit -m "feat: add Sofascore team-name alias table"
```

---

### Task 3: Live match-state SQLite store

**Files:**
- Create: `src/goles/sofascore/store.py`
- Test: `tests/test_sofascore_store.py`

**Interfaces:**
- Produces: `DEFAULT_LIVE_MATCH_STATE_DB_PATH: Path`, `get_connection(db_path=DEFAULT_LIVE_MATCH_STATE_DB_PATH) -> sqlite3.Connection`, `init_db(conn) -> None`, `persist_shot(conn, sofascore_shot_id, sofascore_event_id, fetched_at, home_team, away_team, team, minute, xg, is_goal, shot_type, situation=None, location_x=None, location_y=None, body_part=None) -> bool` (returns `True` if a new row was inserted, `False` if `sofascore_shot_id` already existed), `persist_card(conn, sofascore_event_id, fetched_at, home_team, away_team, team, minute, card_type) -> bool` (same idempotent-insert contract, keyed on `(sofascore_event_id, team, minute)`).

**Design note:** `home_team`/`away_team` (the normalized team names for the whole fixture) are denormalized onto every shot/card row — same pattern already used by `goles.betfair.odds_store`'s `odds_snapshots` table — rather than requiring a separate join table. Without them, rows keyed only by a numeric `sofascore_event_id` would be nearly meaningless once multiple matches accumulate; this is also what makes `team_aliases.py` (Task 2) actually get used, by the poller in Task 4.

- [ ] **Step 1: Write the failing tests**

`tests/test_sofascore_store.py`:
```python
from goles.sofascore.store import get_connection, init_db, persist_card, persist_shot


def test_init_db_creates_shots_and_cards_tables():
    conn = get_connection(":memory:")
    init_db(conn)
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cur.fetchall()}
    assert {"shots", "cards"} <= tables


def test_persist_shot_inserts_a_row_and_returns_true():
    conn = get_connection(":memory:")
    init_db(conn)
    inserted = persist_shot(
        conn,
        sofascore_shot_id=7684954,
        sofascore_event_id=12813015,
        fetched_at="2026-07-12T00:00:00+00:00",
        home_team="Arsenal",
        away_team="Chelsea",
        team="home",
        minute=20,
        xg=0.185,
        is_goal=True,
        shot_type="goal",
        situation="corner",
        location_x=5.0,
        location_y=44.1,
        body_part="head",
    )
    assert inserted is True
    row = conn.execute(
        "SELECT home_team, away_team, team, minute, xg, is_goal, shot_type, situation FROM shots"
    ).fetchone()
    assert row == ("Arsenal", "Chelsea", "home", 20, 0.185, 1, "goal", "corner")


def test_persist_shot_is_idempotent_on_sofascore_shot_id():
    conn = get_connection(":memory:")
    init_db(conn)
    kwargs = dict(
        sofascore_shot_id=7684954,
        sofascore_event_id=12813015,
        fetched_at="2026-07-12T00:00:00+00:00",
        home_team="Arsenal",
        away_team="Chelsea",
        team="home",
        minute=20,
        xg=0.185,
        is_goal=True,
        shot_type="goal",
    )
    first = persist_shot(conn, **kwargs)
    second = persist_shot(conn, **kwargs)
    assert first is True
    assert second is False
    count = conn.execute("SELECT COUNT(*) FROM shots").fetchone()[0]
    assert count == 1


def test_persist_card_inserts_a_row_and_returns_true():
    conn = get_connection(":memory:")
    init_db(conn)
    inserted = persist_card(
        conn, sofascore_event_id=12813015, fetched_at="2026-07-12T00:00:00+00:00",
        home_team="Arsenal", away_team="Chelsea", team="away", minute=55, card_type="red",
    )
    assert inserted is True
    row = conn.execute("SELECT home_team, away_team, team, minute, card_type FROM cards").fetchone()
    assert row == ("Arsenal", "Chelsea", "away", 55, "red")


def test_persist_card_is_idempotent_on_event_team_minute():
    conn = get_connection(":memory:")
    init_db(conn)
    kwargs = dict(
        sofascore_event_id=12813015, fetched_at="2026-07-12T00:00:00+00:00",
        home_team="Arsenal", away_team="Chelsea", team="away", minute=55, card_type="red",
    )
    first = persist_card(conn, **kwargs)
    second = persist_card(conn, **kwargs)
    assert first is True
    assert second is False
    count = conn.execute("SELECT COUNT(*) FROM cards").fetchone()[0]
    assert count == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\pytest.exe tests/test_sofascore_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'goles.sofascore.store'`

- [ ] **Step 3: Write the implementation**

`src/goles/sofascore/store.py`:
```python
from __future__ import annotations

import sqlite3
from pathlib import Path

DEFAULT_LIVE_MATCH_STATE_DB_PATH = Path("data") / "live_match_state.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS shots (
    shot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    sofascore_shot_id INTEGER NOT NULL UNIQUE,
    sofascore_event_id INTEGER NOT NULL,
    fetched_at TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    team TEXT NOT NULL,
    minute INTEGER NOT NULL,
    xg REAL NOT NULL,
    is_goal INTEGER NOT NULL,
    shot_type TEXT NOT NULL,
    situation TEXT,
    location_x REAL,
    location_y REAL,
    body_part TEXT
);

CREATE TABLE IF NOT EXISTS cards (
    card_id INTEGER PRIMARY KEY AUTOINCREMENT,
    sofascore_event_id INTEGER NOT NULL,
    fetched_at TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    team TEXT NOT NULL,
    minute INTEGER NOT NULL,
    card_type TEXT NOT NULL,
    UNIQUE(sofascore_event_id, team, minute)
);
"""


def get_connection(db_path: str | Path = DEFAULT_LIVE_MATCH_STATE_DB_PATH) -> sqlite3.Connection:
    """Opens (creating parent directories if needed) the live match-state
    SQLite database -- a separate file from both data/goles.db (historical
    training data) and the VPS-side live_odds.db."""
    if str(db_path) != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(db_path))


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def persist_shot(
    conn: sqlite3.Connection,
    sofascore_shot_id: int,
    sofascore_event_id: int,
    fetched_at: str,
    home_team: str,
    away_team: str,
    team: str,
    minute: int,
    xg: float,
    is_goal: bool,
    shot_type: str,
    situation: str | None = None,
    location_x: float | None = None,
    location_y: float | None = None,
    body_part: str | None = None,
) -> bool:
    """Inserts a shot row keyed on Sofascore's own stable per-shot id.
    `home_team`/`away_team` (already normalized by the caller) are
    denormalized onto every row so a row is meaningful on its own, without
    a join back to some other fixtures table this plan doesn't build.
    Returns True if a new row was inserted, False if sofascore_shot_id was
    already present (idempotent re-polling -- the poller re-fetches the
    full shotmap every cycle)."""
    cursor = conn.execute(
        """INSERT OR IGNORE INTO shots
           (sofascore_shot_id, sofascore_event_id, fetched_at, home_team, away_team, team, minute,
            xg, is_goal, shot_type, situation, location_x, location_y, body_part)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            sofascore_shot_id, sofascore_event_id, fetched_at, home_team, away_team, team, minute,
            xg, int(is_goal), shot_type, situation, location_x, location_y, body_part,
        ),
    )
    conn.commit()
    return cursor.rowcount > 0


def persist_card(
    conn: sqlite3.Connection,
    sofascore_event_id: int,
    fetched_at: str,
    home_team: str,
    away_team: str,
    team: str,
    minute: int,
    card_type: str,
) -> bool:
    """Inserts a red-card row keyed on (event, team, minute) -- incidents
    have no stable per-item id from Sofascore, but two red cards for the
    same team in the same real match minute is not a realistic collision.
    Returns True if newly inserted, False if already present."""
    cursor = conn.execute(
        """INSERT OR IGNORE INTO cards (sofascore_event_id, fetched_at, home_team, away_team, team, minute, card_type)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (sofascore_event_id, fetched_at, home_team, away_team, team, minute, card_type),
    )
    conn.commit()
    return cursor.rowcount > 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\pytest.exe tests/test_sofascore_store.py -v`
Expected: 5 passed

- [ ] **Step 5: Run the full suite, then commit**

Run: `.venv\Scripts\pytest.exe -q` → expected all pass, no regressions.

```powershell
git add src/goles/sofascore/store.py tests/test_sofascore_store.py
git commit -m "feat: add live match-state SQLite store"
```

---

### Task 4: Poller (discovery, one poll cycle, VPS sync, entrypoint)

**Files:**
- Create: `src/goles/sofascore/poller.py`
- Test: `tests/test_sofascore_poller.py`

**Interfaces:**
- Consumes: `goles.sofascore.client.{list_live_events, get_shotmap, get_incidents}`, `goles.sofascore.team_aliases.normalize_sofascore_team_name`, `goles.sofascore.store.{get_connection, init_db, persist_shot, persist_card, DEFAULT_LIVE_MATCH_STATE_DB_PATH}`.
- Produces: `TRACKED_TOURNAMENTS: list[str]`, `RED_CARD_INCIDENT_CLASSES: set[str]`, `discover_tracked_live_events(client) -> list[dict]`, `poll_once(client, conn, live_events: list[dict]) -> None`, `sync_to_vps(db_path=DEFAULT_LIVE_MATCH_STATE_DB_PATH) -> None`.

- [ ] **Step 1: Write the failing tests**

`tests/test_sofascore_poller.py`:
```python
from unittest.mock import Mock, patch

from goles.sofascore.poller import discover_tracked_live_events, poll_once, sync_to_vps
from goles.sofascore.store import get_connection, init_db


def test_discover_tracked_live_events_filters_by_exact_tournament_name():
    client = Mock()
    with patch("goles.sofascore.poller.list_live_events", return_value=[
        {"id": 1, "tournament": {"name": "Premier League"}},
        {"id": 2, "tournament": {"name": "Scottish Premiership"}},
        {"id": 3, "tournament": {"name": "Bundesliga"}},
    ]):
        events = discover_tracked_live_events(client)
    assert [e["id"] for e in events] == [1, 3]


def test_poll_once_persists_shots_and_red_cards():
    event = {
        "id": 12813015,
        "homeTeam": {"name": "Arsenal"},
        "awayTeam": {"name": "Chelsea"},
        "tournament": {"name": "Premier League"},
    }
    shots = [
        {
            "id": 7684954, "time": 20, "xg": 0.185, "shotType": "goal",
            "situation": "corner", "isHome": True,
            "playerCoordinates": {"x": 5.0, "y": 44.1}, "bodyPart": "head",
        },
        {
            "id": 7684839, "time": 10, "xg": 0.056, "shotType": "miss",
            "situation": "regular", "isHome": False,
            "playerCoordinates": {"x": 10.0, "y": 30.0}, "bodyPart": "right-foot",
        },
    ]
    incidents = [
        {"time": 45, "incidentType": "period"},
        {"time": 55, "incidentType": "card", "incidentClass": "red", "isHome": False},
        {"time": 60, "incidentType": "card", "incidentClass": "yellow", "isHome": True},
    ]
    conn = get_connection(":memory:")
    init_db(conn)
    client = Mock()

    with patch("goles.sofascore.poller.get_shotmap", return_value=shots):
        with patch("goles.sofascore.poller.get_incidents", return_value=incidents):
            poll_once(client, conn, [event])

    shot_rows = conn.execute(
        "SELECT home_team, away_team, team, minute, is_goal FROM shots ORDER BY minute"
    ).fetchall()
    assert shot_rows == [
        ("Arsenal", "Chelsea", "away", 10, 0),
        ("Arsenal", "Chelsea", "home", 20, 1),
    ]

    card_rows = conn.execute("SELECT home_team, away_team, team, minute, card_type FROM cards").fetchall()
    assert card_rows == [("Arsenal", "Chelsea", "away", 55, "red")]  # only the red card, not the yellow


def test_sync_to_vps_invokes_scp_with_expected_arguments():
    with patch("goles.sofascore.poller.subprocess.run") as mock_run:
        sync_to_vps(db_path="data/live_match_state.db")
    mock_run.assert_called_once()
    args, kwargs = mock_run.call_args
    command = args[0]
    assert command[0] == "scp"
    assert "data/live_match_state.db" in command
    assert command[-1].endswith(":/root/goles-live-match-state/live_match_state.db")
    assert kwargs["check"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\pytest.exe tests/test_sofascore_poller.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'goles.sofascore.poller'`

- [ ] **Step 3: Write the implementation**

`src/goles/sofascore/poller.py`:
```python
from __future__ import annotations

import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import tls_requests

from goles.sofascore.client import get_incidents, get_shotmap, list_live_events
from goles.sofascore.store import DEFAULT_LIVE_MATCH_STATE_DB_PATH, get_connection, init_db, persist_card, persist_shot
from goles.sofascore.team_aliases import normalize_sofascore_team_name

TRACKED_TOURNAMENTS = ["Premier League", "Bundesliga"]
# Assumed from community documentation of Sofascore's incident vocabulary,
# NOT yet confirmed against a real observed red card (none occurred in the
# live match sampled during design). Verify against a real occurrence
# during Task 5's manual verification step and correct if wrong.
RED_CARD_INCIDENT_CLASSES = {"red", "yellowRed"}
POLL_INTERVAL_SECONDS = 60

VPS_HOST = "root@85.239.245.73"
VPS_SSH_KEY = str(Path.home() / ".ssh" / "id_ed25519_goles_vps")
VPS_REMOTE_PATH = "/root/goles-live-match-state/live_match_state.db"


def discover_tracked_live_events(client) -> list[dict]:
    """Returns live events whose tournament name exactly matches one of
    TRACKED_TOURNAMENTS (exact match, not substring -- avoids false
    positives like "Scottish Premiership" matching "Premier League")."""
    events = list_live_events(client)
    return [e for e in events if e.get("tournament", {}).get("name") in TRACKED_TOURNAMENTS]


def poll_once(client, conn, live_events: list[dict]) -> None:
    """Fetches the shotmap and incidents for every tracked live event and
    persists new shots and red cards (idempotent -- see store.py). Each
    event's home/away team names are normalized once and denormalized
    onto every row for that event."""
    fetched_at = datetime.now(timezone.utc).isoformat()
    for event in live_events:
        event_id = event["id"]
        home_team = normalize_sofascore_team_name(event["homeTeam"]["name"])
        away_team = normalize_sofascore_team_name(event["awayTeam"]["name"])

        shots = get_shotmap(client, event_id)
        for shot in shots:
            team = "home" if shot.get("isHome") else "away"
            coordinates = shot.get("playerCoordinates") or {}
            persist_shot(
                conn,
                sofascore_shot_id=shot["id"],
                sofascore_event_id=event_id,
                fetched_at=fetched_at,
                home_team=home_team,
                away_team=away_team,
                team=team,
                minute=shot["time"],
                xg=shot["xg"],
                is_goal=shot.get("shotType") == "goal",
                shot_type=shot.get("shotType", ""),
                situation=shot.get("situation"),
                location_x=coordinates.get("x"),
                location_y=coordinates.get("y"),
                body_part=shot.get("bodyPart"),
            )

        incidents = get_incidents(client, event_id)
        for incident in incidents:
            if incident.get("incidentType") != "card":
                continue
            incident_class = incident.get("incidentClass")
            if incident_class not in RED_CARD_INCIDENT_CLASSES:
                continue
            team = "home" if incident.get("isHome") else "away"
            persist_card(
                conn, event_id, fetched_at, home_team, away_team, team, incident["time"], incident_class
            )


def sync_to_vps(db_path: str | Path = DEFAULT_LIVE_MATCH_STATE_DB_PATH) -> None:
    """Copies the local live-match-state SQLite file to the VPS via scp,
    reusing the dedicated SSH key already set up for VPS access. Raises on
    failure -- main()'s broad except catches it and retries next cycle."""
    subprocess.run(
        [
            "scp", "-i", VPS_SSH_KEY, "-o", "StrictHostKeyChecking=accept-new",
            str(db_path), f"{VPS_HOST}:{VPS_REMOTE_PATH}",
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )


def main() -> None:
    """Persistent entrypoint: polls forever, syncing to the VPS after each
    cycle. Discovery/poll/sync failures are caught, printed, and retried
    on the next loop iteration rather than crashing the process."""
    client = tls_requests.Client()
    conn = get_connection()
    init_db(conn)

    while True:
        try:
            live_events = discover_tracked_live_events(client)
            print(f"{len(live_events)} partidos en vivo encontrados en las ligas trackeadas.")
            if live_events:
                poll_once(client, conn, live_events)
            sync_to_vps()
        except Exception as exc:
            print(f"ADVERTENCIA: fallo en el ciclo de polling ({exc}), se reintenta en el proximo ciclo.")
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\pytest.exe tests/test_sofascore_poller.py -v`
Expected: 3 passed

- [ ] **Step 5: Run the full suite, then commit**

Run: `.venv\Scripts\pytest.exe -q` → expected all pass, no regressions.

```powershell
git add src/goles/sofascore/poller.py tests/test_sofascore_poller.py
git commit -m "feat: add Sofascore live match-state poller"
```

---

### Task 5: Deploy on the home PC via Windows Task Scheduler (manual verification)

**Files:** none (operational setup).

No automated tests — this is an operational deployment step, same precedent as `ingest_odds.py`/`ingest_cards.py` and the Betfair poller's Dokploy deployment.

- [ ] **Step 1: Create the remote directory on the VPS for the synced file**

Run:
```bash
ssh -i ~/.ssh/id_ed25519_goles_vps root@85.239.245.73 "mkdir -p /root/goles-live-match-state"
```

- [ ] **Step 2: Register the Windows Scheduled Task**

Run (PowerShell, as the same user who will be logged in when this should run):
```powershell
$action = New-ScheduledTaskAction -Execute "$PWD\.venv\Scripts\python.exe" -Argument "-m goles.sofascore.poller" -WorkingDirectory "$PWD"
$trigger = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit (New-TimeSpan -Days 0)
Register-ScheduledTask -TaskName "GolesSofascorePoller" -Action $action -Trigger $trigger -Settings $settings -Description "Live Sofascore shot/card scraper for the goles project"
```
Expected: the task appears in Task Scheduler (`taskschd.msc`) under name `GolesSofascorePoller`, with the restart-on-failure settings applied.

- [ ] **Step 3: Start it now (don't wait for next logon) and verify it's running**

Run: `Start-ScheduledTask -TaskName "GolesSofascorePoller"`
Run: `Get-ScheduledTaskInfo -TaskName "GolesSofascorePoller"` — expected `LastTaskResult` is `0` or the task shows as running (a running long-lived process has no result yet).

- [ ] **Step 4: Verify real data is flowing**

Wait a few minutes (long enough for at least one poll cycle against any currently-live match in the tracked leagues — there may be none if it's not matchday; if so, confirm the process is at least running and printing the "0 partidos en vivo" line rather than crashing, and re-verify with real data during an actual live match window).

Run: `.venv\Scripts\python.exe -c "import sqlite3; conn = sqlite3.connect('data/live_match_state.db'); print(conn.execute('SELECT COUNT(*) FROM shots').fetchone()); print(conn.execute('SELECT COUNT(*) FROM cards').fetchone())"`

Confirm the file also landed on the VPS: `ssh -i ~/.ssh/id_ed25519_goles_vps root@85.239.245.73 "ls -la /root/goles-live-match-state/"`.

- [ ] **Step 5: Verify the red-card filter against a real occurrence, once observed**

The first time a real red card occurs in a tracked live match while this is running, query `cards` and cross-check against the match's actual events (e.g. via the Sofascore website) to confirm `RED_CARD_INCIDENT_CLASSES = {"red", "yellowRed"}` in `src/goles/sofascore/poller.py` is correct. If Sofascore's real `incidentClass` value differs, fix the constant, add a regression test with the real observed value, and commit the fix.

- [ ] **Step 6: Record the deployment state**

Append a `## Estado de despliegue` section to `docs/superpowers/specs/2026-07-12-sofascore-live-scraper-design.md` with: confirmation the Task Scheduler task runs and restarts correctly, the observed shot/card counts from a real poll window, confirmation the VPS sync works, and the outcome of the red-card-class verification (Step 5) whenever a real red card is first observed. Commit:

```powershell
git add docs/superpowers/specs/2026-07-12-sofascore-live-scraper-design.md
git commit -m "docs: record Sofascore live scraper deployment state"
```
