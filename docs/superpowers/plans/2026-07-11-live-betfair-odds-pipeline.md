# Live Betfair Odds Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and deploy a standalone service that logs into Betfair Exchange (non-interactive certificate auth), continuously polls match-odds and over/under-2.5 prices for the two tracked leagues, computes no-vig probabilities, and persists timestamped snapshots to its own SQLite database on the VPS — the foundation Phase 2's live inference and Telegram bot will later read from.

**Architecture:** A new subpackage `src/goles/betfair/` (`auth.py`, `client.py`, `odds_store.py`, `team_aliases.py`, `poller.py`), fully separate from the training pipeline. Every Betfair HTTP interaction is unit-tested via mocks (no test needs real credentials or network access). It ships as its own Docker image, deployed as a new, isolated Application in the Dokploy project (`Claudator-Goles`) already created on the VPS, running `python -m goles.betfair.poller` with `restart: always`.

**Tech Stack:** Python 3.11+, `requests` (already a project dependency — no new dependencies), `sqlite3` (stdlib), `pytest`. Docker for deployment.

## Global Constraints

- No automated bet placement anywhere in this code — every call is read-only (`listCompetitions`/`listMarketCatalogue`/`listMarketBook`), never `placeOrders`.
- Every Betfair HTTP call must be mocked in tests — no test in this plan may require real network access or real Betfair credentials (same rule the rest of this project already follows).
- `live_odds.db` is a **new, separate SQLite database** from `data/goles.db` (the historical training DB) — default path `data/live_odds.db`, also gitignored (already covered by the existing `data/` entry in `.gitignore`).
- Never commit certificate files, keys, or real Betfair credentials to git. The certificate is generated directly on the VPS (Task 1) and never touches the developer's machine or the repo.
- Field names used for Betfair's JSON responses in this plan (`selectionId`, `runnerName`, `ex.availableToBack[].price`, `description.marketType`, `event.name`/`event.id`, `competition.id`/`name`) follow Betfair's documented API-NG conventions but have **not yet been verified against a real authenticated response** — the user's Betfair delayed App Key and certificate upload (external prerequisites, see the design spec) are still pending. Task 8's manual verification step is the first real check of these assumptions; if real responses differ, the parsing functions in Task 6 are the ones to adjust.
- The REST-style Betfair Exchange endpoint is `https://api.betfair.com/exchange/betting/rest/v1.0/{operation}/`, confirmed from Betfair's own official sample code repository (`betfair/API-NG-sample-code`) — POST with a plain JSON body per operation (not a JSON-RPC envelope).
- Session lifetime is undocumented by Betfair; `BetfairSession` re-logs in reactively (once up front, once more on any non-200 response) rather than assuming a fixed expiry.

---

### Task 1: Generate the Betfair client certificate on the VPS

**Files:** none (operational step, run over the SSH connection already established to `root@85.239.245.73`).

No automated tests — this is a one-time operational step, same precedent as this project's other manual-verification-only scripts.

- [ ] **Step 1: Generate a 2048-bit self-signed certificate on the VPS**

Run over SSH:
```bash
ssh -i ~/.ssh/id_ed25519_goles_vps root@85.239.245.73 \
  "mkdir -p /root/goles-betfair-certs && cd /root/goles-betfair-certs && \
   openssl req -x509 -newkey rsa:2048 -keyout client-2048.key -out client-2048.crt \
   -days 3650 -nodes -subj '/CN=goles-betfair-bot' && chmod 600 client-2048.key && \
   echo CERT_GENERATED"
```
Expected: prints `CERT_GENERATED`, and `/root/goles-betfair-certs/client-2048.key` + `client-2048.crt` exist on the VPS. The `.key` file never leaves the VPS.

- [ ] **Step 2: Retrieve the public certificate for the user to upload**

Run: `ssh -i ~/.ssh/id_ed25519_goles_vps root@85.239.245.73 "cat /root/goles-betfair-certs/client-2048.crt"`

The `.crt` content (public, safe to display) is given to the user so they can paste it into their Betfair account's security settings page (Betfair account → My Account → Security → API access → upload certificate). This step, and requesting the delayed Application Key via the Betfair Developer Program portal, are the two external prerequisites only the user can complete — the rest of this plan proceeds without waiting for them.

---

### Task 2: `BetfairSession` — non-interactive certificate login with reactive re-login

**Files:**
- Create: `src/goles/betfair/__init__.py` (empty)
- Create: `src/goles/betfair/auth.py`
- Test: `tests/test_betfair_auth.py`

**Interfaces:**
- Produces: `BetfairAuthError(Exception)`, `cert_login(app_key, username, password, cert_file, key_file, login_url=LOGIN_URL) -> str`, `BetfairSession(app_key, username, password, cert_file, key_file, login_url=LOGIN_URL)` with method `request(method: str, url: str, **kwargs) -> requests.Response`.

- [ ] **Step 1: Write the failing tests**

`tests/test_betfair_auth.py`:
```python
from unittest.mock import Mock, patch

import pytest

from goles.betfair.auth import BetfairAuthError, BetfairSession, cert_login


def _mock_response(json_body, status_code=200):
    response = Mock()
    response.status_code = status_code
    response.json.return_value = json_body
    response.raise_for_status = Mock()
    if status_code >= 400:
        response.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    return response


def test_cert_login_returns_session_token_on_success():
    with patch("goles.betfair.auth.requests.post", return_value=_mock_response(
        {"sessionToken": "abc123", "loginStatus": "SUCCESS"}
    )) as mock_post:
        token = cert_login("appkey", "user", "pass", "cert.crt", "cert.key")
    assert token == "abc123"
    mock_post.assert_called_once_with(
        "https://identitysso-cert.betfair.com/api/certlogin",
        cert=("cert.crt", "cert.key"),
        headers={"X-Application": "appkey", "Content-Type": "application/x-www-form-urlencoded"},
        data={"username": "user", "password": "pass"},
        timeout=30,
    )


def test_cert_login_raises_on_non_success_status():
    with patch("goles.betfair.auth.requests.post", return_value=_mock_response(
        {"sessionToken": None, "loginStatus": "INVALID_USERNAME_OR_PASSWORD"}
    )):
        with pytest.raises(BetfairAuthError, match="INVALID_USERNAME_OR_PASSWORD"):
            cert_login("appkey", "user", "wrongpass", "cert.crt", "cert.key")


def test_betfair_session_logs_in_once_then_reuses_token():
    login_response = _mock_response({"sessionToken": "tok1", "loginStatus": "SUCCESS"})
    api_response = _mock_response({"result": "ok"})
    with patch("goles.betfair.auth.requests.post", return_value=login_response) as mock_post:
        with patch("goles.betfair.auth.requests.request", return_value=api_response) as mock_request:
            session = BetfairSession("appkey", "user", "pass", "cert.crt", "cert.key")
            session.request("POST", "https://example.test/op1/", json={"a": 1})
            session.request("POST", "https://example.test/op2/", json={"b": 2})
    assert mock_post.call_count == 1  # logged in only once
    assert mock_request.call_count == 2
    _, kwargs = mock_request.call_args_list[0]
    assert kwargs["headers"]["X-Application"] == "appkey"
    assert kwargs["headers"]["X-Authentication"] == "tok1"


def test_betfair_session_relogs_in_once_on_non_200_response():
    login_response = _mock_response({"sessionToken": "tok1", "loginStatus": "SUCCESS"})
    relogin_response = _mock_response({"sessionToken": "tok2", "loginStatus": "SUCCESS"})
    failed_response = _mock_response({"error": "expired"}, status_code=401)
    success_response = _mock_response({"result": "ok"})
    with patch("goles.betfair.auth.requests.post", side_effect=[login_response, relogin_response]):
        with patch("goles.betfair.auth.requests.request", side_effect=[failed_response, success_response]) as mock_request:
            session = BetfairSession("appkey", "user", "pass", "cert.crt", "cert.key")
            response = session.request("POST", "https://example.test/op1/", json={"a": 1})
    assert response is success_response
    assert mock_request.call_count == 2
    _, kwargs = mock_request.call_args_list[1]
    assert kwargs["headers"]["X-Authentication"] == "tok2"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\pytest.exe tests/test_betfair_auth.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'goles.betfair'`

- [ ] **Step 3: Write the implementation**

`src/goles/betfair/__init__.py`: empty file.

`src/goles/betfair/auth.py`:
```python
from __future__ import annotations

import requests

LOGIN_URL = "https://identitysso-cert.betfair.com/api/certlogin"


class BetfairAuthError(Exception):
    """Raised when Betfair's certificate login does not return
    loginStatus == SUCCESS (e.g. INVALID_USERNAME_OR_PASSWORD,
    ACCOUNT_ALREADY_LOCKED)."""


def cert_login(
    app_key: str,
    username: str,
    password: str,
    cert_file: str,
    key_file: str,
    login_url: str = LOGIN_URL,
) -> str:
    """Performs Betfair's non-interactive (bot) certificate login and
    returns the session token. Raises BetfairAuthError on any
    loginStatus other than SUCCESS."""
    response = requests.post(
        login_url,
        cert=(cert_file, key_file),
        headers={
            "X-Application": app_key,
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={"username": username, "password": password},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("loginStatus") != "SUCCESS":
        raise BetfairAuthError(f"Betfair login failed: {payload.get('loginStatus')}")
    return payload["sessionToken"]


class BetfairSession:
    """Holds a Betfair session token and re-authenticates automatically.
    Betfair does not document session lifetime, so this re-logs in
    reactively -- once before the first request, and again exactly once
    if a request comes back with a non-200 status -- rather than
    assuming a fixed expiry duration."""

    def __init__(
        self,
        app_key: str,
        username: str,
        password: str,
        cert_file: str,
        key_file: str,
        login_url: str = LOGIN_URL,
    ) -> None:
        self.app_key = app_key
        self.username = username
        self.password = password
        self.cert_file = cert_file
        self.key_file = key_file
        self.login_url = login_url
        self._session_token: str | None = None

    def _login(self) -> str:
        self._session_token = cert_login(
            self.app_key, self.username, self.password, self.cert_file, self.key_file, self.login_url
        )
        return self._session_token

    def request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Issues an authenticated request against the Exchange API,
        logging in first if there's no session yet, and retrying exactly
        once (with a fresh login) if the first attempt comes back with a
        non-200 status."""
        token = self._session_token or self._login()
        headers = kwargs.pop("headers", {}) or {}
        headers["X-Application"] = self.app_key
        headers["X-Authentication"] = token
        timeout = kwargs.pop("timeout", 30)
        response = requests.request(method, url, headers=headers, timeout=timeout, **kwargs)
        if response.status_code != 200:
            token = self._login()
            headers["X-Authentication"] = token
            response = requests.request(method, url, headers=headers, timeout=timeout, **kwargs)
        return response
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\pytest.exe tests/test_betfair_auth.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```powershell
git add src/goles/betfair/__init__.py src/goles/betfair/auth.py tests/test_betfair_auth.py
git commit -m "feat: add Betfair non-interactive certificate login with reactive re-login"
```

---

### Task 3: Competition and market discovery client

**Files:**
- Create: `src/goles/betfair/client.py`
- Test: `tests/test_betfair_client.py`

**Interfaces:**
- Consumes: `goles.betfair.auth.BetfairSession` (only its `.request(method, url, **kwargs)` method — tests pass a stub object, no real `BetfairSession` needed).
- Produces: `BASE_URL: str`, `list_competitions(session, event_type_id="1") -> list[dict]`, `find_competition_id(session, name_fragment, event_type_id="1") -> str | None`, `list_market_catalogue(session, competition_ids, market_types, max_results=200) -> list[dict]`, `list_market_book(session, market_ids) -> list[dict]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_betfair_client.py`:
```python
from unittest.mock import Mock

from goles.betfair.client import (
    find_competition_id,
    list_competitions,
    list_market_book,
    list_market_catalogue,
)


def _stub_session(json_body):
    response = Mock()
    response.json.return_value = json_body
    response.raise_for_status = Mock()
    session = Mock()
    session.request = Mock(return_value=response)
    return session


def test_list_competitions_extracts_competition_dicts():
    session = _stub_session([
        {"competition": {"id": "1", "name": "English Premier League"}, "marketCount": 40},
        {"competition": {"id": "2", "name": "German Bundesliga"}, "marketCount": 30},
    ])
    competitions = list_competitions(session)
    assert competitions == [
        {"id": "1", "name": "English Premier League"},
        {"id": "2", "name": "German Bundesliga"},
    ]
    session.request.assert_called_once_with(
        "POST",
        "https://api.betfair.com/exchange/betting/rest/v1.0/listCompetitions/",
        json={"filter": {"eventTypeIds": ["1"]}},
    )


def test_find_competition_id_matches_by_case_insensitive_substring():
    session = _stub_session([
        {"competition": {"id": "1", "name": "English Premier League"}},
        {"competition": {"id": "2", "name": "German Bundesliga"}},
    ])
    assert find_competition_id(session, "premier league") == "1"
    assert find_competition_id(session, "Bundesliga") == "2"


def test_find_competition_id_returns_none_when_no_match():
    session = _stub_session([{"competition": {"id": "1", "name": "English Premier League"}}])
    assert find_competition_id(session, "La Liga") is None


def test_list_market_catalogue_sends_expected_filter():
    session = _stub_session([{"marketId": "1.123", "event": {"id": "e1", "name": "Team A v Team B"}}])
    result = list_market_catalogue(session, ["1", "2"], ["MATCH_ODDS", "OVER_UNDER_25"])
    assert result == [{"marketId": "1.123", "event": {"id": "e1", "name": "Team A v Team B"}}]
    session.request.assert_called_once_with(
        "POST",
        "https://api.betfair.com/exchange/betting/rest/v1.0/listMarketCatalogue/",
        json={
            "filter": {
                "eventTypeIds": ["1"],
                "competitionIds": ["1", "2"],
                "marketTypeCodes": ["MATCH_ODDS", "OVER_UNDER_25"],
            },
            "maxResults": 200,
            "marketProjection": ["EVENT", "MARKET_START_TIME", "RUNNER_DESCRIPTION", "MARKET_DESCRIPTION"],
        },
    )


def test_list_market_book_sends_expected_filter():
    session = _stub_session([{"marketId": "1.123", "runners": []}])
    result = list_market_book(session, ["1.123", "1.456"])
    assert result == [{"marketId": "1.123", "runners": []}]
    session.request.assert_called_once_with(
        "POST",
        "https://api.betfair.com/exchange/betting/rest/v1.0/listMarketBook/",
        json={"marketIds": ["1.123", "1.456"], "priceProjection": {"priceData": ["EX_BEST_OFFERS"]}},
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\pytest.exe tests/test_betfair_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'goles.betfair.client'`

- [ ] **Step 3: Write the implementation**

`src/goles/betfair/client.py`:
```python
from __future__ import annotations

BASE_URL = "https://api.betfair.com/exchange/betting/rest/v1.0"


def list_competitions(session, event_type_id: str = "1") -> list[dict]:
    """Returns the list of {"id", "name"} competition dicts for the given
    Betfair eventTypeId ("1" = Soccer)."""
    response = session.request(
        "POST",
        f"{BASE_URL}/listCompetitions/",
        json={"filter": {"eventTypeIds": [event_type_id]}},
    )
    response.raise_for_status()
    return [entry["competition"] for entry in response.json()]


def find_competition_id(session, name_fragment: str, event_type_id: str = "1") -> str | None:
    """Finds the first competition whose name contains `name_fragment`
    (case-insensitive), or None if none match. Used instead of hardcoding
    competition ids, which would be an unverified guess."""
    for competition in list_competitions(session, event_type_id):
        if name_fragment.lower() in competition["name"].lower():
            return competition["id"]
    return None


def list_market_catalogue(
    session, competition_ids: list[str], market_types: list[str], max_results: int = 200
) -> list[dict]:
    """Returns market catalogue entries (marketId, event, runners) for the
    given competitions and market type codes (e.g. MATCH_ODDS,
    OVER_UNDER_25), scoped to Soccer (eventTypeId "1")."""
    response = session.request(
        "POST",
        f"{BASE_URL}/listMarketCatalogue/",
        json={
            "filter": {
                "eventTypeIds": ["1"],
                "competitionIds": competition_ids,
                "marketTypeCodes": market_types,
            },
            "maxResults": max_results,
            "marketProjection": ["EVENT", "MARKET_START_TIME", "RUNNER_DESCRIPTION", "MARKET_DESCRIPTION"],
        },
    )
    response.raise_for_status()
    return response.json()


def list_market_book(session, market_ids: list[str]) -> list[dict]:
    """Returns current best-offers prices for the given market ids."""
    response = session.request(
        "POST",
        f"{BASE_URL}/listMarketBook/",
        json={"marketIds": market_ids, "priceProjection": {"priceData": ["EX_BEST_OFFERS"]}},
    )
    response.raise_for_status()
    return response.json()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\pytest.exe tests/test_betfair_client.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```powershell
git add src/goles/betfair/client.py tests/test_betfair_client.py
git commit -m "feat: add Betfair competition/market discovery client"
```

---

### Task 4: Live odds SQLite store

**Files:**
- Create: `src/goles/betfair/odds_store.py`
- Test: `tests/test_betfair_odds_store.py`

**Interfaces:**
- Produces: `DEFAULT_LIVE_ODDS_DB_PATH: Path`, `get_connection(db_path=DEFAULT_LIVE_ODDS_DB_PATH) -> sqlite3.Connection`, `init_db(conn) -> None`, `persist_snapshot(conn, fetched_at, betfair_event_id, home_team, away_team, market_type, raw_json, home_wp=None, draw_wp=None, away_wp=None, over_wp=None) -> None`.

- [ ] **Step 1: Write the failing tests**

`tests/test_betfair_odds_store.py`:
```python
from goles.betfair.odds_store import get_connection, init_db, persist_snapshot


def test_init_db_creates_odds_snapshots_table():
    conn = get_connection(":memory:")
    init_db(conn)
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cur.fetchall()}
    assert "odds_snapshots" in tables


def test_persist_snapshot_inserts_a_row():
    conn = get_connection(":memory:")
    init_db(conn)
    persist_snapshot(
        conn,
        fetched_at="2026-07-11T12:00:00+00:00",
        betfair_event_id="e1",
        home_team="Arsenal",
        away_team="Chelsea",
        market_type="MATCH_ODDS",
        raw_json="{}",
        home_wp=0.5,
        draw_wp=0.3,
        away_wp=0.2,
    )
    row = conn.execute(
        "SELECT home_team, away_team, market_type, home_wp, draw_wp, away_wp, over_wp FROM odds_snapshots"
    ).fetchone()
    assert row == ("Arsenal", "Chelsea", "MATCH_ODDS", 0.5, 0.3, 0.2, None)


def test_persist_snapshot_allows_multiple_rows_for_same_event():
    conn = get_connection(":memory:")
    init_db(conn)
    persist_snapshot(conn, "2026-07-11T12:00:00+00:00", "e1", "Arsenal", "Chelsea", "MATCH_ODDS", "{}")
    persist_snapshot(conn, "2026-07-11T12:01:00+00:00", "e1", "Arsenal", "Chelsea", "MATCH_ODDS", "{}")
    count = conn.execute("SELECT COUNT(*) FROM odds_snapshots").fetchone()[0]
    assert count == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\pytest.exe tests/test_betfair_odds_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'goles.betfair.odds_store'`

- [ ] **Step 3: Write the implementation**

`src/goles/betfair/odds_store.py`:
```python
from __future__ import annotations

import sqlite3
from pathlib import Path

DEFAULT_LIVE_ODDS_DB_PATH = Path("data") / "live_odds.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS odds_snapshots (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    fetched_at TEXT NOT NULL,
    betfair_event_id TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    market_type TEXT NOT NULL,
    home_wp REAL,
    draw_wp REAL,
    away_wp REAL,
    over_wp REAL,
    raw_json TEXT NOT NULL
);
"""


def get_connection(db_path: str | Path = DEFAULT_LIVE_ODDS_DB_PATH) -> sqlite3.Connection:
    """Opens (creating parent directories if needed) the live-odds SQLite
    database -- a separate file from data/goles.db, which holds historical
    training data only."""
    if str(db_path) != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(db_path))


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def persist_snapshot(
    conn: sqlite3.Connection,
    fetched_at: str,
    betfair_event_id: str,
    home_team: str,
    away_team: str,
    market_type: str,
    raw_json: str,
    home_wp: float | None = None,
    draw_wp: float | None = None,
    away_wp: float | None = None,
    over_wp: float | None = None,
) -> None:
    """Inserts one timestamped odds snapshot row. Each poll cycle inserts a
    new row rather than updating in place, so the full history of odds
    movement for a fixture is preserved."""
    conn.execute(
        """INSERT INTO odds_snapshots
           (fetched_at, betfair_event_id, home_team, away_team, market_type,
            home_wp, draw_wp, away_wp, over_wp, raw_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (fetched_at, betfair_event_id, home_team, away_team, market_type, home_wp, draw_wp, away_wp, over_wp, raw_json),
    )
    conn.commit()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\pytest.exe tests/test_betfair_odds_store.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```powershell
git add src/goles/betfair/odds_store.py tests/test_betfair_odds_store.py
git commit -m "feat: add live odds snapshot SQLite store"
```

---

### Task 5: Betfair team-name alias table

**Files:**
- Create: `src/goles/betfair/team_aliases.py`
- Test: `tests/test_betfair_team_aliases.py`

**Interfaces:**
- Produces: `BETFAIR_TEAM_NAME_ALIASES: dict[str, str]`, `normalize_betfair_team_name(name: str) -> str`.

- [ ] **Step 1: Write the failing tests**

`tests/test_betfair_team_aliases.py`:
```python
from goles.betfair.team_aliases import BETFAIR_TEAM_NAME_ALIASES, normalize_betfair_team_name


def test_normalize_betfair_team_name_passes_through_unmapped_names():
    assert normalize_betfair_team_name("Arsenal") == "Arsenal"
    assert normalize_betfair_team_name("Some Unmapped Team") == "Some Unmapped Team"


def test_betfair_team_name_aliases_has_no_identity_entries():
    for betfair_name, our_name in BETFAIR_TEAM_NAME_ALIASES.items():
        assert betfair_name != our_name
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\pytest.exe tests/test_betfair_team_aliases.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'goles.betfair.team_aliases'`

- [ ] **Step 3: Write the implementation**

`src/goles/betfair/team_aliases.py`:
```python
from __future__ import annotations

# Starter table -- unlike TEAM_NAME_ALIASES in goles/loaders/football_data.py
# (built by diffing a complete, real historical dataset), this one cannot
# yet be verified against real Betfair event names: the delayed App Key and
# certificate login are still pending (see the design spec). Extend this
# table with real observed aliases once the poller runs against production
# and logs an unmatched fixture -- never guess an entry without having seen
# the real Betfair name it maps from.
BETFAIR_TEAM_NAME_ALIASES: dict[str, str] = {}


def normalize_betfair_team_name(name: str) -> str:
    """Maps a Betfair event/runner team name to our Understat-sourced team
    name. Names not in BETFAIR_TEAM_NAME_ALIASES are assumed identical and
    returned unchanged -- callers must treat a name that still doesn't
    resolve to a known team as unmatched and skip it loudly, never
    fuzzy-match."""
    return BETFAIR_TEAM_NAME_ALIASES.get(name, name)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\pytest.exe tests/test_betfair_team_aliases.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```powershell
git add src/goles/betfair/team_aliases.py tests/test_betfair_team_aliases.py
git commit -m "feat: add Betfair team-name alias table"
```

---

### Task 6: Poller core logic (parsing, probability computation, one poll cycle)

**Files:**
- Create: `src/goles/betfair/poller.py`
- Test: `tests/test_betfair_poller.py`

**Interfaces:**
- Consumes: `goles.betfair.auth.BetfairSession`, `goles.betfair.client.{find_competition_id, list_market_catalogue, list_market_book}`, `goles.betfair.odds_store.persist_snapshot`, `goles.betfair.team_aliases.normalize_betfair_team_name`, `goles.loaders.football_data.{compute_no_vig_probabilities, compute_no_vig_two_way}`.
- Produces: `TRACKED_COMPETITIONS: list[str]`, `MATCH_ODDS_TYPE`/`OVER_UNDER_TYPE: str`, `parse_team_names_from_event(event_name: str) -> tuple[str, str]`, `extract_best_back_prices(market_book: dict) -> dict[int, float] | None`, `compute_match_odds_probabilities(runner_name_by_id, prices_by_id, home_name, away_name) -> tuple[float, float, float] | None`, `compute_over_under_probabilities(runner_name_by_id, prices_by_id) -> tuple[float, float] | None`, `discover_tracked_markets(session) -> list[dict]`, `poll_once(session, conn, market_catalogue: list[dict]) -> None`.

- [ ] **Step 1: Write the failing tests**

`tests/test_betfair_poller.py`:
```python
from unittest.mock import Mock

import pytest

from goles.betfair.odds_store import get_connection, init_db
from goles.betfair.poller import (
    compute_match_odds_probabilities,
    compute_over_under_probabilities,
    extract_best_back_prices,
    parse_team_names_from_event,
    poll_once,
)


def test_parse_team_names_from_event_splits_on_v():
    assert parse_team_names_from_event("Arsenal v Chelsea") == ("Arsenal", "Chelsea")


def test_parse_team_names_from_event_raises_on_unexpected_format():
    with pytest.raises(ValueError):
        parse_team_names_from_event("Arsenal - Chelsea")


def test_extract_best_back_prices_returns_price_by_selection_id():
    market_book = {
        "runners": [
            {"selectionId": 1, "ex": {"availableToBack": [{"price": 2.5, "size": 100}]}},
            {"selectionId": 2, "ex": {"availableToBack": [{"price": 3.0, "size": 50}]}},
        ]
    }
    assert extract_best_back_prices(market_book) == {1: 2.5, 2: 3.0}


def test_extract_best_back_prices_returns_none_when_a_runner_has_no_price():
    market_book = {
        "runners": [
            {"selectionId": 1, "ex": {"availableToBack": []}},
            {"selectionId": 2, "ex": {"availableToBack": [{"price": 3.0, "size": 50}]}},
        ]
    }
    assert extract_best_back_prices(market_book) is None


def test_compute_match_odds_probabilities_resolves_home_draw_away():
    runner_name_by_id = {1: "Arsenal", 2: "The Draw", 3: "Chelsea"}
    prices_by_id = {1: 1.5, 2: 4.0, 3: 6.0}
    probs = compute_match_odds_probabilities(runner_name_by_id, prices_by_id, "Arsenal", "Chelsea")
    assert probs is not None
    home_wp, draw_wp, away_wp = probs
    assert abs((home_wp + draw_wp + away_wp) - 1.0) < 1e-9
    assert home_wp > away_wp


def test_compute_match_odds_probabilities_returns_none_when_team_not_found():
    runner_name_by_id = {1: "Some Other Team", 2: "The Draw", 3: "Chelsea"}
    prices_by_id = {1: 1.5, 2: 4.0, 3: 6.0}
    assert compute_match_odds_probabilities(runner_name_by_id, prices_by_id, "Arsenal", "Chelsea") is None


def test_compute_over_under_probabilities_resolves_over_and_under():
    runner_name_by_id = {1: "Over 2.5 Goals", 2: "Under 2.5 Goals"}
    prices_by_id = {1: 1.9, 2: 1.95}
    probs = compute_over_under_probabilities(runner_name_by_id, prices_by_id)
    assert probs is not None
    over_wp, under_wp = probs
    assert abs((over_wp + under_wp) - 1.0) < 1e-9


def test_compute_over_under_probabilities_returns_none_when_runners_not_found():
    runner_name_by_id = {1: "Something Else", 2: "Under 2.5 Goals"}
    prices_by_id = {1: 1.9, 2: 1.95}
    assert compute_over_under_probabilities(runner_name_by_id, prices_by_id) is None


def test_poll_once_persists_a_snapshot_for_a_valid_match_odds_market():
    market_catalogue = [
        {
            "marketId": "1.111",
            "marketType": "MATCH_ODDS",
            "event": {"id": "e1", "name": "Arsenal v Chelsea"},
            "runners": [
                {"selectionId": 1, "runnerName": "Arsenal"},
                {"selectionId": 2, "runnerName": "The Draw"},
                {"selectionId": 3, "runnerName": "Chelsea"},
            ],
        }
    ]
    market_book = {
        "marketId": "1.111",
        "runners": [
            {"selectionId": 1, "ex": {"availableToBack": [{"price": 1.5}]}},
            {"selectionId": 2, "ex": {"availableToBack": [{"price": 4.0}]}},
            {"selectionId": 3, "ex": {"availableToBack": [{"price": 6.0}]}},
        ],
    }
    session = Mock()
    conn = get_connection(":memory:")
    init_db(conn)

    import goles.betfair.poller as poller_module
    poller_module.list_market_book = Mock(return_value=[market_book])

    poll_once(session, conn, market_catalogue)

    row = conn.execute("SELECT home_team, away_team, market_type FROM odds_snapshots").fetchone()
    assert row == ("Arsenal", "Chelsea", "MATCH_ODDS")


def test_poll_once_skips_market_with_no_available_prices_without_raising():
    market_catalogue = [
        {
            "marketId": "1.111",
            "marketType": "MATCH_ODDS",
            "event": {"id": "e1", "name": "Arsenal v Chelsea"},
            "runners": [
                {"selectionId": 1, "runnerName": "Arsenal"},
                {"selectionId": 2, "runnerName": "The Draw"},
                {"selectionId": 3, "runnerName": "Chelsea"},
            ],
        }
    ]
    market_book = {
        "marketId": "1.111",
        "runners": [{"selectionId": 1, "ex": {"availableToBack": []}}],
    }
    session = Mock()
    conn = get_connection(":memory:")
    init_db(conn)

    import goles.betfair.poller as poller_module
    poller_module.list_market_book = Mock(return_value=[market_book])

    poll_once(session, conn, market_catalogue)  # must not raise

    count = conn.execute("SELECT COUNT(*) FROM odds_snapshots").fetchone()[0]
    assert count == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\pytest.exe tests/test_betfair_poller.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'goles.betfair.poller'`

- [ ] **Step 3: Write the implementation**

`src/goles/betfair/poller.py`:
```python
from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import datetime, timezone

from goles.betfair.auth import BetfairSession
from goles.betfair.client import find_competition_id, list_market_book, list_market_catalogue
from goles.betfair.odds_store import get_connection, init_db, persist_snapshot
from goles.betfair.team_aliases import normalize_betfair_team_name
from goles.loaders.football_data import compute_no_vig_probabilities, compute_no_vig_two_way

TRACKED_COMPETITIONS = ["Premier League", "Bundesliga"]
MATCH_ODDS_TYPE = "MATCH_ODDS"
OVER_UNDER_TYPE = "OVER_UNDER_25"
POLL_INTERVAL_SECONDS = 60
FIXTURE_REFRESH_INTERVAL_SECONDS = 900
DRAW_RUNNER_NAME = "The Draw"
OVER_RUNNER_NAME = "Over 2.5 Goals"
UNDER_RUNNER_NAME = "Under 2.5 Goals"


def parse_team_names_from_event(event_name: str) -> tuple[str, str]:
    """Splits Betfair's soccer event name convention ("Home Team v Away
    Team") into (home, away). Raises ValueError on any other format --
    fail loud rather than guess."""
    parts = event_name.split(" v ")
    if len(parts) != 2:
        raise ValueError(f"Unexpected event name format: {event_name!r}")
    return parts[0], parts[1]


def extract_best_back_prices(market_book: dict) -> dict[int, float] | None:
    """Returns {selectionId: best_back_price} for every runner in the
    market book, or None if any runner currently has no available-to-back
    price (an empty/not-yet-liquid market) -- callers must skip such
    markets rather than compute odds from partial data."""
    prices: dict[int, float] = {}
    for runner in market_book.get("runners", []):
        available = runner.get("ex", {}).get("availableToBack", [])
        if not available:
            return None
        prices[runner["selectionId"]] = available[0]["price"]
    return prices


def compute_match_odds_probabilities(
    runner_name_by_id: dict[int, str],
    prices_by_id: dict[int, float],
    home_name: str,
    away_name: str,
) -> tuple[float, float, float] | None:
    """Resolves the MATCH_ODDS market's three runners (home team, draw,
    away team) by name and returns their no-vig (home, draw, away) win
    probabilities, or None if any of the three can't be matched by name
    (via normalize_betfair_team_name) -- never guessed."""
    home_price = draw_price = away_price = None
    for selection_id, price in prices_by_id.items():
        runner_name = normalize_betfair_team_name(runner_name_by_id.get(selection_id, ""))
        if runner_name == DRAW_RUNNER_NAME:
            draw_price = price
        elif runner_name == home_name:
            home_price = price
        elif runner_name == away_name:
            away_price = price
    if home_price is None or draw_price is None or away_price is None:
        return None
    return compute_no_vig_probabilities(home_price, draw_price, away_price)


def compute_over_under_probabilities(
    runner_name_by_id: dict[int, str], prices_by_id: dict[int, float]
) -> tuple[float, float] | None:
    """Resolves the OVER_UNDER_25 market's two runners and returns their
    no-vig (over, under) probabilities, or None if either runner can't be
    matched by name."""
    over_price = under_price = None
    for selection_id, price in prices_by_id.items():
        runner_name = runner_name_by_id.get(selection_id, "")
        if runner_name == OVER_RUNNER_NAME:
            over_price = price
        elif runner_name == UNDER_RUNNER_NAME:
            under_price = price
    if over_price is None or under_price is None:
        return None
    return compute_no_vig_two_way(over_price, under_price)


def discover_tracked_markets(session: BetfairSession) -> list[dict]:
    """Finds the competition ids for TRACKED_COMPETITIONS and returns the
    open MATCH_ODDS + OVER_UNDER_25 market catalogue entries for them.
    A competition that can't be found (name changed, no markets currently
    listed) is simply skipped -- not an error, since coverage naturally
    varies with the football calendar."""
    competition_ids = []
    for name in TRACKED_COMPETITIONS:
        competition_id = find_competition_id(session, name)
        if competition_id is not None:
            competition_ids.append(competition_id)
    if not competition_ids:
        return []
    return list_market_catalogue(session, competition_ids, [MATCH_ODDS_TYPE, OVER_UNDER_TYPE])


def poll_once(session: BetfairSession, conn: sqlite3.Connection, market_catalogue: list[dict]) -> None:
    """Fetches current prices for every market in market_catalogue and
    persists one snapshot row per market with resolvable prices. Markets
    with no available back prices yet, or whose teams/runners can't be
    matched, are skipped with a printed warning -- never silently."""
    market_ids = [m["marketId"] for m in market_catalogue]
    if not market_ids:
        return
    market_books = list_market_book(session, market_ids)
    books_by_id = {b["marketId"]: b for b in market_books}
    fetched_at = datetime.now(timezone.utc).isoformat()

    for catalogue_entry in market_catalogue:
        market_id = catalogue_entry["marketId"]
        market_book = books_by_id.get(market_id)
        if market_book is None:
            continue

        event = catalogue_entry.get("event", {})
        try:
            home_name, away_name = parse_team_names_from_event(event.get("name", ""))
        except ValueError:
            print(f"ADVERTENCIA: no se pudo separar equipos de '{event.get('name')}', se omite mercado {market_id}.")
            continue

        runner_name_by_id = {r["selectionId"]: r["runnerName"] for r in catalogue_entry.get("runners", [])}
        prices_by_id = extract_best_back_prices(market_book)
        if prices_by_id is None:
            continue

        market_type = catalogue_entry.get("marketType") or catalogue_entry.get("description", {}).get("marketType")
        if market_type == MATCH_ODDS_TYPE:
            probs = compute_match_odds_probabilities(runner_name_by_id, prices_by_id, home_name, away_name)
            if probs is None:
                print(
                    f"ADVERTENCIA: no se pudieron resolver equipos/empate en mercado MATCH_ODDS "
                    f"{market_id} ('{event.get('name')}'), se omite."
                )
                continue
            home_wp, draw_wp, away_wp = probs
            persist_snapshot(
                conn, fetched_at, event.get("id", ""), home_name, away_name, MATCH_ODDS_TYPE,
                json.dumps(market_book), home_wp=home_wp, draw_wp=draw_wp, away_wp=away_wp,
            )
        elif market_type == OVER_UNDER_TYPE:
            over_probs = compute_over_under_probabilities(runner_name_by_id, prices_by_id)
            if over_probs is None:
                print(
                    f"ADVERTENCIA: no se pudieron resolver runners over/under en mercado "
                    f"{market_id} ('{event.get('name')}'), se omite."
                )
                continue
            over_wp, _ = over_probs
            persist_snapshot(
                conn, fetched_at, event.get("id", ""), home_name, away_name, OVER_UNDER_TYPE,
                json.dumps(market_book), over_wp=over_wp,
            )


def main() -> None:
    """Persistent entrypoint: logs in, discovers tracked fixtures, then
    polls forever. Any failure during discovery or a poll cycle is caught,
    printed, and retried on the next loop iteration rather than crashing
    the process -- a missing/invalid required env var, in contrast, fails
    immediately and loudly at startup (a real configuration error, not
    something to silently retry)."""
    app_key = os.environ["BETFAIR_APP_KEY"]
    username = os.environ["BETFAIR_USERNAME"]
    password = os.environ["BETFAIR_PASSWORD"]
    cert_file = os.environ.get("BETFAIR_CERT_FILE", "/run/secrets/betfair/client-2048.crt")
    key_file = os.environ.get("BETFAIR_KEY_FILE", "/run/secrets/betfair/client-2048.key")

    session = BetfairSession(app_key, username, password, cert_file, key_file)
    conn = get_connection()
    init_db(conn)

    market_catalogue: list[dict] = []
    last_discovery = 0.0

    while True:
        now = time.monotonic()
        if not market_catalogue or (now - last_discovery) > FIXTURE_REFRESH_INTERVAL_SECONDS:
            try:
                market_catalogue = discover_tracked_markets(session)
                last_discovery = now
                print(f"{len(market_catalogue)} mercados encontrados en las ligas trackeadas.")
            except Exception as exc:
                print(f"ADVERTENCIA: fallo al descubrir mercados ({exc}), se reintenta en el proximo ciclo.")
        if market_catalogue:
            try:
                poll_once(session, conn, market_catalogue)
            except Exception as exc:
                print(f"ADVERTENCIA: fallo en el ciclo de polling ({exc}), se reintenta en el proximo ciclo.")
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\pytest.exe tests/test_betfair_poller.py -v`
Expected: 9 passed

- [ ] **Step 5: Run the full suite**

Run: `.venv\Scripts\pytest.exe -q`
Expected: all pass (previous count + 23 new across Tasks 2-6)

- [ ] **Step 6: Commit**

```powershell
git add src/goles/betfair/poller.py tests/test_betfair_poller.py
git commit -m "feat: add live-odds poller (discovery, parsing, one poll cycle, entrypoint)"
```

---

### Task 7: Dockerize the poller

**Files:**
- Create: `Dockerfile.betfair`
- Create: `.dockerignore`

No automated tests — verified by a real `docker build` (Step 2).

- [ ] **Step 1: Write the Dockerfile**

`Dockerfile.betfair`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .

ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "goles.betfair.poller"]
```

`.dockerignore`:
```
.venv/
data/
__pycache__/
*.pyc
.pytest_cache/
.git/
docs/
tests/
```

- [ ] **Step 2: Build the image on the VPS to confirm it builds cleanly**

Run over SSH (build directly on the VPS rather than requiring Docker on the developer's Windows machine):
```bash
ssh -i ~/.ssh/id_ed25519_goles_vps root@85.239.245.73 "rm -rf /root/goles-build-test && mkdir -p /root/goles-build-test"
scp -i ~/.ssh/id_ed25519_goles_vps -r "src" "pyproject.toml" "Dockerfile.betfair" root@85.239.245.73:/root/goles-build-test/
ssh -i ~/.ssh/id_ed25519_goles_vps root@85.239.245.73 \
  "cd /root/goles-build-test && docker build -f Dockerfile.betfair -t goles-betfair-poller-test . && echo BUILD_OK"
```
Expected: `BUILD_OK` printed, confirming the image builds without error on the actual target environment. This is a throwaway local build purely to validate the Dockerfile before Task 8 configures Dokploy to build it from the git repo directly.

- [ ] **Step 3: Clean up the throwaway build test and commit the Dockerfile**

```bash
ssh -i ~/.ssh/id_ed25519_goles_vps root@85.239.245.73 "docker rmi goles-betfair-poller-test; rm -rf /root/goles-build-test"
```
```powershell
git add Dockerfile.betfair .dockerignore
git commit -m "feat: add Dockerfile for the live-odds poller service"
git push origin master
```

---

### Task 8: Deploy as a new Dokploy Application

**Files:** none (Dokploy dashboard configuration, using the already-open browser session).

No automated tests — manual verification only (Step 3).

- [ ] **Step 1: Create the Application service in the existing `Claudator-Goles` / `production` Dokploy project**

Using the Dokploy dashboard (already open in the browser at `85.239.245.73:3000`, project `Claudator-Goles` → `production`, currently empty): click **Create Service** → **Application**. Name it `betfair-odds-poller`. Configure:
- Source: GitHub repo `claxdio/Claudator-Goles`, branch `master`.
- Build type: Dockerfile, path `Dockerfile.betfair`.
- Restart policy: always (Dokploy's default for Applications).

- [ ] **Step 2: Configure volumes, file mounts, and environment variables**

- Add a volume mounting a persistent path to `/app/data` inside the container, so `live_odds.db` survives redeploys.
- Add file mounts for the certificate generated in Task 1: host path `/root/goles-betfair-certs/client-2048.crt` → container path `/run/secrets/betfair/client-2048.crt`, and the `.key` file the same way to `/run/secrets/betfair/client-2048.key`.
- Add environment variables `BETFAIR_APP_KEY`, `BETFAIR_USERNAME`, `BETFAIR_PASSWORD`. **Leave these as placeholder/empty values for now** — the real values depend on the user completing the two external prerequisites from the design spec (requesting the delayed App Key, and the account still needing the certificate uploaded). Do not fabricate credentials.

- [ ] **Step 3: Deploy and verify the failure mode is graceful, not silent**

Click Deploy. Confirm in the Dokploy logs view that the container starts and then exits/crash-loops with a clear `KeyError: 'BETFAIR_APP_KEY'` (or similar, if a placeholder empty string was used instead of leaving the variable unset — either way the failure must be an obvious, readable error, not a silent hang or an unrelated stack trace). This confirms the deployment pipeline (build, volume, file mounts) works end-to-end; the poller will start functioning for real once the user finishes the two external prerequisites and updates the three Betfair environment variables with real values — that final real-credentials run is outside this plan's scope (no code changes needed for it, just re-entering env vars in the Dokploy UI).

- [ ] **Step 4: Record the deployment state**

Append a short "## Estado de despliegue" note to `docs/superpowers/specs/2026-07-11-live-betfair-odds-pipeline-design.md` recording: the Dokploy application name/URL, confirmation the build succeeded, and the exact two remaining external prerequisites blocking a real end-to-end run. Commit:

```powershell
git add docs/superpowers/specs/2026-07-11-live-betfair-odds-pipeline-design.md
git commit -m "docs: record live-odds poller Dokploy deployment state"
```
