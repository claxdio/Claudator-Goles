# Chile Shadow Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the home-PC Sofascore live poller to Chilean utids 11653/1240, run the already-trained Chilean goal model in real time on those matches, persist every prediction, and send Telegram alerts explicitly labeled "modo prueba — sin edge confirmado, solo observación" — never as a betting signal.

**Architecture:** One new leaf module (`src/goles/telegram.py`), one new engine module (`src/goles/sofascore/shadow.py`), two new tables in the existing live store (`shadow_predictions`, `shadow_alerts`), and integration edits to `src/goles/sofascore/poller.py` (utid discovery, Chilean own-xG at persist time, per-cycle shadow inference). Live feature assembly must mirror `dataset.build_dataset`'s row assembly exactly — enforced by a test that pushes one synthetic match through both paths and asserts identical feature dicts.

**Tech Stack:** Python 3.11+, LightGBM (existing), `tls_requests` (existing — also used for the Telegram Bot API), `sqlite3`, `pytest`. No new dependencies, no paid services.

## Global Constraints

- Chilean competition filtering **must use `tournament.uniqueTournament.id`** (11653 Liga de Primera, 1240 Liga de Ascenso — reuse `TRACKED_UTIDS` from `goles/sofascore/backfill.py`), never tournament-name matching (verified Paraguay collision).
- Every shadow alert **must start with the banner** `🧪 MODO PRUEBA — sin edge confirmado, solo observación`. No betting/value/stake framing anywhere in message copy.
- Live inference only at estimated match minute **20–80 inclusive** (the model's training cutoff range).
- Poisson feature blend in live serving is **0.1** — must equal `train_gbt_chile.POISSON_COMPARISON_BLEND` (train/serve consistency).
- Alert threshold: calibrated probability ≥ **0.30**; cooldown **15 minutes** per (event, team), persisted in the DB (restart-safe).
- Telegram/model/DB failures must degrade loudly but never crash the poller; missing `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID` env vars → alerts print to stdout only.
- Model artifacts: xG booster at `goles.train_xg.XG_MODEL_PATH`, goal model at `goles.train_gbt_chile.MODEL_DIR`, priors DB at `goles.sofascore.backfill.CHILE_DB_PATH` — import these constants, never re-hardcode paths.
- All network I/O mocked in tests (duck-typed clients/boosters); `apply_platt_scaling` with `(a=1.0, b=0.0)` is the identity — use it in tests.
- Existing tests (162) must keep passing unmodified except where a task explicitly edits them.
- Deployment target is the **home PC poller process** (not Dokploy): deploy = commit, push, restart the local process.

---

### Task 1: Telegram delivery module

**Files:**
- Create: `src/goles/telegram.py`
- Test: `tests/test_telegram.py`

**Interfaces:**
- Produces: `TELEGRAM_API_BASE = "https://api.telegram.org"`, `send_message(client, token: str, chat_id: str, text: str) -> bool`. The `client` is duck-typed with a `.post(url, json=...)` method returning an object with `.status_code` (a `tls_requests.Client` in production).

- [ ] **Step 1: Write the failing tests**

`tests/test_telegram.py`:
```python
from unittest.mock import Mock

from goles.telegram import TELEGRAM_API_BASE, send_message


def test_send_message_posts_to_bot_api_and_returns_true_on_200():
    response = Mock()
    response.status_code = 200
    client = Mock()
    client.post = Mock(return_value=response)

    ok = send_message(client, "123:ABC", "-100999", "hola")

    assert ok is True
    client.post.assert_called_once_with(
        f"{TELEGRAM_API_BASE}/bot123:ABC/sendMessage",
        json={"chat_id": "-100999", "text": "hola"},
    )


def test_send_message_returns_false_on_http_error():
    response = Mock()
    response.status_code = 403
    client = Mock()
    client.post = Mock(return_value=response)
    assert send_message(client, "123:ABC", "-100999", "hola") is False


def test_send_message_returns_false_when_request_raises():
    client = Mock()
    client.post = Mock(side_effect=RuntimeError("red caida"))
    assert send_message(client, "123:ABC", "-100999", "hola") is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\pytest.exe tests/test_telegram.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'goles.telegram'`

- [ ] **Step 3: Write the implementation**

`src/goles/telegram.py`:
```python
from __future__ import annotations

TELEGRAM_API_BASE = "https://api.telegram.org"


def send_message(client, token: str, chat_id: str, text: str) -> bool:
    """Sends `text` to `chat_id` via the Telegram Bot API. Returns False
    (with a printed warning) on any failure -- Telegram being down must
    never crash the caller (the live poller)."""
    try:
        response = client.post(
            f"{TELEGRAM_API_BASE}/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text},
        )
        if response.status_code != 200:
            print(f"ADVERTENCIA: Telegram respondio {response.status_code}, mensaje no enviado.")
            return False
        return True
    except Exception as exc:
        print(f"ADVERTENCIA: fallo el envio a Telegram ({exc}).")
        return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\pytest.exe tests/test_telegram.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```powershell
git add src/goles/telegram.py tests/test_telegram.py
git commit -m "feat: add Telegram Bot API delivery module"
```

---

### Task 2: Shadow module — Chilean event detection + match-minute estimation

**Files:**
- Create: `src/goles/sofascore/shadow.py`
- Test: `tests/test_sofascore_shadow.py`

**Interfaces:**
- Consumes: `goles.sofascore.backfill.TRACKED_UTIDS` (`{11653: "CHI-Liga de Primera", 1240: "CHI-Liga de Ascenso"}`).
- Produces: `is_chilean_event(event: dict) -> bool`, `estimate_match_minute(event: dict, now_ts: float) -> int | None`. (Later tasks extend this same file.)

- [ ] **Step 1: Write the failing tests**

`tests/test_sofascore_shadow.py` (the `time`/`status` shapes below are real observed Sofascore live-event payloads from the design verification, not invented):
```python
from goles.sofascore.shadow import estimate_match_minute, is_chilean_event

NOW = 1_783_910_000.0


def _live_event(utid=11653, status_type="inprogress", description="2nd half",
                initial=2700, elapsed_seconds=600):
    return {
        "id": 15421086,
        "tournament": {"name": "Liga de Primera", "uniqueTournament": {"id": utid}},
        "homeTeam": {"name": "Colo-Colo"},
        "awayTeam": {"name": "Cobresal"},
        "status": {"type": status_type, "description": description},
        "time": {
            "currentPeriodStartTimestamp": NOW - elapsed_seconds,
            "initial": initial,
            "max": 5400,
        },
    }


def test_is_chilean_event_matches_both_tracked_utids_only():
    assert is_chilean_event(_live_event(utid=11653)) is True
    assert is_chilean_event(_live_event(utid=1240)) is True
    assert is_chilean_event(_live_event(utid=17)) is False  # Premier League
    assert is_chilean_event({"id": 1, "tournament": {"name": "Liga de Primera"}}) is False


def test_estimate_minute_second_half():
    # 10 minutes into the second half (initial 2700 s = 45 min) -> minute 56
    event = _live_event(initial=2700, elapsed_seconds=600)
    assert estimate_match_minute(event, NOW) == 56


def test_estimate_minute_first_half():
    event = _live_event(description="1st half", initial=0, elapsed_seconds=300)
    assert estimate_match_minute(event, NOW) == 6


def test_estimate_minute_clamps_to_90():
    event = _live_event(initial=2700, elapsed_seconds=4000)  # deep injury time
    assert estimate_match_minute(event, NOW) == 90


def test_estimate_minute_none_at_halftime():
    # currentPeriodStartTimestamp is stale during the break -- skip inference
    event = _live_event(description="Halftime")
    assert estimate_match_minute(event, NOW) is None


def test_estimate_minute_none_when_time_dict_empty():
    # observed for real on some lower-coverage live events
    event = _live_event()
    event["time"] = {}
    assert estimate_match_minute(event, NOW) is None


def test_estimate_minute_none_when_not_inprogress():
    event = _live_event(status_type="finished")
    assert estimate_match_minute(event, NOW) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\pytest.exe tests/test_sofascore_shadow.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'goles.sofascore.shadow'`

- [ ] **Step 3: Write the implementation**

`src/goles/sofascore/shadow.py`:
```python
from __future__ import annotations

from goles.sofascore.backfill import TRACKED_UTIDS


def is_chilean_event(event: dict) -> bool:
    """Filter by uniqueTournament id, NEVER by tournament name (Paraguay's
    second tier shares Chile's old league name -- verified collision)."""
    utid = event.get("tournament", {}).get("uniqueTournament", {}).get("id")
    return utid in TRACKED_UTIDS


def estimate_match_minute(event: dict, now_ts: float) -> int | None:
    """Current match minute from a live event's `time` fields (verified
    shape 2026-07-12: `initial` is seconds already on the clock at period
    start -- 0 first half, 2700 second half -- and
    `currentPeriodStartTimestamp` is the period's unix start). Returns
    None when inference should be skipped: match not in progress, at
    halftime (the timestamp is stale during the break), or the `time`
    dict is empty (observed on some lower-coverage events)."""
    status = event.get("status", {})
    if status.get("type") != "inprogress" or status.get("description") == "Halftime":
        return None
    time_info = event.get("time") or {}
    period_start = time_info.get("currentPeriodStartTimestamp")
    initial = time_info.get("initial")
    if period_start is None or initial is None:
        return None
    elapsed_minutes = int((now_ts - period_start) // 60)
    minute = initial // 60 + elapsed_minutes + 1
    return max(1, min(minute, 90))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\pytest.exe tests/test_sofascore_shadow.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```powershell
git add src/goles/sofascore/shadow.py tests/test_sofascore_shadow.py
git commit -m "feat: add Chilean live-event detection and match-minute estimation"
```

---

### Task 3: Shadow module — live feature assembly, priors, prediction

**Files:**
- Modify: `src/goles/sofascore/shadow.py` (extend Task 2's file)
- Test: `tests/test_sofascore_shadow.py` (append)

**Interfaces:**
- Consumes: `goles.features.{compute_ml_features, compute_state_at_minute}`, `goles.model.{dynamic_lambda, prob_goal_in_window}`, `goles.backtest.{RECENT_WINDOW_MINUTES, HORIZON_MINUTES}`, `goles.dataset.FEATURE_NAMES`, `goles.gbt_model.{raw_predictions, apply_platt_scaling}`, `goles.priors.team_match_xg`, `goles.sofascore.translate.{translate_shot, UnknownVocabularyError}`, `goles.xg_model.predict_xg`.
- Produces: `POISSON_BLEND = 0.1`, `translate_live_shots(sofa_shots: list[dict], xg_booster) -> list[dict]` (returns shot dicts with keys `minute, team, xg, is_goal, location_x, location_y, situation, shot_type, last_action`), `chile_prior_xg(chile_conn, team_name: str, season: str) -> float`, `chile_rest_days(chile_conn, team_name: str, season: str, today_iso: str) -> float`, `build_live_features(shots, cards, cutoff_minute: int, team: str, prior: float, own_rest_days: float, opp_rest_days: float) -> dict[str, float]`, `predict_goal_prob(booster, platt: tuple[float, float], features: dict[str, float]) -> float`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_sofascore_shadow.py`:
```python
import json

from goles.dataset import FEATURE_NAMES, build_dataset
from goles.db import get_connection as history_connection, init_db as history_init_db
from goles.loaders.understat import persist_shots
from goles.sofascore.shadow import (
    build_live_features,
    chile_prior_xg,
    chile_rest_days,
    predict_goal_prob,
    translate_live_shots,
)


class _FakeXgBooster:
    def predict(self, X):
        return [0.123] * len(X)


def test_translate_live_shots_computes_own_xg_and_flips_own_goals():
    sofa_shots = [
        {"id": 1, "time": 12, "shotType": "goal", "situation": "assisted", "isHome": True,
         "playerCoordinates": {"x": 8.0, "y": 50.0}, "bodyPart": "right-foot"},
        {"id": 2, "time": 30, "shotType": "goal", "situation": "own-goal", "goalType": "own",
         "isHome": True, "playerCoordinates": {"x": 4.0, "y": 50.0}, "bodyPart": "other"},
        {"id": 3, "time": 40, "shotType": "miss", "situation": "shootout",  # unknown vocab -> dropped
         "isHome": False, "playerCoordinates": {"x": 11.0, "y": 44.0}, "bodyPart": "head"},
    ]
    shots = translate_live_shots(sofa_shots, _FakeXgBooster())
    assert len(shots) == 2
    assert shots[0]["team"] == "home" and abs(shots[0]["xg"] - 0.123) < 1e-9
    # own goal: shooter was home -> goal credited to away, xg forced to 0.0
    assert shots[1]["team"] == "away" and shots[1]["xg"] == 0.0 and shots[1]["is_goal"] is True
    assert all(s["last_action"] is None for s in shots)


def _chile_db_with_history():
    conn = history_connection(":memory:")
    history_init_db(conn)
    records = []
    # Two finished Colo-Colo matches in season 2026 (matches the backfill's shape)
    for match_id, date, xg1, xg2 in ((900, "2026-06-28", 1.5, 0.7), (901, "2026-07-05", 2.5, 0.9)):
        records += [
            {"match_id": match_id, "league": "CHI-Liga de Primera", "season": "2026",
             "date": date, "home_team": "Colo-Colo", "away_team": "Cobresal",
             "minute": 10, "team": "home", "xg": xg1, "is_goal": False,
             "location_x": 0.9, "location_y": 0.5, "situation": "OpenPlay", "shot_type": "RightFoot"},
            {"match_id": match_id, "league": "CHI-Liga de Primera", "season": "2026",
             "date": date, "home_team": "Colo-Colo", "away_team": "Cobresal",
             "minute": 70, "team": "away", "xg": xg2, "is_goal": False,
             "location_x": 0.9, "location_y": 0.5, "situation": "OpenPlay", "shot_type": "Head"},
        ]
    persist_shots(conn, records)
    return conn


def test_chile_prior_xg_is_mean_xg_over_season_matches():
    conn = _chile_db_with_history()
    assert abs(chile_prior_xg(conn, "Colo-Colo", "2026") - 2.0) < 1e-9   # (1.5 + 2.5) / 2
    assert abs(chile_prior_xg(conn, "Cobresal", "2026") - 0.8) < 1e-9    # (0.7 + 0.9) / 2


def test_chile_prior_and_rest_days_degrade_for_unknown_team():
    conn = _chile_db_with_history()
    assert chile_prior_xg(conn, "Equipo Fantasma", "2026") == 0.0
    assert chile_rest_days(conn, "Equipo Fantasma", "2026", "2026-07-12") == 7.0


def test_chile_rest_days_measures_from_last_match_date():
    conn = _chile_db_with_history()
    assert chile_rest_days(conn, "Colo-Colo", "2026", "2026-07-12") == 7.0  # 07-05 -> 07-12


def test_live_features_match_training_pipeline_exactly():
    """The consistency gate: one synthetic match through the training path
    (persist_shots -> build_dataset) and the live path (build_live_features)
    must produce identical feature dicts for both teams."""
    conn = history_connection(":memory:")
    history_init_db(conn)
    shots_common = [
        {"minute": 8, "team": "home", "xg": 0.31, "is_goal": True,
         "location_x": 0.93, "location_y": 0.48, "situation": "OpenPlay", "shot_type": "RightFoot"},
        {"minute": 15, "team": "away", "xg": 0.05, "is_goal": False,
         "location_x": 0.78, "location_y": 0.30, "situation": "FromCorner", "shot_type": "Head"},
        {"minute": 19, "team": "home", "xg": 0.12, "is_goal": False,
         "location_x": 0.88, "location_y": 0.55, "situation": "SetPiece", "shot_type": "LeftFoot"},
    ]
    records = [
        dict(s, match_id=700, league="CHI-Liga de Primera", season="2026",
             date="2026-07-10", home_team="Colo-Colo", away_team="Cobresal")
        for s in shots_common
    ]
    persist_shots(conn, records)
    rows = build_dataset(conn, cutoff_minutes=[20], blend=0.1)
    assert len(rows) == 2  # one per team

    live_shots = [dict(s, last_action=None) for s in shots_common]
    for row in rows:
        live = build_live_features(
            live_shots, [], 20, row.team,
            prior=0.0, own_rest_days=7.0, opp_rest_days=7.0,  # matchday-1 values, same as build_dataset
        )
        assert live == row.features


def test_predict_goal_prob_orders_features_and_applies_platt():
    class _SpyBooster:
        def __init__(self):
            self.seen = None
        def predict(self, X):
            self.seen = X
            return [0.4] * len(X)

    booster = _SpyBooster()
    features = {name: float(i) for i, name in enumerate(FEATURE_NAMES)}
    prob = predict_goal_prob(booster, (1.0, 0.0), features)  # (1, 0) = identity Platt
    assert abs(prob - 0.4) < 1e-9
    assert list(booster.seen[0]) == [float(i) for i in range(len(FEATURE_NAMES))]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\pytest.exe tests/test_sofascore_shadow.py -v`
Expected: the 7 Task 2 tests pass; the new tests FAIL with `ImportError: cannot import name 'build_live_features'`

- [ ] **Step 3: Write the implementation**

Append to `src/goles/sofascore/shadow.py` (and extend its imports):
```python
from datetime import date as _date

from goles.backtest import HORIZON_MINUTES, RECENT_WINDOW_MINUTES
from goles.dataset import FEATURE_NAMES
from goles.features import compute_ml_features, compute_state_at_minute
from goles.gbt_model import apply_platt_scaling, raw_predictions
from goles.model import dynamic_lambda, prob_goal_in_window
from goles.priors import team_match_xg
from goles.sofascore.translate import UnknownVocabularyError, translate_shot
from goles.xg_model import predict_xg

# Must equal train_gbt_chile.POISSON_COMPARISON_BLEND -- the poisson_prob
# feature the Chilean model was trained on used this blend, so live serving
# must compute it identically (train/serve consistency).
POISSON_BLEND = 0.1


def translate_live_shots(sofa_shots: list[dict], xg_booster) -> list[dict]:
    """Raw Sofascore shotmap -> the exact shot-dict shape the training
    pipeline's load_match_shots produces (minute/team/xg/is_goal/location/
    situation/shot_type/last_action), with our own computed xG (Sofascore
    publishes xg: null for Chile) and the own-goal flip applied. Unknown
    vocabulary fails loud per shot and drops it, same as the backfill."""
    shots = []
    for sofa_shot in sofa_shots:
        try:
            t = translate_shot(sofa_shot)
        except UnknownVocabularyError as exc:
            print(f"  ADVERTENCIA (sombra): {exc}, tiro omitido.")
            continue
        xg = 0.0 if t["is_own_goal"] else predict_xg(xg_booster, t)
        shots.append(
            {
                "minute": t["minute"], "team": "home" if t["is_home"] else "away",
                "xg": xg, "is_goal": t["is_goal"],
                "location_x": t["location_x"], "location_y": t["location_y"],
                "situation": t["situation"], "shot_type": t["shot_type"],
                "last_action": None,
            }
        )
    return shots


def _chile_team_id(chile_conn, team_name: str) -> int | None:
    row = chile_conn.execute(
        "SELECT team_id FROM teams WHERE name = ?", (team_name,)
    ).fetchone()
    return row[0] if row else None


def chile_prior_xg(chile_conn, team_name: str, season: str) -> float:
    """Pre-match prior: mean xG per match across the team's backfilled
    matches this season (all finished, so all strictly earlier than any
    live match). Team names match exactly -- both the live feed and the
    backfill store Sofascore's own names. Unknown team -> 0.0, the same
    neutral prior trailing_xg_per90 gives on matchday 1."""
    team_id = _chile_team_id(chile_conn, team_name)
    if team_id is None:
        return 0.0
    match_ids = [
        row[0]
        for row in chile_conn.execute(
            """SELECT match_id FROM matches
               WHERE season = ? AND (home_team_id = ? OR away_team_id = ?)""",
            (season, team_id, team_id),
        ).fetchall()
    ]
    if not match_ids:
        return 0.0
    return sum(team_match_xg(chile_conn, mid, team_id) for mid in match_ids) / len(match_ids)


def chile_rest_days(chile_conn, team_name: str, season: str, today_iso: str) -> float:
    """Days since the team's most recent backfilled match this season.
    Unknown team or no matches -> 7.0, the same default build_dataset uses
    when days_since_last_match returns None."""
    team_id = _chile_team_id(chile_conn, team_name)
    if team_id is None:
        return 7.0
    row = chile_conn.execute(
        """SELECT MAX(date) FROM matches
           WHERE season = ? AND (home_team_id = ? OR away_team_id = ?)""",
        (season, team_id, team_id),
    ).fetchone()
    if row is None or row[0] is None:
        return 7.0
    return float((_date.fromisoformat(today_iso) - _date.fromisoformat(row[0])).days)


def build_live_features(
    shots: list[dict],
    cards: list[dict],
    cutoff_minute: int,
    team: str,
    prior: float,
    own_rest_days: float,
    opp_rest_days: float,
) -> dict[str, float]:
    """Assembles the full FEATURE_NAMES dict for one (team, minute) -- a
    line-for-line mirror of dataset.build_dataset's row assembly (market
    features hard-zero: no Chilean odds existed in training either).
    Guarded by test_live_features_match_training_pipeline_exactly."""
    features = dict(compute_ml_features(shots, cutoff_minute, team, cards=cards))
    state = compute_state_at_minute(shots, cutoff_minute, window=RECENT_WINDOW_MINUTES)
    recent_xg = state.home_xg_last15 if team == "home" else state.away_xg_last15
    lam = dynamic_lambda(
        pre_match_xg_per90=prior,
        in_match_xg_recent=recent_xg,
        recent_window_minutes=RECENT_WINDOW_MINUTES,
        horizon_minutes=HORIZON_MINUTES,
        blend=POISSON_BLEND,
    )
    features["own_rest_days"] = own_rest_days
    features["opp_rest_days"] = opp_rest_days
    features["own_market_wp"] = 0.0
    features["opp_market_wp"] = 0.0
    features["market_draw_wp"] = 0.0
    features["market_over25_wp"] = 0.0
    features["trailing_prior_xg"] = prior
    features["poisson_prob"] = prob_goal_in_window(lam)
    return features


def predict_goal_prob(booster, platt: tuple[float, float], features: dict[str, float]) -> float:
    X = [[features[name] for name in FEATURE_NAMES]]
    raw = raw_predictions(booster, X)
    a, b = platt
    return apply_platt_scaling(raw, a, b)[0]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\pytest.exe tests/test_sofascore_shadow.py -v`
Expected: 13 passed. If `test_live_features_match_training_pipeline_exactly` fails, the live assembly diverged from `build_dataset` — fix `build_live_features`, never the test.

- [ ] **Step 5: Run the full suite, then commit**

Run: `.venv\Scripts\pytest.exe -q` → all pass.

```powershell
git add src/goles/sofascore/shadow.py tests/test_sofascore_shadow.py
git commit -m "feat: add live feature assembly with train/serve consistency gate"
```

---

### Task 4: Shadow prediction/alert persistence in the live store

**Files:**
- Modify: `src/goles/sofascore/store.py`
- Test: `tests/test_sofascore_store.py` (append; the file already tests `persist_shot`/`persist_card`)

**Interfaces:**
- Produces: two new tables in the `SCHEMA` string (`shadow_predictions`, `shadow_alerts`), `persist_shadow_prediction(conn, sofascore_event_id: int, fetched_at: str, home_team: str, away_team: str, team: str, minute: int, probability: float, features_json: str) -> None`, `record_shadow_alert(conn, sofascore_event_id: int, team: str, minute: int, probability: float, sent_at: str) -> None`, `last_shadow_alert_minute(conn, sofascore_event_id: int, team: str) -> int | None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_sofascore_store.py`:
```python
from goles.sofascore.store import (
    last_shadow_alert_minute,
    persist_shadow_prediction,
    record_shadow_alert,
)


def test_persist_shadow_prediction_round_trips():
    conn = get_connection(":memory:")
    init_db(conn)
    persist_shadow_prediction(
        conn, 15421086, "2026-07-12T22:00:00+00:00", "Colo-Colo", "Cobresal",
        "away", 63, 0.34, '{"minute": 63.0}',
    )
    row = conn.execute(
        """SELECT sofascore_event_id, team, minute, probability, features_json
           FROM shadow_predictions"""
    ).fetchone()
    assert row == (15421086, "away", 63, 0.34, '{"minute": 63.0}')


def test_last_shadow_alert_minute_is_none_before_any_alert_then_latest():
    conn = get_connection(":memory:")
    init_db(conn)
    assert last_shadow_alert_minute(conn, 15421086, "away") is None
    record_shadow_alert(conn, 15421086, "away", 40, 0.31, "2026-07-12T21:40:00+00:00")
    record_shadow_alert(conn, 15421086, "away", 63, 0.34, "2026-07-12T22:03:00+00:00")
    record_shadow_alert(conn, 15421086, "home", 70, 0.30, "2026-07-12T22:10:00+00:00")
    assert last_shadow_alert_minute(conn, 15421086, "away") == 63  # per-team, latest
    assert last_shadow_alert_minute(conn, 99999, "away") is None   # per-event
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\pytest.exe tests/test_sofascore_store.py -v`
Expected: existing tests pass; new ones FAIL with `ImportError: cannot import name 'persist_shadow_prediction'`

- [ ] **Step 3: Write the implementation**

In `src/goles/sofascore/store.py`, append to the `SCHEMA` string (inside the triple-quoted literal, after the `cards` table):
```sql
CREATE TABLE IF NOT EXISTS shadow_predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    sofascore_event_id INTEGER NOT NULL,
    fetched_at TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    team TEXT NOT NULL,
    minute INTEGER NOT NULL,
    probability REAL NOT NULL,
    features_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS shadow_alerts (
    alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
    sofascore_event_id INTEGER NOT NULL,
    team TEXT NOT NULL,
    minute INTEGER NOT NULL,
    probability REAL NOT NULL,
    sent_at TEXT NOT NULL
);
```

Then append the functions:
```python
def persist_shadow_prediction(
    conn: sqlite3.Connection,
    sofascore_event_id: int,
    fetched_at: str,
    home_team: str,
    away_team: str,
    team: str,
    minute: int,
    probability: float,
    features_json: str,
) -> None:
    """Every shadow prediction is persisted (not just alerted ones) --
    this table is the future live-serve validation set: after finished
    matches land in goles_chile.db via the resumable backfill, joining on
    sofascore_event_id measures the live path's real BSS."""
    conn.execute(
        """INSERT INTO shadow_predictions
           (sofascore_event_id, fetched_at, home_team, away_team, team, minute, probability, features_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (sofascore_event_id, fetched_at, home_team, away_team, team, minute, probability, features_json),
    )
    conn.commit()


def record_shadow_alert(
    conn: sqlite3.Connection,
    sofascore_event_id: int,
    team: str,
    minute: int,
    probability: float,
    sent_at: str,
) -> None:
    conn.execute(
        """INSERT INTO shadow_alerts (sofascore_event_id, team, minute, probability, sent_at)
           VALUES (?, ?, ?, ?, ?)""",
        (sofascore_event_id, team, minute, probability, sent_at),
    )
    conn.commit()


def last_shadow_alert_minute(
    conn: sqlite3.Connection, sofascore_event_id: int, team: str
) -> int | None:
    """Latest alerted minute for (event, team) -- the cooldown state.
    Persisted in the DB so a poller restart can't re-spam Telegram."""
    row = conn.execute(
        "SELECT MAX(minute) FROM shadow_alerts WHERE sofascore_event_id = ? AND team = ?",
        (sofascore_event_id, team),
    ).fetchone()
    return row[0] if row and row[0] is not None else None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\pytest.exe tests/test_sofascore_store.py -v`
Expected: all pass (existing + 2 new)

- [ ] **Step 5: Run the full suite, then commit**

Run: `.venv\Scripts\pytest.exe -q` → all pass.

```powershell
git add src/goles/sofascore/store.py tests/test_sofascore_store.py
git commit -m "feat: add shadow prediction and alert tables to the live store"
```

---

### Task 5: Poller integration — utid discovery, Chilean own-xG, shadow cycle, alerts

**Files:**
- Modify: `src/goles/sofascore/shadow.py` (alerting + `shadow_cycle`)
- Modify: `src/goles/sofascore/poller.py`
- Test: `tests/test_sofascore_shadow.py` (append), `tests/test_sofascore_poller.py` (append + one edit)

**Interfaces:**
- Consumes: everything from Tasks 1–4; `goles.sofascore.client.get_shotmap`; `goles.train_xg.XG_MODEL_PATH`; `goles.train_gbt_chile.MODEL_DIR`; `goles.sofascore.backfill.CHILE_DB_PATH`; `goles.persistence.load_model`; `goles.xg_model.load_xg_model`; `goles.db.get_connection` (aliased, for the Chile DB); `goles.telegram.send_message`.
- Produces: `SHADOW_ALERT_THRESHOLD = 0.30`, `SHADOW_ALERT_COOLDOWN_MINUTES = 15`, `INFERENCE_MIN_MINUTE = 20`, `INFERENCE_MAX_MINUTE = 80`, `SHADOW_BANNER`, `should_alert(probability: float, minute: int, last_alert_minute: int | None) -> bool`, `format_alert(home, away, league, minute, home_score, away_score, team_label, probability) -> str`, `shadow_cycle(client, live_conn, chile_conn, xg_booster, goal_booster, platt, send_alert, chilean_events, now_ts, season) -> None` — all in `shadow.py`. In `poller.py`: `discover_tracked_live_events` extended with utid matching, `poll_once(client, conn, live_events, xg_booster=None)`, `main()` wired.

- [ ] **Step 1: Write the failing shadow tests**

Append to `tests/test_sofascore_shadow.py`:
```python
from unittest.mock import Mock, patch

from goles.sofascore.shadow import (
    SHADOW_ALERT_THRESHOLD,
    SHADOW_BANNER,
    format_alert,
    shadow_cycle,
    should_alert,
)
from goles.sofascore.store import get_connection as live_connection, init_db as live_init_db


def test_should_alert_threshold_and_cooldown():
    assert should_alert(0.30, 50, None) is True
    assert should_alert(0.29, 50, None) is False
    assert should_alert(0.35, 50, 40) is False   # 10 min since last alert < 15
    assert should_alert(0.35, 56, 40) is True    # 16 min -> cooldown over


def test_format_alert_carries_the_test_mode_banner_and_no_betting_framing():
    text = format_alert("Colo-Colo", "Cobresal", "CHI-Liga de Primera", 63, 1, 0, "Cobresal", 0.34)
    assert text.startswith(SHADOW_BANNER)
    assert "Colo-Colo 1-0 Cobresal" in text
    assert "Min 63'" in text and "Cobresal" in text and "34%" in text
    # the banner itself says "sin edge confirmado" -- what must never appear
    # is betting/value framing:
    for forbidden in ("apuesta", "apostar", "stake", "cuota", "value"):
        assert forbidden not in text.lower()


class _ProbBooster:
    """Duck-typed goal booster returning a fixed raw probability; with
    Platt (1.0, 0.0) -- the identity -- the calibrated prob equals it."""
    def __init__(self, prob):
        self.prob = prob
    def predict(self, X):
        return [self.prob] * len(X)


def _shadow_setup():
    live_conn = live_connection(":memory:")
    live_init_db(live_conn)
    chile_conn = _chile_db_with_history()
    sent = []
    event = _live_event(initial=2700, elapsed_seconds=600)  # minute 56
    sofa_shots = [
        {"id": 1, "time": 12, "shotType": "goal", "situation": "assisted", "isHome": True,
         "playerCoordinates": {"x": 8.0, "y": 50.0}, "bodyPart": "right-foot"},
    ]
    return live_conn, chile_conn, sent, event, sofa_shots


def test_shadow_cycle_persists_predictions_for_both_teams_and_alerts_over_threshold():
    live_conn, chile_conn, sent, event, sofa_shots = _shadow_setup()
    with patch("goles.sofascore.shadow.get_shotmap", return_value=sofa_shots):
        shadow_cycle(
            Mock(), live_conn, chile_conn, _FakeXgBooster(), _ProbBooster(0.9), (1.0, 0.0),
            sent.append, [event], NOW, "2026",
        )
    preds = live_conn.execute(
        "SELECT team, minute, probability FROM shadow_predictions ORDER BY team"
    ).fetchall()
    assert len(preds) == 2
    assert preds[0][1] == 56 and abs(preds[0][2] - 0.9) < 1e-9
    assert len(sent) == 2 and all(m.startswith(SHADOW_BANNER) for m in sent)
    alerts = live_conn.execute("SELECT team, minute FROM shadow_alerts").fetchall()
    assert len(alerts) == 2


def test_shadow_cycle_respects_cooldown_across_cycles():
    live_conn, chile_conn, sent, event, sofa_shots = _shadow_setup()
    with patch("goles.sofascore.shadow.get_shotmap", return_value=sofa_shots):
        args = (Mock(), live_conn, chile_conn, _FakeXgBooster(), _ProbBooster(0.9), (1.0, 0.0), sent.append)
        shadow_cycle(*args, [event], NOW, "2026")
        shadow_cycle(*args, [event], NOW + 60, "2026")  # next cycle, 1 min later
    assert len(sent) == 2  # no re-alerts within the cooldown
    assert live_conn.execute("SELECT COUNT(*) FROM shadow_predictions").fetchone()[0] == 4


def test_shadow_cycle_below_threshold_persists_but_never_alerts():
    live_conn, chile_conn, sent, event, sofa_shots = _shadow_setup()
    with patch("goles.sofascore.shadow.get_shotmap", return_value=sofa_shots):
        shadow_cycle(
            Mock(), live_conn, chile_conn, _FakeXgBooster(), _ProbBooster(0.05), (1.0, 0.0),
            sent.append, [event], NOW, "2026",
        )
    assert sent == []
    assert live_conn.execute("SELECT COUNT(*) FROM shadow_predictions").fetchone()[0] == 2


def test_shadow_cycle_skips_out_of_range_minutes_and_isolates_event_failures():
    live_conn, chile_conn, sent, _, sofa_shots = _shadow_setup()
    early = _live_event(description="1st half", initial=0, elapsed_seconds=300)  # minute 6 < 20
    broken = _live_event()
    broken["id"] = 424242
    ok = _live_event(initial=2700, elapsed_seconds=600)
    ok["id"] = 555555

    def fake_shotmap(client, event_id):
        if event_id == 424242:
            raise RuntimeError("404")
        return sofa_shots

    with patch("goles.sofascore.shadow.get_shotmap", side_effect=fake_shotmap):
        shadow_cycle(
            Mock(), live_conn, chile_conn, _FakeXgBooster(), _ProbBooster(0.9), (1.0, 0.0),
            sent.append, [early, broken, ok], NOW, "2026",
        )
    event_ids = {r[0] for r in live_conn.execute(
        "SELECT sofascore_event_id FROM shadow_predictions").fetchall()}
    assert event_ids == {555555}  # early skipped, broken isolated, ok processed
```

- [ ] **Step 2: Write the failing poller tests**

Append to `tests/test_sofascore_poller.py`:
```python
def test_discover_tracked_live_events_also_matches_chilean_utids():
    client = Mock()
    with patch("goles.sofascore.poller.list_live_events", return_value=[
        {"id": 1, "tournament": {"name": "Premier League", "uniqueTournament": {"id": 17}}},
        {"id": 2, "tournament": {"name": "Liga de Primera", "uniqueTournament": {"id": 11653}}},
        {"id": 3, "tournament": {"name": "Liga de Ascenso", "uniqueTournament": {"id": 1240}}},
        {"id": 4, "tournament": {"name": "Primera División B", "uniqueTournament": {"id": 22759}}},  # Paraguay!
    ]):
        events = discover_tracked_live_events(client)
    assert [e["id"] for e in events] == [1, 2, 3]


def test_poll_once_computes_own_xg_for_chilean_shots_with_null_xg():
    event = {
        "id": 777,
        "homeTeam": {"name": "Colo-Colo"},
        "awayTeam": {"name": "Cobresal"},
        "tournament": {"name": "Liga de Primera", "uniqueTournament": {"id": 11653}},
    }
    shots = [
        {"id": 1, "time": 12, "xg": None, "shotType": "goal", "situation": "assisted",
         "isHome": True, "playerCoordinates": {"x": 8.0, "y": 50.0}, "bodyPart": "right-foot"},
        {"id": 2, "time": 40, "xg": None, "shotType": "miss", "situation": "shootout",  # unknown vocab
         "isHome": False, "playerCoordinates": {"x": 11.0, "y": 44.0}, "bodyPart": "head"},
    ]

    class _FakeXgBooster:
        def predict(self, X):
            return [0.123] * len(X)

    conn = get_connection(":memory:")
    init_db(conn)
    with patch("goles.sofascore.poller.get_shotmap", return_value=shots):
        with patch("goles.sofascore.poller.get_incidents", return_value=[]):
            poll_once(Mock(), conn, [event], xg_booster=_FakeXgBooster())

    rows = conn.execute("SELECT sofascore_shot_id, xg FROM shots").fetchall()
    assert rows == [(1, 0.123)]  # shot 2 dropped loudly (unknown vocab), shot 1 got our xG
```

- [ ] **Step 3: Run both test files to verify the new tests fail**

Run: `.venv\Scripts\pytest.exe tests/test_sofascore_shadow.py tests/test_sofascore_poller.py -v`
Expected: new tests FAIL (`ImportError: cannot import name 'shadow_cycle'`; `TypeError: poll_once() got an unexpected keyword argument 'xg_booster'`); all previously passing tests still pass.

- [ ] **Step 4: Implement the shadow alerting + cycle**

Append to `src/goles/sofascore/shadow.py` (extend imports with `import json`, `from datetime import datetime, timezone`, `from goles.sofascore.client import get_shotmap`, `from goles.sofascore.store import last_shadow_alert_minute, persist_shadow_prediction, record_shadow_alert`):
```python
SHADOW_ALERT_THRESHOLD = 0.30
SHADOW_ALERT_COOLDOWN_MINUTES = 15
INFERENCE_MIN_MINUTE = 20
INFERENCE_MAX_MINUTE = 80
SHADOW_BANNER = "\U0001f9ea MODO PRUEBA — sin edge confirmado, solo observación"


def should_alert(probability: float, minute: int, last_alert_minute: int | None) -> bool:
    if probability < SHADOW_ALERT_THRESHOLD:
        return False
    if last_alert_minute is not None and minute - last_alert_minute < SHADOW_ALERT_COOLDOWN_MINUTES:
        return False
    return True


def format_alert(
    home: str, away: str, league: str, minute: int,
    home_score: int, away_score: int, team_label: str, probability: float,
) -> str:
    """Observation-only copy: the banner is non-negotiable and nothing in
    the message may frame this as a betting signal (no odds, no stakes,
    no value language) -- the model's Chilean BSS is ~0 and the message
    must not pretend otherwise."""
    return (
        f"{SHADOW_BANNER}\n"
        f"⚽ {home} {home_score}-{away_score} {away} ({league})\n"
        f"Min {minute}' — P(gol de {team_label} en próx. 15 min): {probability:.0%}"
    )


def shadow_cycle(
    client,
    live_conn,
    chile_conn,
    xg_booster,
    goal_booster,
    platt: tuple[float, float],
    send_alert,
    chilean_events: list[dict],
    now_ts: float,
    season: str,
) -> None:
    """One shadow-inference pass over the live Chilean events: estimate the
    minute, translate the shotmap with our own xG, assemble the exact
    training-time feature vector for each side, persist both predictions,
    and alert (threshold + cooldown) via send_alert(text). Per-event
    failures are isolated -- one broken event never blocks the rest."""
    fetched_at = datetime.fromtimestamp(now_ts, tz=timezone.utc).isoformat()
    today_iso = fetched_at[:10]
    for event in chilean_events:
        event_id = event["id"]
        try:
            minute = estimate_match_minute(event, now_ts)
            if minute is None or not (INFERENCE_MIN_MINUTE <= minute <= INFERENCE_MAX_MINUTE):
                continue
            home = event["homeTeam"]["name"]
            away = event["awayTeam"]["name"]
            league = TRACKED_UTIDS[event["tournament"]["uniqueTournament"]["id"]]

            shots = translate_live_shots(get_shotmap(client, event_id), xg_booster)
            cards = [
                {"team": row[0], "minute": row[1]}
                for row in live_conn.execute(
                    "SELECT team, minute FROM cards WHERE sofascore_event_id = ?", (event_id,)
                ).fetchall()
            ]
            home_score = sum(1 for s in shots if s["team"] == "home" and s["is_goal"])
            away_score = sum(1 for s in shots if s["team"] == "away" and s["is_goal"])
            priors = {"home": chile_prior_xg(chile_conn, home, season),
                      "away": chile_prior_xg(chile_conn, away, season)}
            rests = {"home": chile_rest_days(chile_conn, home, season, today_iso),
                     "away": chile_rest_days(chile_conn, away, season, today_iso)}

            for team, team_label in (("home", home), ("away", away)):
                opponent = "away" if team == "home" else "home"
                features = build_live_features(
                    shots, cards, minute, team,
                    prior=priors[team], own_rest_days=rests[team], opp_rest_days=rests[opponent],
                )
                probability = predict_goal_prob(goal_booster, platt, features)
                persist_shadow_prediction(
                    live_conn, event_id, fetched_at, home, away, team, minute,
                    probability, json.dumps(features),
                )
                if should_alert(probability, minute, last_shadow_alert_minute(live_conn, event_id, team)):
                    send_alert(format_alert(home, away, league, minute, home_score, away_score, team_label, probability))
                    record_shadow_alert(live_conn, event_id, team, minute, probability, fetched_at)
        except Exception as exc:
            print(f"ADVERTENCIA: fallo el modo sombra para el evento {event_id} ({exc}), se continua.")
```

- [ ] **Step 5: Implement the poller changes**

In `src/goles/sofascore/poller.py`:

1. Extend the imports:
```python
import os

from goles.db import get_connection as history_connection
from goles.persistence import load_model
from goles.sofascore.backfill import CHILE_DB_PATH
from goles.sofascore.shadow import is_chilean_event, shadow_cycle
from goles.sofascore.translate import translate_shot
from goles.telegram import send_message
from goles.train_gbt_chile import MODEL_DIR as CHILE_MODEL_DIR
from goles.train_xg import XG_MODEL_PATH
from goles.xg_model import load_xg_model, predict_xg
```

2. Replace `discover_tracked_live_events`:
```python
def discover_tracked_live_events(client) -> list[dict]:
    """Returns live events matching TRACKED_TOURNAMENTS by exact name
    (EPL/Bundesliga -- unchanged) plus Chilean events by uniqueTournament
    id (name matching is forbidden for Chile: Paraguay's second tier
    shares the old league name, verified collision)."""
    events = list_live_events(client)
    return [
        e for e in events
        if e.get("tournament", {}).get("name") in TRACKED_TOURNAMENTS or is_chilean_event(e)
    ]
```

3. In `poll_once`, change the signature to `def poll_once(client, conn, live_events: list[dict], xg_booster=None) -> None:` and, inside the shot loop, replace `xg=shot["xg"],` with a computed value. Add right before `persist_shot(`:
```python
                    xg_value = shot.get("xg")
                    if xg_value is None and is_chilean_event(event) and xg_booster is not None:
                        # Sofascore publishes xg: null for Chile and the
                        # store's xg column is NOT NULL -- compute our own,
                        # same as the historical backfill (own goals: 0.0).
                        translated = translate_shot(shot)
                        xg_value = 0.0 if translated["is_own_goal"] else predict_xg(xg_booster, translated)
```
and pass `xg=xg_value,` to `persist_shot`. (A `translate_shot` raise on unknown vocabulary lands in the existing per-shot `except`, dropping only that shot with a warning — the behavior `test_poll_once_computes_own_xg_for_chilean_shots_with_null_xg` locks in.)

4. Replace `main()`:
```python
def main() -> None:
    """Persistent entrypoint: polls forever, syncing to the VPS after each
    cycle. Shadow mode (Chilean live inference + observation-only Telegram
    alerts) activates only if its artifacts load; any failure disables it
    loudly while the EPL/Bundesliga poller keeps running untouched."""
    client = tls_requests.Client()
    conn = get_connection()
    init_db(conn)

    xg_booster = goal_booster = platt = chile_conn = None
    try:
        xg_booster = load_xg_model(XG_MODEL_PATH)
        goal_booster, platt = load_model(CHILE_MODEL_DIR)
        chile_conn = history_connection(CHILE_DB_PATH)
        print("Modo sombra Chile: activado (utids 11653/1240).")
    except Exception as exc:
        print(f"ADVERTENCIA: modo sombra desactivado ({exc}) -- el poller EPL/Bundesliga sigue normal.")

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if token and chat_id:
        def send_alert(text: str) -> None:
            if not send_message(client, token, chat_id, text):
                print("ADVERTENCIA: alerta sombra no enviada a Telegram; solo consola:\n" + text)
    else:
        print("ADVERTENCIA: TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID no configurados -- alertas sombra solo en consola.")

        def send_alert(text: str) -> None:
            print(text)

    while True:
        try:
            live_events = discover_tracked_live_events(client)
            print(f"{len(live_events)} partidos en vivo encontrados en las ligas trackeadas.")
            if live_events:
                poll_once(client, conn, live_events, xg_booster=xg_booster)
                if chile_conn is not None:
                    chilean = [e for e in live_events if is_chilean_event(e)]
                    if chilean:
                        now = datetime.now(timezone.utc)
                        shadow_cycle(
                            client, conn, chile_conn, xg_booster, goal_booster, platt,
                            send_alert, chilean, now.timestamp(), str(now.year),
                        )
            sync_to_vps()
        except Exception as exc:
            print(f"ADVERTENCIA: fallo en el ciclo de polling ({exc}), se reintenta en el proximo ciclo.")
        time.sleep(POLL_INTERVAL_SECONDS)
```

- [ ] **Step 6: Run both test files, then the full suite**

Run: `.venv\Scripts\pytest.exe tests/test_sofascore_shadow.py tests/test_sofascore_poller.py -v`
Expected: all pass (Task 2+3 tests, 6 new shadow tests, existing poller tests, 2 new poller tests).

Run: `.venv\Scripts\pytest.exe -q`
Expected: all pass (162 pre-existing + ~20 new across Tasks 1–5).

- [ ] **Step 7: Commit and push**

```powershell
git add src/goles/sofascore/shadow.py src/goles/sofascore/poller.py tests/test_sofascore_shadow.py tests/test_sofascore_poller.py
git commit -m "feat: Chile shadow mode -- live inference with observation-only Telegram alerts"
git push origin master
```

(The poller runs on the home PC, not Dokploy — pushing does not deploy it. Restart happens in Step 8.)

- [ ] **Step 8: Manual live verification (home PC, during a Chilean match)**

1. Set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in the poller's environment (ask the user for the values — they exist but are not in the repo).
2. Restart the poller: `.venv\Scripts\python.exe -m goles.sofascore.poller` during a Liga de Primera or Liga de Ascenso matchday (fixtures: most weekends; check Sofascore).
3. Verify in order: startup prints `Modo sombra Chile: activado`; during a live Chilean match, `shadow_predictions` rows accumulate (~2/min once minute ≥ 20): `sqlite3 data/live_match_state.db "SELECT minute, team, ROUND(probability,3) FROM shadow_predictions ORDER BY prediction_id DESC LIMIT 10"`; if any probability crosses 0.30, a Telegram message arrives starting with the 🧪 banner; EPL polling and `sync_to_vps` keep working (no new warnings).
4. Record the verification result (event id, prediction count, any alert screenshots/notes) in a `## Resultado` section appended to `docs/superpowers/specs/2026-07-12-chile-shadow-mode-design.md`, then:

```powershell
git add docs/superpowers/specs/2026-07-12-chile-shadow-mode-design.md
git commit -m "docs: record shadow-mode live verification result"
git push origin master
```
