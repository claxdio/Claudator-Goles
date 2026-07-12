# Chile Trial: Own xG Model + Backfill + Retrain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train our own shot→xG model on the 106,538 labeled Understat shots we already have, use it to backfill 5 seasons of Chilean Liga de Primera + Liga de Ascenso shot data from Sofascore into a new historical database, then retrain and backtest the goal-prediction model on Chilean data — answering "does the model work for Chile?" before any live infrastructure is built.

**Architecture:** Three new modules (`src/goles/xg_model.py`, `src/goles/sofascore/translate.py`, `src/goles/sofascore/backfill.py`) plus two scripts (`src/goles/train_xg.py`, `src/goles/train_gbt_chile.py`). Chilean shots are stored **already translated into Understat conventions** (coordinates 0–1 toward goal, Understat situation/shot_type vocabulary) in a separate `data/goles_chile.db` that reuses the existing `goles.db` schema verbatim — so `features.py`, `priors.py`, `dataset.py`, and the whole training pipeline work on Chilean data unchanged.

**Tech Stack:** Python 3.11+, LightGBM (already a dependency), `tls_requests` (already a dependency), `sqlite3`, `pytest`. No new dependencies, no paid services.

## Global Constraints

- All Sofascore HTTP calls mocked in tests (duck-typed stub client) — no test needs network access.
- Chilean competition filtering **must use Sofascore uniqueTournament ids** — Liga de Primera = `11653`, Liga de Ascenso = `1240` — never tournament-name matching (Paraguay's second tier is also named "Primera División B", verified collision).
- Copa Chile (utid 1221) is **excluded** — verified: no shotmap data exists for it on Sofascore.
- `data/goles_chile.db` is a new, separate database reusing the existing `goles.db` schema verbatim; `matches.understat_id` stores the Sofascore event id (documented column reuse). It must never touch `data/goles.db`.
- Stored Chilean shots use **Understat conventions**: `location_x`/`location_y` in 0–1 toward the attacking goal, `situation` ∈ {OpenPlay, FromCorner, SetPiece, DirectFreekick, Penalty}, `shot_type` ∈ {RightFoot, LeftFoot, Head, OtherBodyPart}.
- Unknown Sofascore vocabulary values fail loud (raise → caller logs and skips that shot), never silently guessed.
- Penalties: excluded from xG-model training; at prediction time a penalty is assigned fixed xG **0.76** (empirical conversion rate).
- The xG model never uses the shot outcome (`shotType` = goal/save/miss) as an input feature — that's the label side.
- Backfill and any real Sofascore fetch run **from the home PC only** (datacenter IPs are blocked — verified).
- No paid services.
- Existing tests (142) must keep passing unmodified.

---

### Task 1: xG model module + training script

**Files:**
- Create: `src/goles/xg_model.py`
- Create: `src/goles/train_xg.py`
- Test: `tests/test_xg_model.py`

**Interfaces:**
- Produces: `PENALTY_XG = 0.76`, `XG_FEATURE_NAMES: list[str]`, `shot_to_features(shot: dict) -> list[float]` (shot is an Understat-convention dict with `location_x`, `location_y`, `situation`, `shot_type`), `train_xg_model(shots: list[dict]) -> lgb.Booster` (shots additionally carry `is_goal`), `predict_xg(booster, shot: dict) -> float` (returns `PENALTY_XG` for penalties without consulting the booster), `save_xg_model(booster, path)` / `load_xg_model(path)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_xg_model.py`:
```python
import math

from goles.xg_model import (
    PENALTY_XG,
    XG_FEATURE_NAMES,
    predict_xg,
    shot_to_features,
    train_xg_model,
)


def _shot(x, y, situation="OpenPlay", shot_type="RightFoot", is_goal=False):
    return {
        "location_x": x, "location_y": y,
        "situation": situation, "shot_type": shot_type, "is_goal": is_goal,
    }


def test_shot_to_features_length_matches_names():
    features = shot_to_features(_shot(0.9, 0.5))
    assert len(features) == len(XG_FEATURE_NAMES)


def test_distance_is_zero_at_goal_center_and_grows_with_distance():
    near = shot_to_features(_shot(0.99, 0.5))
    far = shot_to_features(_shot(0.5, 0.5))
    dist_idx = XG_FEATURE_NAMES.index("distance_m")
    assert near[dist_idx] < far[dist_idx]
    assert near[dist_idx] < 2.0  # ~1 meter out, dead center


def test_angle_is_larger_from_the_center_than_from_the_byline():
    center = shot_to_features(_shot(0.88, 0.5))
    wide = shot_to_features(_shot(0.88, 0.1))
    angle_idx = XG_FEATURE_NAMES.index("angle_rad")
    assert center[angle_idx] > wide[angle_idx]


def test_train_and_predict_learns_that_close_beats_far():
    import random

    random.seed(7)
    shots = []
    # synthetic but directionally-real data: close shots score more often
    for _ in range(2000):
        close = random.random() < 0.5
        x = random.uniform(0.88, 0.98) if close else random.uniform(0.55, 0.75)
        goal = random.random() < (0.35 if close else 0.03)
        shots.append(_shot(x, random.uniform(0.35, 0.65), is_goal=goal))
    booster = train_xg_model(shots)
    xg_close = predict_xg(booster, _shot(0.93, 0.5))
    xg_far = predict_xg(booster, _shot(0.6, 0.5))
    assert xg_close > xg_far
    assert 0.0 <= xg_far <= 1.0


def test_penalties_get_fixed_xg_and_are_excluded_from_training():
    shots = [_shot(0.9, 0.5, is_goal=True) for _ in range(50)]
    shots += [_shot(0.9, 0.5, is_goal=False) for _ in range(50)]
    shots += [_shot(0.88, 0.5, situation="Penalty", is_goal=True) for _ in range(400)]
    booster = train_xg_model(shots)
    assert predict_xg(booster, _shot(0.88, 0.5, situation="Penalty")) == PENALTY_XG
    # non-penalty prediction should reflect only the 50/50 non-penalty data,
    # not be dragged toward 1.0 by the 400 penalty goals
    assert predict_xg(booster, _shot(0.9, 0.5)) < 0.8
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\pytest.exe tests/test_xg_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'goles.xg_model'`

- [ ] **Step 3: Write the implementation**

`src/goles/xg_model.py`:
```python
from __future__ import annotations

import math
from pathlib import Path

import lightgbm as lgb
import numpy as np

# Empirical penalty conversion rate. Penalties are excluded from training
# (a location model trained on open play should not extrapolate to them)
# and assigned this fixed value at prediction time.
PENALTY_XG = 0.76

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
GOAL_WIDTH_M = 7.32

_SITUATIONS = ["OpenPlay", "FromCorner", "SetPiece", "DirectFreekick"]
_SHOT_TYPES = ["RightFoot", "LeftFoot", "Head", "OtherBodyPart"]

XG_FEATURE_NAMES = (
    ["distance_m", "angle_rad"]
    + [f"situation_{s}" for s in _SITUATIONS]
    + [f"shot_type_{t}" for t in _SHOT_TYPES]
)


def shot_to_features(shot: dict) -> list[float]:
    """Understat-convention shot dict -> xG feature vector. location_x is
    the 0-1 fraction of pitch length toward the attacking goal, location_y
    the 0-1 fraction of pitch width."""
    dx = (1.0 - shot["location_x"]) * PITCH_LENGTH_M
    dy = (shot["location_y"] - 0.5) * PITCH_WIDTH_M
    distance = math.sqrt(dx * dx + dy * dy)

    # Angle subtended by the two goal posts from the shot location.
    half_goal = GOAL_WIDTH_M / 2.0
    denominator = dx * dx + dy * dy - half_goal * half_goal
    if denominator <= 0:
        angle = math.pi / 2  # inside the width of the goal mouth, point blank
    else:
        angle = math.atan2(GOAL_WIDTH_M * dx, denominator)

    features = [distance, angle]
    features += [1.0 if shot.get("situation") == s else 0.0 for s in _SITUATIONS]
    features += [1.0 if shot.get("shot_type") == t else 0.0 for t in _SHOT_TYPES]
    return features


def train_xg_model(shots: list[dict]) -> lgb.Booster:
    """Trains P(goal | shot features) on non-penalty shots. `is_goal` is
    the label; any reference xg on the dicts is never used as an input."""
    usable = [s for s in shots if s.get("situation") != "Penalty"]
    X = np.array([shot_to_features(s) for s in usable], dtype=float)
    y = np.array([int(s["is_goal"]) for s in usable])
    train_set = lgb.Dataset(X, label=y)
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": 31,
        "min_data_in_leaf": 200,
        "learning_rate": 0.05,
        "verbosity": -1,
        "seed": 42,
        "deterministic": True,
    }
    return lgb.train(params, train_set, num_boost_round=300)


def predict_xg(booster: lgb.Booster, shot: dict) -> float:
    if shot.get("situation") == "Penalty":
        return PENALTY_XG
    features = np.array([shot_to_features(shot)], dtype=float)
    return float(booster.predict(features)[0])


def save_xg_model(booster: lgb.Booster, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(path))


def load_xg_model(path: str | Path) -> lgb.Booster:
    return lgb.Booster(model_file=str(path))
```

`src/goles/train_xg.py`:
```python
from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from goles.db import get_connection
from goles.xg_model import predict_xg, save_xg_model, train_xg_model

XG_MODEL_PATH = Path("data") / "model" / "xg_booster.txt"


def main() -> None:
    conn = get_connection()
    rows = conn.execute(
        """SELECT xg, is_goal, location_x, location_y, situation, shot_type
           FROM shots WHERE location_x IS NOT NULL"""
    ).fetchall()
    shots = [
        {
            "understat_xg": r[0], "is_goal": bool(r[1]),
            "location_x": r[2], "location_y": r[3],
            "situation": r[4], "shot_type": r[5],
        }
        for r in rows
    ]
    print(f"{len(shots)} tiros historicos cargados.")

    random.seed(42)
    random.shuffle(shots)
    split = int(len(shots) * 0.8)
    train_shots, valid_shots = shots[:split], shots[split:]

    booster = train_xg_model(train_shots)

    valid_np = [s for s in valid_shots if s["situation"] != "Penalty"]
    ours = np.array([predict_xg(booster, s) for s in valid_np])
    theirs = np.array([s["understat_xg"] for s in valid_np])
    actual = np.array([float(s["is_goal"]) for s in valid_np])

    corr = float(np.corrcoef(ours, theirs)[0, 1])
    mae = float(np.mean(np.abs(ours - theirs)))
    print(f"Validacion ({len(valid_np)} tiros no-penal):")
    print(f"  correlacion con xG de Understat: {corr:.4f}  (esperado >= 0.80)")
    print(f"  MAE vs xG de Understat: {mae:.4f}")
    print(f"  media xG nuestro: {ours.mean():.4f} | media xG Understat: {theirs.mean():.4f} | tasa real de gol: {actual.mean():.4f}")

    save_xg_model(booster, XG_MODEL_PATH)
    print(f"Modelo xG guardado en {XG_MODEL_PATH}.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\pytest.exe tests/test_xg_model.py -v`
Expected: 5 passed

- [ ] **Step 5: Train for real and check quality**

Run: `.venv\Scripts\python.exe -m goles.train_xg`
Expected: correlation with Understat xG **≥ 0.80** and our mean xG within ~0.01 of both Understat's mean and the actual goal rate. If correlation is materially lower, stop and investigate (most likely a feature bug) before Task 2.

- [ ] **Step 6: Run the full suite, then commit**

Run: `.venv\Scripts\pytest.exe -q` → all pass (142 + 5 new).

```powershell
git add src/goles/xg_model.py src/goles/train_xg.py tests/test_xg_model.py
git commit -m "feat: add own xG model trained on Understat historical shots"
```

---

### Task 2: Sofascore→Understat translation layer

**Files:**
- Create: `src/goles/sofascore/translate.py`
- Test: `tests/test_sofascore_translate.py`

**Interfaces:**
- Produces: `SITUATION_MAP: dict[str, str]`, `BODY_PART_MAP: dict[str, str]`, `UnknownVocabularyError(ValueError)`, `translate_shot(sofa_shot: dict) -> dict` returning an Understat-convention dict with keys `minute`, `location_x`, `location_y`, `situation`, `shot_type`, `is_goal`, `is_home`.

- [ ] **Step 1: Write the failing tests**

`tests/test_sofascore_translate.py` (samples below are real observed Sofascore values from the design investigation, not invented):
```python
import pytest

from goles.sofascore.translate import UnknownVocabularyError, translate_shot


def _sofa_shot(**overrides):
    shot = {
        "id": 7684954, "time": 20, "shotType": "goal", "situation": "corner",
        "isHome": True, "playerCoordinates": {"x": 5.0, "y": 44.1},
        "bodyPart": "head",
    }
    shot.update(overrides)
    return shot


def test_translate_maps_coordinates_to_understat_convention():
    out = translate_shot(_sofa_shot())
    # Sofascore x=5 (5% of pitch length from the goal line) -> Understat 0.95
    assert out["location_x"] == pytest.approx(0.95)
    assert out["location_y"] == pytest.approx(0.441)


def test_translate_maps_vocabularies_and_outcome():
    out = translate_shot(_sofa_shot())
    assert out["situation"] == "FromCorner"
    assert out["shot_type"] == "Head"
    assert out["is_goal"] is True
    assert out["minute"] == 20
    assert out["is_home"] is True


def test_translate_open_play_variants_all_map_to_openplay():
    for sofa_situation in ("regular", "assisted", "fast-break"):
        out = translate_shot(_sofa_shot(situation=sofa_situation, shotType="miss"))
        assert out["situation"] == "OpenPlay"
        assert out["is_goal"] is False


def test_translate_set_piece_vocabulary():
    assert translate_shot(_sofa_shot(situation="set-piece"))["situation"] == "SetPiece"
    assert translate_shot(_sofa_shot(situation="free-kick"))["situation"] == "DirectFreekick"
    assert translate_shot(_sofa_shot(situation="penalty"))["situation"] == "Penalty"


def test_translate_fails_loud_on_unknown_situation():
    with pytest.raises(UnknownVocabularyError, match="volea-imaginaria"):
        translate_shot(_sofa_shot(situation="volea-imaginaria"))


def test_translate_tolerates_missing_body_part():
    out = translate_shot(_sofa_shot(bodyPart=None))
    assert out["shot_type"] is None  # xG model one-hots it as all-zero
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\pytest.exe tests/test_sofascore_translate.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`src/goles/sofascore/translate.py`:
```python
from __future__ import annotations

# Observed Sofascore situation vocabulary -> Understat's. Extended
# empirically as the backfill logs unknown values (which fail loud below);
# never guess an entry without a real observed value.
SITUATION_MAP = {
    "regular": "OpenPlay",
    "assisted": "OpenPlay",
    "fast-break": "OpenPlay",
    "corner": "FromCorner",
    "set-piece": "SetPiece",
    "free-kick": "DirectFreekick",
    "penalty": "Penalty",
}

BODY_PART_MAP = {
    "right-foot": "RightFoot",
    "left-foot": "LeftFoot",
    "head": "Head",
    "other": "OtherBodyPart",
}


class UnknownVocabularyError(ValueError):
    """A Sofascore vocabulary value we have never observed -- fail loud so
    the mapping table gets extended deliberately, never guessed."""


def translate_shot(sofa_shot: dict) -> dict:
    """Raw Sofascore shot dict -> Understat-convention dict, so everything
    downstream (xg_model, features.py, the goles.db schema) works on
    Chilean data unchanged. Sofascore x is the % of pitch length measured
    from the opponent's goal line (x=5 is point blank); Understat
    location_x is the 0-1 fraction toward the attacking goal."""
    situation_raw = sofa_shot.get("situation")
    if situation_raw not in SITUATION_MAP:
        raise UnknownVocabularyError(f"situacion Sofascore desconocida: {situation_raw!r}")

    body_raw = sofa_shot.get("bodyPart")
    if body_raw is not None and body_raw not in BODY_PART_MAP:
        raise UnknownVocabularyError(f"bodyPart Sofascore desconocido: {body_raw!r}")

    coordinates = sofa_shot.get("playerCoordinates") or {}
    x = coordinates.get("x")
    y = coordinates.get("y")
    if x is None or y is None:
        raise UnknownVocabularyError("tiro sin coordenadas -- no se puede calcular xG")

    return {
        "minute": sofa_shot["time"],
        "location_x": 1.0 - (x / 100.0),
        "location_y": y / 100.0,
        "situation": SITUATION_MAP[situation_raw],
        "shot_type": BODY_PART_MAP[body_raw] if body_raw is not None else None,
        "is_goal": sofa_shot.get("shotType") == "goal",
        "is_home": bool(sofa_shot.get("isHome")),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\pytest.exe tests/test_sofascore_translate.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```powershell
git add src/goles/sofascore/translate.py tests/test_sofascore_translate.py
git commit -m "feat: add Sofascore-to-Understat shot translation layer"
```

---

### Task 3: Validate translation + xG model against Sofascore's real xG (manual verification)

**Files:**
- Create: `src/goles/validate_xg_vs_sofascore.py`

No automated tests — this is the real-network empirical check of the coordinate mapping (same manual-verification precedent as every ingest script).

- [ ] **Step 1: Write the script**

`src/goles/validate_xg_vs_sofascore.py`:
```python
from __future__ import annotations

import numpy as np
import tls_requests

from goles.sofascore.client import get_shotmap
from goles.sofascore.translate import UnknownVocabularyError, translate_shot
from goles.train_xg import XG_MODEL_PATH
from goles.xg_model import load_xg_model, predict_xg

# A finished top-tier match where Sofascore publishes real per-shot xG
# (FIFA World Cup knockout match observed during design). Any finished
# top-tier event id works -- pass a different one as argv[1] if needed.
DEFAULT_EVENT_ID = 12813015


def main(event_id: int = DEFAULT_EVENT_ID) -> None:
    booster = load_xg_model(XG_MODEL_PATH)
    client = tls_requests.Client()
    shots = get_shotmap(client, event_id)
    print(f"{len(shots)} tiros en el evento {event_id}.")

    ours, theirs = [], []
    skipped = 0
    for shot in shots:
        if shot.get("xg") is None or shot.get("situation") == "penalty":
            continue
        try:
            translated = translate_shot(shot)
        except UnknownVocabularyError as exc:
            print(f"  omitido: {exc}")
            skipped += 1
            continue
        ours.append(predict_xg(booster, translated))
        theirs.append(shot["xg"])

    if len(ours) < 5:
        print("Muy pocos tiros comparables -- probar con otro event id de liga top.")
        return
    ours_np, theirs_np = np.array(ours), np.array(theirs)
    corr = float(np.corrcoef(ours_np, theirs_np)[0, 1])
    print(f"Comparables: {len(ours)} (omitidos: {skipped})")
    print(f"correlacion nuestro-xG vs Sofascore-xG: {corr:.4f}  (esperado >= 0.75)")
    print(f"MAE: {float(np.mean(np.abs(ours_np - theirs_np))):.4f}")
    print(f"medias: nuestro {ours_np.mean():.4f} vs Sofascore {theirs_np.mean():.4f}")


if __name__ == "__main__":
    import sys

    main(int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_EVENT_ID)
```

- [ ] **Step 2: Run it for real (home PC)**

Run: `.venv\Scripts\python.exe -m goles.validate_xg_vs_sofascore`

If the default event has too few shots, pick any recently finished Premier-League-tier match id from Sofascore. **Acceptance: Pearson correlation ≥ 0.75** (the two xG models legitimately differ, so perfect agreement is impossible; strong correlation validates the coordinate mapping). If correlation is < 0.5, the most likely culprit is the coordinate convention in `translate.py` — fix before Task 4 (this is the exact failure this task exists to catch). Also note any `UnknownVocabularyError` values printed — extend `SITUATION_MAP` with real observed values if they appear.

- [ ] **Step 3: Commit**

```powershell
git add src/goles/validate_xg_vs_sofascore.py
git commit -m "feat: add xG-vs-Sofascore empirical validation script"
```

---

### Task 4: Chilean historical backfill

**Files:**
- Create: `src/goles/sofascore/backfill.py`
- Test: `tests/test_sofascore_backfill.py`

**Interfaces:**
- Consumes: `goles.sofascore.client.{get_shotmap, get_incidents}`, `goles.sofascore.translate.{translate_shot, UnknownVocabularyError}`, `goles.xg_model.{load_xg_model, predict_xg}`, `goles.loaders.understat.persist_shots`, `goles.db.{get_connection, init_db}`, existing `cards` table schema.
- Produces: `CHILE_DB_PATH = Path("data") / "goles_chile.db"`, `TRACKED_UTIDS: dict[int, str]` (`{11653: "CHI-Liga de Primera", 1240: "CHI-Liga de Ascenso"}`), `fetch_season_event_ids(client, utid: int, season_id: int) -> list[dict]` (paginates, returns finished events only), `backfill_event(client, conn, booster, event: dict, league: str, season_label: str) -> str` (returns `"ok"`, `"skipped_existing"`, or `"no_shotmap"`), `main()`.

- [ ] **Step 1: Write the failing tests**

`tests/test_sofascore_backfill.py`:
```python
from unittest.mock import Mock, patch

from goles.db import get_connection, init_db
from goles.sofascore.backfill import backfill_event, fetch_season_event_ids


def _finished_event(event_id=101, home="Colo-Colo", away="Cobresal"):
    return {
        "id": event_id,
        "homeTeam": {"name": home},
        "awayTeam": {"name": away},
        "startTimestamp": 1751328000,  # 2025-07-01 UTC
        "status": {"type": "finished"},
    }


SOFA_SHOTS = [
    {"id": 1, "time": 12, "shotType": "goal", "situation": "assisted", "isHome": True,
     "playerCoordinates": {"x": 8.0, "y": 50.0}, "bodyPart": "right-foot"},
    {"id": 2, "time": 70, "shotType": "miss", "situation": "corner", "isHome": False,
     "playerCoordinates": {"x": 11.0, "y": 44.0}, "bodyPart": "head"},
]
SOFA_INCIDENTS = [
    {"time": 55, "incidentType": "card", "incidentClass": "red", "isHome": False},
    {"time": 60, "incidentType": "card", "incidentClass": "yellow", "isHome": True},
]


class _FakeBooster:
    def predict(self, X):
        return [0.123] * len(X)


def test_fetch_season_event_ids_paginates_and_filters_finished():
    pages = [
        {"events": [_finished_event(1), {"id": 2, "status": {"type": "notstarted"}}], "hasNextPage": True},
        {"events": [_finished_event(3)], "hasNextPage": False},
    ]
    responses = []
    for p in pages:
        r = Mock()
        r.status_code = 200
        r.json.return_value = p
        responses.append(r)
    client = Mock()
    client.get = Mock(side_effect=responses)
    events = fetch_season_event_ids(client, 11653, 88493)
    assert [e["id"] for e in events] == [1, 3]
    assert client.get.call_count == 2


def test_backfill_event_persists_shots_with_our_xg_and_red_cards():
    conn = get_connection(":memory:")
    init_db(conn)
    with patch("goles.sofascore.backfill.get_shotmap", return_value=SOFA_SHOTS):
        with patch("goles.sofascore.backfill.get_incidents", return_value=SOFA_INCIDENTS):
            result = backfill_event(
                Mock(), conn, _FakeBooster(), _finished_event(), "CHI-Liga de Primera", "2025"
            )
    assert result == "ok"
    shots = conn.execute(
        "SELECT minute, xg, is_goal, location_x, situation, shot_type FROM shots ORDER BY minute"
    ).fetchall()
    assert len(shots) == 2
    assert shots[0][0] == 12 and shots[0][2] == 1
    assert abs(shots[0][1] - 0.123) < 1e-9  # our computed xG, not Sofascore's null
    assert abs(shots[0][3] - 0.92) < 1e-9  # 1 - 8/100
    assert shots[0][4] == "OpenPlay" and shots[0][5] == "RightFoot"
    cards = conn.execute("SELECT minute FROM cards").fetchall()
    assert cards == [(55,)]  # red only, yellow excluded


def test_backfill_event_skips_already_persisted_matches():
    conn = get_connection(":memory:")
    init_db(conn)
    with patch("goles.sofascore.backfill.get_shotmap", return_value=SOFA_SHOTS):
        with patch("goles.sofascore.backfill.get_incidents", return_value=SOFA_INCIDENTS):
            first = backfill_event(Mock(), conn, _FakeBooster(), _finished_event(), "CHI-Liga de Primera", "2025")
            second = backfill_event(Mock(), conn, _FakeBooster(), _finished_event(), "CHI-Liga de Primera", "2025")
    assert first == "ok"
    assert second == "skipped_existing"
    assert conn.execute("SELECT COUNT(*) FROM shots").fetchone()[0] == 2


def test_backfill_event_reports_missing_shotmap_without_raising():
    conn = get_connection(":memory:")
    init_db(conn)

    def raise_404(client, event_id):
        raise RuntimeError("404 Client Error")

    with patch("goles.sofascore.backfill.get_shotmap", side_effect=raise_404):
        with patch("goles.sofascore.backfill.get_incidents", return_value=[]):
            result = backfill_event(Mock(), conn, _FakeBooster(), _finished_event(), "CHI-Liga de Primera", "2025")
    assert result == "no_shotmap"
    assert conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\pytest.exe tests/test_sofascore_backfill.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`src/goles/sofascore/backfill.py`:
```python
from __future__ import annotations

import sqlite3
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from goles.db import get_connection, init_db
from goles.loaders.understat import persist_shots
from goles.sofascore.client import BASE_URL, get_incidents, get_shotmap
from goles.sofascore.translate import UnknownVocabularyError, translate_shot
from goles.xg_model import load_xg_model, predict_xg

CHILE_DB_PATH = Path("data") / "goles_chile.db"
TRACKED_UTIDS = {11653: "CHI-Liga de Primera", 1240: "CHI-Liga de Ascenso"}
# Chilean seasons are calendar years; Sofascore season ids fetched live in main().
BACKFILL_YEARS = ["2022", "2023", "2024", "2025", "2026"]
REQUEST_DELAY_SECONDS = 0.7
RED_CARD_INCIDENT_CLASSES = {"red", "yellowRed"}

# Census of every card incidentClass observed during the backfill -- this
# is how the poller's assumed red-card vocabulary finally gets verified
# against real data (printed at the end of main()).
observed_card_classes: Counter = Counter()


def fetch_season_event_ids(client, utid: int, season_id: int) -> list[dict]:
    """Paginates /events/last/{page} and returns finished events only."""
    events: list[dict] = []
    page = 0
    while True:
        response = client.get(f"{BASE_URL}/unique-tournament/{utid}/season/{season_id}/events/last/{page}")
        if response.status_code != 200:
            break
        payload = response.json()
        events.extend(e for e in payload.get("events", []) if e.get("status", {}).get("type") == "finished")
        if not payload.get("hasNextPage"):
            break
        page += 1
    return events


def backfill_event(client, conn: sqlite3.Connection, booster, event: dict, league: str, season_label: str) -> str:
    event_id = event["id"]
    existing = conn.execute("SELECT 1 FROM matches WHERE understat_id = ?", (event_id,)).fetchone()
    if existing:
        return "skipped_existing"

    try:
        sofa_shots = get_shotmap(client, event_id)
    except Exception:
        return "no_shotmap"

    date_iso = datetime.fromtimestamp(event["startTimestamp"], tz=timezone.utc).strftime("%Y-%m-%d")
    home = event["homeTeam"]["name"]
    away = event["awayTeam"]["name"]

    records = []
    for sofa_shot in sofa_shots:
        try:
            t = translate_shot(sofa_shot)
        except UnknownVocabularyError as exc:
            print(f"  ADVERTENCIA: {exc} (evento {event_id}), tiro omitido.")
            continue
        records.append(
            {
                "match_id": event_id, "league": league, "season": season_label,
                "date": date_iso, "home_team": home, "away_team": away,
                "minute": t["minute"], "team": "home" if t["is_home"] else "away",
                "xg": predict_xg(booster, t), "is_goal": t["is_goal"],
                "location_x": t["location_x"], "location_y": t["location_y"],
                "situation": t["situation"], "shot_type": t["shot_type"],
            }
        )
    if not records:
        return "no_shotmap"
    persist_shots(conn, records)

    match_row = conn.execute(
        "SELECT match_id, home_team_id, away_team_id FROM matches WHERE understat_id = ?", (event_id,)
    ).fetchone()
    if match_row is not None:
        match_pk, home_id, away_id = match_row
        try:
            incidents = get_incidents(client, event_id)
        except Exception:
            incidents = []
        for incident in incidents:
            if incident.get("incidentType") != "card":
                continue
            incident_class = incident.get("incidentClass")
            observed_card_classes[incident_class] += 1
            if incident_class not in RED_CARD_INCIDENT_CLASSES:
                continue
            team_id = home_id if incident.get("isHome") else away_id
            conn.execute(
                "INSERT INTO cards (match_id, team_id, minute) VALUES (?, ?, ?)",
                (match_pk, team_id, incident["time"]),
            )
        conn.commit()
    return "ok"


def main() -> None:
    import tls_requests

    from goles.train_xg import XG_MODEL_PATH

    booster = load_xg_model(XG_MODEL_PATH)
    client = tls_requests.Client()
    conn = get_connection(CHILE_DB_PATH)
    init_db(conn)

    for utid, league in TRACKED_UTIDS.items():
        response = client.get(f"{BASE_URL}/unique-tournament/{utid}/seasons")
        seasons = {s["year"]: s["id"] for s in response.json().get("seasons", [])}
        for year in BACKFILL_YEARS:
            if year not in seasons:
                print(f"{league} {year}: temporada no disponible en Sofascore, se omite.")
                continue
            events = fetch_season_event_ids(client, utid, seasons[year])
            print(f"{league} {year}: {len(events)} partidos terminados.")
            tally = Counter()
            for event in events:
                tally[backfill_event(client, conn, booster, event, league, year)] += 1
                time.sleep(REQUEST_DELAY_SECONDS)
            print(f"  -> {dict(tally)}")

    print("\nCenso de incidentClass de tarjetas observadas (verificacion empirica del vocabulario):")
    for cls, count in observed_card_classes.most_common():
        print(f"  {cls}: {count}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\pytest.exe tests/test_sofascore_backfill.py -v`
Expected: 4 passed

- [ ] **Step 5: Run the full suite, then commit the code**

Run: `.venv\Scripts\pytest.exe -q` → all pass.

```powershell
git add src/goles/sofascore/backfill.py tests/test_sofascore_backfill.py
git commit -m "feat: add Chilean historical backfill from Sofascore with own xG"
```

- [ ] **Step 6: Run the real backfill (home PC, ~2 hours)**

Run: `.venv\Scripts\python.exe -m goles.sofascore.backfill`

Expected: ~2,000–3,000 matches across both divisions and 5 seasons (some older Ascenso seasons may have thinner shotmap coverage — `no_shotmap` counts per season tell us exactly how much usable data each season has; a season that's mostly `no_shotmap` simply contributes less training data, which the tally makes visible rather than silent). **Read the incidentClass census carefully** — it empirically settles the red-card vocabulary; if it shows values other than `red`/`yellowRed`/`yellow` (e.g. a different string for second yellow), update `RED_CARD_INCIDENT_CLASSES` here AND in `src/goles/sofascore/poller.py`, and note it in the spec.

- [ ] **Step 7: Sanity-check the Chilean database**

Run a quick query: total matches per league/season, total shots, mean shots/match (expect ~20–30), mean our-xG per match (expect ~1.0–1.6 per team per match, consistent with typical goal rates), red cards count (expect roughly 1 per 8–12 matches in Chilean football). If mean xG per match is wildly off (e.g. > 3 per team), suspect the coordinate mapping and stop.

---

### Task 5: Retrain the goal model on Chilean data + backtest (decision gate)

**Files:**
- Create: `src/goles/train_gbt_chile.py`

No automated tests (training script — same precedent as `train_gbt.py`/`train_gbt_replication.py`).

- [ ] **Step 1: Write the script**

`src/goles/train_gbt_chile.py` — copy `src/goles/train_gbt.py` verbatim, then apply exactly these changes:
1. `from goles.sofascore.backfill import CHILE_DB_PATH` and `conn = get_connection(CHILE_DB_PATH)` instead of the default DB.
2. `TEST_SEASON = "2026"`, `VALIDATION_SEASON = "2025"` (train = 2022–2024).
3. `MODEL_DIR = Path("data") / "model_chile"` (never overwrite the EPL/Bundesliga artifacts).
4. Update the header comment: Chilean data has no market odds (all `market_*` features are 0.0 — the model will simply never split on them) and no lastAction-derived features; the Poisson-baseline comparison uses the same blend as the main script.

- [ ] **Step 2: Run it for real**

Run: `.venv\Scripts\python.exe -m goles.train_gbt_chile`

- [ ] **Step 3: Record the result and decide (the gate)**

Append a `## Resultado` section to `docs/superpowers/specs/2026-07-12-chile-own-xg-and-backfill-design.md` with: dataset sizes, BSS, calibration table, top feature importances, and the xG-model validation numbers from Tasks 1/3. **Decision rule:** BSS meaningfully above 0 (≳ 0.01) with sane calibration → proceed to Phase D (live + Telegram, new spec). BSS ≈ 0 or negative → the Chile trial stops here and we wait for August's EPL/Bundesliga restart (where the full feature set including market odds exists). Honest prior: expect below the EPL's 0.0335 given the missing market features.

```powershell
git add src/goles/train_gbt_chile.py docs/superpowers/specs/2026-07-12-chile-own-xg-and-backfill-design.md
git commit -m "feat: retrain goal model on Chilean data and record backtest result"
```

---

### Task 6 (optional, parallel): start collecting live Chilean odds on the VPS

**Files:**
- Modify: `src/goles/betfair/poller.py` (the `TRACKED_COMPETITIONS` constant)
- Test: `tests/test_betfair_poller.py` (only if the constant's semantics change — a pure list addition needs no new test)

- [ ] **Step 1: Add the Chilean competition**

In `src/goles/betfair/poller.py` change:
```python
TRACKED_COMPETITIONS = ["Premier League", "Bundesliga", "Chilean Primera Division"]
```
(`find_competition_id` matches by case-insensitive substring against Betfair's competition list; "Chilean Primera Division" is Betfair's naming — if the poller's next discovery cycle logs it as not found, check the Dokploy logs for the exact live name and adjust.)

- [ ] **Step 2: Run the suite, commit, push (Dokploy auto-deploys on push)**

Run: `.venv\Scripts\pytest.exe -q` → all pass.
```powershell
git add src/goles/betfair/poller.py
git commit -m "feat: track Chilean Primera Division odds on the Betfair poller"
git push origin master
```

- [ ] **Step 3: Verify in Dokploy logs**

After the auto-deploy, the Logs tab should show a higher "mercados encontrados" count when Chilean matches have open markets. These odds accumulate in `live_odds.db` for a future market-aware Chilean retrain (weeks of data needed — explicitly NOT part of this plan's decision gate).
