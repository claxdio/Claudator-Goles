# Enriquecimiento de Features (situation, coordenadas, lastAction) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the goal-construction signals identified by the sports-science review (shots from inside the box, set-piece vs open-play xG split, line-breaking actions, transition/counter-attack shots) to the training pipeline — using raw-cache fields we already have on disk but currently discard — then retrain LightGBM and honestly compare against the current validated baseline (BSS 0.0193 on 2324 test / 0.0167 on 2223 replication).

**Architecture:** The soccerdata-normalized DataFrame corrupts two of the fields we need (verified empirically: 19,312 shots lose `Head`/`OtherBodyPart` → NA in `body_part`; all 1,198 penalties lose `Penalty` → NA in `situation`) and does not expose `lastAction` at all. So enrichment fields come straight from the raw cached match JSONs (`match_{id}.json`, verified schema: each shot dict has `id`, `situation`, `shotType`, `lastAction`, `X`, `Y`), joined to the DataFrame rows by `shot_id`. The `shots` table gains 5 nullable columns; `shots_to_records`/`persist_shots`/`load_match_shots` carry them through; `compute_ml_features` gains 7 new features (28 total with the 2 dataset-level ones), written defensively with `.get()` so every existing test fixture keeps working unchanged; then both training scripts re-run for the honest before/after comparison.

**Tech Stack:** Unchanged — Python 3.11+, sqlite3, pandas, lightgbm, pytest. No new dependencies.

## Global Constraints

- No paid services, no new network downloads: every enrichment field already exists in the local raw cache (`C:\Users\Claudio\soccerdata\data\Understat\match_*.json`, 4,116 files). The cache directory must be resolved programmatically (`soccerdata._config.DATA_DIR / "Understat"`), never hardcoded, so the pipeline works on the VPS later.
- **Do NOT read `situation`/`body_part` from the soccerdata-normalized DataFrame** — they are corrupted there (Head→NA, Penalty→NA, verified against raw JSON). `location_x`/`location_y` ARE clean in the DataFrame and may be read from it. `situation`/`shot_type`/`last_action` come only from raw JSON via the `shot_id` join.
- Backward compatibility is mandatory: all 55 existing tests must keep passing without modification. New shot-dict keys are optional — every consumer uses `.get(key)` with a sane default, and `shots_to_records`'s `shot_details` parameter defaults to `None`.
- `data/goles.db` is disposable derived state (gitignored, rebuildable from cache in ~1-2 minutes with zero network). The migration strategy is: update the `SCHEMA` in `db.py`, delete the old DB file, re-run `python -m goles.ingest_history`. No ALTER-TABLE migration code (YAGNI — nothing else consumes this DB).
- Feature-count discipline (research: low-thousands-of-matches datasets overfit with bloated feature sets): exactly 7 new features, bringing `compute_ml_features` from 19 → 26 keys and `FEATURE_NAMES` from 21 → 28. No more.
- Live-replicability tags: box/set-piece features have live equivalents (Sofascore/FotMob live shotmaps expose per-shot coordinates, situation, body part). The two `lastAction`-derived features (`own_linebreak_shots`, `own_transition_shots`) are **Understat-only (post-match)** — they are included to measure their value, but must be commented as "experimental: no live equivalent yet" so Phase 2 knows they may need dropping or substituting.
- Box threshold: a shot is "inside the box" when `location_x >= 0.84` (Understat X is normalized 0-1 toward the attacking goal; the penalty area starts at 16.5m from the goal line on a ~105m pitch → 1 − 16.5/105 ≈ 0.843).
- Temporal discipline unchanged from the GBT plan: test seasons never touch training or calibration; the retrain comparison (Task 5) uses the identical splits as the runs it is compared against.

---

### Task 1: Schema + raw-cache detail loader + enriched records

**Files:**
- Modify: `src/goles/db.py` (add 5 columns to the `shots` table in `SCHEMA`)
- Modify: `src/goles/loaders/understat.py` (new `load_shot_details_from_cache`; extend `shots_to_records` and `persist_shots`)
- Test: `tests/test_understat_loader.py` (append)

**Interfaces:**
- Produces: `load_shot_details_from_cache(cache_dir: Path) -> dict[int, dict]` (shot id → `{"situation": str, "shot_type": str, "last_action": str}`), `shots_to_records(shots_df, shot_details: dict[int, dict] | None = None) -> list[dict]` (records gain `location_x`, `location_y`, `situation`, `shot_type`, `last_action` keys — `None` when unavailable), `persist_shots` stores the new columns.
- Consumes: existing `get_or_create_team`; raw cache JSON shape verified empirically: `{"shots": {"h": [...], "a": [...]}}` with each shot carrying string-typed `id`, `situation`, `shotType`, `lastAction`.

- [ ] **Step 1: Update the schema in `src/goles/db.py`**

Replace the `shots` table definition inside `SCHEMA` with:

```sql
CREATE TABLE IF NOT EXISTS shots (
    shot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL REFERENCES matches(match_id),
    minute INTEGER NOT NULL,
    team_id INTEGER NOT NULL REFERENCES teams(team_id),
    xg REAL NOT NULL,
    is_goal INTEGER NOT NULL,
    location_x REAL,
    location_y REAL,
    situation TEXT,
    shot_type TEXT,
    last_action TEXT
);
```

(No other table changes. Columns are nullable so tests that insert bare shots keep working.)

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_understat_loader.py`:

```python
def test_shots_to_records_carries_location_and_details():
    df = _make_understat_like_df(
        [
            {
                "league": "ENG-Premier League", "season": "2324",
                "game": "2023-08-11 Arsenal-Chelsea", "team": "Arsenal", "player": "Player A",
                "game_id": 901, "shot_id": 5001, "minute": 23, "xg": 0.15,
                "result": "Missed Shot", "location_x": 0.91, "location_y": 0.48,
            },
        ]
    )
    details = {5001: {"situation": "OpenPlay", "shot_type": "Head", "last_action": "Throughball"}}
    records = shots_to_records(df, shot_details=details)
    rec = records[0]
    assert rec["location_x"] == 0.91
    assert rec["location_y"] == 0.48
    assert rec["situation"] == "OpenPlay"
    assert rec["shot_type"] == "Head"
    assert rec["last_action"] == "Throughball"


def test_shots_to_records_defaults_details_to_none_when_absent():
    df = _make_understat_like_df(
        [
            {
                "league": "ENG-Premier League", "season": "2324",
                "game": "2023-08-11 Arsenal-Chelsea", "team": "Arsenal", "player": "Player A",
                "game_id": 902, "shot_id": 5002, "minute": 10, "xg": 0.1,
                "result": "Goal", "location_x": 0.88, "location_y": 0.5,
            },
        ]
    )
    records = shots_to_records(df)  # no shot_details at all
    rec = records[0]
    assert rec["situation"] is None
    assert rec["shot_type"] is None
    assert rec["last_action"] is None
    assert rec["location_x"] == 0.88


def test_persist_shots_stores_enrichment_columns():
    conn = get_connection(":memory:")
    init_db(conn)
    records = [
        {
            "match_id": 903, "league": "TEST", "season": "2324", "date": "2023-09-01",
            "home_team": "Team A", "away_team": "Team B",
            "minute": 30, "team": "home", "xg": 0.4, "is_goal": True,
            "location_x": 0.9, "location_y": 0.45,
            "situation": "FromCorner", "shot_type": "Head", "last_action": "Cross",
        },
    ]
    persist_shots(conn, records)
    row = conn.execute(
        "SELECT location_x, location_y, situation, shot_type, last_action FROM shots"
    ).fetchone()
    assert row == (0.9, 0.45, "FromCorner", "Head", "Cross")


def test_load_shot_details_from_cache_reads_raw_match_json(tmp_path):
    import json

    match_file = tmp_path / "match_777.json"
    match_file.write_text(json.dumps({
        "shots": {
            "h": [{"id": "111", "situation": "OpenPlay", "shotType": "LeftFoot",
                   "lastAction": "Pass", "minute": "10"}],
            "a": [{"id": "222", "situation": "Penalty", "shotType": "RightFoot",
                   "lastAction": "Standard", "minute": "55"}],
        }
    }), encoding="utf-8")
    (tmp_path / "league_1_season_2023.json").write_text("{}", encoding="utf-8")  # non-match file, must be ignored

    details = load_shot_details_from_cache(tmp_path)
    assert details[111] == {"situation": "OpenPlay", "shot_type": "LeftFoot", "last_action": "Pass"}
    assert details[222] == {"situation": "Penalty", "shot_type": "RightFoot", "last_action": "Standard"}
    assert len(details) == 2
```

Also add the import at the top of the file: `from goles.loaders.understat import load_shot_details_from_cache` (extend the existing import line).

Note: `_make_understat_like_df` fixtures gain `shot_id`/`location_x`/`location_y` columns — existing tests that don't include them keep working because the implementation reads them with `row_dict.get(...)`.

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv\Scripts\pytest.exe tests/test_understat_loader.py -v`
Expected: FAIL — `ImportError: cannot import name 'load_shot_details_from_cache'`

- [ ] **Step 4: Implement**

In `src/goles/loaders/understat.py`:

(a) Add imports at top: `import json` and `from pathlib import Path`.

(b) Add the cache reader:

```python
def load_shot_details_from_cache(cache_dir: Path) -> dict[int, dict]:
    """Reads the raw cached Understat match JSONs (match_{id}.json) and
    returns a mapping shot_id -> {situation, shot_type, last_action}.

    This exists because the soccerdata-normalized DataFrame corrupts these
    fields (verified empirically: 'Head'/'OtherBodyPart' body parts and
    'Penalty' situations all become NA in the normalized output, and
    lastAction is not exposed at all) -- the raw JSON is the only reliable
    source. location_x/location_y are NOT read here because they survive
    normalization intact and come from the DataFrame instead."""
    details: dict[int, dict] = {}
    for match_file in Path(cache_dir).glob("match_*.json"):
        with open(match_file, encoding="utf-8") as fh:
            data = json.load(fh)
        shots = data.get("shots", {})
        if not isinstance(shots, dict):
            continue
        for shot in shots.get("h", []) + shots.get("a", []):
            details[int(shot["id"])] = {
                "situation": shot.get("situation"),
                "shot_type": shot.get("shotType"),
                "last_action": shot.get("lastAction"),
            }
    return details
```

(c) Change `shots_to_records`'s signature to `def shots_to_records(shots_df: pd.DataFrame, shot_details: dict[int, dict] | None = None) -> list[dict]:` and extend its docstring with one paragraph: records now also carry `location_x`, `location_y` (from the DataFrame, may be None), and `situation`/`shot_type`/`last_action` (from `shot_details` joined by shot_id, all None when `shot_details` is None or the id is missing).

(d) Inside the per-row loop, after computing `xg`/`is_goal`, add:

```python
            raw_shot_id = row_dict.get("shot_id")
            detail = (shot_details or {}).get(int(raw_shot_id)) if raw_shot_id is not None else None
            loc_x = row_dict.get("location_x")
            loc_y = row_dict.get("location_y")
```

and extend the appended record dict with:

```python
                    "location_x": float(loc_x) if loc_x is not None and not pd.isna(loc_x) else None,
                    "location_y": float(loc_y) if loc_y is not None and not pd.isna(loc_y) else None,
                    "situation": detail.get("situation") if detail else None,
                    "shot_type": detail.get("shot_type") if detail else None,
                    "last_action": detail.get("last_action") if detail else None,
```

(e) In `persist_shots`, replace the shots INSERT with:

```python
        conn.execute(
            """INSERT INTO shots
               (match_id, minute, team_id, xg, is_goal,
                location_x, location_y, situation, shot_type, last_action)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                match_id, rec["minute"], team_id, rec["xg"], int(rec["is_goal"]),
                rec.get("location_x"), rec.get("location_y"),
                rec.get("situation"), rec.get("shot_type"), rec.get("last_action"),
            ),
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv\Scripts\pytest.exe tests/test_understat_loader.py -v`
Expected: all pass (11 existing + 4 new = 15)

- [ ] **Step 6: Run the full suite**

Run: `.venv\Scripts\pytest.exe -q`
Expected: 59 passed, no regressions

- [ ] **Step 7: Commit**

```powershell
git add src/goles/db.py src/goles/loaders/understat.py tests/test_understat_loader.py
git commit -m "feat: carry shot coordinates, situation, shot type and last action through ingestion"
```

---

### Task 2: Re-ingest with enrichment (real data, manual verification)

**Files:**
- Modify: `src/goles/ingest_history.py`

**Interfaces:**
- Consumes: Task 1's `load_shot_details_from_cache` + extended `shots_to_records`.
- Produces: a repopulated `data/goles.db` with enrichment columns filled for all 106,538 shots.

No automated tests (same precedent as the original ingest script).

- [ ] **Step 1: Update the script**

In `src/goles/ingest_history.py`, replace `main()`'s body between the fetch and persist calls so it becomes:

```python
def main() -> None:
    conn = get_connection()
    init_db(conn)

    print(f"Descargando datos de Understat para {LEAGUES} temporadas {SEASONS}...")
    print("La primera corrida sin cache puede tardar bastante (~1 partido/seg).")
    shots_df = fetch_understat_shots(LEAGUES, SEASONS)

    from soccerdata._config import DATA_DIR

    cache_dir = DATA_DIR / "Understat"
    print(f"Leyendo detalles de tiro (situation/shotType/lastAction) del cache crudo en {cache_dir}...")
    shot_details = load_shot_details_from_cache(cache_dir)
    print(f"{len(shot_details)} tiros con detalle crudo encontrados.")

    records = shots_to_records(shots_df, shot_details=shot_details)
    enriched = sum(1 for r in records if r["situation"] is not None)
    print(f"{len(records)} eventos de tiro descargados ({enriched} con situation enriquecida). Guardando...")
    persist_shots(conn, records)
    print("Listo.")
```

Add `load_shot_details_from_cache` to the existing `from goles.loaders.understat import ...` line.

- [ ] **Step 2: Rebuild the database for real**

Run (PowerShell, venv activated):
```powershell
Remove-Item data\goles.db
python -m goles.ingest_history
```
Expected: no network (all 4,116 matches cached), completes in a couple of minutes. The "con situation enriquecida" count should be ≈106,538 (nearly every shot except own-goal rows whose ids still join fine — expect >99% coverage; if it prints a dramatically lower number, the shot_id join is broken — stop and investigate rather than proceeding).

Then verify directly:
```powershell
python -c "from goles.db import get_connection; c = get_connection(); print(c.execute('SELECT COUNT(*), SUM(situation IS NOT NULL), SUM(location_x IS NOT NULL), SUM(location_x >= 0.84) FROM shots').fetchone()); print(c.execute('SELECT situation, COUNT(*) FROM shots GROUP BY situation').fetchall())"
```
Expected: total 106,538; situation/location coverage near-total; situation breakdown matching the raw counts (OpenPlay ≈77,577, FromCorner ≈17,302, SetPiece ≈6,873, DirectFreekick ≈3,588, Penalty ≈1,198 — plus NULLs for the 75 own-goal rows, which have no meaningful shot detail for the scoring side).

- [ ] **Step 3: Confirm no regressions, then commit**

Run: `.venv\Scripts\pytest.exe -q` → expected 59 passed.

```powershell
git add src/goles/ingest_history.py
git commit -m "feat: enrich ingestion with raw-cache shot details and rebuild database"
```

---

### Task 3: Carry enrichment through `load_match_shots`

**Files:**
- Modify: `src/goles/backtest.py` (`load_match_shots` selects the new columns)
- Test: `tests/test_backtest.py` (append one test)

**Interfaces:**
- Produces: shot dicts from `load_match_shots` gain optional keys `location_x`, `location_y`, `situation`, `shot_type`, `last_action` (values may be None). All existing consumers unaffected (they only read the original 4 keys).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_backtest.py`:

```python
def test_load_match_shots_carries_enrichment_fields():
    from goles.backtest import load_match_shots

    conn = get_connection(":memory:")
    init_db(conn)
    persist_shots(conn, [
        {
            "match_id": 601, "league": "TEST", "season": "2526", "date": "2025-08-01",
            "home_team": "Team A", "away_team": "Team B",
            "minute": 12, "team": "home", "xg": 0.3, "is_goal": False,
            "location_x": 0.9, "location_y": 0.5,
            "situation": "OpenPlay", "shot_type": "Head", "last_action": "Cross",
        },
    ])
    match_id, home_id, away_id = conn.execute(
        "SELECT match_id, home_team_id, away_team_id FROM matches"
    ).fetchone()
    shots = load_match_shots(conn, match_id, home_id, away_id)
    assert shots[0]["location_x"] == 0.9
    assert shots[0]["situation"] == "OpenPlay"
    assert shots[0]["shot_type"] == "Head"
    assert shots[0]["last_action"] == "Cross"
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv\Scripts\pytest.exe tests/test_backtest.py -v -k enrichment`
Expected: FAIL with `KeyError: 'location_x'`

- [ ] **Step 3: Implement**

Replace `load_match_shots`'s body in `src/goles/backtest.py`:

```python
def load_match_shots(
    conn: sqlite3.Connection, match_id: int, home_team_id: int, away_team_id: int
) -> list[dict]:
    rows = conn.execute(
        """SELECT minute, team_id, xg, is_goal,
                  location_x, location_y, situation, shot_type, last_action
           FROM shots WHERE match_id = ? ORDER BY minute""",
        (match_id,),
    ).fetchall()
    shots = []
    for minute, team_id, xg, is_goal, loc_x, loc_y, situation, shot_type, last_action in rows:
        team = "home" if team_id == home_team_id else "away"
        shots.append(
            {
                "minute": minute, "team": team, "xg": xg, "is_goal": bool(is_goal),
                "location_x": loc_x, "location_y": loc_y,
                "situation": situation, "shot_type": shot_type, "last_action": last_action,
            }
        )
    return shots
```

- [ ] **Step 4: Run the full suite**

Run: `.venv\Scripts\pytest.exe -q`
Expected: 60 passed

- [ ] **Step 5: Commit**

```powershell
git add src/goles/backtest.py tests/test_backtest.py
git commit -m "feat: expose shot enrichment fields from load_match_shots"
```

---

### Task 4: 7 new features in `compute_ml_features` + dataset wiring

**Files:**
- Modify: `src/goles/features.py` (extend `compute_ml_features` — 19 → 26 keys)
- Modify: `src/goles/dataset.py` (`FEATURE_NAMES` 21 → 28)
- Test: `tests/test_features.py` (append)

**Interfaces:**
- Produces: `compute_ml_features` additionally returns: `own_box_xg_total`, `opp_box_xg_total`, `own_box_shots_recent`, `own_setpiece_xg`, `opp_setpiece_xg`, `own_linebreak_shots`, `own_transition_shots`. `FEATURE_NAMES` lists all 26 + `trailing_prior_xg` + `poisson_prob` = 28, order matters for LightGBM.
- Consumes: shot dicts with the optional enrichment keys (missing/None-safe via `.get()`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_features.py`:

```python
ENRICHED_SHOTS = [
    {"minute": 10, "team": "home", "xg": 0.05, "is_goal": False,
     "location_x": 0.70, "situation": "OpenPlay", "last_action": "Pass"},
    {"minute": 20, "team": "home", "xg": 0.30, "is_goal": False,
     "location_x": 0.90, "situation": "OpenPlay", "last_action": "Throughball"},
    {"minute": 30, "team": "home", "xg": 0.10, "is_goal": False,
     "location_x": 0.88, "situation": "FromCorner", "last_action": "Cross"},
    {"minute": 55, "team": "home", "xg": 0.20, "is_goal": True,
     "location_x": 0.95, "situation": "OpenPlay", "last_action": "BallRecovery"},
    {"minute": 40, "team": "away", "xg": 0.15, "is_goal": False,
     "location_x": 0.86, "situation": "SetPiece", "last_action": "Aerial"},
]


def test_box_features_use_x_threshold():
    f = compute_ml_features(ENRICHED_SHOTS, cutoff_minute=60, team="home")
    # home box shots (x >= 0.84): 0.30 @20, 0.10 @30, 0.20 @55 -> xg total 0.60
    assert abs(f["own_box_xg_total"] - 0.60) < 1e-9
    # away box shots: the 0.15 @40 -> 0.15
    assert abs(f["opp_box_xg_total"] - 0.15) < 1e-9
    # recent window (45,60]: only the @55 box shot
    assert f["own_box_shots_recent"] == 1.0


def test_setpiece_xg_split():
    f = compute_ml_features(ENRICHED_SHOTS, cutoff_minute=60, team="home")
    # home set-piece situations (FromCorner/SetPiece/DirectFreekick): the 0.10 corner shot
    assert abs(f["own_setpiece_xg"] - 0.10) < 1e-9
    # away: the 0.15 SetPiece shot
    assert abs(f["opp_setpiece_xg"] - 0.15) < 1e-9


def test_linebreak_and_transition_counts():
    f = compute_ml_features(ENRICHED_SHOTS, cutoff_minute=60, team="home")
    assert f["own_linebreak_shots"] == 1.0  # the Throughball
    assert f["own_transition_shots"] == 1.0  # the BallRecovery


def test_enrichment_features_default_to_zero_without_enriched_data():
    f = compute_ml_features(ML_SAMPLE_SHOTS, cutoff_minute=65, team="home")
    assert f["own_box_xg_total"] == 0.0
    assert f["own_setpiece_xg"] == 0.0
    assert f["own_linebreak_shots"] == 0.0
    assert f["own_transition_shots"] == 0.0
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv\Scripts\pytest.exe tests/test_features.py -v -k "box or setpiece or linebreak or enrichment"`
Expected: FAIL with `KeyError: 'own_box_xg_total'`

- [ ] **Step 3: Implement**

In `src/goles/features.py`, inside `compute_ml_features`, after the `time_since_goal` computation add:

```python
    BOX_X_THRESHOLD = 0.84  # Understat X is 0-1 toward the attacking goal; box edge ~= 1 - 16.5/105
    SETPIECE_SITUATIONS = {"SetPiece", "FromCorner", "DirectFreekick"}
    LINEBREAK_ACTIONS = {"Throughball", "TakeOn"}
    TRANSITION_ACTIONS = {"BallRecovery", "Rebound"}

    def _is_box(s: dict) -> bool:
        x = s.get("location_x")
        return x is not None and x >= BOX_X_THRESHOLD

    own_box_xg_total = sum(s["xg"] for s in own_shots if _is_box(s))
    opp_box_xg_total = sum(s["xg"] for s in opp_shots if _is_box(s))
    own_box_shots_recent = float(
        sum(1 for s in own_shots if _is_box(s) and s["minute"] > recent_window_start)
    )
    own_setpiece_xg = sum(s["xg"] for s in own_shots if s.get("situation") in SETPIECE_SITUATIONS)
    opp_setpiece_xg = sum(s["xg"] for s in opp_shots if s.get("situation") in SETPIECE_SITUATIONS)
    # Experimental: derived from Understat's lastAction, which has NO live
    # equivalent in the Sofascore/FotMob feeds -- if these prove valuable,
    # Phase 2 must find a live proxy or retrain without them.
    own_linebreak_shots = float(sum(1 for s in own_shots if s.get("last_action") in LINEBREAK_ACTIONS))
    own_transition_shots = float(sum(1 for s in own_shots if s.get("last_action") in TRANSITION_ACTIONS))
```

and extend the returned dict with:

```python
        "own_box_xg_total": own_box_xg_total,
        "opp_box_xg_total": opp_box_xg_total,
        "own_box_shots_recent": own_box_shots_recent,
        "own_setpiece_xg": own_setpiece_xg,
        "opp_setpiece_xg": opp_setpiece_xg,
        "own_linebreak_shots": own_linebreak_shots,
        "own_transition_shots": own_transition_shots,
```

In `src/goles/dataset.py`, add the same 7 names to `FEATURE_NAMES`, inserted immediately before `"trailing_prior_xg"` (order: the list must exactly equal `compute_ml_features`'s keys plus the final two dataset-level entries).

- [ ] **Step 4: Run the full suite**

Run: `.venv\Scripts\pytest.exe -q`
Expected: 64 passed (the `test_build_dataset_produces_one_row_per_match_team_cutoff` test asserts `set(FEATURE_NAMES) == set(r.features.keys())`, which validates the wiring automatically).

- [ ] **Step 5: Commit**

```powershell
git add src/goles/features.py src/goles/dataset.py tests/test_features.py
git commit -m "feat: add box, set-piece, line-break and transition features from enriched shot data"
```

---

### Task 5: Retrain and honest before/after comparison (real data, manual verification)

**Files:**
- None new — runs the existing `src/goles/train_gbt.py` and `src/goles/train_gbt_replication.py` unmodified (they pick up the new features automatically through `build_dataset`/`FEATURE_NAMES`).

- [ ] **Step 1: Run both training scripts**

```powershell
python -m goles.train_gbt
python -m goles.train_gbt_replication
```
Expected: no network, a few minutes each (dataset build dominates).

- [ ] **Step 2: Compare honestly against the recorded baseline**

Baseline to beat (recorded in the GBT plan's Próximos pasos): **BSS 0.0193** (test 2324) and **BSS 0.0167** (réplica test 2223). Read the new numbers:
- If both runs improve (or one improves and the other holds), the enrichment is a win — record the new numbers and the new feature-importance ranking (watch specifically where `own_box_xg_total` and `own_setpiece_xg` land — the sports-science thesis predicts they should matter).
- If results are flat (within ~±0.002), the honest conclusion is that Understat's xG already encoded most of this signal (xG is itself built from location/situation/body part) — keep the features (they cost nothing and may help the live phase where xG models differ), but say so plainly.
- If either run gets meaningfully WORSE, the features are adding noise at this data scale — revert Task 4's `FEATURE_NAMES` additions (keep the data plumbing from Tasks 1-3, which is valuable regardless) and record why.
- Also check: if the two `lastAction`-derived experimental features rank near the bottom of importance in both runs, note that dropping them for Phase 2 (where they have no live equivalent anyway) would be cost-free.

- [ ] **Step 3: Record the outcome and commit**

Append the real before/after numbers to this plan's "Resultado" section (add it at the end of this file), then:

```powershell
git add docs/superpowers/plans/2026-07-09-goal-predictor-feature-enrichment.md
git commit -m "docs: record feature-enrichment before/after training results"
```

## Próximos pasos (fuera de alcance de este plan)

Whatever the retrain outcome, the next milestones remain (from the GBT plan's Próximos pasos): model persistence (booster + Platt params) and the Phase 2 live pipeline. If the enrichment wins, Phase 2's live feature computation must map each feature to its Sofascore/FotMob live-shotmap equivalent (coordinates/situation/body part are available live; `lastAction` is not — see the experimental tags in `compute_ml_features`). The Squawka lead (possible free Opta-sourced progression stats) stays parked until the user asks for it.
