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
