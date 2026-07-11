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
