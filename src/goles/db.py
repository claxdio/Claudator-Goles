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
    draw_elo_wp REAL,
    market_home_wp REAL,
    market_draw_wp REAL,
    market_away_wp REAL,
    market_over25_wp REAL
);

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
