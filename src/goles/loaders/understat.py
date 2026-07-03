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
