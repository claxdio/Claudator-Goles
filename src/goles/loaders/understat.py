from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
import soccerdata as sd

from goles.db import get_or_create_team


def fetch_understat_shots(leagues: list[str], seasons: list[str]) -> pd.DataFrame:
    """Fetch shot-level events for the given leagues/seasons from Understat
    via the `soccerdata` wrapper. Returns the raw DataFrame produced by
    `soccerdata`'s Understat reader: `league`, `season`, `game`, `team`, and
    `player` are MultiIndex levels (not columns), there is no direct
    home_team/away_team column, and shot quality is exposed as a lowercase
    `xg` column (plus `game_id`, `minute`, `result`, etc.). See
    `shots_to_records`'s docstring for how this shape gets normalized into
    plain records."""
    reader = sd.Understat(leagues=leagues, seasons=seasons)
    return reader.read_shot_events()


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


def shots_to_records(shots_df: pd.DataFrame, shot_details: dict[int, dict] | None = None) -> list[dict]:
    """Normalize an Understat shot-events DataFrame (as returned by
    `fetch_understat_shots`) into plain dict records with keys: match_id,
    league, season, home_team, away_team, date, minute, team ("home"/"away"),
    xg, is_goal.

    Records also carry `location_x`, `location_y` (read straight from the
    DataFrame, may be None if the columns are absent) and `situation`,
    `shot_type`, `last_action` (looked up in `shot_details` by shot_id, all
    None when `shot_details` is None or the id is missing from it).

    The real soccerdata Understat reader returns `league`, `season`, `game`,
    and `team` as MultiIndex levels (not columns), has no direct home_team/
    away_team column, and uses a lowercase `xg` column. This function resets
    the index and derives home/away by matching the two team names that
    appear as index values for a given game_id against the `game` string,
    which has the form "{date} {home_team}-{away_team}" — matched exactly
    against the two known team names (not a naive split on "-") so that
    hyphenated team names (e.g. "Stoke-on-Trent") don't break disambiguation.
    The date portion (before the first space) is captured verbatim as the
    "date" field.

    A match can legitimately have shot rows from only ONE team: a side that
    records zero shots in the whole game never appears as an index value
    (e.g. Bournemouth 0-1 Manchester City on 2019-03-02, Understat game
    9486, where Bournemouth took no shots). In that case the single known
    team name is matched as a prefix ("{team}-…") or suffix ("…-{team}") of
    the game string's teams part to decide its side, and the opponent's
    name is recovered from the remainder of the game string.

    Own goals: Understat records an own goal as a shot row with
    `result == "Own Goal"` attributed to the *shooting* player's own team
    (e.g. a Manchester United defender's own goal is attributed to
    team="Manchester United"), but the goal actually counts for the
    *opposing* side. For these rows, `is_goal` is set True, `xg` is forced
    to 0.0 (an own goal isn't a shot-quality event for the scoring side),
    and `team` is flipped to the side opposite the shooting team's side.
    """
    df = shots_df.reset_index()

    records = []
    for game_id, game_shots in df.groupby("game_id"):
        teams_in_game = game_shots["team"].unique().tolist()
        game_str = game_shots["game"].iloc[0]
        date_part, teams_part = game_str.split(" ", 1)
        if len(teams_in_game) == 2:
            team_a, team_b = teams_in_game
            if teams_part == f"{team_a}-{team_b}":
                home_team, away_team = team_a, team_b
            elif teams_part == f"{team_b}-{team_a}":
                home_team, away_team = team_b, team_a
            else:
                raise ValueError(
                    f"could not match teams {teams_in_game} against game string {game_str!r}"
                )
        elif len(teams_in_game) == 1:
            # One side took zero shots, so only the other team appears in the
            # shot rows. Decide the known team's side by matching it against
            # the start/end of the game string and recover the opponent's
            # name from the remainder.
            (team,) = teams_in_game
            is_home = teams_part.startswith(f"{team}-")
            is_away = teams_part.endswith(f"-{team}")
            if is_home == is_away:
                raise ValueError(
                    f"could not match team {team!r} against game string {game_str!r}"
                )
            if is_home:
                home_team = team
                away_team = teams_part[len(team) + 1 :]
            else:
                home_team = teams_part[: -(len(team) + 1)]
                away_team = team
        else:
            raise ValueError(
                f"expected 1 or 2 teams for game_id={game_id}, got {teams_in_game}"
            )

        league = game_shots["league"].iloc[0]
        season = game_shots["season"].iloc[0]

        for row in game_shots.itertuples(index=False):
            row_dict = row._asdict()
            team_side = "home" if row_dict["team"] == home_team else "away"
            is_own_goal = row_dict["result"] == "Own Goal"
            if is_own_goal:
                # The shot is attributed to the shooter's own team, but the
                # goal counts for the opposing side, so flip it here. xg is
                # forced to 0.0 since an own goal isn't a shot-quality event
                # for the scoring side.
                team_side = "away" if team_side == "home" else "home"
                xg = 0.0
                is_goal = True
            else:
                xg = float(row_dict["xg"])
                is_goal = row_dict["result"] == "Goal"
            raw_shot_id = row_dict.get("shot_id")
            detail = (shot_details or {}).get(int(raw_shot_id)) if raw_shot_id is not None else None
            loc_x = row_dict.get("location_x")
            loc_y = row_dict.get("location_y")
            records.append(
                {
                    "match_id": row_dict["game_id"],
                    "league": league,
                    "season": season,
                    "home_team": home_team,
                    "away_team": away_team,
                    "date": date_part,
                    "minute": int(row_dict["minute"]),
                    "team": team_side,
                    "xg": xg,
                    "is_goal": is_goal,
                    "location_x": float(loc_x) if loc_x is not None and not pd.isna(loc_x) else None,
                    "location_y": float(loc_y) if loc_y is not None and not pd.isna(loc_y) else None,
                    "situation": detail.get("situation") if detail else None,
                    "shot_type": detail.get("shot_type") if detail else None,
                    "last_action": detail.get("last_action") if detail else None,
                }
            )
    return records


def persist_shots(conn: sqlite3.Connection, records: list[dict]) -> None:
    """Insert normalized shot records into the database, creating the
    parent match row (and teams) on first sight of a given `match_id`, and
    incrementing the match's goal tally for every shot with is_goal=True.

    Safe to call repeatedly across separate calls for the same match_id:
    if a match already exists in `matches` when its first record in this
    call is processed, all shots for that match_id are skipped entirely
    (no new `shots` rows, no goal-tally updates) rather than re-persisted.
    This relies on shots for a given match_id always arriving together in
    a single `records` list from the loader (one match is never split
    across multiple `persist_shots` calls), so "match already exists"
    reliably means "already fully persisted" and partial persistence
    isn't a real scenario here."""
    # Maps understat match_id -> internal match_id for matches created
    # earlier in *this* call, or -> _SKIP for matches found to already exist
    # in the database before this call started (meaning all their shots
    # were already fully persisted in a previous call).
    _SKIP = object()
    known_matches: dict[int, object] = {}

    for rec in records:
        match_id_key = rec["match_id"]
        cached = known_matches.get(match_id_key)
        if cached is _SKIP:
            continue

        if cached is not None:
            match_id = cached
        else:
            row = conn.execute(
                "SELECT match_id FROM matches WHERE understat_id = ?", (match_id_key,)
            ).fetchone()
            if row is not None:
                known_matches[match_id_key] = _SKIP
                continue

            home_id = get_or_create_team(conn, rec["home_team"])
            away_id = get_or_create_team(conn, rec["away_team"])
            conn.execute(
                """INSERT INTO matches
                   (understat_id, league, season, date, home_team_id, away_team_id)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    match_id_key,
                    rec["league"],
                    rec["season"],
                    rec.get("date", ""),
                    home_id,
                    away_id,
                ),
            )
            row = conn.execute(
                "SELECT match_id FROM matches WHERE understat_id = ?", (match_id_key,)
            ).fetchone()
            match_id = row[0]
            known_matches[match_id_key] = match_id

        home_id = get_or_create_team(conn, rec["home_team"])
        away_id = get_or_create_team(conn, rec["away_team"])
        team_id = home_id if rec["team"] == "home" else away_id
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
        if rec["is_goal"]:
            goal_col = "home_goals" if rec["team"] == "home" else "away_goals"
            conn.execute(
                f"UPDATE matches SET {goal_col} = {goal_col} + 1 WHERE match_id = ?",
                (match_id,),
            )
    conn.commit()


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
