from __future__ import annotations

import sqlite3


def team_match_xg(conn: sqlite3.Connection, match_id: int, team_id: int) -> float:
    """Total shot xG recorded for `team_id` in a single match."""
    row = conn.execute(
        "SELECT COALESCE(SUM(xg), 0.0) FROM shots WHERE match_id = ? AND team_id = ?",
        (match_id, team_id),
    ).fetchone()
    return row[0]


def team_matches_chronological(
    conn: sqlite3.Connection, team_id: int, league: str, season: str
) -> list[tuple[int, str]]:
    """Returns (match_id, date) for every match `team_id` played in
    (league, season), ordered by date ascending."""
    rows = conn.execute(
        """SELECT match_id, date FROM matches
           WHERE league = ? AND season = ? AND (home_team_id = ? OR away_team_id = ?)
           ORDER BY date ASC""",
        (league, season, team_id, team_id),
    ).fetchall()
    return [(row[0], row[1]) for row in rows]


def trailing_xg_per90(
    conn: sqlite3.Connection,
    team_id: int,
    league: str,
    season: str,
    before_match_id: int,
) -> float:
    """Average xG scored by `team_id` per match, across all of its matches
    in (league, season) that occurred strictly before `before_match_id` by
    date. This is a proper pre-match prior: it never looks at
    `before_match_id` itself or any later match, unlike a same-match xG
    total.

    Returns 0.0 if there is no strictly-earlier match for this team this
    season (e.g. matchday 1) — a neutral prior rather than a crash.

    Raises ValueError if `before_match_id` is not among `team_id`'s matches
    in (league, season).
    """
    matches = team_matches_chronological(conn, team_id, league, season)
    match_dates = {mid: date for mid, date in matches}
    if before_match_id not in match_dates:
        raise ValueError(
            f"match_id={before_match_id} not found for team_id={team_id} "
            f"in league={league!r} season={season!r}"
        )
    before_date = match_dates[before_match_id]
    prior_match_ids = [mid for mid, date in matches if date < before_date]
    if not prior_match_ids:
        return 0.0
    total_xg = sum(team_match_xg(conn, mid, team_id) for mid in prior_match_ids)
    return total_xg / len(prior_match_ids)


def days_since_last_match(
    conn: sqlite3.Connection,
    team_id: int,
    league: str,
    season: str,
    before_match_id: int,
) -> float | None:
    """Days between `team_id`'s previous match in (league, season) and the
    match identified by `before_match_id`. Returns None for a team's first
    match of the season (no prior fixture to measure a gap from) -- callers
    should treat None as "no rest-day signal available", not zero."""
    from datetime import date as _date

    matches = team_matches_chronological(conn, team_id, league, season)
    match_dates = {mid: date for mid, date in matches}
    if before_match_id not in match_dates:
        raise ValueError(
            f"match_id={before_match_id} not found for team_id={team_id} "
            f"in league={league!r} season={season!r}"
        )
    before_date = match_dates[before_match_id]
    prior_dates = sorted(date for mid, date in matches if date < before_date)
    if not prior_dates:
        return None
    last_date = prior_dates[-1]
    d1 = _date.fromisoformat(last_date)
    d2 = _date.fromisoformat(before_date)
    return float((d2 - d1).days)
