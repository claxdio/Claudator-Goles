import pytest

from goles.db import get_connection, get_or_create_team, init_db
from goles.priors import team_matches_chronological, trailing_xg_per90, days_since_last_match


def _insert_match(conn, understat_id, league, season, date, home_id, away_id):
    conn.execute(
        """INSERT INTO matches
           (understat_id, league, season, date, home_team_id, away_team_id)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (understat_id, league, season, date, home_id, away_id),
    )
    return conn.execute(
        "SELECT match_id FROM matches WHERE understat_id = ?", (understat_id,)
    ).fetchone()[0]


def _insert_shot(conn, match_id, team_id, xg):
    conn.execute(
        "INSERT INTO shots (match_id, minute, team_id, xg, is_goal) VALUES (?, 10, ?, ?, 0)",
        (match_id, team_id, xg),
    )


def test_team_matches_chronological_orders_by_date():
    conn = get_connection(":memory:")
    init_db(conn)
    arsenal = get_or_create_team(conn, "Arsenal")
    chelsea = get_or_create_team(conn, "Chelsea")
    fulham = get_or_create_team(conn, "Fulham")
    m2 = _insert_match(conn, 2, "ENG-Premier League", "2324", "2023-09-01", arsenal, fulham)
    m1 = _insert_match(conn, 1, "ENG-Premier League", "2324", "2023-08-11", arsenal, chelsea)
    conn.commit()

    matches = team_matches_chronological(conn, arsenal, "ENG-Premier League", "2324")
    assert matches == [(m1, "2023-08-11"), (m2, "2023-09-01")]


def test_trailing_xg_per90_returns_zero_for_teams_first_match_of_season():
    conn = get_connection(":memory:")
    init_db(conn)
    arsenal = get_or_create_team(conn, "Arsenal")
    chelsea = get_or_create_team(conn, "Chelsea")
    m1 = _insert_match(conn, 1, "ENG-Premier League", "2324", "2023-08-11", arsenal, chelsea)
    conn.commit()

    assert trailing_xg_per90(conn, arsenal, "ENG-Premier League", "2324", m1) == 0.0


def test_trailing_xg_per90_averages_only_strictly_earlier_matches():
    conn = get_connection(":memory:")
    init_db(conn)
    arsenal = get_or_create_team(conn, "Arsenal")
    chelsea = get_or_create_team(conn, "Chelsea")
    fulham = get_or_create_team(conn, "Fulham")
    everton = get_or_create_team(conn, "Everton")

    m1 = _insert_match(conn, 1, "ENG-Premier League", "2324", "2023-08-11", arsenal, chelsea)
    m2 = _insert_match(conn, 2, "ENG-Premier League", "2324", "2023-08-19", arsenal, fulham)
    m3 = _insert_match(conn, 3, "ENG-Premier League", "2324", "2023-08-26", arsenal, everton)

    _insert_shot(conn, m1, arsenal, 1.0)
    _insert_shot(conn, m1, arsenal, 0.5)  # match 1 total xg for arsenal = 1.5
    _insert_shot(conn, m2, arsenal, 2.5)  # match 2 total xg for arsenal = 2.5
    _insert_shot(conn, m3, arsenal, 9.0)  # match 3 is the one being predicted -- must be excluded
    conn.commit()

    # before match 3: average of matches 1 and 2 = (1.5 + 2.5) / 2 = 2.0
    result = trailing_xg_per90(conn, arsenal, "ENG-Premier League", "2324", m3)
    assert abs(result - 2.0) < 1e-9

    # before match 2: only match 1 counts = 1.5
    result2 = trailing_xg_per90(conn, arsenal, "ENG-Premier League", "2324", m2)
    assert abs(result2 - 1.5) < 1e-9


def test_trailing_xg_per90_raises_for_unknown_match():
    conn = get_connection(":memory:")
    init_db(conn)
    arsenal = get_or_create_team(conn, "Arsenal")
    chelsea = get_or_create_team(conn, "Chelsea")
    _insert_match(conn, 1, "ENG-Premier League", "2324", "2023-08-11", arsenal, chelsea)
    conn.commit()

    with pytest.raises(ValueError):
        trailing_xg_per90(conn, arsenal, "ENG-Premier League", "2324", before_match_id=999)


def test_days_since_last_match_computes_the_gap():
    conn = get_connection(":memory:")
    init_db(conn)
    arsenal = get_or_create_team(conn, "Arsenal")
    chelsea = get_or_create_team(conn, "Chelsea")
    fulham = get_or_create_team(conn, "Fulham")

    m1 = _insert_match(conn, 1, "ENG-Premier League", "2324", "2023-08-11", arsenal, chelsea)
    m2 = _insert_match(conn, 2, "ENG-Premier League", "2324", "2023-08-19", arsenal, fulham)
    conn.commit()

    gap = days_since_last_match(conn, arsenal, "ENG-Premier League", "2324", m2)
    assert gap == 8.0


def test_days_since_last_match_returns_none_for_first_match_of_season():
    conn = get_connection(":memory:")
    init_db(conn)
    arsenal = get_or_create_team(conn, "Arsenal")
    chelsea = get_or_create_team(conn, "Chelsea")
    m1 = _insert_match(conn, 1, "ENG-Premier League", "2324", "2023-08-11", arsenal, chelsea)
    conn.commit()

    assert days_since_last_match(conn, arsenal, "ENG-Premier League", "2324", m1) is None
