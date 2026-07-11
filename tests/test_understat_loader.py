from unittest.mock import MagicMock, patch

import numpy
import pandas as pd
import pytest

from goles.db import get_connection, get_or_create_team, init_db
from goles.loaders.understat import (
    fetch_understat_shots,
    load_red_cards_from_cache,
    load_shot_details_from_cache,
    persist_red_cards,
    persist_shots,
    shots_to_records,
)


def test_fetch_understat_shots_delegates_to_soccerdata_reader():
    fake_df = pd.DataFrame([{"game_id": 1}])
    mock_reader = MagicMock()
    mock_reader.read_shot_events.return_value = fake_df
    with patch("goles.loaders.understat.sd.Understat", return_value=mock_reader) as mock_cls:
        result = fetch_understat_shots(["ENG-Premier League"], ["2023-24"])
    mock_cls.assert_called_once_with(leagues=["ENG-Premier League"], seasons=["2023-24"])
    assert result is fake_df


def _make_understat_like_df(rows):
    df = pd.DataFrame(rows)
    return df.set_index(["league", "season", "game", "team", "player"])


def test_shots_to_records_normalizes_team_side_and_goal_flag():
    df = _make_understat_like_df(
        [
            {
                "league": "ENG-Premier League", "season": "2324",
                "game": "2023-08-11 Arsenal-Chelsea", "team": "Arsenal", "player": "Player A",
                "game_id": 101, "minute": 23, "xg": 0.15, "result": "Missed Shot",
            },
            {
                "league": "ENG-Premier League", "season": "2324",
                "game": "2023-08-11 Arsenal-Chelsea", "team": "Chelsea", "player": "Player B",
                "game_id": 101, "minute": 41, "xg": 0.42, "result": "Goal",
            },
        ]
    )
    records = shots_to_records(df)
    assert records[0]["team"] == "home"
    assert records[0]["home_team"] == "Arsenal"
    assert records[0]["away_team"] == "Chelsea"
    assert records[0]["is_goal"] is False
    assert records[1]["team"] == "away"
    assert records[1]["is_goal"] is True
    assert records[1]["xg"] == 0.42


def test_shots_to_records_disambiguates_hyphenated_team_names():
    df = _make_understat_like_df(
        [
            {
                "league": "ENG-Premier League", "season": "2324",
                "game": "2023-08-11 Stoke-on-Trent-Newcastle United",
                "team": "Stoke-on-Trent", "player": "Player A",
                "game_id": 202, "minute": 10, "xg": 0.2, "result": "Goal",
            },
            {
                "league": "ENG-Premier League", "season": "2324",
                "game": "2023-08-11 Stoke-on-Trent-Newcastle United",
                "team": "Newcastle United", "player": "Player B",
                "game_id": 202, "minute": 55, "xg": 0.3, "result": "Missed Shot",
            },
        ]
    )
    records = shots_to_records(df)
    home_record = next(r for r in records if r["minute"] == 10)
    assert home_record["team"] == "home"
    assert home_record["home_team"] == "Stoke-on-Trent"
    assert home_record["away_team"] == "Newcastle United"


def test_shots_to_records_flips_own_goal_to_the_conceding_team():
    df = _make_understat_like_df(
        [
            {
                "league": "ENG-Premier League", "season": "2324",
                "game": "2023-08-19 Tottenham-Manchester United",
                "team": "Manchester United", "player": "Lisandro Martinez",
                "game_id": 303, "minute": 82, "xg": 0.0, "result": "Own Goal",
            },
            {
                "league": "ENG-Premier League", "season": "2324",
                "game": "2023-08-19 Tottenham-Manchester United",
                "team": "Tottenham", "player": "Player X",
                "game_id": 303, "minute": 10, "xg": 0.3, "result": "Missed Shot",
            },
        ]
    )
    records = shots_to_records(df)
    own_goal_record = next(r for r in records if r["minute"] == 82)
    # Shot was attributed to Manchester United (the away team), but the
    # own goal counts for the home team (Tottenham), so team flips to home.
    assert own_goal_record["team"] == "home"
    assert own_goal_record["is_goal"] is True
    assert own_goal_record["xg"] == 0.0
    assert own_goal_record["date"] == "2023-08-19"


def test_persist_shots_creates_match_and_shots_and_updates_score():
    conn = get_connection(":memory:")
    init_db(conn)
    records = [
        {
            "match_id": 101, "league": "ENG-Premier League", "season": "2023-24",
            "home_team": "Arsenal", "away_team": "Chelsea",
            "minute": 23, "team": "home", "xg": 0.15, "is_goal": False,
        },
        {
            "match_id": 101, "league": "ENG-Premier League", "season": "2023-24",
            "home_team": "Arsenal", "away_team": "Chelsea",
            "minute": 41, "team": "away", "xg": 0.42, "is_goal": True,
        },
    ]
    persist_shots(conn, records)
    row = conn.execute(
        "SELECT home_goals, away_goals FROM matches WHERE understat_id = 101"
    ).fetchone()
    assert row == (0, 1)
    shot_count = conn.execute("SELECT COUNT(*) FROM shots").fetchone()[0]
    assert shot_count == 2


def test_persist_shots_is_idempotent_for_the_same_match_id():
    conn = get_connection(":memory:")
    init_db(conn)
    record = {
        "match_id": 202, "league": "ENG-Premier League", "season": "2023-24",
        "home_team": "Arsenal", "away_team": "Chelsea",
        "minute": 10, "team": "home", "xg": 0.2, "is_goal": False,
    }
    persist_shots(conn, [record])
    persist_shots(conn, [record])
    match_count = conn.execute(
        "SELECT COUNT(*) FROM matches WHERE understat_id = 202"
    ).fetchone()[0]
    assert match_count == 1


def test_persist_shots_second_call_is_a_true_no_op_for_an_already_persisted_match():
    """Simulates re-running the CLI loader on a match that was already fully
    persisted: shots and match/goal state from the first call must be left
    completely untouched, even if the second call is (incorrectly) given a
    different set of shot records for the same match_id."""
    conn = get_connection(":memory:")
    init_db(conn)

    match_a_first_call = [
        {
            "match_id": 303, "league": "ENG-Premier League", "season": "2023-24",
            "home_team": "Arsenal", "away_team": "Chelsea",
            "minute": 5, "team": "home", "xg": 0.1, "is_goal": False,
        },
        {
            "match_id": 303, "league": "ENG-Premier League", "season": "2023-24",
            "home_team": "Arsenal", "away_team": "Chelsea",
            "minute": 30, "team": "home", "xg": 0.5, "is_goal": True,
        },
    ]
    persist_shots(conn, match_a_first_call)

    shots_after_first = conn.execute("SELECT COUNT(*) FROM shots").fetchone()[0]
    goals_after_first = conn.execute(
        "SELECT home_goals, away_goals FROM matches WHERE understat_id = 303"
    ).fetchone()
    assert shots_after_first == 2
    assert goals_after_first == (1, 0)

    # Simulate a naive re-run: same match_id, but different shot records
    # (as would happen if a fresh fetch from Understat were re-persisted).
    match_a_second_call_different_shots = [
        {
            "match_id": 303, "league": "ENG-Premier League", "season": "2023-24",
            "home_team": "Arsenal", "away_team": "Chelsea",
            "minute": 60, "team": "away", "xg": 0.3, "is_goal": True,
        },
        {
            "match_id": 303, "league": "ENG-Premier League", "season": "2023-24",
            "home_team": "Arsenal", "away_team": "Chelsea",
            "minute": 75, "team": "away", "xg": 0.2, "is_goal": True,
        },
    ]
    persist_shots(conn, match_a_second_call_different_shots)

    shots_after_second = conn.execute("SELECT COUNT(*) FROM shots").fetchone()[0]
    goals_after_second = conn.execute(
        "SELECT home_goals, away_goals FROM matches WHERE understat_id = 303"
    ).fetchone()
    assert shots_after_second == 2
    assert goals_after_second == (1, 0)


def test_shots_to_records_handles_home_team_with_zero_shots_from_away_team():
    df = _make_understat_like_df(
        [
            {
                "league": "ENG-Premier League", "season": "1819",
                "game": "2019-03-02 Bournemouth-Manchester City",
                "team": "Bournemouth", "player": "Player A",
                "game_id": 9486, "minute": 12, "xg": 0.05, "result": "Missed Shot",
            },
        ]
    )
    records = shots_to_records(df)
    assert len(records) == 1
    assert records[0]["team"] == "home"
    assert records[0]["home_team"] == "Bournemouth"
    assert records[0]["away_team"] == "Manchester City"


def test_shots_to_records_handles_away_team_with_zero_shots_from_home_team():
    df = _make_understat_like_df(
        [
            {
                "league": "ENG-Premier League", "season": "1819",
                "game": "2019-03-02 Bournemouth-Manchester City",
                "team": "Manchester City", "player": "Player B",
                "game_id": 9487, "minute": 67, "xg": 0.8, "result": "Goal",
            },
        ]
    )
    records = shots_to_records(df)
    assert len(records) == 1
    assert records[0]["team"] == "away"
    assert records[0]["home_team"] == "Bournemouth"
    assert records[0]["away_team"] == "Manchester City"


def test_shots_to_records_raises_when_lone_team_matches_both_prefix_and_suffix():
    df = _make_understat_like_df(
        [
            {
                "league": "ENG-Premier League", "season": "1819",
                "game": "2019-03-02 City-Manchester-City",
                "team": "City", "player": "Player A",
                "game_id": 9488, "minute": 20, "xg": 0.1, "result": "Missed Shot",
            },
        ]
    )
    with pytest.raises(ValueError):
        shots_to_records(df)


def test_shots_to_records_flips_own_goal_when_only_the_scoring_team_has_shots():
    df = _make_understat_like_df(
        [
            {
                "league": "ENG-Premier League", "season": "2324",
                "game": "2023-08-19 Tottenham-Manchester United",
                "team": "Manchester United", "player": "Lisandro Martinez",
                "game_id": 9489, "minute": 82, "xg": 0.0, "result": "Own Goal",
            },
        ]
    )
    records = shots_to_records(df)
    assert len(records) == 1
    own_goal_record = records[0]
    # Manchester United (away) is the only team with shot rows, but its own
    # goal counts for Tottenham (home), which took no shots at all in this
    # game — the flip must still land on the correct side.
    assert own_goal_record["team"] == "home"
    assert own_goal_record["home_team"] == "Tottenham"
    assert own_goal_record["away_team"] == "Manchester United"
    assert own_goal_record["is_goal"] is True
    assert own_goal_record["xg"] == 0.0


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


def test_persist_shots_stores_understat_id_as_integer_not_blob_for_numpy_match_id():
    """Regression test for a real data-integrity bug: shots_to_records used to
    pass through row_dict["game_id"] unconverted, which is a numpy.int64 (a
    pandas groupby key), not a plain Python int. sqlite3 silently binds
    numpy.int64 parameters with BLOB storage class instead of coercing them
    to INTEGER, which meant matches.understat_id was stored as an 8-byte BLOB
    for all rows -- silently breaking every later `WHERE understat_id = ?`
    lookup done with a plain Python int (e.g. persist_red_cards). This test
    simulates a record built with a numpy.int64 match_id directly (as would
    happen if shots_to_records regressed) and asserts persist_shots still
    ends up with INTEGER storage class in the database."""
    conn = get_connection(":memory:")
    init_db(conn)
    records = [
        {
            "match_id": numpy.int64(12345), "league": "TEST", "season": "2324",
            "home_team": "Team A", "away_team": "Team B",
            "minute": 10, "team": "home", "xg": 0.1, "is_goal": False,
        },
    ]
    persist_shots(conn, records)
    # Look up the internal match_id without filtering by understat_id -- if
    # the bug were present, understat_id would be a BLOB and a `WHERE
    # understat_id = 12345` filter would never match, masking the failure.
    internal_match_id = conn.execute("SELECT match_id FROM matches").fetchone()[0]
    row = conn.execute(
        "SELECT typeof(understat_id) FROM matches WHERE match_id = ?", (internal_match_id,)
    ).fetchone()
    assert row == ("integer",)


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


def test_load_red_cards_from_cache_reads_rosters(tmp_path):
    import json

    (tmp_path / "match_501.json").write_text(
        json.dumps(
            {
                "rosters": {
                    "h": {
                        "1": {"player": "Player A", "time": "65", "red_card": "1"},
                        "2": {"player": "Player B", "time": "90", "red_card": "0"},
                    },
                    "a": {
                        "3": {"player": "Player C", "time": "78", "red_card": "1"},
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    red_cards = load_red_cards_from_cache(tmp_path)

    assert red_cards[501] == [
        {"team_h_a": "h", "minute": 65},
        {"team_h_a": "a", "minute": 78},
    ]


def test_load_red_cards_from_cache_skips_matches_with_no_red_cards(tmp_path):
    import json

    (tmp_path / "match_502.json").write_text(
        json.dumps({"rosters": {"h": {"1": {"player": "Player A", "time": "90", "red_card": "0"}}, "a": {}}}),
        encoding="utf-8",
    )

    red_cards = load_red_cards_from_cache(tmp_path)

    assert 502 not in red_cards


def test_persist_red_cards_matches_by_understat_id_and_side():
    conn = get_connection(":memory:")
    init_db(conn)
    home_id = get_or_create_team(conn, "Team A")
    away_id = get_or_create_team(conn, "Team B")
    conn.execute(
        """INSERT INTO matches (understat_id, league, season, date, home_team_id, away_team_id)
           VALUES (501, 'TEST', '2324', '2023-08-11', ?, ?)""",
        (home_id, away_id),
    )
    conn.commit()

    red_cards_by_game_id = {501: [{"team_h_a": "h", "minute": 65}, {"team_h_a": "a", "minute": 78}]}
    processed, not_found = persist_red_cards(conn, red_cards_by_game_id)

    assert processed == 1
    assert not_found == 0
    rows = conn.execute("SELECT team_id, minute FROM cards ORDER BY minute").fetchall()
    assert rows == [(home_id, 65), (away_id, 78)]


def test_persist_red_cards_counts_unmatched_game_ids():
    conn = get_connection(":memory:")
    init_db(conn)
    red_cards_by_game_id = {999: [{"team_h_a": "h", "minute": 30}]}
    processed, not_found = persist_red_cards(conn, red_cards_by_game_id)
    assert processed == 0
    assert not_found == 1


def test_persist_red_cards_is_idempotent():
    conn = get_connection(":memory:")
    init_db(conn)
    home_id = get_or_create_team(conn, "Team A")
    away_id = get_or_create_team(conn, "Team B")
    conn.execute(
        """INSERT INTO matches (understat_id, league, season, date, home_team_id, away_team_id)
           VALUES (501, 'TEST', '2324', '2023-08-11', ?, ?)""",
        (home_id, away_id),
    )
    conn.commit()
    red_cards_by_game_id = {501: [{"team_h_a": "h", "minute": 65}]}
    persist_red_cards(conn, red_cards_by_game_id)
    persist_red_cards(conn, red_cards_by_game_id)  # second call: no-op for this match

    count = conn.execute("SELECT COUNT(*) FROM cards").fetchone()[0]
    assert count == 1
