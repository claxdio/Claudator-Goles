from unittest.mock import MagicMock, patch

import pandas as pd

from goles.db import get_connection, init_db
from goles.loaders.understat import (
    fetch_understat_shots,
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


def test_shots_to_records_normalizes_team_side_and_goal_flag():
    df = pd.DataFrame(
        [
            {
                "game_id": 101, "league": "ENG-Premier League", "season": "2023-24",
                "home_team": "Arsenal", "away_team": "Chelsea", "team": "Arsenal",
                "minute": 23, "xG": 0.15, "result": "MissedShots",
            },
            {
                "game_id": 101, "league": "ENG-Premier League", "season": "2023-24",
                "home_team": "Arsenal", "away_team": "Chelsea", "team": "Chelsea",
                "minute": 41, "xG": 0.42, "result": "Goal",
            },
        ]
    )
    records = shots_to_records(df)
    assert records[0]["team"] == "home"
    assert records[0]["is_goal"] is False
    assert records[1]["team"] == "away"
    assert records[1]["is_goal"] is True
    assert records[1]["xg"] == 0.42


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
