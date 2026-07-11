from unittest.mock import Mock, patch

import pandas as pd

from goles.loaders.football_data import (
    TEAM_NAME_ALIASES,
    fetch_odds,
    normalize_team_name,
)

SAMPLE_CSV = (
    "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,AvgH,AvgD,AvgA,Avg>2.5,Avg<2.5\n"
    "E0,11/08/2023,Burnley,Man City,0,3,9.02,5.35,1.35,1.90,1.95\n"
    "E0,12/08/2023,Arsenal,Nott'm Forest,2,1,1.18,7.64,15.67,1.75,2.10\n"
)


def test_normalize_team_name_maps_known_aliases():
    assert normalize_team_name("Man City") == "Manchester City"
    assert normalize_team_name("Nott'm Forest") == "Nottingham Forest"
    assert normalize_team_name("Dortmund") == "Borussia Dortmund"


def test_normalize_team_name_passes_through_unmapped_names():
    assert normalize_team_name("Arsenal") == "Arsenal"
    assert normalize_team_name("Burnley") == "Burnley"


def test_team_name_aliases_has_no_identity_entries():
    # every value should differ from its key -- identity mappings belong in
    # "pass through unchanged", not cluttering the alias table
    for fd_name, our_name in TEAM_NAME_ALIASES.items():
        assert fd_name != our_name


def test_fetch_odds_concatenates_leagues_and_labels_them():
    mock_response = Mock()
    mock_response.text = SAMPLE_CSV
    mock_response.raise_for_status = Mock()
    with patch("goles.loaders.football_data.requests.get", return_value=mock_response) as mock_get:
        df = fetch_odds({"ENG-Premier League": "E0"}, ["2324"])
    mock_get.assert_called_once_with(
        "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
    )
    assert len(df) == 2
    assert (df["understat_league"] == "ENG-Premier League").all()
    assert list(df["HomeTeam"]) == ["Burnley", "Arsenal"]


import pandas as pd
import pytest

from goles.db import get_connection, get_or_create_team, init_db
from goles.loaders.football_data import (
    compute_no_vig_probabilities,
    compute_no_vig_two_way,
    persist_odds,
)


def test_compute_no_vig_probabilities_sums_to_one_and_favors_favorite():
    home_wp, draw_wp, away_wp = compute_no_vig_probabilities(1.35, 5.35, 9.02)
    assert abs((home_wp + draw_wp + away_wp) - 1.0) < 1e-9
    assert home_wp > away_wp  # 1.35 is the shortest (favorite) price


def test_compute_no_vig_two_way_sums_to_one():
    over_wp, under_wp = compute_no_vig_two_way(1.90, 1.95)
    assert abs((over_wp + under_wp) - 1.0) < 1e-9


def test_persist_odds_matches_by_date_and_normalized_team_names():
    conn = get_connection(":memory:")
    init_db(conn)
    home_id = get_or_create_team(conn, "Manchester City")
    away_id = get_or_create_team(conn, "Burnley")
    conn.execute(
        """INSERT INTO matches (understat_id, league, season, date, home_team_id, away_team_id)
           VALUES (1, 'ENG-Premier League', '2324', '2023-08-11', ?, ?)""",
        (away_id, home_id),  # Burnley home, Man City away -- matches SAMPLE_CSV row 1
    )
    conn.commit()

    odds_df = pd.DataFrame(
        [
            {
                "Date": "11/08/2023", "HomeTeam": "Burnley", "AwayTeam": "Man City",
                "AvgH": 9.02, "AvgD": 5.35, "AvgA": 1.35, "Avg>2.5": 1.90, "Avg<2.5": 1.95,
                "understat_league": "ENG-Premier League", "understat_season": "2324",
            },
        ]
    )
    matched, unmatched = persist_odds(conn, odds_df)
    assert matched == 1
    assert unmatched == 0

    row = conn.execute(
        "SELECT market_home_wp, market_draw_wp, market_away_wp, market_over25_wp FROM matches"
    ).fetchone()
    assert all(v is not None for v in row)
    assert abs(row[0] + row[1] + row[2] - 1.0) < 1e-9


def test_persist_odds_counts_unmatched_rows_without_raising():
    conn = get_connection(":memory:")
    init_db(conn)
    # no matches inserted at all -- every odds row should be unmatched
    odds_df = pd.DataFrame(
        [
            {
                "Date": "11/08/2023", "HomeTeam": "Burnley", "AwayTeam": "Man City",
                "AvgH": 9.02, "AvgD": 5.35, "AvgA": 1.35, "Avg>2.5": 1.90, "Avg<2.5": 1.95,
                "understat_league": "ENG-Premier League", "understat_season": "2324",
            },
        ]
    )
    matched, unmatched = persist_odds(conn, odds_df)
    assert matched == 0
    assert unmatched == 1


def test_persist_odds_falls_back_to_betbrain_column_names():
    conn = get_connection(":memory:")
    init_db(conn)
    home_id = get_or_create_team(conn, "Manchester City")
    away_id = get_or_create_team(conn, "Burnley")
    conn.execute(
        """INSERT INTO matches (understat_id, league, season, date, home_team_id, away_team_id)
           VALUES (1, 'ENG-Premier League', '1819', '2018-08-11', ?, ?)""",
        (away_id, home_id),
    )
    conn.commit()

    odds_df = pd.DataFrame(
        [
            {
                "Date": "11/08/2018", "HomeTeam": "Burnley", "AwayTeam": "Man City",
                "AvgH": float("nan"), "AvgD": float("nan"), "AvgA": float("nan"),
                "Avg>2.5": float("nan"), "Avg<2.5": float("nan"),
                "BbAvH": 9.02, "BbAvD": 5.35, "BbAvA": 1.35,
                "BbAv>2.5": 1.90, "BbAv<2.5": 1.95,
                "understat_league": "ENG-Premier League", "understat_season": "1819",
            },
        ]
    )
    matched, unmatched = persist_odds(conn, odds_df)
    assert matched == 1
    assert unmatched == 0
    row = conn.execute("SELECT market_home_wp FROM matches").fetchone()
    assert row[0] is not None


def test_persist_odds_counts_nan_odds_as_unmatched_even_when_match_found():
    conn = get_connection(":memory:")
    init_db(conn)
    home_id = get_or_create_team(conn, "Manchester City")
    away_id = get_or_create_team(conn, "Burnley")
    conn.execute(
        """INSERT INTO matches (understat_id, league, season, date, home_team_id, away_team_id)
           VALUES (1, 'ENG-Premier League', '1819', '2018-08-11', ?, ?)""",
        (away_id, home_id),
    )
    conn.commit()

    odds_df = pd.DataFrame(
        [
            {
                "Date": "11/08/2018", "HomeTeam": "Burnley", "AwayTeam": "Man City",
                "AvgH": float("nan"), "AvgD": float("nan"), "AvgA": float("nan"),
                "Avg>2.5": float("nan"), "Avg<2.5": float("nan"),
                # no BbAv* fallback columns present at all in this row
                "understat_league": "ENG-Premier League", "understat_season": "1819",
            },
        ]
    )
    matched, unmatched = persist_odds(conn, odds_df)
    assert matched == 0
    assert unmatched == 1
    row = conn.execute("SELECT market_home_wp FROM matches").fetchone()
    assert row[0] is None
