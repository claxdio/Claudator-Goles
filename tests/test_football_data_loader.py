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
