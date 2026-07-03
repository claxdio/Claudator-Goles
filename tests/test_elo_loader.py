from unittest.mock import Mock, patch

from goles.loaders.elo import elo_win_draw_probabilities, fetch_elo_ratings

SAMPLE_CSV = (
    "Rank,Club,Country,Level,Elo,From,To\n"
    "1,Man City,ENG,1,2010.5,2026-06-25,2026-07-02\n"
    "2,Bayern,GER,1,1980.2,2026-06-25,2026-07-02\n"
)


def test_fetch_elo_ratings_parses_csv():
    mock_response = Mock()
    mock_response.text = SAMPLE_CSV
    mock_response.raise_for_status = Mock()
    with patch("goles.loaders.elo.requests.get", return_value=mock_response) as mock_get:
        df = fetch_elo_ratings("2026-07-02")
    mock_get.assert_called_once_with("http://api.clubelo.com/2026-07-02", timeout=30)
    assert list(df["Club"]) == ["Man City", "Bayern"]
    assert df.loc[df["Club"] == "Man City", "Elo"].iloc[0] == 2010.5


def test_elo_win_draw_probabilities_favor_stronger_team():
    home_win, draw, away_win = elo_win_draw_probabilities(home_elo=2000, away_elo=1700)
    assert home_win > away_win
    assert abs((home_win + draw + away_win) - 1.0) < 1e-9


def test_elo_win_draw_probabilities_close_match_has_home_edge_and_nonzero_draw():
    home_win, draw, away_win = elo_win_draw_probabilities(home_elo=1800, away_elo=1800)
    assert home_win > away_win
    assert draw > 0
