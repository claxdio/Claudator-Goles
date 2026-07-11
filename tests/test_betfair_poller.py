from unittest.mock import Mock, patch

import pytest

from goles.betfair.odds_store import get_connection, init_db
from goles.betfair.poller import (
    compute_match_odds_probabilities,
    compute_over_under_probabilities,
    discover_tracked_markets,
    extract_best_back_prices,
    parse_team_names_from_event,
    poll_once,
)


def test_parse_team_names_from_event_splits_on_v():
    assert parse_team_names_from_event("Arsenal v Chelsea") == ("Arsenal", "Chelsea")


def test_parse_team_names_from_event_raises_on_unexpected_format():
    with pytest.raises(ValueError):
        parse_team_names_from_event("Arsenal - Chelsea")


def test_extract_best_back_prices_returns_price_by_selection_id():
    market_book = {
        "runners": [
            {"selectionId": 1, "ex": {"availableToBack": [{"price": 2.5, "size": 100}]}},
            {"selectionId": 2, "ex": {"availableToBack": [{"price": 3.0, "size": 50}]}},
        ]
    }
    assert extract_best_back_prices(market_book) == {1: 2.5, 2: 3.0}


def test_extract_best_back_prices_returns_none_when_a_runner_has_no_price():
    market_book = {
        "runners": [
            {"selectionId": 1, "ex": {"availableToBack": []}},
            {"selectionId": 2, "ex": {"availableToBack": [{"price": 3.0, "size": 50}]}},
        ]
    }
    assert extract_best_back_prices(market_book) is None


def test_compute_match_odds_probabilities_resolves_home_draw_away():
    runner_name_by_id = {1: "Arsenal", 2: "The Draw", 3: "Chelsea"}
    prices_by_id = {1: 1.5, 2: 4.0, 3: 6.0}
    probs = compute_match_odds_probabilities(runner_name_by_id, prices_by_id, "Arsenal", "Chelsea")
    assert probs is not None
    home_wp, draw_wp, away_wp = probs
    assert abs((home_wp + draw_wp + away_wp) - 1.0) < 1e-9
    assert home_wp > away_wp


def test_compute_match_odds_probabilities_returns_none_when_team_not_found():
    runner_name_by_id = {1: "Some Other Team", 2: "The Draw", 3: "Chelsea"}
    prices_by_id = {1: 1.5, 2: 4.0, 3: 6.0}
    assert compute_match_odds_probabilities(runner_name_by_id, prices_by_id, "Arsenal", "Chelsea") is None


def test_compute_over_under_probabilities_resolves_over_and_under():
    runner_name_by_id = {1: "Over 2.5 Goals", 2: "Under 2.5 Goals"}
    prices_by_id = {1: 1.9, 2: 1.95}
    probs = compute_over_under_probabilities(runner_name_by_id, prices_by_id)
    assert probs is not None
    over_wp, under_wp = probs
    assert abs((over_wp + under_wp) - 1.0) < 1e-9


def test_compute_over_under_probabilities_returns_none_when_runners_not_found():
    runner_name_by_id = {1: "Something Else", 2: "Under 2.5 Goals"}
    prices_by_id = {1: 1.9, 2: 1.95}
    assert compute_over_under_probabilities(runner_name_by_id, prices_by_id) is None


def test_poll_once_persists_a_snapshot_for_a_valid_match_odds_market():
    market_catalogue = [
        {
            "marketId": "1.111",
            "marketType": "MATCH_ODDS",
            "event": {"id": "e1", "name": "Arsenal v Chelsea"},
            "runners": [
                {"selectionId": 1, "runnerName": "Arsenal"},
                {"selectionId": 2, "runnerName": "The Draw"},
                {"selectionId": 3, "runnerName": "Chelsea"},
            ],
        }
    ]
    market_book = {
        "marketId": "1.111",
        "runners": [
            {"selectionId": 1, "ex": {"availableToBack": [{"price": 1.5}]}},
            {"selectionId": 2, "ex": {"availableToBack": [{"price": 4.0}]}},
            {"selectionId": 3, "ex": {"availableToBack": [{"price": 6.0}]}},
        ],
    }
    session = Mock()
    conn = get_connection(":memory:")
    init_db(conn)

    with patch("goles.betfair.poller.list_market_book", return_value=[market_book]):
        poll_once(session, conn, market_catalogue)

    row = conn.execute("SELECT home_team, away_team, market_type FROM odds_snapshots").fetchone()
    assert row == ("Arsenal", "Chelsea", "MATCH_ODDS")


def test_poll_once_skips_market_with_no_available_prices_without_raising(capsys):
    market_catalogue = [
        {
            "marketId": "1.111",
            "marketType": "MATCH_ODDS",
            "event": {"id": "e1", "name": "Arsenal v Chelsea"},
            "runners": [
                {"selectionId": 1, "runnerName": "Arsenal"},
                {"selectionId": 2, "runnerName": "The Draw"},
                {"selectionId": 3, "runnerName": "Chelsea"},
            ],
        }
    ]
    market_book = {
        "marketId": "1.111",
        "runners": [{"selectionId": 1, "ex": {"availableToBack": []}}],
    }
    session = Mock()
    conn = get_connection(":memory:")
    init_db(conn)

    with patch("goles.betfair.poller.list_market_book", return_value=[market_book]):
        poll_once(session, conn, market_catalogue)  # must not raise

    count = conn.execute("SELECT COUNT(*) FROM odds_snapshots").fetchone()[0]
    assert count == 0
    assert "ADVERTENCIA" in capsys.readouterr().out


def test_poll_once_warns_when_market_absent_from_list_market_book_response(capsys):
    """A market present in market_catalogue but absent from the
    list_market_book response (e.g. suspended/removed) must be skipped with
    a printed ADVERTENCIA warning -- never silently."""
    market_catalogue = [
        {
            "marketId": "1.111",
            "marketType": "MATCH_ODDS",
            "event": {"id": "e1", "name": "Arsenal v Chelsea"},
            "runners": [
                {"selectionId": 1, "runnerName": "Arsenal"},
                {"selectionId": 2, "runnerName": "The Draw"},
                {"selectionId": 3, "runnerName": "Chelsea"},
            ],
        }
    ]
    session = Mock()
    conn = get_connection(":memory:")
    init_db(conn)

    with patch("goles.betfair.poller.list_market_book", return_value=[]):
        poll_once(session, conn, market_catalogue)  # must not raise

    count = conn.execute("SELECT COUNT(*) FROM odds_snapshots").fetchone()[0]
    assert count == 0
    assert "ADVERTENCIA" in capsys.readouterr().out


def test_poll_once_normalizes_home_away_names_via_alias_table():
    """Once BETFAIR_TEAM_NAME_ALIASES gets a real entry, the runner name
    (already normalized inside compute_match_odds_probabilities) must be
    compared against an equally-normalized home/away name -- otherwise the
    raw, unnormalized event-derived name would never match again."""
    market_catalogue = [
        {
            "marketId": "1.111",
            "marketType": "MATCH_ODDS",
            "event": {"id": "e1", "name": "Man Utd v Chelsea"},
            "runners": [
                {"selectionId": 1, "runnerName": "Man Utd"},
                {"selectionId": 2, "runnerName": "The Draw"},
                {"selectionId": 3, "runnerName": "Chelsea"},
            ],
        }
    ]
    market_book = {
        "marketId": "1.111",
        "runners": [
            {"selectionId": 1, "ex": {"availableToBack": [{"price": 1.5}]}},
            {"selectionId": 2, "ex": {"availableToBack": [{"price": 4.0}]}},
            {"selectionId": 3, "ex": {"availableToBack": [{"price": 6.0}]}},
        ],
    }
    session = Mock()
    conn = get_connection(":memory:")
    init_db(conn)

    with (
        patch.dict(
            "goles.betfair.team_aliases.BETFAIR_TEAM_NAME_ALIASES",
            {"Man Utd": "Manchester United"},
            clear=True,
        ),
        patch("goles.betfair.poller.list_market_book", return_value=[market_book]),
    ):
        poll_once(session, conn, market_catalogue)

    row = conn.execute("SELECT home_team, away_team, market_type FROM odds_snapshots").fetchone()
    assert row == ("Manchester United", "Chelsea", "MATCH_ODDS")


def test_discover_tracked_markets_returns_catalogue_when_all_competitions_found():
    with (
        patch("goles.betfair.poller.find_competition_id", side_effect=["id-pl", "id-bl"]) as mock_find,
        patch("goles.betfair.poller.list_market_catalogue", return_value=[{"marketId": "1.111"}]) as mock_list,
    ):
        session = Mock()
        result = discover_tracked_markets(session)

    assert result == [{"marketId": "1.111"}]
    assert mock_find.call_count == 2
    mock_list.assert_called_once_with(session, ["id-pl", "id-bl"], ["MATCH_ODDS", "OVER_UNDER_25"])


def test_discover_tracked_markets_skips_competition_not_found():
    with (
        patch("goles.betfair.poller.find_competition_id", side_effect=[None, "id-bl"]),
        patch("goles.betfair.poller.list_market_catalogue", return_value=[{"marketId": "1.222"}]) as mock_list,
    ):
        session = Mock()
        result = discover_tracked_markets(session)

    assert result == [{"marketId": "1.222"}]
    mock_list.assert_called_once_with(session, ["id-bl"], ["MATCH_ODDS", "OVER_UNDER_25"])


def test_discover_tracked_markets_returns_empty_list_when_no_competitions_found():
    with (
        patch("goles.betfair.poller.find_competition_id", return_value=None),
        patch("goles.betfair.poller.list_market_catalogue") as mock_list,
    ):
        session = Mock()
        result = discover_tracked_markets(session)

    assert result == []
    mock_list.assert_not_called()
