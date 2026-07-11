from unittest.mock import Mock

from goles.betfair.client import (
    find_competition_id,
    list_competitions,
    list_market_book,
    list_market_catalogue,
)


def _stub_session(json_body):
    response = Mock()
    response.json.return_value = json_body
    response.raise_for_status = Mock()
    session = Mock()
    session.request = Mock(return_value=response)
    return session


def test_list_competitions_extracts_competition_dicts():
    session = _stub_session([
        {"competition": {"id": "1", "name": "English Premier League"}, "marketCount": 40},
        {"competition": {"id": "2", "name": "German Bundesliga"}, "marketCount": 30},
    ])
    competitions = list_competitions(session)
    assert competitions == [
        {"id": "1", "name": "English Premier League"},
        {"id": "2", "name": "German Bundesliga"},
    ]
    session.request.assert_called_once_with(
        "POST",
        "https://api.betfair.com/exchange/betting/rest/v1.0/listCompetitions/",
        json={"filter": {"eventTypeIds": ["1"]}},
    )


def test_find_competition_id_matches_by_case_insensitive_substring():
    session = _stub_session([
        {"competition": {"id": "1", "name": "English Premier League"}},
        {"competition": {"id": "2", "name": "German Bundesliga"}},
    ])
    assert find_competition_id(session, "premier league") == "1"
    assert find_competition_id(session, "Bundesliga") == "2"


def test_find_competition_id_returns_none_when_no_match():
    session = _stub_session([{"competition": {"id": "1", "name": "English Premier League"}}])
    assert find_competition_id(session, "La Liga") is None


def test_list_market_catalogue_sends_expected_filter():
    session = _stub_session([{"marketId": "1.123", "event": {"id": "e1", "name": "Team A v Team B"}}])
    result = list_market_catalogue(session, ["1", "2"], ["MATCH_ODDS", "OVER_UNDER_25"])
    assert result == [{"marketId": "1.123", "event": {"id": "e1", "name": "Team A v Team B"}}]
    session.request.assert_called_once_with(
        "POST",
        "https://api.betfair.com/exchange/betting/rest/v1.0/listMarketCatalogue/",
        json={
            "filter": {
                "eventTypeIds": ["1"],
                "competitionIds": ["1", "2"],
                "marketTypeCodes": ["MATCH_ODDS", "OVER_UNDER_25"],
            },
            "maxResults": 200,
            "marketProjection": ["EVENT", "MARKET_START_TIME", "RUNNER_DESCRIPTION", "MARKET_DESCRIPTION"],
        },
    )


def test_list_market_book_sends_expected_filter():
    session = _stub_session([{"marketId": "1.123", "runners": []}])
    result = list_market_book(session, ["1.123", "1.456"])
    assert result == [{"marketId": "1.123", "runners": []}]
    session.request.assert_called_once_with(
        "POST",
        "https://api.betfair.com/exchange/betting/rest/v1.0/listMarketBook/",
        json={"marketIds": ["1.123", "1.456"], "priceProjection": {"priceData": ["EX_BEST_OFFERS"]}},
    )
