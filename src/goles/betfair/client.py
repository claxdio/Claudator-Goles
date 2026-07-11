from __future__ import annotations

BASE_URL = "https://api.betfair.com/exchange/betting/rest/v1.0"


def list_competitions(session, event_type_id: str = "1") -> list[dict]:
    """Returns the list of {"id", "name"} competition dicts for the given
    Betfair eventTypeId ("1" = Soccer)."""
    response = session.request(
        "POST",
        f"{BASE_URL}/listCompetitions/",
        json={"filter": {"eventTypeIds": [event_type_id]}},
    )
    response.raise_for_status()
    return [entry["competition"] for entry in response.json()]


def find_competition_id(session, name_fragment: str, event_type_id: str = "1") -> str | None:
    """Finds the first competition whose name contains `name_fragment`
    (case-insensitive), or None if none match. Used instead of hardcoding
    competition ids, which would be an unverified guess."""
    for competition in list_competitions(session, event_type_id):
        if name_fragment.lower() in competition["name"].lower():
            return competition["id"]
    return None


def list_market_catalogue(
    session, competition_ids: list[str], market_types: list[str], max_results: int = 200
) -> list[dict]:
    """Returns market catalogue entries (marketId, event, runners) for the
    given competitions and market type codes (e.g. MATCH_ODDS,
    OVER_UNDER_25), scoped to Soccer (eventTypeId "1")."""
    response = session.request(
        "POST",
        f"{BASE_URL}/listMarketCatalogue/",
        json={
            "filter": {
                "eventTypeIds": ["1"],
                "competitionIds": competition_ids,
                "marketTypeCodes": market_types,
            },
            "maxResults": max_results,
            "marketProjection": ["EVENT", "MARKET_START_TIME", "RUNNER_DESCRIPTION", "MARKET_DESCRIPTION"],
        },
    )
    response.raise_for_status()
    return response.json()


def list_market_book(session, market_ids: list[str]) -> list[dict]:
    """Returns current best-offers prices for the given market ids."""
    response = session.request(
        "POST",
        f"{BASE_URL}/listMarketBook/",
        json={"marketIds": market_ids, "priceProjection": {"priceData": ["EX_BEST_OFFERS"]}},
    )
    response.raise_for_status()
    return response.json()
