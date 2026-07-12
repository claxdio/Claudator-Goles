from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import datetime, timezone

from goles.betfair.auth import BetfairSession
from goles.betfair.client import find_competition_id, list_market_book, list_market_catalogue
from goles.betfair.odds_store import get_connection, init_db, persist_snapshot
from goles.betfair.team_aliases import normalize_betfair_team_name
from goles.loaders.football_data import compute_no_vig_probabilities, compute_no_vig_two_way

TRACKED_COMPETITIONS = ["Premier League", "Bundesliga", "Chilean Primera Division"]
MATCH_ODDS_TYPE = "MATCH_ODDS"
OVER_UNDER_TYPE = "OVER_UNDER_25"
POLL_INTERVAL_SECONDS = 60
FIXTURE_REFRESH_INTERVAL_SECONDS = 900
DRAW_RUNNER_NAME = "The Draw"
OVER_RUNNER_NAME = "Over 2.5 Goals"
UNDER_RUNNER_NAME = "Under 2.5 Goals"


def parse_team_names_from_event(event_name: str) -> tuple[str, str]:
    """Splits Betfair's soccer event name convention ("Home Team v Away
    Team") into (home, away). Raises ValueError on any other format --
    fail loud rather than guess."""
    parts = event_name.split(" v ")
    if len(parts) != 2:
        raise ValueError(f"Unexpected event name format: {event_name!r}")
    return parts[0], parts[1]


def extract_best_back_prices(market_book: dict) -> dict[int, float] | None:
    """Returns {selectionId: best_back_price} for every runner in the
    market book, or None if any runner currently has no available-to-back
    price (an empty/not-yet-liquid market) -- callers must skip such
    markets rather than compute odds from partial data."""
    prices: dict[int, float] = {}
    for runner in market_book.get("runners", []):
        available = runner.get("ex", {}).get("availableToBack", [])
        if not available:
            return None
        prices[runner["selectionId"]] = available[0]["price"]
    return prices


def compute_match_odds_probabilities(
    runner_name_by_id: dict[int, str],
    prices_by_id: dict[int, float],
    home_name: str,
    away_name: str,
) -> tuple[float, float, float] | None:
    """Resolves the MATCH_ODDS market's three runners (home team, draw,
    away team) by name and returns their no-vig (home, draw, away) win
    probabilities, or None if any of the three can't be matched by name
    (via normalize_betfair_team_name) -- never guessed."""
    home_price = draw_price = away_price = None
    for selection_id, price in prices_by_id.items():
        runner_name = normalize_betfair_team_name(runner_name_by_id.get(selection_id, ""))
        if runner_name == DRAW_RUNNER_NAME:
            draw_price = price
        elif runner_name == home_name:
            home_price = price
        elif runner_name == away_name:
            away_price = price
    if home_price is None or draw_price is None or away_price is None:
        return None
    return compute_no_vig_probabilities(home_price, draw_price, away_price)


def compute_over_under_probabilities(
    runner_name_by_id: dict[int, str], prices_by_id: dict[int, float]
) -> tuple[float, float] | None:
    """Resolves the OVER_UNDER_25 market's two runners and returns their
    no-vig (over, under) probabilities, or None if either runner can't be
    matched by name."""
    over_price = under_price = None
    for selection_id, price in prices_by_id.items():
        runner_name = runner_name_by_id.get(selection_id, "")
        if runner_name == OVER_RUNNER_NAME:
            over_price = price
        elif runner_name == UNDER_RUNNER_NAME:
            under_price = price
    if over_price is None or under_price is None:
        return None
    return compute_no_vig_two_way(over_price, under_price)


def discover_tracked_markets(session: BetfairSession) -> list[dict]:
    """Finds the competition ids for TRACKED_COMPETITIONS and returns the
    open MATCH_ODDS + OVER_UNDER_25 market catalogue entries for them.
    A competition that can't be found (name changed, no markets currently
    listed) is simply skipped -- not an error, since coverage naturally
    varies with the football calendar."""
    competition_ids = []
    for name in TRACKED_COMPETITIONS:
        competition_id = find_competition_id(session, name)
        if competition_id is not None:
            competition_ids.append(competition_id)
    if not competition_ids:
        return []
    return list_market_catalogue(session, competition_ids, [MATCH_ODDS_TYPE, OVER_UNDER_TYPE])


def poll_once(session: BetfairSession, conn: sqlite3.Connection, market_catalogue: list[dict]) -> None:
    """Fetches current prices for every market in market_catalogue and
    persists one snapshot row per market with resolvable prices. Markets
    with no available back prices yet, or whose teams/runners can't be
    matched, are skipped with a printed warning -- never silently."""
    market_ids = [m["marketId"] for m in market_catalogue]
    if not market_ids:
        return
    market_books = list_market_book(session, market_ids)
    books_by_id = {b["marketId"]: b for b in market_books}
    fetched_at = datetime.now(timezone.utc).isoformat()

    for catalogue_entry in market_catalogue:
        market_id = catalogue_entry["marketId"]
        market_book = books_by_id.get(market_id)
        if market_book is None:
            print(
                f"ADVERTENCIA: no se encontro market book para el mercado {market_id} "
                f"('{catalogue_entry.get('event', {}).get('name')}'), se omite."
            )
            continue

        event = catalogue_entry.get("event", {})
        try:
            home_name, away_name = parse_team_names_from_event(event.get("name", ""))
        except ValueError:
            print(f"ADVERTENCIA: no se pudo separar equipos de '{event.get('name')}', se omite mercado {market_id}.")
            continue
        home_name = normalize_betfair_team_name(home_name)
        away_name = normalize_betfair_team_name(away_name)

        runner_name_by_id = {r["selectionId"]: r["runnerName"] for r in catalogue_entry.get("runners", [])}
        prices_by_id = extract_best_back_prices(market_book)
        if prices_by_id is None:
            print(f"ADVERTENCIA: no hay precios disponibles para el mercado {market_id} ('{event.get('name')}'), se omite.")
            continue

        market_type = catalogue_entry.get("marketType") or catalogue_entry.get("description", {}).get("marketType")
        if market_type == MATCH_ODDS_TYPE:
            probs = compute_match_odds_probabilities(runner_name_by_id, prices_by_id, home_name, away_name)
            if probs is None:
                print(
                    f"ADVERTENCIA: no se pudieron resolver equipos/empate en mercado MATCH_ODDS "
                    f"{market_id} ('{event.get('name')}'), se omite."
                )
                continue
            home_wp, draw_wp, away_wp = probs
            persist_snapshot(
                conn, fetched_at, event.get("id", ""), home_name, away_name, MATCH_ODDS_TYPE,
                json.dumps(market_book), home_wp=home_wp, draw_wp=draw_wp, away_wp=away_wp,
            )
        elif market_type == OVER_UNDER_TYPE:
            over_probs = compute_over_under_probabilities(runner_name_by_id, prices_by_id)
            if over_probs is None:
                print(
                    f"ADVERTENCIA: no se pudieron resolver runners over/under en mercado "
                    f"{market_id} ('{event.get('name')}'), se omite."
                )
                continue
            over_wp, _ = over_probs
            persist_snapshot(
                conn, fetched_at, event.get("id", ""), home_name, away_name, OVER_UNDER_TYPE,
                json.dumps(market_book), over_wp=over_wp,
            )


def main() -> None:
    """Persistent entrypoint: logs in, discovers tracked fixtures, then
    polls forever. Any failure during discovery or a poll cycle is caught,
    printed, and retried on the next loop iteration rather than crashing
    the process -- a missing/invalid required env var, in contrast, fails
    immediately and loudly at startup (a real configuration error, not
    something to silently retry)."""
    app_key = os.environ["BETFAIR_APP_KEY"]
    username = os.environ["BETFAIR_USERNAME"]
    password = os.environ["BETFAIR_PASSWORD"]
    cert_file = os.environ.get("BETFAIR_CERT_FILE", "/run/secrets/betfair/client-2048.crt")
    key_file = os.environ.get("BETFAIR_KEY_FILE", "/run/secrets/betfair/client-2048.key")
    proxy_url = os.environ.get("BETFAIR_SOCKS_PROXY_URL")

    session = BetfairSession(app_key, username, password, cert_file, key_file, proxy_url=proxy_url)
    conn = get_connection()
    init_db(conn)

    market_catalogue: list[dict] = []
    last_discovery = 0.0

    while True:
        now = time.monotonic()
        if not market_catalogue or (now - last_discovery) > FIXTURE_REFRESH_INTERVAL_SECONDS:
            try:
                market_catalogue = discover_tracked_markets(session)
                last_discovery = now
                print(f"{len(market_catalogue)} mercados encontrados en las ligas trackeadas.")
            except Exception as exc:
                print(f"ADVERTENCIA: fallo al descubrir mercados ({exc}), se reintenta en el proximo ciclo.")
        if market_catalogue:
            try:
                poll_once(session, conn, market_catalogue)
            except Exception as exc:
                print(f"ADVERTENCIA: fallo en el ciclo de polling ({exc}), se reintenta en el proximo ciclo.")
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
