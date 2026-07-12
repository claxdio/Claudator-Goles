from __future__ import annotations

import sqlite3
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from goles.db import get_connection, init_db
from goles.loaders.understat import persist_shots
from goles.sofascore.client import BASE_URL, get_incidents, get_shotmap
from goles.sofascore.translate import UnknownVocabularyError, translate_shot
from goles.xg_model import load_xg_model, predict_xg

CHILE_DB_PATH = Path("data") / "goles_chile.db"
TRACKED_UTIDS = {11653: "CHI-Liga de Primera", 1240: "CHI-Liga de Ascenso"}
# Chilean seasons are calendar years; Sofascore season ids fetched live in main().
BACKFILL_YEARS = ["2022", "2023", "2024", "2025", "2026"]
REQUEST_DELAY_SECONDS = 0.7
RED_CARD_INCIDENT_CLASSES = {"red", "yellowRed"}

# Census of every card incidentClass observed during the backfill -- this
# is how the poller's assumed red-card vocabulary finally gets verified
# against real data (printed at the end of main()).
observed_card_classes: Counter = Counter()


def fetch_season_event_ids(client, utid: int, season_id: int) -> list[dict]:
    """Paginates /events/last/{page} and returns finished events only."""
    events: list[dict] = []
    page = 0
    while True:
        response = client.get(f"{BASE_URL}/unique-tournament/{utid}/season/{season_id}/events/last/{page}")
        if response.status_code != 200:
            break
        payload = response.json()
        events.extend(e for e in payload.get("events", []) if e.get("status", {}).get("type") == "finished")
        if not payload.get("hasNextPage"):
            break
        page += 1
    return events


def backfill_event(client, conn: sqlite3.Connection, booster, event: dict, league: str, season_label: str) -> str:
    event_id = event["id"]
    existing = conn.execute("SELECT 1 FROM matches WHERE understat_id = ?", (event_id,)).fetchone()
    if existing:
        return "skipped_existing"

    try:
        sofa_shots = get_shotmap(client, event_id)
    except Exception:
        return "no_shotmap"

    date_iso = datetime.fromtimestamp(event["startTimestamp"], tz=timezone.utc).strftime("%Y-%m-%d")
    home = event["homeTeam"]["name"]
    away = event["awayTeam"]["name"]

    records = []
    for sofa_shot in sofa_shots:
        try:
            t = translate_shot(sofa_shot)
        except UnknownVocabularyError as exc:
            print(f"  ADVERTENCIA: {exc} (evento {event_id}), tiro omitido.")
            continue
        records.append(
            {
                "match_id": event_id, "league": league, "season": season_label,
                "date": date_iso, "home_team": home, "away_team": away,
                "minute": t["minute"], "team": "home" if t["is_home"] else "away",
                "xg": predict_xg(booster, t), "is_goal": t["is_goal"],
                "location_x": t["location_x"], "location_y": t["location_y"],
                "situation": t["situation"], "shot_type": t["shot_type"],
            }
        )
    if not records:
        return "no_shotmap"
    persist_shots(conn, records)

    match_row = conn.execute(
        "SELECT match_id, home_team_id, away_team_id FROM matches WHERE understat_id = ?", (event_id,)
    ).fetchone()
    if match_row is not None:
        match_pk, home_id, away_id = match_row
        try:
            incidents = get_incidents(client, event_id)
        except Exception:
            incidents = []
        for incident in incidents:
            if incident.get("incidentType") != "card":
                continue
            incident_class = incident.get("incidentClass")
            observed_card_classes[incident_class] += 1
            if incident_class not in RED_CARD_INCIDENT_CLASSES:
                continue
            team_id = home_id if incident.get("isHome") else away_id
            conn.execute(
                "INSERT INTO cards (match_id, team_id, minute) VALUES (?, ?, ?)",
                (match_pk, team_id, incident["time"]),
            )
        conn.commit()
    return "ok"


def main() -> None:
    import tls_requests

    from goles.train_xg import XG_MODEL_PATH

    booster = load_xg_model(XG_MODEL_PATH)
    client = tls_requests.Client()
    conn = get_connection(CHILE_DB_PATH)
    init_db(conn)

    for utid, league in TRACKED_UTIDS.items():
        response = client.get(f"{BASE_URL}/unique-tournament/{utid}/seasons")
        seasons = {s["year"]: s["id"] for s in response.json().get("seasons", [])}
        for year in BACKFILL_YEARS:
            if year not in seasons:
                print(f"{league} {year}: temporada no disponible en Sofascore, se omite.")
                continue
            events = fetch_season_event_ids(client, utid, seasons[year])
            print(f"{league} {year}: {len(events)} partidos terminados.")
            tally = Counter()
            for event in events:
                tally[backfill_event(client, conn, booster, event, league, year)] += 1
                time.sleep(REQUEST_DELAY_SECONDS)
            print(f"  -> {dict(tally)}")

    print("\nCenso de incidentClass de tarjetas observadas (verificacion empirica del vocabulario):")
    for cls, count in observed_card_classes.most_common():
        print(f"  {cls}: {count}")


if __name__ == "__main__":
    main()
