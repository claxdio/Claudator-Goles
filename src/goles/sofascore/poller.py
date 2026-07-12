from __future__ import annotations

import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import tls_requests

from goles.sofascore.client import get_incidents, get_shotmap, list_live_events
from goles.sofascore.store import DEFAULT_LIVE_MATCH_STATE_DB_PATH, get_connection, init_db, persist_card, persist_shot
from goles.sofascore.team_aliases import normalize_sofascore_team_name

TRACKED_TOURNAMENTS = ["Premier League", "Bundesliga"]
# Assumed from community documentation of Sofascore's incident vocabulary,
# NOT yet confirmed against a real observed red card (none occurred in the
# live match sampled during design). Verify against a real occurrence
# during Task 5's manual verification step and correct if wrong.
RED_CARD_INCIDENT_CLASSES = {"red", "yellowRed"}
POLL_INTERVAL_SECONDS = 60

VPS_HOST = "root@85.239.245.73"
VPS_SSH_KEY = str(Path.home() / ".ssh" / "id_ed25519_goles_vps")
VPS_REMOTE_PATH = "/root/goles-live-match-state/live_match_state.db"


def discover_tracked_live_events(client) -> list[dict]:
    """Returns live events whose tournament name exactly matches one of
    TRACKED_TOURNAMENTS (exact match, not substring -- avoids false
    positives like "Scottish Premiership" matching "Premier League")."""
    events = list_live_events(client)
    return [e for e in events if e.get("tournament", {}).get("name") in TRACKED_TOURNAMENTS]


def poll_once(client, conn, live_events: list[dict]) -> None:
    """Fetches the shotmap and incidents for every tracked live event and
    persists new shots and red cards (idempotent -- see store.py). Each
    event's home/away team names are normalized once and denormalized
    onto every row for that event."""
    fetched_at = datetime.now(timezone.utc).isoformat()
    for event in live_events:
        event_id = event["id"]
        try:
            home_team = normalize_sofascore_team_name(event["homeTeam"]["name"])
            away_team = normalize_sofascore_team_name(event["awayTeam"]["name"])
        except Exception as exc:
            print(f"ADVERTENCIA: fallo al procesar el evento {event_id} ({exc}), se continua con el resto.")
            continue

        # Shots and incidents are fetched from two independent Sofascore
        # endpoints with independent data coverage -- observed for real:
        # some matches 404 on shotmap but still have working incidents
        # (e.g. lower-tier Copa Chile ties). A failure fetching one must
        # not discard whatever the other one has.
        try:
            shots = get_shotmap(client, event_id)
            for shot in shots:
                try:
                    team = "home" if shot.get("isHome") else "away"
                    coordinates = shot.get("playerCoordinates") or {}
                    persist_shot(
                        conn,
                        sofascore_shot_id=shot["id"],
                        sofascore_event_id=event_id,
                        fetched_at=fetched_at,
                        home_team=home_team,
                        away_team=away_team,
                        team=team,
                        minute=shot["time"],
                        xg=shot["xg"],
                        is_goal=shot.get("shotType") == "goal",
                        shot_type=shot.get("shotType", ""),
                        situation=shot.get("situation"),
                        location_x=coordinates.get("x"),
                        location_y=coordinates.get("y"),
                        body_part=shot.get("bodyPart"),
                    )
                except Exception as exc:
                    shot_id = shot.get("id", "?")
                    print(
                        f"ADVERTENCIA: fallo al procesar el tiro {shot_id} del evento {event_id} ({exc}), se omite."
                    )
        except Exception as exc:
            print(f"ADVERTENCIA: fallo al obtener el shotmap del evento {event_id} ({exc}), se omite.")

        try:
            incidents = get_incidents(client, event_id)
            for incident in incidents:
                try:
                    if incident.get("incidentType") != "card":
                        continue
                    incident_class = incident.get("incidentClass")
                    if incident_class not in RED_CARD_INCIDENT_CLASSES:
                        continue
                    team = "home" if incident.get("isHome") else "away"
                    persist_card(
                        conn, event_id, fetched_at, home_team, away_team, team, incident["time"], incident_class
                    )
                except Exception as exc:
                    print(
                        f"ADVERTENCIA: fallo al procesar un incidente del evento {event_id} ({exc}), se omite."
                    )
        except Exception as exc:
            print(f"ADVERTENCIA: fallo al obtener los incidentes del evento {event_id} ({exc}), se omite.")


def sync_to_vps(db_path: str | Path = DEFAULT_LIVE_MATCH_STATE_DB_PATH) -> None:
    """Copies the local live-match-state SQLite file to the VPS via scp,
    reusing the dedicated SSH key already set up for VPS access. Raises on
    failure -- main()'s broad except catches it and retries next cycle."""
    try:
        subprocess.run(
            [
                "scp", "-i", VPS_SSH_KEY, "-o", "StrictHostKeyChecking=accept-new",
                str(db_path), f"{VPS_HOST}:{VPS_REMOTE_PATH}",
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="replace") if exc.stderr else "(no stderr captured)"
        raise RuntimeError(f"scp failed (exit {exc.returncode}): {stderr}") from exc


def main() -> None:
    """Persistent entrypoint: polls forever, syncing to the VPS after each
    cycle. Discovery/poll/sync failures are caught, printed, and retried
    on the next loop iteration rather than crashing the process."""
    client = tls_requests.Client()
    conn = get_connection()
    init_db(conn)

    while True:
        try:
            live_events = discover_tracked_live_events(client)
            print(f"{len(live_events)} partidos en vivo encontrados en las ligas trackeadas.")
            if live_events:
                poll_once(client, conn, live_events)
            sync_to_vps()
        except Exception as exc:
            print(f"ADVERTENCIA: fallo en el ciclo de polling ({exc}), se reintenta en el proximo ciclo.")
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
