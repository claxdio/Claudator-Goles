from __future__ import annotations

BASE_URL = "https://api.sofascore.com/api/v1"


def list_live_events(client) -> list[dict]:
    """Returns all currently live football events worldwide (unfiltered by
    league -- callers filter by tournament name, see poller.py)."""
    response = client.get(f"{BASE_URL}/sport/football/events/live")
    response.raise_for_status()
    return response.json().get("events", [])


def get_shotmap(client, event_id: int) -> list[dict]:
    """Returns the raw shot list for a live or completed event."""
    response = client.get(f"{BASE_URL}/event/{event_id}/shotmap")
    response.raise_for_status()
    return response.json().get("shotmap", [])


def get_incidents(client, event_id: int) -> list[dict]:
    """Returns the raw incident list (goals, cards, periods, ...) for a
    live or completed event."""
    response = client.get(f"{BASE_URL}/event/{event_id}/incidents")
    response.raise_for_status()
    return response.json().get("incidents", [])
