from unittest.mock import Mock, patch

from goles.sofascore.poller import discover_tracked_live_events, poll_once, sync_to_vps
from goles.sofascore.store import get_connection, init_db


def test_discover_tracked_live_events_filters_by_exact_tournament_name():
    client = Mock()
    with patch("goles.sofascore.poller.list_live_events", return_value=[
        {"id": 1, "tournament": {"name": "Premier League"}},
        {"id": 2, "tournament": {"name": "Scottish Premiership"}},
        {"id": 3, "tournament": {"name": "Bundesliga"}},
    ]):
        events = discover_tracked_live_events(client)
    assert [e["id"] for e in events] == [1, 3]


def test_poll_once_persists_shots_and_red_cards():
    event = {
        "id": 12813015,
        "homeTeam": {"name": "Arsenal"},
        "awayTeam": {"name": "Chelsea"},
        "tournament": {"name": "Premier League"},
    }
    shots = [
        {
            "id": 7684954, "time": 20, "xg": 0.185, "shotType": "goal",
            "situation": "corner", "isHome": True,
            "playerCoordinates": {"x": 5.0, "y": 44.1}, "bodyPart": "head",
        },
        {
            "id": 7684839, "time": 10, "xg": 0.056, "shotType": "miss",
            "situation": "regular", "isHome": False,
            "playerCoordinates": {"x": 10.0, "y": 30.0}, "bodyPart": "right-foot",
        },
    ]
    incidents = [
        {"time": 45, "incidentType": "period"},
        {"time": 55, "incidentType": "card", "incidentClass": "red", "isHome": False},
        {"time": 60, "incidentType": "card", "incidentClass": "yellow", "isHome": True},
    ]
    conn = get_connection(":memory:")
    init_db(conn)
    client = Mock()

    with patch("goles.sofascore.poller.get_shotmap", return_value=shots):
        with patch("goles.sofascore.poller.get_incidents", return_value=incidents):
            poll_once(client, conn, [event])

    shot_rows = conn.execute(
        "SELECT home_team, away_team, team, minute, is_goal FROM shots ORDER BY minute"
    ).fetchall()
    assert shot_rows == [
        ("Arsenal", "Chelsea", "away", 10, 0),
        ("Arsenal", "Chelsea", "home", 20, 1),
    ]

    card_rows = conn.execute("SELECT home_team, away_team, team, minute, card_type FROM cards").fetchall()
    assert card_rows == [("Arsenal", "Chelsea", "away", 55, "red")]  # only the red card, not the yellow


def test_sync_to_vps_invokes_scp_with_expected_arguments():
    with patch("goles.sofascore.poller.subprocess.run") as mock_run:
        sync_to_vps(db_path="data/live_match_state.db")
    mock_run.assert_called_once()
    args, kwargs = mock_run.call_args
    command = args[0]
    assert command[0] == "scp"
    assert "data/live_match_state.db" in command
    assert command[-1].endswith(":/root/goles-live-match-state/live_match_state.db")
    assert kwargs["check"] is True
