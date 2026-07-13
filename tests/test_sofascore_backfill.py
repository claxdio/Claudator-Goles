from unittest.mock import Mock, patch

from goles.db import get_connection, init_db
from goles.sofascore.backfill import backfill_event, fetch_season_event_ids


def _finished_event(event_id=101, home="Colo-Colo", away="Cobresal"):
    return {
        "id": event_id,
        "homeTeam": {"name": home},
        "awayTeam": {"name": away},
        "startTimestamp": 1751328000,  # 2025-07-01 UTC
        "status": {"type": "finished"},
    }


SOFA_SHOTS = [
    {"id": 1, "time": 12, "shotType": "goal", "situation": "assisted", "isHome": True,
     "playerCoordinates": {"x": 8.0, "y": 50.0}, "bodyPart": "right-foot"},
    {"id": 2, "time": 70, "shotType": "miss", "situation": "corner", "isHome": False,
     "playerCoordinates": {"x": 11.0, "y": 44.0}, "bodyPart": "head"},
]
SOFA_INCIDENTS = [
    {"time": 55, "incidentType": "card", "incidentClass": "red", "isHome": False, "player": {"name": "Test Player"}},
    {"time": 60, "incidentType": "card", "incidentClass": "yellow", "isHome": True, "player": {"name": "Test Player 2"}},
]


class _FakeBooster:
    def predict(self, X):
        return [0.123] * len(X)


def test_fetch_season_event_ids_paginates_and_filters_finished():
    pages = [
        {"events": [_finished_event(1), {"id": 2, "status": {"type": "notstarted"}}], "hasNextPage": True},
        {"events": [_finished_event(3)], "hasNextPage": False},
    ]
    responses = []
    for p in pages:
        r = Mock()
        r.status_code = 200
        r.json.return_value = p
        responses.append(r)
    client = Mock()
    client.get = Mock(side_effect=responses)
    events = fetch_season_event_ids(client, 11653, 88493)
    assert [e["id"] for e in events] == [1, 3]
    assert client.get.call_count == 2


def test_backfill_event_persists_shots_with_our_xg_and_red_cards():
    conn = get_connection(":memory:")
    init_db(conn)
    with patch("goles.sofascore.backfill.get_shotmap", return_value=SOFA_SHOTS):
        with patch("goles.sofascore.backfill.get_incidents", return_value=SOFA_INCIDENTS):
            result = backfill_event(
                Mock(), conn, _FakeBooster(), _finished_event(), "CHI-Liga de Primera", "2025"
            )
    assert result == "ok"
    shots = conn.execute(
        "SELECT minute, xg, is_goal, location_x, situation, shot_type FROM shots ORDER BY minute"
    ).fetchall()
    assert len(shots) == 2
    assert shots[0][0] == 12 and shots[0][2] == 1
    assert abs(shots[0][1] - 0.123) < 1e-9  # our computed xG, not Sofascore's null
    assert abs(shots[0][3] - 0.92) < 1e-9  # 1 - 8/100
    assert shots[0][4] == "OpenPlay" and shots[0][5] == "RightFoot"
    cards = conn.execute("SELECT minute FROM cards").fetchall()
    assert cards == [(55,)]  # red only, yellow excluded


def test_backfill_event_skips_already_persisted_matches():
    conn = get_connection(":memory:")
    init_db(conn)
    with patch("goles.sofascore.backfill.get_shotmap", return_value=SOFA_SHOTS):
        with patch("goles.sofascore.backfill.get_incidents", return_value=SOFA_INCIDENTS):
            first = backfill_event(Mock(), conn, _FakeBooster(), _finished_event(), "CHI-Liga de Primera", "2025")
            second = backfill_event(Mock(), conn, _FakeBooster(), _finished_event(), "CHI-Liga de Primera", "2025")
    assert first == "ok"
    assert second == "skipped_existing"
    assert conn.execute("SELECT COUNT(*) FROM shots").fetchone()[0] == 2


def test_backfill_event_reports_missing_shotmap_without_raising():
    conn = get_connection(":memory:")
    init_db(conn)

    def raise_404(client, event_id):
        raise RuntimeError("404 Client Error")

    with patch("goles.sofascore.backfill.get_shotmap", side_effect=raise_404):
        with patch("goles.sofascore.backfill.get_incidents", return_value=[]):
            result = backfill_event(Mock(), conn, _FakeBooster(), _finished_event(), "CHI-Liga de Primera", "2025")
    assert result == "no_shotmap"
    assert conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 0


OWN_GOAL_SHOTS = [
    {"id": 1, "time": 4, "shotType": "goal", "situation": "own-goal", "goalType": "own",
     "isHome": False, "playerCoordinates": {"x": 4.0, "y": 50.0}, "bodyPart": "other"},
]


def test_backfill_event_credits_own_goal_to_the_right_team_with_zero_xg():
    conn = get_connection(":memory:")
    init_db(conn)
    with patch("goles.sofascore.backfill.get_shotmap", return_value=OWN_GOAL_SHOTS):
        with patch("goles.sofascore.backfill.get_incidents", return_value=[]):
            result = backfill_event(
                Mock(), conn, _FakeBooster(), _finished_event(), "CHI-Liga de Primera", "2025"
            )
    assert result == "ok"
    home_goals, away_goals = conn.execute(
        "SELECT home_goals, away_goals FROM matches"
    ).fetchone()
    # shooter (own-goal scorer) was away (isHome=False) -> goal counts for home
    assert home_goals == 1 and away_goals == 0
    xg = conn.execute("SELECT xg FROM shots").fetchone()[0]
    assert xg == 0.0  # never fed through the booster


def test_backfill_event_excludes_manager_red_cards():
    # Real observed Sofascore shape: a manager/staff dismissal has a
    # "manager" key instead of "player" and a negative "time" (post-match
    # ejection) -- doesn't put the team down a man on the pitch, must not
    # be counted the same as a player red card.
    incidents_with_manager_card = SOFA_INCIDENTS + [
        {"time": -5, "incidentType": "card", "incidentClass": "red", "isHome": True,
         "manager": {"name": "Test Coach"}},
    ]
    conn = get_connection(":memory:")
    init_db(conn)
    with patch("goles.sofascore.backfill.get_shotmap", return_value=SOFA_SHOTS):
        with patch("goles.sofascore.backfill.get_incidents", return_value=incidents_with_manager_card):
            backfill_event(Mock(), conn, _FakeBooster(), _finished_event(), "CHI-Liga de Primera", "2025")
    cards = conn.execute("SELECT minute FROM cards").fetchall()
    assert cards == [(55,)]  # only the real player red card; manager dismissal excluded
