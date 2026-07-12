from unittest.mock import Mock

from goles.sofascore.client import get_incidents, get_shotmap, list_live_events


def _stub_client(json_body):
    response = Mock()
    response.json.return_value = json_body
    response.raise_for_status = Mock()
    client = Mock()
    client.get = Mock(return_value=response)
    return client


def test_list_live_events_returns_events_list():
    client = _stub_client({"events": [{"id": 1, "tournament": {"name": "Premier League"}}]})
    events = list_live_events(client)
    assert events == [{"id": 1, "tournament": {"name": "Premier League"}}]
    client.get.assert_called_once_with("https://api.sofascore.com/api/v1/sport/football/events/live")


def test_list_live_events_returns_empty_list_when_missing_key():
    client = _stub_client({})
    assert list_live_events(client) == []


def test_get_shotmap_returns_shotmap_list():
    client = _stub_client({"shotmap": [{"id": 123, "time": 10, "xg": 0.2}]})
    shots = get_shotmap(client, 999)
    assert shots == [{"id": 123, "time": 10, "xg": 0.2}]
    client.get.assert_called_once_with("https://api.sofascore.com/api/v1/event/999/shotmap")


def test_get_incidents_returns_incidents_list():
    client = _stub_client({"incidents": [{"time": 45, "incidentType": "period"}]})
    incidents = get_incidents(client, 999)
    assert incidents == [{"time": 45, "incidentType": "period"}]
    client.get.assert_called_once_with("https://api.sofascore.com/api/v1/event/999/incidents")
