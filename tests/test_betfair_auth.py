from unittest.mock import Mock, patch

import pytest

from goles.betfair.auth import BetfairAuthError, BetfairSession, cert_login


def _mock_response(json_body, status_code=200):
    response = Mock()
    response.status_code = status_code
    response.json.return_value = json_body
    response.raise_for_status = Mock()
    if status_code >= 400:
        response.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    return response


def test_cert_login_returns_session_token_on_success():
    with patch("goles.betfair.auth.requests.post", return_value=_mock_response(
        {"sessionToken": "abc123", "loginStatus": "SUCCESS"}
    )) as mock_post:
        token = cert_login("appkey", "user", "pass", "cert.crt", "cert.key")
    assert token == "abc123"
    mock_post.assert_called_once_with(
        "https://identitysso-cert.betfair.com/api/certlogin",
        cert=("cert.crt", "cert.key"),
        headers={"X-Application": "appkey", "Content-Type": "application/x-www-form-urlencoded"},
        data={"username": "user", "password": "pass"},
        timeout=30,
    )


def test_cert_login_raises_on_non_success_status():
    with patch("goles.betfair.auth.requests.post", return_value=_mock_response(
        {"sessionToken": None, "loginStatus": "INVALID_USERNAME_OR_PASSWORD"}
    )):
        with pytest.raises(BetfairAuthError, match="INVALID_USERNAME_OR_PASSWORD"):
            cert_login("appkey", "user", "wrongpass", "cert.crt", "cert.key")


def test_betfair_session_logs_in_once_then_reuses_token():
    login_response = _mock_response({"sessionToken": "tok1", "loginStatus": "SUCCESS"})
    api_response = _mock_response({"result": "ok"})
    with patch("goles.betfair.auth.requests.post", return_value=login_response) as mock_post:
        with patch("goles.betfair.auth.requests.request", return_value=api_response) as mock_request:
            session = BetfairSession("appkey", "user", "pass", "cert.crt", "cert.key")
            session.request("POST", "https://example.test/op1/", json={"a": 1})
            session.request("POST", "https://example.test/op2/", json={"b": 2})
    assert mock_post.call_count == 1  # logged in only once
    assert mock_request.call_count == 2
    _, kwargs = mock_request.call_args_list[0]
    assert kwargs["headers"]["X-Application"] == "appkey"
    assert kwargs["headers"]["X-Authentication"] == "tok1"


def test_betfair_session_relogs_in_once_on_non_200_response():
    login_response = _mock_response({"sessionToken": "tok1", "loginStatus": "SUCCESS"})
    relogin_response = _mock_response({"sessionToken": "tok2", "loginStatus": "SUCCESS"})
    failed_response = _mock_response({"error": "expired"}, status_code=401)
    success_response = _mock_response({"result": "ok"})
    with patch("goles.betfair.auth.requests.post", side_effect=[login_response, relogin_response]):
        with patch("goles.betfair.auth.requests.request", side_effect=[failed_response, success_response]) as mock_request:
            session = BetfairSession("appkey", "user", "pass", "cert.crt", "cert.key")
            response = session.request("POST", "https://example.test/op1/", json={"a": 1})
    assert response is success_response
    assert mock_request.call_count == 2
    _, kwargs = mock_request.call_args_list[1]
    assert kwargs["headers"]["X-Authentication"] == "tok2"


def test_cert_login_routes_through_proxy_when_given():
    with patch("goles.betfair.auth.requests.post", return_value=_mock_response(
        {"sessionToken": "abc123", "loginStatus": "SUCCESS"}
    )) as mock_post:
        cert_login("appkey", "user", "pass", "cert.crt", "cert.key", proxy_url="socks5h://127.0.0.1:1080")
    _, kwargs = mock_post.call_args
    assert kwargs["proxies"] == {"http": "socks5h://127.0.0.1:1080", "https": "socks5h://127.0.0.1:1080"}


def test_betfair_session_routes_api_calls_through_proxy_when_given():
    login_response = _mock_response({"sessionToken": "tok1", "loginStatus": "SUCCESS"})
    api_response = _mock_response({"result": "ok"})
    with patch("goles.betfair.auth.requests.post", return_value=login_response) as mock_post:
        with patch("goles.betfair.auth.requests.request", return_value=api_response) as mock_request:
            session = BetfairSession(
                "appkey", "user", "pass", "cert.crt", "cert.key", proxy_url="socks5h://127.0.0.1:1080"
            )
            session.request("POST", "https://example.test/op1/", json={"a": 1})
    _, login_kwargs = mock_post.call_args
    assert login_kwargs["proxies"] == {"http": "socks5h://127.0.0.1:1080", "https": "socks5h://127.0.0.1:1080"}
    _, request_kwargs = mock_request.call_args
    assert request_kwargs["proxies"] == {"http": "socks5h://127.0.0.1:1080", "https": "socks5h://127.0.0.1:1080"}
