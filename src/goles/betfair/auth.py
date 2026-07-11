from __future__ import annotations

import requests

LOGIN_URL = "https://identitysso-cert.betfair.com/api/certlogin"


class BetfairAuthError(Exception):
    """Raised when Betfair's certificate login does not return
    loginStatus == SUCCESS (e.g. INVALID_USERNAME_OR_PASSWORD,
    ACCOUNT_ALREADY_LOCKED)."""


def cert_login(
    app_key: str,
    username: str,
    password: str,
    cert_file: str,
    key_file: str,
    login_url: str = LOGIN_URL,
) -> str:
    """Performs Betfair's non-interactive (bot) certificate login and
    returns the session token. Raises BetfairAuthError on any
    loginStatus other than SUCCESS."""
    response = requests.post(
        login_url,
        cert=(cert_file, key_file),
        headers={
            "X-Application": app_key,
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={"username": username, "password": password},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("loginStatus") != "SUCCESS":
        raise BetfairAuthError(f"Betfair login failed: {payload.get('loginStatus')}")
    return payload["sessionToken"]


class BetfairSession:
    """Holds a Betfair session token and re-authenticates automatically.
    Betfair does not document session lifetime, so this re-logs in
    reactively -- once before the first request, and again exactly once
    if a request comes back with a non-200 status -- rather than
    assuming a fixed expiry duration."""

    def __init__(
        self,
        app_key: str,
        username: str,
        password: str,
        cert_file: str,
        key_file: str,
        login_url: str = LOGIN_URL,
    ) -> None:
        self.app_key = app_key
        self.username = username
        self.password = password
        self.cert_file = cert_file
        self.key_file = key_file
        self.login_url = login_url
        self._session_token: str | None = None

    def _login(self) -> str:
        self._session_token = cert_login(
            self.app_key, self.username, self.password, self.cert_file, self.key_file, self.login_url
        )
        return self._session_token

    def request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Issues an authenticated request against the Exchange API,
        logging in first if there's no session yet, and retrying exactly
        once (with a fresh login) if the first attempt comes back with a
        non-200 status."""
        token = self._session_token or self._login()
        headers = kwargs.pop("headers", {}) or {}
        headers["X-Application"] = self.app_key
        headers["X-Authentication"] = token
        timeout = kwargs.pop("timeout", 30)
        response = requests.request(method, url, headers=headers, timeout=timeout, **kwargs)
        if response.status_code != 200:
            token = self._login()
            headers["X-Authentication"] = token
            response = requests.request(method, url, headers=headers, timeout=timeout, **kwargs)
        return response
