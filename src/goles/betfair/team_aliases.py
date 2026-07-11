from __future__ import annotations

# Starter table -- unlike TEAM_NAME_ALIASES in goles/loaders/football_data.py
# (built by diffing a complete, real historical dataset), this one cannot
# yet be verified against real Betfair event names: the delayed App Key and
# certificate login are still pending (see the design spec). Extend this
# table with real observed aliases once the poller runs against production
# and logs an unmatched fixture -- never guess an entry without having seen
# the real Betfair name it maps from.
BETFAIR_TEAM_NAME_ALIASES: dict[str, str] = {}


def normalize_betfair_team_name(name: str) -> str:
    """Maps a Betfair event/runner team name to our Understat-sourced team
    name. Names not in BETFAIR_TEAM_NAME_ALIASES are assumed identical and
    returned unchanged -- callers must treat a name that still doesn't
    resolve to a known team as unmatched and skip it loudly, never
    fuzzy-match."""
    return BETFAIR_TEAM_NAME_ALIASES.get(name, name)
