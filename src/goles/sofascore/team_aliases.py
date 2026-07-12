from __future__ import annotations

# Starter table -- Sofascore's exact team-name strings for our tracked
# teams haven't been observed yet (same honest-starter-table precedent as
# goles/betfair/team_aliases.py). Extend this table with real observed
# aliases once the poller runs against production and a fixture's team
# name doesn't match our Understat-sourced `teams` table -- never guess an
# entry without having seen the real Sofascore name it maps from.
SOFASCORE_TEAM_NAME_ALIASES: dict[str, str] = {}


def normalize_sofascore_team_name(name: str) -> str:
    """Maps a Sofascore team name to our Understat-sourced team name.
    Names not in SOFASCORE_TEAM_NAME_ALIASES are assumed identical and
    returned unchanged."""
    return SOFASCORE_TEAM_NAME_ALIASES.get(name, name)
