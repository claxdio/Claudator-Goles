from __future__ import annotations

import io

import pandas as pd
import requests

LEAGUE_CODES = {
    "ENG-Premier League": "E0",
    "GER-Bundesliga": "D1",
}

# Built by fetching all 6 seasons (1819-2324) of both leagues from
# football-data.co.uk and diffing team names against our real `teams`
# table (populated from Understat). Every name that differs is listed
# here explicitly -- this is a complete, verified table for the
# leagues/seasons this project currently tracks, not a guess. A team not
# in this table and not identical to our own naming will fail to match
# in Task 3's persistence step, loudly, rather than being silently
# fuzzy-matched.
TEAM_NAME_ALIASES = {
    # Premier League
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Newcastle": "Newcastle United",
    "Nott'm Forest": "Nottingham Forest",
    "West Brom": "West Bromwich Albion",
    "Wolves": "Wolverhampton Wanderers",
    # Bundesliga
    "Bielefeld": "Arminia Bielefeld",
    "Dortmund": "Borussia Dortmund",
    "Ein Frankfurt": "Eintracht Frankfurt",
    "FC Koln": "FC Cologne",
    "Fortuna Dusseldorf": "Fortuna Duesseldorf",
    "Greuther Furth": "Greuther Fuerth",
    "Hannover": "Hannover 96",
    "Heidenheim": "FC Heidenheim",
    "Hertha": "Hertha Berlin",
    "Leverkusen": "Bayer Leverkusen",
    "M'gladbach": "Borussia M.Gladbach",
    "Mainz": "Mainz 05",
    "Nurnberg": "Nuernberg",
    "RB Leipzig": "RasenBallsport Leipzig",
    "Stuttgart": "VfB Stuttgart",
}


def normalize_team_name(name: str) -> str:
    """Maps a football-data.co.uk team name to our Understat-sourced team
    name. Names not in TEAM_NAME_ALIASES are assumed identical between the
    two sources and returned unchanged."""
    return TEAM_NAME_ALIASES.get(name, name)


def fetch_odds(leagues: dict[str, str], seasons: list[str]) -> pd.DataFrame:
    """Fetches football-data.co.uk match/odds CSVs directly (bypassing
    soccerdata's MatchHistory reader, whose TLS-fingerprinting client is
    currently blocked -- verified 503 -- by this specific site; a plain
    requests.get with a standard User-Agent works). `leagues` maps our
    league name (e.g. "ENG-Premier League") to football-data.co.uk's
    short code (e.g. "E0"). Returns the concatenated raw CSV rows across
    every league/season with an added `understat_league` column."""
    frames = []
    for league_name, code in leagues.items():
        for season in seasons:
            url = f"https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
            response.raise_for_status()
            df = pd.read_csv(io.StringIO(response.text))
            df["understat_league"] = league_name
            df["understat_season"] = season
            frames.append(df)
    return pd.concat(frames, ignore_index=True)
