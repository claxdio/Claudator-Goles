from __future__ import annotations

import io
import sqlite3

import pandas as pd
import requests

from goles.db import get_or_create_team

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


def compute_no_vig_probabilities(
    odds_home: float, odds_draw: float, odds_away: float
) -> tuple[float, float, float]:
    """Converts three decimal odds into de-margined (no-vig) probabilities
    that sum to exactly 1.0, by normalizing the raw 1/odds values (whose
    sum exceeds 1.0 by the bookmaker's overround)."""
    raw = [1.0 / odds_home, 1.0 / odds_draw, 1.0 / odds_away]
    total = sum(raw)
    return raw[0] / total, raw[1] / total, raw[2] / total


def compute_no_vig_two_way(odds_a: float, odds_b: float) -> tuple[float, float]:
    """Same de-margining for a two-outcome market (e.g. over/under 2.5)."""
    raw = [1.0 / odds_a, 1.0 / odds_b]
    total = sum(raw)
    return raw[0] / total, raw[1] / total


def _to_iso_date(football_data_date: str) -> str:
    """Converts football-data.co.uk's DD/MM/YYYY to our ISO YYYY-MM-DD."""
    day, month, year = football_data_date.split("/")
    if len(year) == 2:
        year = "20" + year
    return f"{year}-{month.zfill(2)}-{day.zfill(2)}"


def persist_odds(conn: sqlite3.Connection, odds_df: pd.DataFrame) -> tuple[int, int]:
    """Normalizes team names, converts dates, computes no-vig probabilities,
    and updates matching rows in `matches` (joined by league + season +
    date + home/away team name, since football-data.co.uk has no id
    compatible with our understat_id). Rows with no matching database row
    are counted as unmatched but do not raise -- the caller decides what
    coverage is acceptable. Returns (matched_count, unmatched_count)."""
    matched = 0
    unmatched = 0
    for row_dict in odds_df.to_dict("records"):
        home_name = normalize_team_name(row_dict["HomeTeam"])
        away_name = normalize_team_name(row_dict["AwayTeam"])
        date_iso = _to_iso_date(row_dict["Date"])

        home_row = conn.execute("SELECT team_id FROM teams WHERE name = ?", (home_name,)).fetchone()
        away_row = conn.execute("SELECT team_id FROM teams WHERE name = ?", (away_name,)).fetchone()
        if home_row is None or away_row is None:
            unmatched += 1
            continue
        home_id, away_id = home_row[0], away_row[0]

        match_row = conn.execute(
            """SELECT match_id FROM matches
               WHERE league = ? AND season = ? AND date = ?
                 AND home_team_id = ? AND away_team_id = ?""",
            (row_dict["understat_league"], row_dict["understat_season"], date_iso, home_id, away_id),
        ).fetchone()
        if match_row is None:
            unmatched += 1
            continue

        home_wp, draw_wp, away_wp = compute_no_vig_probabilities(
            row_dict["AvgH"], row_dict["AvgD"], row_dict["AvgA"]
        )
        over_wp, _ = compute_no_vig_two_way(row_dict["Avg>2.5"], row_dict["Avg<2.5"])
        conn.execute(
            """UPDATE matches
               SET market_home_wp = ?, market_draw_wp = ?, market_away_wp = ?, market_over25_wp = ?
               WHERE match_id = ?""",
            (home_wp, draw_wp, away_wp, over_wp, match_row[0]),
        )
        matched += 1
    conn.commit()
    return matched, unmatched
