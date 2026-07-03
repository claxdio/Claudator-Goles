from __future__ import annotations

import io

import pandas as pd
import requests

CLUBELO_BASE_URL = "http://api.clubelo.com"


def fetch_elo_ratings(date: str) -> pd.DataFrame:
    """Fetch all clubs' Elo ratings as of `date` (format YYYY-MM-DD) from
    ClubElo's free, unauthenticated CSV endpoint."""
    response = requests.get(f"{CLUBELO_BASE_URL}/{date}", timeout=30)
    response.raise_for_status()
    return pd.read_csv(io.StringIO(response.text))


def elo_win_draw_probabilities(
    home_elo: float, away_elo: float, home_advantage: float = 100.0
) -> tuple[float, float, float]:
    """Convert two Elo ratings into (home_win_prob, draw_prob, away_win_prob)
    using the standard Elo logistic expectation, adjusted for a fixed
    home-advantage offset (in Elo points) and a fixed draw band."""
    diff = (home_elo + home_advantage) - away_elo
    home_or_draw_edge = 1 / (1 + 10 ** (-diff / 400))
    draw_band = 0.25
    home_win = max(home_or_draw_edge - draw_band / 2, 0.0)
    away_win = max((1 - home_or_draw_edge) - draw_band / 2, 0.0)
    draw = max(1.0 - home_win - away_win, 0.0)
    total = home_win + draw + away_win
    return home_win / total, draw / total, away_win / total
