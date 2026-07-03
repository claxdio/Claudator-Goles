from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from goles.backtest import (
    CUTOFF_MINUTES,
    DEFAULT_BLEND,
    HORIZON_MINUTES,
    RECENT_WINDOW_MINUTES,
    load_match_shots,
)
from goles.features import compute_ml_features, compute_state_at_minute, goal_in_window
from goles.model import dynamic_lambda, prob_goal_in_window
from goles.priors import trailing_xg_per90

FEATURE_NAMES = [
    "is_home",
    "minute",
    "minutes_remaining",
    "score_diff",
    "score_diff_x_minutes_remaining",
    "own_xg_total",
    "opp_xg_total",
    "xg_diff",
    "own_xg_rate",
    "opp_xg_rate",
    "own_max_shot_xg",
    "opp_max_shot_xg",
    "own_big_chances",
    "opp_big_chances",
    "own_recent_xg",
    "opp_recent_xg",
    "own_trend",
    "own_time_since_shot",
    "time_since_goal",
    "trailing_prior_xg",
    "poisson_prob",
]


@dataclass
class DatasetRow:
    match_id: int
    season: str
    team: str
    cutoff: int
    features: dict[str, float]
    label: bool


def build_dataset(
    conn: sqlite3.Connection,
    cutoff_minutes: list[int] = CUTOFF_MINUTES,
    blend: float = DEFAULT_BLEND,
) -> list[DatasetRow]:
    """Builds one row per (match, team, cutoff) across every match stored
    in the database, computing the full ML feature set (Task 2) plus the
    existing trailing-xG prior and Poisson prediction as two additional
    features, and the goal-in-next-15-minutes label."""
    rows: list[DatasetRow] = []
    matches = conn.execute(
        "SELECT match_id, home_team_id, away_team_id, league, season FROM matches"
    ).fetchall()

    for match_id, home_team_id, away_team_id, league, season in matches:
        shots = load_match_shots(conn, match_id, home_team_id, away_team_id)
        if not shots:
            continue
        for team, team_id in (("home", home_team_id), ("away", away_team_id)):
            prior = trailing_xg_per90(conn, team_id, league, season, match_id)
            for cutoff in cutoff_minutes:
                ml_features = compute_ml_features(shots, cutoff, team)

                state = compute_state_at_minute(shots, cutoff, window=RECENT_WINDOW_MINUTES)
                recent_xg = state.home_xg_last15 if team == "home" else state.away_xg_last15
                lam = dynamic_lambda(
                    pre_match_xg_per90=prior,
                    in_match_xg_recent=recent_xg,
                    recent_window_minutes=RECENT_WINDOW_MINUTES,
                    horizon_minutes=HORIZON_MINUTES,
                    blend=blend,
                )
                poisson_prob = prob_goal_in_window(lam)

                full_features = dict(ml_features)
                full_features["trailing_prior_xg"] = prior
                full_features["poisson_prob"] = poisson_prob

                label = goal_in_window(shots, cutoff, HORIZON_MINUTES, team)
                rows.append(
                    DatasetRow(
                        match_id=match_id,
                        season=season,
                        team=team,
                        cutoff=cutoff,
                        features=full_features,
                        label=label,
                    )
                )
    return rows


def split_by_season(
    rows: list[DatasetRow], test_season: str, validation_season: str
) -> tuple[list[DatasetRow], list[DatasetRow], list[DatasetRow]]:
    """Splits rows into (train, validation, test) by season: `test_season`
    is held out entirely for final evaluation, `validation_season` is used
    only for calibration fitting, and every other season is training data.
    Raises ValueError if `test_season == validation_season`."""
    if test_season == validation_season:
        raise ValueError("test_season and validation_season must differ")
    train = [r for r in rows if r.season not in (test_season, validation_season)]
    validation = [r for r in rows if r.season == validation_season]
    test = [r for r in rows if r.season == test_season]
    return train, validation, test


def rows_to_arrays(rows: list[DatasetRow]) -> tuple[list[list[float]], list[int]]:
    """Converts DatasetRows into a feature matrix (columns ordered per
    FEATURE_NAMES) and a label vector, ready for LightGBM."""
    X = [[r.features[name] for name in FEATURE_NAMES] for r in rows]
    y = [int(r.label) for r in rows]
    return X, y
