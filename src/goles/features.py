from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MatchState:
    minute: int
    home_score: int
    away_score: int
    home_xg_last15: float
    away_xg_last15: float
    home_shots_last15: int
    away_shots_last15: int


def compute_state_at_minute(
    shots: list[dict], cutoff_minute: int, window: int = 15
) -> MatchState:
    """Reconstruct match state as of `cutoff_minute`, using only shots with
    minute <= cutoff_minute (so a backtest never sees the future)."""
    home_score = sum(
        1 for s in shots if s["minute"] <= cutoff_minute and s["team"] == "home" and s["is_goal"]
    )
    away_score = sum(
        1 for s in shots if s["minute"] <= cutoff_minute and s["team"] == "away" and s["is_goal"]
    )

    window_start = cutoff_minute - window
    home_xg = sum(
        s["xg"] for s in shots if window_start < s["minute"] <= cutoff_minute and s["team"] == "home"
    )
    away_xg = sum(
        s["xg"] for s in shots if window_start < s["minute"] <= cutoff_minute and s["team"] == "away"
    )
    home_shots = sum(
        1 for s in shots if window_start < s["minute"] <= cutoff_minute and s["team"] == "home"
    )
    away_shots = sum(
        1 for s in shots if window_start < s["minute"] <= cutoff_minute and s["team"] == "away"
    )

    return MatchState(
        minute=cutoff_minute,
        home_score=home_score,
        away_score=away_score,
        home_xg_last15=home_xg,
        away_xg_last15=away_xg,
        home_shots_last15=home_shots,
        away_shots_last15=away_shots,
    )


def goal_in_window(shots: list[dict], cutoff_minute: int, horizon: int, team: str) -> bool:
    """Did `team` score a goal in (cutoff_minute, cutoff_minute + horizon]?
    Used only to *label* historical data for backtesting/training — never
    call this with information a live model wouldn't have yet."""
    return any(
        s["team"] == team and s["is_goal"] and cutoff_minute < s["minute"] <= cutoff_minute + horizon
        for s in shots
    )
