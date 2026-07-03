from goles.features import compute_state_at_minute, goal_in_window

SAMPLE_SHOTS = [
    {"minute": 10, "team": "home", "xg": 0.10, "is_goal": False},
    {"minute": 34, "team": "home", "xg": 0.35, "is_goal": True},
    {"minute": 50, "team": "away", "xg": 0.20, "is_goal": False},
    {"minute": 63, "team": "away", "xg": 0.55, "is_goal": True},
    {"minute": 70, "team": "home", "xg": 0.18, "is_goal": False},
    {"minute": 74, "team": "home", "xg": 0.60, "is_goal": True},
]


def test_compute_state_at_minute_counts_goals_scored_so_far():
    state = compute_state_at_minute(SAMPLE_SHOTS, cutoff_minute=65, window=15)
    assert state.home_score == 1
    assert state.away_score == 1


def test_compute_state_at_minute_rolling_window_only_includes_recent_shots():
    state = compute_state_at_minute(SAMPLE_SHOTS, cutoff_minute=65, window=15)
    # window is (50, 65]; only the away shot at minute 63 falls inside it
    assert state.away_shots_last15 == 1
    assert abs(state.away_xg_last15 - 0.55) < 1e-9
    assert state.home_shots_last15 == 0
    assert abs(state.home_xg_last15 - 0.0) < 1e-9


def test_goal_in_window_detects_future_goal_for_team():
    assert goal_in_window(SAMPLE_SHOTS, cutoff_minute=65, horizon=10, team="home") is True
    assert goal_in_window(SAMPLE_SHOTS, cutoff_minute=65, horizon=10, team="away") is False


def test_goal_in_window_excludes_goals_outside_horizon():
    # the home goal at minute 34 is outside the (20, 30] window
    assert goal_in_window(SAMPLE_SHOTS, cutoff_minute=20, horizon=10, team="home") is False
