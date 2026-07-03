import math

import pytest

from goles.model import dynamic_lambda, prob_goal_in_window


def test_dynamic_lambda_pure_prior_when_blend_zero():
    lam = dynamic_lambda(
        pre_match_xg_per90=1.8, in_match_xg_recent=5.0,
        recent_window_minutes=15, horizon_minutes=15, blend=0.0,
    )
    assert abs(lam - (1.8 / 90 * 15)) < 1e-9


def test_dynamic_lambda_pure_observed_when_blend_one():
    lam = dynamic_lambda(
        pre_match_xg_per90=1.8, in_match_xg_recent=0.6,
        recent_window_minutes=15, horizon_minutes=15, blend=1.0,
    )
    assert abs(lam - 0.6) < 1e-9


def test_dynamic_lambda_rejects_non_positive_window():
    with pytest.raises(ValueError):
        dynamic_lambda(1.8, 0.5, recent_window_minutes=0, horizon_minutes=15)


def test_prob_goal_in_window_matches_poisson_formula():
    assert abs(prob_goal_in_window(0.0) - 0.0) < 1e-9
    lam = 0.4
    assert abs(prob_goal_in_window(lam) - (1 - math.exp(-lam))) < 1e-9


def test_prob_goal_in_window_rejects_negative_expected_goals():
    with pytest.raises(ValueError):
        prob_goal_in_window(-0.1)
