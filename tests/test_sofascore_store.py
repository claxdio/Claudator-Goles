from goles.sofascore.store import get_connection, init_db, persist_card, persist_shot


def test_init_db_creates_shots_and_cards_tables():
    conn = get_connection(":memory:")
    init_db(conn)
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cur.fetchall()}
    assert {"shots", "cards"} <= tables


def test_persist_shot_inserts_a_row_and_returns_true():
    conn = get_connection(":memory:")
    init_db(conn)
    inserted = persist_shot(
        conn,
        sofascore_shot_id=7684954,
        sofascore_event_id=12813015,
        fetched_at="2026-07-12T00:00:00+00:00",
        home_team="Arsenal",
        away_team="Chelsea",
        team="home",
        minute=20,
        xg=0.185,
        is_goal=True,
        shot_type="goal",
        situation="corner",
        location_x=5.0,
        location_y=44.1,
        body_part="head",
    )
    assert inserted is True
    row = conn.execute(
        "SELECT home_team, away_team, team, minute, xg, is_goal, shot_type, situation FROM shots"
    ).fetchone()
    assert row == ("Arsenal", "Chelsea", "home", 20, 0.185, 1, "goal", "corner")


def test_persist_shot_is_idempotent_on_sofascore_shot_id():
    conn = get_connection(":memory:")
    init_db(conn)
    kwargs = dict(
        sofascore_shot_id=7684954,
        sofascore_event_id=12813015,
        fetched_at="2026-07-12T00:00:00+00:00",
        home_team="Arsenal",
        away_team="Chelsea",
        team="home",
        minute=20,
        xg=0.185,
        is_goal=True,
        shot_type="goal",
    )
    first = persist_shot(conn, **kwargs)
    second = persist_shot(conn, **kwargs)
    assert first is True
    assert second is False
    count = conn.execute("SELECT COUNT(*) FROM shots").fetchone()[0]
    assert count == 1


def test_persist_card_inserts_a_row_and_returns_true():
    conn = get_connection(":memory:")
    init_db(conn)
    inserted = persist_card(
        conn, sofascore_event_id=12813015, fetched_at="2026-07-12T00:00:00+00:00",
        home_team="Arsenal", away_team="Chelsea", team="away", minute=55, card_type="red",
    )
    assert inserted is True
    row = conn.execute("SELECT home_team, away_team, team, minute, card_type FROM cards").fetchone()
    assert row == ("Arsenal", "Chelsea", "away", 55, "red")


def test_persist_card_is_idempotent_on_event_team_minute():
    conn = get_connection(":memory:")
    init_db(conn)
    kwargs = dict(
        sofascore_event_id=12813015, fetched_at="2026-07-12T00:00:00+00:00",
        home_team="Arsenal", away_team="Chelsea", team="away", minute=55, card_type="red",
    )
    first = persist_card(conn, **kwargs)
    second = persist_card(conn, **kwargs)
    assert first is True
    assert second is False
    count = conn.execute("SELECT COUNT(*) FROM cards").fetchone()[0]
    assert count == 1
