from goles.betfair.odds_store import get_connection, init_db, persist_snapshot


def test_init_db_creates_odds_snapshots_table():
    conn = get_connection(":memory:")
    init_db(conn)
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cur.fetchall()}
    assert "odds_snapshots" in tables


def test_persist_snapshot_inserts_a_row():
    conn = get_connection(":memory:")
    init_db(conn)
    persist_snapshot(
        conn,
        fetched_at="2026-07-11T12:00:00+00:00",
        betfair_event_id="e1",
        home_team="Arsenal",
        away_team="Chelsea",
        market_type="MATCH_ODDS",
        raw_json="{}",
        home_wp=0.5,
        draw_wp=0.3,
        away_wp=0.2,
    )
    row = conn.execute(
        "SELECT home_team, away_team, market_type, home_wp, draw_wp, away_wp, over_wp FROM odds_snapshots"
    ).fetchone()
    assert row == ("Arsenal", "Chelsea", "MATCH_ODDS", 0.5, 0.3, 0.2, None)


def test_persist_snapshot_allows_multiple_rows_for_same_event():
    conn = get_connection(":memory:")
    init_db(conn)
    persist_snapshot(conn, "2026-07-11T12:00:00+00:00", "e1", "Arsenal", "Chelsea", "MATCH_ODDS", "{}")
    persist_snapshot(conn, "2026-07-11T12:01:00+00:00", "e1", "Arsenal", "Chelsea", "MATCH_ODDS", "{}")
    count = conn.execute("SELECT COUNT(*) FROM odds_snapshots").fetchone()[0]
    assert count == 2
