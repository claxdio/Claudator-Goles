from goles.db import get_connection, init_db, get_or_create_team


def test_init_db_creates_expected_tables():
    conn = get_connection(":memory:")
    init_db(conn)
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cur.fetchall()}
    assert {"teams", "matches", "shots", "elo_ratings"} <= tables


def test_get_or_create_team_is_idempotent():
    conn = get_connection(":memory:")
    init_db(conn)
    id1 = get_or_create_team(conn, "Arsenal")
    id2 = get_or_create_team(conn, "Arsenal")
    assert id1 == id2
    id3 = get_or_create_team(conn, "Chelsea")
    assert id3 != id1
