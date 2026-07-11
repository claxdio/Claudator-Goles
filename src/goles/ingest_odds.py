from __future__ import annotations

from goles.db import get_connection, init_db
from goles.loaders.football_data import LEAGUE_CODES, fetch_odds, persist_odds

SEASONS = ["1819", "1920", "2021", "2122", "2223", "2324"]


def main() -> None:
    conn = get_connection()
    init_db(conn)

    print(f"Descargando cuotas de football-data.co.uk para {list(LEAGUE_CODES.keys())} temporadas {SEASONS}...")
    odds_df = fetch_odds(LEAGUE_CODES, SEASONS)
    print(f"{len(odds_df)} filas de cuotas descargadas. Emparejando contra partidos existentes...")

    matched, unmatched = persist_odds(conn, odds_df)
    total = matched + unmatched
    coverage = matched / total if total else 0.0
    print(f"Emparejados: {matched}/{total} ({coverage:.1%}). Sin emparejar: {unmatched}.")
    if coverage < 0.95:
        print(
            "ADVERTENCIA: cobertura por debajo del 95%. Revisar TEAM_NAME_ALIASES antes de "
            "usar estas features -- puede haber un equipo nuevo sin mapear."
        )


if __name__ == "__main__":
    main()
