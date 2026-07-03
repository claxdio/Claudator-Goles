from __future__ import annotations

from goles.backtest import print_report, run_backtest
from goles.db import get_connection, init_db
from goles.loaders.understat import fetch_understat_shots, persist_shots, shots_to_records

LEAGUES = ["ENG-Premier League", "GER-Bundesliga"]
SEASONS = ["2324"]


def main() -> None:
    conn = get_connection()
    init_db(conn)

    print(f"Descargando datos de Understat para {LEAGUES} temporada {SEASONS}...")
    shots_df = fetch_understat_shots(LEAGUES, SEASONS)
    records = shots_to_records(shots_df)
    print(f"{len(records)} eventos de tiro descargados. Guardando en la base de datos...")
    persist_shots(conn, records)

    print("Corriendo backtest para el equipo local...")
    result = run_backtest(conn, team="home")
    print_report(result)


if __name__ == "__main__":
    main()
