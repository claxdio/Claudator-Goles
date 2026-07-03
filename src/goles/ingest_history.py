from __future__ import annotations

from goles.db import get_connection, init_db
from goles.loaders.understat import fetch_understat_shots, persist_shots, shots_to_records

LEAGUES = ["ENG-Premier League", "GER-Bundesliga"]
# Six seasons gives ~4,100 total matches across both leagues -- enough that
# a held-out test season and a held-out validation season each still leave
# a training split comfortably above the ~1,500-match threshold below which
# gradient-boosted trees tend to memorize rather than generalize (see this
# plan's Global Constraints).
SEASONS = ["1819", "1920", "2021", "2122", "2223", "2324"]


def main() -> None:
    conn = get_connection()
    init_db(conn)

    print(f"Descargando datos de Understat para {LEAGUES} temporadas {SEASONS}...")
    print("La primera corrida sin cache puede tardar bastante (~1 partido/seg).")
    shots_df = fetch_understat_shots(LEAGUES, SEASONS)
    records = shots_to_records(shots_df)
    print(f"{len(records)} eventos de tiro descargados. Guardando en la base de datos...")
    persist_shots(conn, records)
    print("Listo.")


if __name__ == "__main__":
    main()
