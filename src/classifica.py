"""
Fase 1 - Modulo Classifica.

Calcola la classifica di una stagione di Serie A a partire dallo storico
partite pulito (data/processed/serieA_matches.csv), con la possibilità di
fermarsi a una certa data (utile per calcolare la classifica "al momento
della partita X", propedeutico al feature engineering).

Regole: 3 punti vittoria, 1 pareggio, 0 sconfitta. Ordinamento per punti,
poi differenza reti, poi gol fatti (semplificato: la regola ufficiale della
Lega Serie A prevede anche gli scontri diretti in caso di parità in due
squadre, non implementati qui in Fase 1 - nota per iterazioni future).

Uso:
    python src/classifica.py --season 2023/2024
"""

import argparse
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
PROCESSED = BASE / "data" / "processed" / "serieA_matches.csv"


def calcola_classifica(matches: pd.DataFrame, season: str, upto_date: str | None = None) -> pd.DataFrame:
    """Calcola la classifica per una stagione, opzionalmente fino a una data inclusa."""
    m = matches[matches["season"] == season].copy()
    if upto_date is not None:
        m = m[m["date"] <= pd.Timestamp(upto_date)]
    if m.empty:
        raise ValueError(f"Nessuna partita trovata per la stagione {season} (upto_date={upto_date})")

    teams = sorted(set(m["home_team"]) | set(m["away_team"]))
    stats = {t: {"PG": 0, "V": 0, "N": 0, "P": 0, "GF": 0, "GS": 0} for t in teams}

    for _, row in m.iterrows():
        h, a = row["home_team"], row["away_team"]
        hg, ag = int(row["home_goals"]), int(row["away_goals"])

        stats[h]["PG"] += 1
        stats[a]["PG"] += 1
        stats[h]["GF"] += hg
        stats[h]["GS"] += ag
        stats[a]["GF"] += ag
        stats[a]["GS"] += hg

        if hg > ag:
            stats[h]["V"] += 1
            stats[a]["P"] += 1
        elif hg < ag:
            stats[a]["V"] += 1
            stats[h]["P"] += 1
        else:
            stats[h]["N"] += 1
            stats[a]["N"] += 1

    rows = []
    for t, s in stats.items():
        punti = s["V"] * 3 + s["N"]
        diff = s["GF"] - s["GS"]
        rows.append({"Squadra": t, "Punti": punti, "PG": s["PG"], "V": s["V"], "N": s["N"], "P": s["P"],
                      "GF": s["GF"], "GS": s["GS"], "DR": diff})

    classifica = pd.DataFrame(rows).sort_values(
        by=["Punti", "DR", "GF"], ascending=False
    ).reset_index(drop=True)
    classifica.index += 1
    classifica.index.name = "Pos"
    return classifica


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", default="2023/2024", help="Es. 2023/2024")
    parser.add_argument("--upto-date", default=None, help="Formato YYYY-MM-DD, opzionale")
    args = parser.parse_args()

    matches = pd.read_csv(PROCESSED, parse_dates=["date"])
    classifica = calcola_classifica(matches, args.season, args.upto_date)
    pd.set_option("display.width", 120)
    print(f"\nClassifica Serie A {args.season}" + (f" (fino al {args.upto_date})" if args.upto_date else ""))
    print(classifica.to_string())


if __name__ == "__main__":
    main()
