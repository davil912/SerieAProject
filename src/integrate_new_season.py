"""
Fase 4 (estensione) - Integra una stagione scaricata manualmente dal formato
originale Football-Data.co.uk (colonne Div,Date,Time,HomeTeam,...,FTHG,FTAG,
FTR,...) nello storico pulito del progetto (data/processed/serieA_matches.csv).

A differenza delle stagioni precedenti (arrivate gia' complete di Elo/forma
dal mirror GitHub), qui Elo e forma NON sono forniti dalla fonte: vengono
calcolati da noi (src/elo_updater.py) a partire dall'ultimo valore noto per
ciascuna squadra (Serie A o, se piu' recente, Serie B - utile per le
neopromosse). Vedi FASE4_REPORT.md per la spiegazione completa.

IDEMPOTENTE (Fase 6): pensato per essere rilanciato piu' volte durante una
STESSA stagione in corso (es. ogni settimana, con il file "stagione ad oggi"
scaricato di nuovo da football-data.co.uk, che cresce partita dopo partita).
Le partite gia' presenti (stessa stagione, stessa coppia home_team/away_team -
in un girone all'italiana una coppia gioca in casa una volta sola a stagione)
vengono riconosciute e saltate: si integrano SOLO le partite davvero nuove,
con Elo/forma che ripartono correttamente dall'ultimo stato gia' salvato
(niente doppio conteggio, niente Elo "ricominciato da capo" ogni settimana).

Uso:
    python src/integrate_new_season.py <path_csv_football-data> <stagione es. 2026/2027>
"""

import sys
from pathlib import Path
from collections import deque
import pandas as pd
from elo_updater import seed_ratings, update_ratings

BASE = Path(__file__).resolve().parent.parent
PROCESSED_A = BASE / "data" / "processed" / "serieA_matches.csv"
PROCESSED_B = BASE / "data" / "processed" / "serieB_matches.csv"

RENAME_MAP = {
    "Date": "date", "Time": "time", "HomeTeam": "home_team", "AwayTeam": "away_team",
    "FTHG": "home_goals", "FTAG": "away_goals", "FTR": "result",
    "HTHG": "home_goals_ht", "HTAG": "away_goals_ht", "HTR": "result_ht",
    "HS": "home_shots", "AS": "away_shots", "HST": "home_shots_target", "AST": "away_shots_target",
    "HF": "home_fouls", "AF": "away_fouls", "HC": "home_corners", "AC": "away_corners",
    "HY": "home_yellow", "AY": "away_yellow", "HR": "home_red", "AR": "away_red",
    "B365H": "odd_home", "B365D": "odd_draw", "B365A": "odd_away",
    "B365>2.5": "odd_over25", "B365<2.5": "odd_under25",
}


def points(result: str, is_home: bool) -> int:
    if result == "D":
        return 1
    if (result == "H") == is_home:
        return 3
    return 0


def main():
    if len(sys.argv) != 3:
        print("Uso: python src/integrate_new_season.py <path_csv> <stagione es. 2025/2026>")
        sys.exit(1)
    csv_path, season = sys.argv[1], sys.argv[2]

    new = pd.read_csv(csv_path)
    new = new[list(RENAME_MAP.keys())].rename(columns=RENAME_MAP)
    new["date"] = pd.to_datetime(new["date"], dayfirst=True)
    new = new.sort_values("date").reset_index(drop=True)
    new["season"] = season
    new["league"] = "Serie A"

    serieA_hist = pd.read_csv(PROCESSED_A, parse_dates=["date"])
    serieB_hist = pd.read_csv(PROCESSED_B, parse_dates=["date"])

    # --- Idempotenza (Fase 6): tieni solo le partite di `season` non ancora presenti ---
    already = serieA_hist[serieA_hist["season"] == season]
    already_keys = set(zip(already["home_team"], already["away_team"]))
    n_before = len(new)
    new = new[~new.apply(lambda r: (r["home_team"], r["away_team"]) in already_keys, axis=1)].reset_index(drop=True)
    if len(new) < n_before:
        print(f"{n_before - len(new)} partite di {season} erano gia' presenti in {PROCESSED_A.name} "
              f"(script idempotente): vengono integrate solo le {len(new)} nuove.\n")
    if len(new) == 0:
        print("Nessuna nuova partita da integrare. Nulla da fare.")
        return

    teams = sorted(set(new["home_team"]) | set(new["away_team"]))

    # --- Elo: seed dall'ultimo valore noto (Serie A o Serie B), poi aggiornato partita per partita ---
    elo = seed_ratings(serieA_hist, serieB_hist, teams)
    print("Elo di partenza (seed):")
    for t in teams:
        print(f"  {t:15s} {elo[t]:.1f}")

    # --- Forma: seed dagli ultimi (fino a) 5 risultati di ciascuna squadra, Serie A+B combinate ---
    combined_hist = pd.concat([serieA_hist, serieB_hist], ignore_index=True).sort_values("date")
    form_hist = {t: deque(maxlen=5) for t in teams}
    for t in teams:
        t_matches = combined_hist[(combined_hist["home_team"] == t) | (combined_hist["away_team"] == t)].tail(5)
        for r in t_matches.itertuples(index=False):
            is_home = r.home_team == t
            form_hist[t].append(points(r.result, is_home))

    home_elo_col, away_elo_col = [], []
    home_f3_col, away_f3_col, home_f5_col, away_f5_col = [], [], [], []

    for r in new.itertuples(index=False):
        h, a = r.home_team, r.away_team

        # feature PRE-partita (stato prima di processare il risultato corrente)
        home_elo_col.append(elo[h])
        away_elo_col.append(elo[a])
        home_f3_col.append(sum(list(form_hist[h])[-3:]))
        away_f3_col.append(sum(list(form_hist[a])[-3:]))
        home_f5_col.append(sum(form_hist[h]))
        away_f5_col.append(sum(form_hist[a]))

        # aggiornamento stato DOPO la partita
        elo[h], elo[a] = update_ratings(elo[h], elo[a], r.result)
        form_hist[h].append(points(r.result, True))
        form_hist[a].append(points(r.result, False))

    new["home_elo"] = home_elo_col
    new["away_elo"] = away_elo_col
    new["home_form3"] = home_f3_col
    new["away_form3"] = away_f3_col
    new["home_form5"] = home_f5_col
    new["away_form5"] = away_f5_col

    # allinea le colonne a quelle del dataset esistente
    for col in serieA_hist.columns:
        if col not in new.columns:
            new[col] = pd.NA
    new = new[serieA_hist.columns]

    updated = pd.concat([serieA_hist, new], ignore_index=True).sort_values("date").reset_index(drop=True)
    updated.to_csv(PROCESSED_A, index=False)
    print(f"\nAggiunte {len(new)} partite (stagione {season}) a {PROCESSED_A}")
    print("Elo finale a fine stagione integrata:")
    for t in teams:
        print(f"  {t:15s} {elo[t]:.1f}")


if __name__ == "__main__":
    main()
