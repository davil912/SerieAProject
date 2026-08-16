"""
Fase 2 - Feature engineering.

Costruisce, per ogni partita, un set di feature calcolate SOLO con informazioni
disponibili prima del calcio d'inizio (nessun leakage temporale). Le feature
Elo e forma (già presenti nel dataset pulito) sono state verificate essere
pre-partita; qui vengono aggiunte: differenza Elo/forma, scontri diretti
(H2H), giorni di riposo, avanzamento di stagione, e un indicatore di
vantaggio-casa "di periodo" (che cattura il trend calante osservato in Fase 1).

Le quote di mercato (odd_home/draw/away) NON vengono usate come feature di
input al modello: vengono tenute da parte come benchmark esterno di
validazione (Fase 2 - modello baseline), per un confronto onesto.

Uso:
    python src/feature_builder.py
"""

from pathlib import Path
from collections import defaultdict, deque
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parent.parent
PROCESSED = BASE / "data" / "processed" / "serieA_matches.csv"
OUT_PATH = BASE / "data" / "processed" / "serieA_features.csv"

# Le prime stagioni hanno dati grezzi incompleti (vedi FASE1_REPORT.md) -> escluse.
MIN_SEASON = "2005/2006"

H2H_WINDOW = 5          # numero di scontri diretti precedenti da considerare
HOME_ADV_WINDOW = 200   # finestra mobile (partite) per il vantaggio-casa "di periodo"


def build_features(matches: pd.DataFrame) -> pd.DataFrame:
    df = matches[matches["season"] >= MIN_SEASON].copy()
    df = df.sort_values("date").reset_index(drop=True)

    # ---------- Feature vettoriali (già pronte colonna per colonna) ----------
    df["elo_diff"] = df["home_elo"] - df["away_elo"]
    df["form3_diff"] = df["home_form3"] - df["away_form3"]
    df["form5_diff"] = df["home_form5"] - df["away_form5"]

    # Vantaggio-casa "di periodo": % vittorie casa nelle ultime HOME_ADV_WINDOW
    # partite del campionato, calcolata SENZA includere la partita corrente
    # (shift(1) sull'expanding/rolling mean) -> cattura il trend calante nel tempo.
    is_home_win = (df["result"] == "H").astype(float)
    df["home_advantage_recent"] = (
        is_home_win.shift(1).rolling(HOME_ADV_WINDOW, min_periods=30).mean()
    )
    # fallback per le primissime partite del dataset dove la finestra è vuota
    df["home_advantage_recent"] = df["home_advantage_recent"].fillna(is_home_win.expanding().mean().shift(1))

    # ---------- Feature stateful (richiedono di scorrere le partite in ordine) ----------
    last_match_date = {}          # squadra -> ultima data in cui ha giocato
    games_played_season = defaultdict(int)  # (squadra, stagione) -> partite già giocate in stagione
    h2h_history = defaultdict(lambda: deque(maxlen=H2H_WINDOW))  # coppia squadre -> ultimi risultati

    rest_home, rest_away = [], []
    gp_home, gp_away = [], []
    h2h_home_ppg = []   # punti-a-partita della squadra di casa negli ultimi H2H_WINDOW scontri diretti
    h2h_n = []            # quanti scontri diretti disponibili (0-5)

    for row in df.itertuples(index=False):
        h, a, season, date = row.home_team, row.away_team, row.season, row.date

        # --- giorni di riposo ---
        rest_home.append((date - last_match_date[h]).days if h in last_match_date else np.nan)
        rest_away.append((date - last_match_date[a]).days if a in last_match_date else np.nan)

        # --- partite già giocate in stagione (avanzamento stagione) ---
        gp_home.append(games_played_season[(h, season)])
        gp_away.append(games_played_season[(a, season)])

        # --- scontri diretti (H2H): punti-a-partita della squadra di CASA attuale ---
        pair_key = tuple(sorted([h, a]))
        history = h2h_history[pair_key]
        if len(history) == 0:
            h2h_home_ppg.append(np.nan)
        else:
            pts = []
            for (past_home, past_result) in history:
                # ricalcolo i punti dal punto di vista della squadra `h` (casa oggi)
                if past_home == h:
                    pts.append(3 if past_result == "H" else 1 if past_result == "D" else 0)
                else:
                    pts.append(3 if past_result == "A" else 1 if past_result == "D" else 0)
            h2h_home_ppg.append(sum(pts) / len(pts))
        h2h_n.append(len(history))

        # --- aggiornamento stato per le partite successive ---
        last_match_date[h] = date
        last_match_date[a] = date
        games_played_season[(h, season)] += 1
        games_played_season[(a, season)] += 1
        h2h_history[pair_key].append((h, row.result))

    df["rest_days_home"] = rest_home
    df["rest_days_away"] = rest_away
    df["rest_days_diff"] = df["rest_days_home"] - df["rest_days_away"]
    df["games_played_home"] = gp_home
    df["games_played_away"] = gp_away
    df["season_progress"] = (df["games_played_home"] + df["games_played_away"]) / 2 / 38.0
    df["h2h_home_ppg"] = h2h_home_ppg
    df["h2h_n_precedenti"] = h2h_n

    # Odds -> probabilità implicite di mercato (tenute SOLO per il benchmark, non come feature di training)
    inv = 1 / df[["odd_home", "odd_draw", "odd_away"]]
    overround = inv.sum(axis=1)
    df["market_prob_home"] = inv["odd_home"] / overround
    df["market_prob_draw"] = inv["odd_draw"] / overround
    df["market_prob_away"] = inv["odd_away"] / overround

    return df


def main():
    matches = pd.read_csv(PROCESSED, parse_dates=["date"])
    feats = build_features(matches)

    feature_cols = [
        "elo_diff", "form3_diff", "form5_diff", "home_advantage_recent",
        "rest_days_home", "rest_days_away", "rest_days_diff",
        "games_played_home", "games_played_away", "season_progress",
        "h2h_home_ppg", "h2h_n_precedenti",
    ]
    print("Feature costruite:", feature_cols)
    print("\nValori mancanti per feature (sulle partite utilizzabili):")
    for c in feature_cols:
        print(f"  {c}: {feats[c].isna().mean()*100:.1f}%")

    feats.to_csv(OUT_PATH, index=False)
    print(f"\nSalvato: {OUT_PATH}  ({len(feats)} partite, {feats['season'].min()} -> {feats['season'].max()})")


if __name__ == "__main__":
    main()
