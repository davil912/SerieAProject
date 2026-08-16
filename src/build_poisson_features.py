"""
Fase 3 - Aggiunge le feature del modello Poisson al dataset di feature.

Per ogni stagione, il modello Poisson viene rifittato usando SOLO le ultime
5 stagioni precedenti (finestra mobile, non tutta la storia) - una squadra
di 15 anni fa non è indicativa della sua forza attuale. Stesso principio
walk-forward già usato altrove nel progetto: nessuna informazione futura
entra nel calcolo delle feature di una partita.

Uso:
    python src/build_poisson_features.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
from poisson_model import fit_poisson_model

BASE = Path(__file__).resolve().parent.parent
MATCHES_PATH = BASE / "data" / "processed" / "serieA_matches.csv"
FEATURES_PATH = BASE / "data" / "processed" / "serieA_features.csv"

MIN_SEASON = "2005/2006"
ROLLING_WINDOW_SEASONS = 5
ALPHA = 0.01


def main():
    matches = pd.read_csv(MATCHES_PATH, parse_dates=["date"])
    matches = matches[matches["season"] >= MIN_SEASON].sort_values("date").reset_index(drop=True)

    seasons = sorted(matches["season"].unique())
    all_rows = []

    for i, season in enumerate(seasons):
        prior_seasons = seasons[:i]
        window = prior_seasons[-ROLLING_WINDOW_SEASONS:]  # ultime (fino a) 5 stagioni precedenti

        season_matches = matches[matches["season"] == season].copy()

        if len(window) == 0:
            # prima stagione in assoluto nel dataset: nessuna storia disponibile
            season_matches["poisson_exp_goals_home"] = np.nan
            season_matches["poisson_exp_goals_away"] = np.nan
            season_matches["poisson_prob_home"] = np.nan
            season_matches["poisson_prob_draw"] = np.nan
            season_matches["poisson_prob_away"] = np.nan
        else:
            train_window_df = matches[matches["season"].isin(window)]
            model = fit_poisson_model(train_window_df, alpha=ALPHA)

            preds = season_matches.apply(
                lambda r: model.match_probs(r["home_team"], r["away_team"]), axis=1, result_type="expand"
            )
            preds.columns = ["poisson_exp_goals_home", "poisson_exp_goals_away",
                              "poisson_prob_home", "poisson_prob_draw", "poisson_prob_away"]
            season_matches = pd.concat([season_matches.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)

        print(f"  stagione {season}: finestra training = {window if window else '(nessuna - prima stagione)'}")
        all_rows.append(season_matches)

    poisson_df = pd.concat(all_rows, ignore_index=True)
    poisson_df["poisson_exp_goals_diff"] = poisson_df["poisson_exp_goals_home"] - poisson_df["poisson_exp_goals_away"]

    poisson_cols = ["date", "home_team", "away_team",
                     "poisson_exp_goals_home", "poisson_exp_goals_away", "poisson_exp_goals_diff",
                     "poisson_prob_home", "poisson_prob_draw", "poisson_prob_away"]

    # Merge con il file di feature esistente (Fase 2), su data+squadre (chiave univoca per partita)
    features = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
    features = features.drop(columns=[c for c in poisson_cols if c in features.columns and c not in ("date", "home_team", "away_team")])
    merged = features.merge(poisson_df[poisson_cols], on=["date", "home_team", "away_team"], how="left")

    print(f"\nPartite totali: {len(merged)}; con feature Poisson disponibili: {merged['poisson_prob_home'].notna().sum()}")
    merged.to_csv(FEATURES_PATH, index=False)
    print(f"Aggiornato: {FEATURES_PATH}")


if __name__ == "__main__":
    main()
