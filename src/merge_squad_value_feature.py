"""
Fase 4 (estensione) - Aggiunge il valore rosa (squadra di casa, trasferta,
differenza) come feature in serieA_features.csv, usando la tabella
stagionale prodotta da build_squad_value_history.py.

Le righe marcate "non affidabili" (poca copertura di valutazioni giocatori
per quella stagione, tipico delle stagioni piu' vecchie) vengono impostate
a NaN piuttosto che tenere un numero fuorviante: XGBoost gestisce i NaN
nativamente, la Logistic Regression li gestisce con l'imputazione gia'
presente nella sua pipeline (vedi train_baseline.py / train_ensemble.py).

Uso:
    python src/merge_squad_value_feature.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
FEATURES_PATH = BASE / "data" / "processed" / "serieA_features.csv"
SQUAD_VALUE_PATH = BASE / "data" / "processed" / "squad_values_by_season.csv"


def main():
    features = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
    squad_value = pd.read_csv(SQUAD_VALUE_PATH)
    squad_value.loc[~squad_value["affidabile"], "valore_rosa_milioni"] = np.nan
    sv = squad_value.set_index(["team", "season"])["valore_rosa_milioni"]

    features = features.drop(columns=[c for c in ["home_squad_value", "away_squad_value", "squad_value_diff"] if c in features.columns])

    features["home_squad_value"] = features.apply(lambda r: sv.get((r["home_team"], r["season"]), np.nan), axis=1)
    features["away_squad_value"] = features.apply(lambda r: sv.get((r["away_team"], r["season"]), np.nan), axis=1)
    features["squad_value_diff"] = features["home_squad_value"] - features["away_squad_value"]

    coverage = features["squad_value_diff"].notna().mean() * 100
    print(f"Copertura squad_value_diff sul dataset di feature completo: {coverage:.1f}%")
    coverage_recent = features[features["season"] >= "2018/2019"]["squad_value_diff"].notna().mean() * 100
    print(f"Copertura dal 2018/19 in poi: {coverage_recent:.1f}%")

    features.to_csv(FEATURES_PATH, index=False)
    print(f"Aggiornato: {FEATURES_PATH}")


if __name__ == "__main__":
    main()
