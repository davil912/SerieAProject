"""
Fase 3 - Modello XGBoost con backtest walk-forward, confrontato con:
  - Logistic Regression (baseline Fase 2)
  - probabilità di mercato (quote bookmaker)
  - baseline banale "vince sempre la casa"

Stesso schema di validazione della Fase 2 (train solo su stagioni passate,
mai split casuale) per un confronto onesto. In più: una versione CALIBRATA
delle probabilità XGBoost, ottenuta tenendo da parte l'ultima stagione del
training come set di calibrazione (mai la stagione di test).

Uso:
    python src/train_xgboost.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import accuracy_score, log_loss
import joblib

BASE = Path(__file__).resolve().parent.parent
FEATURES_PATH = BASE / "data" / "processed" / "serieA_features.csv"
MODELS_DIR = BASE / "models"

FEATURE_COLS = [
    "elo_diff", "home_elo", "away_elo",
    "form3_diff", "form5_diff",
    "home_advantage_recent",
    "rest_days_diff", "season_progress",
    "h2h_home_ppg", "h2h_n_precedenti",
    "poisson_exp_goals_home", "poisson_exp_goals_away", "poisson_exp_goals_diff",
    "poisson_prob_home", "poisson_prob_draw", "poisson_prob_away",
    "home_squad_value", "away_squad_value", "squad_value_diff",
]
CLASSES = ["A", "D", "H"]
FIRST_TEST_SEASON_INDEX = 5

LABEL_TO_INT = {"A": 0, "D": 1, "H": 2}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}


def prepare_xy(df: pd.DataFrame):
    X = df[FEATURE_COLS].copy()
    # XGBoost gestisce nativamente i NaN (li instrada durante lo split degli alberi):
    # non serve imputazione esplicita, a differenza della Logistic Regression.
    y = df["result"].map(LABEL_TO_INT)
    return X, y


def onehot_int(y_int: pd.Series) -> np.ndarray:
    oh = np.zeros((len(y_int), 3))
    oh[np.arange(len(y_int)), y_int.values] = 1
    return oh


def multiclass_brier(y_true_oh, y_prob):
    return float(np.mean(np.sum((y_prob - y_true_oh) ** 2, axis=1)))


def make_xgb():
    # Iperparametri scelti dopo un piccolo confronto manuale (non una grid search
    # esaustiva): con un dataset di questa dimensione (poche migliaia di partite)
    # e un target intrinsecamente rumoroso, alberi poco profondi e molta
    # regolarizzazione hanno dato risultati piu' stabili di configurazioni piu'
    # "aggressive". Vedi FASE3_REPORT.md per il confronto.
    return XGBClassifier(
        n_estimators=60, max_depth=2, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7,
        objective="multi:softprob", num_class=3,
        reg_lambda=5.0, eval_metric="mlogloss",
        n_jobs=4, random_state=42,
    )


def main():
    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
    seasons = sorted(df["season"].unique())
    test_seasons = seasons[FIRST_TEST_SEASON_INDEX:]

    results = []

    for test_season in test_seasons:
        train_full = df[df["season"] < test_season]
        test_df = df[df["season"] == test_season]

        train_full_seasons = sorted(train_full["season"].unique())
        calib_season = train_full_seasons[-1]          # ultima stagione di training -> set di calibrazione
        fit_seasons = train_full_seasons[:-1]            # il resto -> fit vero e proprio

        train_fit_df = train_full[train_full["season"].isin(fit_seasons)]
        calib_df = train_full[train_full["season"] == calib_season]

        X_fit, y_fit = prepare_xy(train_fit_df)
        X_calib, y_calib = prepare_xy(calib_df)
        X_full, y_full = prepare_xy(train_full)
        X_test, y_test = prepare_xy(test_df)
        y_test_oh = onehot_int(y_test)

        # --- XGBoost "raw": addestrato su TUTTO il training disponibile ---
        model_raw = make_xgb()
        model_raw.fit(X_full, y_full)
        proba_raw = model_raw.predict_proba(X_test)

        # --- XGBoost calibrato: fit su fit_seasons, calibrazione su calib_season ---
        model_for_calib = make_xgb()
        model_for_calib.fit(X_fit, y_fit)
        # sigmoid (Platt scaling) invece di isotonic: la calibrazione isotonica richiede
        # molti piu' campioni per classe per non overfittare - con una sola stagione
        # (~380 partite, 3 classi) produceva log-loss peggiore del modello non calibrato.
        calibrated = CalibratedClassifierCV(FrozenEstimator(model_for_calib), method="sigmoid")
        calibrated.fit(X_calib, y_calib)
        proba_calib = calibrated.predict_proba(X_test)
        proba_calib = proba_calib / proba_calib.sum(axis=1, keepdims=True)  # rinormalizza (arrotondamenti float)

        for name, proba in [("XGBoost", proba_raw), ("XGBoost_calibrato", proba_calib)]:
            y_pred = proba.argmax(axis=1)
            acc = accuracy_score(y_test, y_pred)
            ll = log_loss(y_test, proba, labels=[0, 1, 2])
            brier = multiclass_brier(y_test_oh, proba)
            results.append({"season": test_season, "modello": name, "accuracy": acc, "log_loss": ll, "brier": brier})

        # --- Benchmark: mercato e baseline banale (identici alla Fase 2, ricalcolati qui per il confronto diretto) ---
        market_df = test_df[["market_prob_away", "market_prob_draw", "market_prob_home"]].values
        missing = np.isnan(market_df).any(axis=1)
        if missing.any():
            market_df = market_df.copy()
            market_df[missing] = proba_raw[missing][:, [0, 1, 2]]
        acc = accuracy_score(y_test, market_df.argmax(axis=1))
        ll = log_loss(y_test, market_df, labels=[0, 1, 2])
        brier = multiclass_brier(y_test_oh, market_df)
        results.append({"season": test_season, "modello": "Quote_di_mercato", "accuracy": acc, "log_loss": ll, "brier": brier})

        const = np.tile([0.05, 0.05, 0.90], (len(test_df), 1))  # A, D, H
        acc = accuracy_score(y_test, const.argmax(axis=1))
        ll = log_loss(y_test, const, labels=[0, 1, 2])
        brier = multiclass_brier(y_test_oh, const)
        results.append({"season": test_season, "modello": "Baseline_vince_sempre_casa", "accuracy": acc, "log_loss": ll, "brier": brier})

    results_df = pd.DataFrame(results)
    summary = results_df.groupby("modello")[["accuracy", "log_loss", "brier"]].mean().round(4).sort_values("log_loss")

    print(f"Backtest walk-forward su {len(test_seasons)} stagioni ({test_seasons[0]} -> {test_seasons[-1]})\n")
    print("=== Metriche medie sul backtest ===")
    print(summary.to_string())

    # --- Feature importance dal modello XGBoost finale (addestrato su tutti i dati) ---
    X_all, y_all = prepare_xy(df)
    final_model = make_xgb()
    final_model.fit(X_all, y_all)
    importances = pd.Series(final_model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\n=== Feature importance (modello finale XGBoost) ===")
    print(importances.round(4).to_string())

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(final_model, MODELS_DIR / "xgboost_final.pkl")
    results_df.to_csv(BASE / "notebooks" / "backtest_results_fase3.csv", index=False)
    print(f"\nModello salvato in {MODELS_DIR / 'xgboost_final.pkl'}")
    print("Risultati backtest salvati in notebooks/backtest_results_fase3.csv")


if __name__ == "__main__":
    main()
