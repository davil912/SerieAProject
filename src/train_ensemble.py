"""
Fase 4 - Ensemble Logistic Regression + XGBoost.

Anziche' scegliere un vincitore netto tra i due modelli (che in Fase 3 si sono
rivelati molto vicini), si combinano le probabilita' con una media pesata:

    p_ensemble = w * p_logreg + (1 - w) * p_xgboost

Per scegliere il peso w in modo onesto (senza "spiare" tutto il backtest), le
15 stagioni di test vengono divise in due blocchi CRONOLOGICI:
  - stagioni di SELEZIONE (le prime 10, 2010/11-2019/20): usate per scegliere w
  - stagioni di HOLDOUT (le ultime 5, 2020/21-2024/25): mai usate per scegliere
    w, servono solo per valutare la scelta finale in modo imparziale

Uso:
    python src/train_ensemble.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
from xgboost import XGBClassifier

BASE = Path(__file__).resolve().parent.parent
FEATURES_PATH = BASE / "data" / "processed" / "serieA_features.csv"

LOGREG_FEATURES = [
    "elo_diff", "home_elo", "away_elo", "form3_diff", "form5_diff",
    "home_advantage_recent", "rest_days_diff", "season_progress",
    "h2h_home_ppg", "h2h_n_precedenti",
]
XGB_FEATURES = LOGREG_FEATURES + [
    "poisson_exp_goals_home", "poisson_exp_goals_away", "poisson_exp_goals_diff",
    "poisson_prob_home", "poisson_prob_draw", "poisson_prob_away",
]
CLASSES = ["A", "D", "H"]
LABEL_TO_INT = {"A": 0, "D": 1, "H": 2}
FIRST_TEST_SEASON_INDEX = 5
N_SELECTION_SEASONS = 10  # le prime 10 (su 15) stagioni di test -> scelta del peso


def prepare_logreg_xy(df):
    X = df[LOGREG_FEATURES].copy()
    X["h2h_home_ppg"] = X["h2h_home_ppg"].fillna(X["h2h_home_ppg"].median())
    X["h2h_home_ppg"] = X["h2h_home_ppg"].fillna(1.5)
    X["rest_days_diff"] = X["rest_days_diff"].fillna(0)
    X["home_advantage_recent"] = X["home_advantage_recent"].fillna(0.45)
    X = X.fillna(0)
    y = df["result"]
    return X, y


def make_xgb():
    return XGBClassifier(
        n_estimators=60, max_depth=2, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7,
        objective="multi:softprob", num_class=3,
        reg_lambda=5.0, eval_metric="mlogloss",
        n_jobs=4, random_state=42,
    )


def multiclass_brier(y_true_oh, y_prob):
    return float(np.mean(np.sum((y_prob - y_true_oh) ** 2, axis=1)))


def main():
    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
    seasons = sorted(df["season"].unique())
    test_seasons = seasons[FIRST_TEST_SEASON_INDEX:]
    selection_seasons = test_seasons[:N_SELECTION_SEASONS]
    holdout_seasons = test_seasons[N_SELECTION_SEASONS:]

    per_season_proba = {}  # season -> (y_true_int, proba_logreg, proba_xgb)

    for test_season in test_seasons:
        train_df = df[df["season"] < test_season]
        test_df = df[df["season"] == test_season]

        # --- Logistic Regression ---
        X_tr, y_tr = prepare_logreg_xy(train_df)
        X_te, y_te = prepare_logreg_xy(test_df)
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))])
        pipe.fit(X_tr, y_tr)
        proba_lr = pd.DataFrame(pipe.predict_proba(X_te), columns=pipe.classes_)[CLASSES].values

        # --- XGBoost ---
        X_tr_x = train_df[XGB_FEATURES]
        y_tr_x = train_df["result"].map(LABEL_TO_INT)
        X_te_x = test_df[XGB_FEATURES]
        xgb = make_xgb()
        xgb.fit(X_tr_x, y_tr_x)
        proba_xgb_int = xgb.predict_proba(X_te_x)  # colonne in ordine [A, D, H] = [0,1,2]

        y_true_int = test_df["result"].map(LABEL_TO_INT).values
        per_season_proba[test_season] = (y_true_int, proba_lr, proba_xgb_int)

    def eval_weight(w, seasons_subset):
        accs, lls, briers = [], [], []
        for s in seasons_subset:
            y_true_int, p_lr, p_xgb = per_season_proba[s]
            p_ens = w * p_lr + (1 - w) * p_xgb
            y_oh = np.zeros((len(y_true_int), 3))
            y_oh[np.arange(len(y_true_int)), y_true_int] = 1
            accs.append(accuracy_score(y_true_int, p_ens.argmax(axis=1)))
            lls.append(log_loss(y_true_int, p_ens, labels=[0, 1, 2]))
            briers.append(multiclass_brier(y_oh, p_ens))
        return np.mean(accs), np.mean(lls), np.mean(briers)

    print(f"=== Scelta del peso w su {len(selection_seasons)} stagioni di SELEZIONE ({selection_seasons[0]} -> {selection_seasons[-1]}) ===")
    grid = np.arange(0.0, 1.01, 0.1)
    grid_results = []
    for w in grid:
        acc, ll, brier = eval_weight(w, selection_seasons)
        grid_results.append((w, acc, ll, brier))
        print(f"  w_logreg={w:.1f} -> accuracy={acc:.4f}  log_loss={ll:.4f}  brier={brier:.4f}")

    best_w = min(grid_results, key=lambda t: t[2])[0]
    print(f"\nPeso scelto (log-loss minimo su selezione): w_logreg={best_w:.1f}, w_xgboost={1-best_w:.1f}")

    print(f"\n=== Valutazione IMPARZIALE su {len(holdout_seasons)} stagioni di HOLDOUT ({holdout_seasons[0]} -> {holdout_seasons[-1]}), mai viste nella scelta di w ===")
    for name, w in [("Logistic Regression (w=1.0)", 1.0), ("XGBoost (w=0.0)", 0.0), (f"Ensemble (w={best_w:.1f})", best_w)]:
        acc, ll, brier = eval_weight(w, holdout_seasons)
        print(f"  {name:30s} accuracy={acc:.4f}  log_loss={ll:.4f}  brier={brier:.4f}")

    print(f"\n=== Media sulle 15 stagioni totali (per confronto diretto con Fase 2/3) ===")
    all_results = []
    for name, w in [("LogisticRegression", 1.0), ("XGBoost", 0.0), ("Ensemble", best_w)]:
        acc, ll, brier = eval_weight(w, test_seasons)
        all_results.append({"modello": name, "accuracy": acc, "log_loss": ll, "brier": brier})
        print(f"  {name:20s} accuracy={acc:.4f}  log_loss={ll:.4f}  brier={brier:.4f}")

    pd.DataFrame(all_results).to_csv(BASE / "notebooks" / "ensemble_results.csv", index=False)
    pd.DataFrame(grid_results, columns=["w_logreg", "accuracy", "log_loss", "brier"]).to_csv(
        BASE / "notebooks" / "ensemble_weight_search.csv", index=False)
    print(f"\nSalvato notebooks/ensemble_results.csv e notebooks/ensemble_weight_search.csv")


if __name__ == "__main__":
    main()
