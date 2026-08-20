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
    "squad_value_diff", "squad_value_log_ratio",
]
XGB_EXTRA_FEATURES = [
    "poisson_exp_goals_home", "poisson_exp_goals_away", "poisson_exp_goals_diff",
    "poisson_prob_home", "poisson_prob_draw", "poisson_prob_away",
    "home_squad_value", "away_squad_value", "squad_value_diff", "squad_value_log_ratio",
]
XGB_FEATURES = [c for c in LOGREG_FEATURES if c not in ("squad_value_diff", "squad_value_log_ratio")] + XGB_EXTRA_FEATURES
SQUAD_VALUE_COLS = {"home_squad_value", "away_squad_value", "squad_value_diff", "squad_value_log_ratio"}
# Fase 4d: moltiplicatore valore-rosa gia' scelto (vedi FASE4d_VALORE_ROSA_PESO.md) - qui
# fisso, non ri-cercato, per tenere la nuova griglia (Fase 4e, sotto) gestibile.
SQUAD_VALUE_MULTIPLIER = 2.0
# Fase 4e: quanto "pesare di piu'" le stagioni recenti nel training - richiesto dall'utente.
# None = pesi uniformi (comportamento fino a Fase 4d). Altri valori = half-life in stagioni
# (il peso si dimezza ogni N stagioni indietro). Candidati confrontati sulle sole stagioni
# di SELEZIONE, mai sull'holdout (stessa logica gia' usata per w e per il valore rosa).
SEASON_HALF_LIFE_CANDIDATES = [None, 10, 6, 3]
CLASSES = ["A", "D", "H"]
LABEL_TO_INT = {"A": 0, "D": 1, "H": 2}
FIRST_TEST_SEASON_INDEX = 5
N_SELECTION_SEASONS = 10  # le prime 10 (su 15) stagioni di test -> scelta del peso


def feature_weights_vector(multiplier):
    return np.array([multiplier if c in SQUAD_VALUE_COLS else 1.0 for c in XGB_FEATURES])


def season_sample_weights(df: pd.DataFrame, half_life) -> np.ndarray:
    """Peso esponenziale decrescente per stagione (vedi train_baseline.py per i dettagli)."""
    if half_life is None:
        return np.ones(len(df))
    train_seasons_sorted = sorted(df["season"].unique())
    most_recent_idx = len(train_seasons_sorted) - 1
    season_to_idx = {s: i for i, s in enumerate(train_seasons_sorted)}
    seasons_ago = most_recent_idx - df["season"].map(season_to_idx)
    return 0.5 ** (seasons_ago / half_life)


def prepare_logreg_xy(df):
    X = df[LOGREG_FEATURES].copy()
    X["h2h_home_ppg"] = X["h2h_home_ppg"].fillna(X["h2h_home_ppg"].median())
    X["h2h_home_ppg"] = X["h2h_home_ppg"].fillna(1.5)
    X["rest_days_diff"] = X["rest_days_diff"].fillna(0)
    X["home_advantage_recent"] = X["home_advantage_recent"].fillna(0.45)
    X = X.fillna(0)
    y = df["result"]
    return X, y


def make_xgb(feature_weights=None):
    kwargs = {}
    if feature_weights is not None:
        kwargs["feature_weights"] = feature_weights  # costruttore, non fit(): evita warning di deprecazione
    return XGBClassifier(
        n_estimators=60, max_depth=2, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7,
        objective="multi:softprob", num_class=3,
        reg_lambda=5.0, eval_metric="mlogloss",
        n_jobs=4, random_state=42,
        **kwargs,
    )


def multiclass_brier(y_true_oh, y_prob):
    return float(np.mean(np.sum((y_prob - y_true_oh) ** 2, axis=1)))


def main():
    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
    seasons = sorted(df["season"].unique())
    test_seasons = seasons[FIRST_TEST_SEASON_INDEX:]
    selection_seasons = test_seasons[:N_SELECTION_SEASONS]
    holdout_seasons = test_seasons[N_SELECTION_SEASONS:]

    # --- Logistic Regression + XGBoost: rifit per ogni candidato di half-life (Fase 4e),
    # moltiplicatore valore-rosa fisso a SQUAD_VALUE_MULTIPLIER (gia' scelto in Fase 4d) ---
    fw = feature_weights_vector(SQUAD_VALUE_MULTIPLIER)
    proba_lr_by_hl_season = {}
    proba_xgb_by_hl_season = {}
    y_true_by_season = {}
    for hl in SEASON_HALF_LIFE_CANDIDATES:
        proba_lr_by_hl_season[hl] = {}
        proba_xgb_by_hl_season[hl] = {}
        for test_season in test_seasons:
            train_df = df[df["season"] < test_season]
            test_df = df[df["season"] == test_season]
            w_train = season_sample_weights(train_df, hl)

            X_tr, y_tr = prepare_logreg_xy(train_df)
            X_te, y_te = prepare_logreg_xy(test_df)
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))])
            pipe.fit(X_tr, y_tr, clf__sample_weight=w_train)
            proba_lr_by_hl_season[hl][test_season] = pd.DataFrame(pipe.predict_proba(X_te), columns=pipe.classes_)[CLASSES].values

            X_tr_x = train_df[XGB_FEATURES]
            y_tr_x = train_df["result"].map(LABEL_TO_INT)
            X_te_x = test_df[XGB_FEATURES]
            xgb = make_xgb(fw)
            xgb.fit(X_tr_x, y_tr_x, sample_weight=w_train)
            proba_xgb_by_hl_season[hl][test_season] = xgb.predict_proba(X_te_x)  # colonne [A, D, H]

            if hl == SEASON_HALF_LIFE_CANDIDATES[0]:
                y_true_by_season[test_season] = test_df["result"].map(LABEL_TO_INT).values

    def eval_combo(w, hl, seasons_subset):
        accs, lls, briers = [], [], []
        for s in seasons_subset:
            y_true_int = y_true_by_season[s]
            p_lr = proba_lr_by_hl_season[hl][s]
            p_xgb = proba_xgb_by_hl_season[hl][s]
            p_ens = w * p_lr + (1 - w) * p_xgb
            y_oh = np.zeros((len(y_true_int), 3))
            y_oh[np.arange(len(y_true_int)), y_true_int] = 1
            accs.append(accuracy_score(y_true_int, p_ens.argmax(axis=1)))
            lls.append(log_loss(y_true_int, p_ens, labels=[0, 1, 2]))
            briers.append(multiclass_brier(y_oh, p_ens))
        return np.mean(accs), np.mean(lls), np.mean(briers)

    print(f"=== Scelta congiunta di w (peso LogReg) e half-life (peso stagioni recenti) ===")
    print(f"    su {len(selection_seasons)} stagioni di SELEZIONE ({selection_seasons[0]} -> {selection_seasons[-1]})\n")
    w_grid = np.arange(0.0, 1.01, 0.1)
    grid_results = []
    for hl in SEASON_HALF_LIFE_CANDIDATES:
        for w in w_grid:
            acc, ll, brier = eval_combo(w, hl, selection_seasons)
            grid_results.append((w, hl, acc, ll, brier))

    best_w, best_hl, _, best_ll, _ = min(grid_results, key=lambda t: t[3])
    print("Migliori combinazioni per half-life (log-loss minimo su selezione, al variare di w):")
    for hl in SEASON_HALF_LIFE_CANDIDATES:
        rows = [r for r in grid_results if r[1] == hl]
        w_m, _, acc_m, ll_m, brier_m = min(rows, key=lambda t: t[3])
        flag = "  <-- scelto" if hl == best_hl else ""
        hl_label = "nessuno (uniforme)" if hl is None else f"{hl} stagioni"
        print(f"  half_life={hl_label:20s}: miglior w_logreg={w_m:.1f} -> accuracy={acc_m:.4f}  log_loss={ll_m:.4f}  brier={brier_m:.4f}{flag}")

    print(f"\nCombinazione scelta (log-loss minimo su selezione): w_logreg={best_w:.1f}, half_life={best_hl}")

    print(f"\n=== Valutazione IMPARZIALE su {len(holdout_seasons)} stagioni di HOLDOUT ({holdout_seasons[0]} -> {holdout_seasons[-1]}), mai viste nella scelta ===")
    for name, w, hl in [
        ("Logistic Regression (w=1.0, no recency)", 1.0, None),
        ("XGBoost (w=0.0, no recency - baseline)", 0.0, None),
        (f"XGBoost (w=0.0, half_life={best_hl})", 0.0, best_hl),
        ("Ensemble (no recency - baseline Fase 4d)", best_w, None),
        (f"Ensemble (w={best_w:.1f}, half_life={best_hl} - scelto)", best_w, best_hl),
    ]:
        acc, ll, brier = eval_combo(w, hl, holdout_seasons)
        print(f"  {name:45s} accuracy={acc:.4f}  log_loss={ll:.4f}  brier={brier:.4f}")

    print(f"\n=== Media sulle 15 stagioni totali (per confronto diretto con le fasi precedenti) ===")
    all_results = []
    for name, w, hl in [
        ("LogisticRegression", 1.0, best_hl),
        ("XGBoost_no_recency", 0.0, None),
        (f"XGBoost_half_life_{best_hl}", 0.0, best_hl),
        ("Ensemble_no_recency", best_w, None),
        (f"Ensemble_half_life_{best_hl}", best_w, best_hl),
    ]:
        acc, ll, brier = eval_combo(w, hl, test_seasons)
        all_results.append({"modello": name, "accuracy": acc, "log_loss": ll, "brier": brier})
        print(f"  {name:35s} accuracy={acc:.4f}  log_loss={ll:.4f}  brier={brier:.4f}")

    pd.DataFrame(all_results).to_csv(BASE / "notebooks" / "ensemble_results.csv", index=False)
    pd.DataFrame(grid_results, columns=["w_logreg", "season_half_life", "accuracy", "log_loss", "brier"]).to_csv(
        BASE / "notebooks" / "ensemble_weight_search.csv", index=False)
    print(f"\nSalvato notebooks/ensemble_results.csv e notebooks/ensemble_weight_search.csv")
    print(f"\n>>> Per allineare train_baseline.py/train_xgboost.py/predict_season.py: SEASON_HALF_LIFE = {best_hl}, ENSEMBLE_W_LOGREG = {best_w:.1f}")


if __name__ == "__main__":
    main()
