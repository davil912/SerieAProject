"""
Fase 2 - Modello baseline (Logistic Regression) con validazione walk-forward.

Valida la pipeline end-to-end: train SOLO su stagioni passate, test sulla
stagione successiva, si avanza stagione per stagione (mai split casuale).
Confronta il modello con due benchmark:
  - "vince sempre la casa": baseline banale
  - probabilità implicite del mercato (quote bookmaker): benchmark esterno

Uso:
    python src/train_baseline.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
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
    "squad_value_diff", "squad_value_log_ratio",
]
CLASSES = ["A", "D", "H"]  # ordine alfabetico usato da sklearn per LabelEncoder-like mapping
FIRST_TEST_SEASON_INDEX = 5  # le prime 5 stagioni sono usate solo per training iniziale
# Fase 4e: le stagioni piu' vecchie contano meno nel training - richiesto dall'utente.
# half_life=6 -> il peso si dimezza ogni 6 stagioni indietro rispetto all'ultima disponibile
# nel training di quel fold (walk-forward, mai la stagione di test). Scelto confrontando
# [None, 10, 6, 3] su train_ensemble.py (stagioni di SELEZIONE): vedi FASE4e_RECENCY.md.
SEASON_HALF_LIFE = 10


def season_sample_weights(df: pd.DataFrame, half_life=SEASON_HALF_LIFE) -> np.ndarray:
    """Peso esponenziale decrescente per stagione: la stagione piu' recente PRESENTE
    NEL TRAINING (non quella di test) ha peso 1.0, ogni `half_life` stagioni indietro
    il peso si dimezza. half_life=None -> pesi tutti uguali (comportamento originale)."""
    if half_life is None:
        return np.ones(len(df))
    train_seasons_sorted = sorted(df["season"].unique())
    most_recent_idx = len(train_seasons_sorted) - 1
    season_to_idx = {s: i for i, s in enumerate(train_seasons_sorted)}
    seasons_ago = most_recent_idx - df["season"].map(season_to_idx)
    return 0.5 ** (seasons_ago / half_life)


def prepare_xy(df: pd.DataFrame):
    X = df[FEATURE_COLS].copy()
    # imputazione: mancanze concentrate nei precedenti storici (inizio dataset / primo H2H).
    # Se la finestra di training e' troppo corta anche la mediana puo' essere NaN
    # (es. prima stagione, nessuno scontro diretto pregresso): in quel caso si
    # ripiega su un valore neutro fisso (1.5 punti/partita, 0 giorni di riposo extra, ecc.).
    X["h2h_home_ppg"] = X["h2h_home_ppg"].fillna(X["h2h_home_ppg"].median())
    X["h2h_home_ppg"] = X["h2h_home_ppg"].fillna(1.5)
    X["rest_days_diff"] = X["rest_days_diff"].fillna(0)
    X["home_advantage_recent"] = X["home_advantage_recent"].fillna(0.45)
    X = X.fillna(0)  # rete di sicurezza finale su qualunque altra colonna
    y = df["result"]
    return X, y


def multiclass_brier(y_true_onehot: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean(np.sum((y_prob - y_true_onehot) ** 2, axis=1)))


def onehot(y: pd.Series) -> np.ndarray:
    return pd.get_dummies(y)[CLASSES].values.astype(float)


def evaluate_predictions(name, y_true, y_prob_df, results_list, season):
    y_true_oh = onehot(y_true)
    y_pred = y_prob_df[CLASSES].values.argmax(axis=1)
    y_pred_labels = np.array(CLASSES)[y_pred]

    acc = accuracy_score(y_true, y_pred_labels)
    ll = log_loss(y_true, y_prob_df[CLASSES].values, labels=CLASSES)
    brier = multiclass_brier(y_true_oh, y_prob_df[CLASSES].values)

    results_list.append({"season": season, "modello": name, "accuracy": acc, "log_loss": ll, "brier": brier})


def main():
    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
    seasons = sorted(df["season"].unique())
    test_seasons = seasons[FIRST_TEST_SEASON_INDEX:]

    results = []

    for test_season in test_seasons:
        train_df = df[df["season"] < test_season]
        test_df = df[df["season"] == test_season]

        X_train, y_train = prepare_xy(train_df)
        X_test, y_test = prepare_xy(test_df)
        w_train = season_sample_weights(train_df)

        # --- Modello baseline: Logistic Regression multinomiale ---
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000)),
        ])
        pipe.fit(X_train, y_train, clf__sample_weight=w_train)
        proba = pipe.predict_proba(X_test)
        proba_df = pd.DataFrame(proba, columns=pipe.classes_, index=test_df.index)
        evaluate_predictions("LogisticRegression", y_test, proba_df, results, test_season)

        # --- Benchmark 1: vince sempre la casa ---
        const_df = pd.DataFrame(0.0, index=test_df.index, columns=CLASSES)
        const_df["H"] = 1.0
        # per il log-loss usiamo una versione "smussata" (altrimenti log-loss infinito sugli errori)
        const_smoothed = pd.DataFrame(
            np.tile([0.05, 0.05, 0.90], (len(test_df), 1)), columns=["A", "D", "H"], index=test_df.index
        )
        evaluate_predictions("Baseline_vince_sempre_casa", y_test, const_smoothed, results, test_season)

        # --- Benchmark 2: probabilità implicite del mercato (quote bookmaker) ---
        market_df = test_df[["market_prob_away", "market_prob_draw", "market_prob_home"]].copy()
        market_df.columns = ["A", "D", "H"]
        market_df.index = test_df.index
        # se mancano le quote per qualche partita, fallback su Logistic Regression per quella riga
        missing = market_df.isna().any(axis=1)
        if missing.any():
            market_df.loc[missing, CLASSES] = proba_df.loc[missing, CLASSES].values
        evaluate_predictions("Quote_di_mercato", y_test, market_df, results, test_season)

    results_df = pd.DataFrame(results)
    summary = results_df.groupby("modello")[["accuracy", "log_loss", "brier"]].mean().round(4)
    summary = summary.sort_values("log_loss")

    print(f"Backtest walk-forward su {len(test_seasons)} stagioni ({test_seasons[0]} -> {test_seasons[-1]})\n")
    print("=== Metriche medie sul backtest (log_loss piu' basso = meglio) ===")
    print(summary.to_string())

    print("\n=== Dettaglio per stagione (accuracy) ===")
    pivot_acc = results_df.pivot(index="season", columns="modello", values="accuracy").round(3)
    print(pivot_acc.to_string())

    # --- Modello finale addestrato su TUTTI i dati disponibili (per uso successivo / Fase 3) ---
    X_all, y_all = prepare_xy(df)
    final_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000)),
    ])
    final_pipe.fit(X_all, y_all, clf__sample_weight=season_sample_weights(df))

    print("\n=== Coefficienti modello finale (Logistic Regression, feature standardizzate) ===")
    coef_df = pd.DataFrame(final_pipe.named_steps["clf"].coef_, columns=FEATURE_COLS, index=final_pipe.classes_)
    print(coef_df.round(3).to_string())

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(final_pipe, MODELS_DIR / "baseline_logreg.pkl")
    results_df.to_csv(BASE / "notebooks" / "backtest_results.csv", index=False)
    print(f"\nModello salvato in {MODELS_DIR / 'baseline_logreg.pkl'}")
    print(f"Risultati backtest salvati in notebooks/backtest_results.csv")


if __name__ == "__main__":
    main()
