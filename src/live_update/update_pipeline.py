"""
Fase 4 - Pipeline di aggiornamento automatico.

Da eseguire periodicamente (es. una volta a settimana, il lunedi' dopo il
weekend di campionato): scarica di nuovo la fonte dati, individua le partite
NON ancora presenti nello storico locale, le aggiunge, e ricalcola la
classifica della stagione piu' recente disponibile ("classifica live").

LIMITE NOTO (Fase 6) sul download automatico per la stagione 2026/2027 in
corso: il mirror GitHub usato qui sotto (MIRROR_URL) si e' fermato al 2025 e
NON contiene la stagione 2026/2027; football-data.co.uk, l'altra fonte
possibile, blocca le richieste dirette da questo ambiente cloud (403). Il
download automatico qui sotto resta quindi un "best effort" che oggi non
trova nulla di nuovo per la stagione corrente. La via affidabile e' manuale:
scaricare dal sito il CSV "stagione 2026/2027 ad oggi", metterlo in
data/incoming/serieA_2026_27.csv (vedi data/incoming/README.md) e lanciare
    python src/live_update/integrate_new_season.py data/incoming/serieA_2026_27.csv 2026/2027
che e' IDEMPOTENTE: si puo' rilanciare ogni settimana con il file scaricato
di nuovo senza creare duplicati.

Se vengono aggiunte nuove partite (automaticamente o via integrate_new_season.py),
ricorda di rilanciare a mano:
    python src/preprocessing/prepare_data.py
    python src/preprocessing/feature_builder.py
    python src/preprocessing/build_poisson_features.py
    python src/simulation/predict_season.py   # si accorge da solo delle partite 2026/2027 gia' giocate (Fase 6)
    python dashboard/build_dashboard.py
per aggiornare anche il dataset di feature usato dai modelli (non lo si fa
automaticamente qui per tenere il refresh dati separato dal retraining,
come indicato nel piano d'azione originale: il retraining va fatto ogni
poche giornate, non ad ogni singola partita).

Uso:
    python src/live_update/update_pipeline.py
"""

import sys
import urllib.request
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE / "src" / "common"))
from classifica import calcola_classifica  # noqa: E402

MIRROR_URL = "https://raw.githubusercontent.com/xgabora/Club-Football-Match-Data-2000-2025/main/data/Matches.csv"
RAW_DIR = BASE / "data" / "raw"


def download_latest():
    """Scarica di nuovo il dataset sorgente. Ritorna (serieA_df, serieB_df) o (None, None) se non raggiungibile."""
    try:
        tmp_path = RAW_DIR / "_latest_download.csv"
        urllib.request.urlretrieve(MIRROR_URL, tmp_path)
    except Exception as e:
        print(f"[Attenzione] impossibile scaricare la fonte dati ({e}). "
              f"Nota: questa fonte a volte non e' raggiungibile dall'ambiente cloud; "
              f"riprovare da un'altra rete o aggiornare manualmente i CSV in data/raw/.")
        return None, None

    df = pd.read_csv(tmp_path, low_memory=False)
    tmp_path.unlink()
    serie_a = df[df["Division"] == "I1"].copy()
    serie_b = df[df["Division"] == "I2"].copy()
    return serie_a, serie_b


def merge_new_matches(existing_path: Path, new_df: pd.DataFrame) -> int:
    """Accoda a existing_path solo le righe di new_df non gia' presenti (chiave: data+squadre). Ritorna il numero di righe aggiunte."""
    existing = pd.read_csv(existing_path, low_memory=False)
    key_cols = ["MatchDate", "HomeTeam", "AwayTeam"]
    existing_keys = set(map(tuple, existing[key_cols].values))
    new_rows = new_df[~new_df[key_cols].apply(tuple, axis=1).isin(existing_keys)]

    if len(new_rows) > 0:
        combined = pd.concat([existing, new_rows], ignore_index=True)
        combined.to_csv(existing_path, index=False)
    return len(new_rows)


def main():
    print("Scaricamento dati aggiornati...")
    serie_a_new, serie_b_new = download_latest()

    if serie_a_new is not None:
        n_added_a = merge_new_matches(RAW_DIR / "serieA_raw.csv", serie_a_new)
        n_added_b = merge_new_matches(RAW_DIR / "serieB_raw.csv", serie_b_new)
        print(f"Nuove partite Serie A aggiunte: {n_added_a}")
        print(f"Nuove partite Serie B aggiunte: {n_added_b}")
        if n_added_a > 0:
            print("\n[Azione richiesta] Nuove partite trovate: rilancia gli script in "
                  "src/preprocessing/ (prepare_data.py, feature_builder.py, build_poisson_features.py) "
                  "per aggiornare le feature, poi eventualmente quelli in src/training/ per ri-addestrare i modelli.")
    else:
        print("Nessun aggiornamento scaricato in questa esecuzione (vedi messaggio sopra).")

    # --- Classifica live: stagione piu' recente disponibile nei dati puliti ---
    processed_path = BASE / "data" / "processed" / "serieA_matches.csv"
    if processed_path.exists():
        matches = pd.read_csv(processed_path, parse_dates=["date"])
        latest_season = sorted(matches["season"].unique())[-1]
        classifica = calcola_classifica(matches, latest_season)
        print(f"\n=== Classifica live — {latest_season} (dati puliti in data/processed/) ===")
        print(classifica.to_string())
    else:
        print("\n[Nota] data/processed/serieA_matches.csv non trovato: esegui prima "
              "src/preprocessing/prepare_data.py.")


if __name__ == "__main__":
    main()
