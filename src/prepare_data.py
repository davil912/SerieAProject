"""
Fase 1 - Pulizia e unificazione dati Serie A.

Legge i CSV grezzi (data/raw/serieA_raw.csv, data/raw/serieB_raw.csv) scaricati
da Football-Data.co.uk (via mirror GitHub xgabora/Club-Football-Match-Data-2000-2025)
e produce un dataset pulito e pronto per l'analisi/feature engineering in
data/processed/.

Uso:
    python src/prepare_data.py
"""

from pathlib import Path
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# Colonne che ci interessano dal dataset grezzo (le altre, principalmente quote
# di bookmaker aggiuntive e statistiche di dettaglio, restano nel raw e potranno
# essere riprese in seguito per il feature engineering avanzato).
KEEP_COLUMNS = [
    "MatchDate", "MatchTime", "HomeTeam", "AwayTeam",
    "HomeElo", "AwayElo",
    "Form3Home", "Form5Home", "Form3Away", "Form5Away",
    "FTHome", "FTAway", "FTResult",
    "HTHome", "HTAway", "HTResult",
    "HomeShots", "AwayShots", "HomeTarget", "AwayTarget",
    "HomeFouls", "AwayFouls", "HomeCorners", "AwayCorners",
    "HomeYellow", "AwayYellow", "HomeRed", "AwayRed",
    "OddHome", "OddDraw", "OddAway",
    "Over25", "Under25",
]

RENAME_MAP = {
    "MatchDate": "date",
    "MatchTime": "time",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "HomeElo": "home_elo",
    "AwayElo": "away_elo",
    "Form3Home": "home_form3",
    "Form5Home": "home_form5",
    "Form3Away": "away_form3",
    "Form5Away": "away_form5",
    "FTHome": "home_goals",
    "FTAway": "away_goals",
    "FTResult": "result",          # H / D / A
    "HTHome": "home_goals_ht",
    "HTAway": "away_goals_ht",
    "HTResult": "result_ht",
    "HomeShots": "home_shots",
    "AwayShots": "away_shots",
    "HomeTarget": "home_shots_target",
    "AwayTarget": "away_shots_target",
    "HomeFouls": "home_fouls",
    "AwayFouls": "away_fouls",
    "HomeCorners": "home_corners",
    "AwayCorners": "away_corners",
    "HomeYellow": "home_yellow",
    "AwayYellow": "away_yellow",
    "HomeRed": "home_red",
    "AwayRed": "away_red",
    "OddHome": "odd_home",
    "OddDraw": "odd_draw",
    "OddAway": "odd_away",
    "Over25": "odd_over25",
    "Under25": "odd_under25",
}


def _season_label_from_start(date: pd.Timestamp) -> str:
    """Etichetta stagione a partire dalla data di INIZIO stagione (non della singola partita)."""
    if date.month >= 7:
        return f"{date.year}/{date.year + 1}"
    else:
        return f"{date.year - 1}/{date.year}"


def assign_seasons_by_gap(df: pd.DataFrame, gap_days: int = 45) -> pd.Series:
    """
    Assegna la stagione basandosi sui blocchi temporali di partite consecutive,
    invece che sul semplice mese del calendario. Necessario perché alcune stagioni
    (es. 2019/2020, terminata ad agosto 2020 per il COVID) sforano i confini
    "naturali" agosto-maggio e andrebbero altrimenti assegnate alla stagione
    successiva per errore.

    Un gap di oltre `gap_days` giorni tra due partite consecutive (dell'intero
    campionato) viene considerato l'inizio di una nuova stagione.
    """
    dates_sorted = df["date"].sort_values()
    gaps = dates_sorted.diff().dt.days
    new_block = (gaps > gap_days) | gaps.isna()
    block_id = new_block.cumsum()
    block_id.index = dates_sorted.index

    block_start_date = dates_sorted.groupby(block_id).transform("min")
    season_labels = block_start_date.apply(_season_label_from_start)

    # riallinea all'ordine/indice originale del dataframe
    return season_labels.reindex(df.index)


def load_and_clean(raw_path: Path, league_label: str) -> pd.DataFrame:
    df = pd.read_csv(raw_path, low_memory=False)
    df = df[[c for c in KEEP_COLUMNS if c in df.columns]].rename(columns=RENAME_MAP)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team", "result"])

    df["season"] = assign_seasons_by_gap(df)
    df["league"] = league_label

    df = df.sort_values("date").reset_index(drop=True)

    # Sanity check: il risultato (result) deve essere coerente con i gol segnati
    inferred = pd.Series("D", index=df.index)
    inferred[df["home_goals"] > df["away_goals"]] = "H"
    inferred[df["home_goals"] < df["away_goals"]] = "A"
    mismatches = (inferred != df["result"]).sum()
    if mismatches:
        print(f"  [attenzione] {mismatches} righe con esito incoerente rispetto ai gol: corretto automaticamente")
        df["result"] = inferred

    return df


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Pulizia dati Serie A...")
    serie_a = load_and_clean(RAW_DIR / "serieA_raw.csv", "Serie A")
    print(f"  {len(serie_a)} partite valide, stagioni {serie_a['season'].min()} -> {serie_a['season'].max()}")

    print("Pulizia dati Serie B (per gestione neopromosse)...")
    serie_b = load_and_clean(RAW_DIR / "serieB_raw.csv", "Serie B")
    print(f"  {len(serie_b)} partite valide, stagioni {serie_b['season'].min()} -> {serie_b['season'].max()}")

    serie_a.to_csv(PROCESSED_DIR / "serieA_matches.csv", index=False)
    serie_b.to_csv(PROCESSED_DIR / "serieB_matches.csv", index=False)

    # Report valori mancanti sulle colonne principali
    print("\nValori mancanti (Serie A) sulle colonne chiave:")
    key_cols = ["home_elo", "away_elo", "odd_home", "odd_draw", "odd_away",
                "home_shots", "away_shots"]
    for col in key_cols:
        if col in serie_a.columns:
            missing = serie_a[col].isna().mean() * 100
            print(f"  {col}: {missing:.1f}% mancanti")

    print(f"\nSalvato: {PROCESSED_DIR / 'serieA_matches.csv'}")
    print(f"Salvato: {PROCESSED_DIR / 'serieB_matches.csv'}")


if __name__ == "__main__":
    main()
