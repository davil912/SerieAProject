"""
Fase 4 (estensione) - Motore Elo proprio, per continuare l'aggiornamento del
rating quando la fonte esterna (ClubElo, via il mirror GitHub) non ha ancora
pubblicato la stagione più recente.

Formula Elo standard con vantaggio-casa espresso in punti Elo (non è la
metodologia esatta di ClubElo, che pesa anche il margine di vittoria - qui è
una versione semplificata ma coerente, usata SOLO per continuare i rating
a partire dall'ultimo valore noto per ciascuna squadra).

    E_home = 1 / (1 + 10^(-(R_home + HOME_ADV - R_away) / 400))
    R_home_new = R_home + K * (S_home - E_home)

K=20 e HOME_ADV=100 sono i valori "standard" usati storicamente anche dal
World Football Elo Ratings.
"""

import pandas as pd

K = 20
HOME_ADV = 100
DEFAULT_RATING = 1500.0  # usato solo se una squadra non ha alcuna storia (ne' Serie A ne' Serie B)


def expected_score(r_a: float, r_b: float, home_adv: float = 0.0) -> float:
    return 1.0 / (1.0 + 10 ** (-((r_a + home_adv) - r_b) / 400))


def update_ratings(r_home: float, r_away: float, result: str, k: float = K, home_adv: float = HOME_ADV):
    """result: 'H', 'D', o 'A'. Ritorna (nuovo_r_home, nuovo_r_away)."""
    s_home = 1.0 if result == "H" else 0.5 if result == "D" else 0.0
    e_home = expected_score(r_home, r_away, home_adv)
    r_home_new = r_home + k * (s_home - e_home)
    r_away_new = r_away + k * ((1 - s_home) - (1 - e_home))
    return r_home_new, r_away_new


def seed_ratings(serieA_matches: pd.DataFrame, serieB_matches: pd.DataFrame, teams: list) -> dict:
    """Recupera il rating Elo AGGIORNATO A OGGI per ciascuna squadra: prima cerca
    l'ultima partita in Serie A, poi (se piu' recente o se assente) in Serie B,
    infine ricade su un valore di default per le squadre senza alcuna storia.

    Nota (Fase 6): le colonne home_elo/away_elo salvate nello storico sono il
    rating PRE-partita (cosi' com'erano al calcio d'inizio, per evitare
    leakage temporale nelle feature di training). Per ottenere il rating
    "oggi" bisogna quindi applicare un ultimo update_ratings() con il
    risultato reale di quella partita - stessa correzione gia' usata in
    src/predict_season.py (compute_current_elo). Senza questo passaggio,
    ogni volta che si integra una nuova stagione/giornata (Fase 4, Fase 6)
    si riparte da un Elo "vecchio di una partita" per ciascuna squadra."""
    ratings = {}
    for team in teams:
        candidates = []
        for df, label in [(serieA_matches, "A"), (serieB_matches, "B")]:
            m = df[(df["home_team"] == team) | (df["away_team"] == team)].sort_values("date")
            if len(m):
                last = m.iloc[-1]
                is_home = last["home_team"] == team
                r_home_post, r_away_post = update_ratings(last["home_elo"], last["away_elo"], last["result"])
                elo_post = r_home_post if is_home else r_away_post
                candidates.append((last["date"], elo_post, label))
        if candidates:
            candidates.sort(key=lambda c: c[0])
            best_date, best_elo, best_label = candidates[-1]
            ratings[team] = float(best_elo)
        else:
            ratings[team] = DEFAULT_RATING
    return ratings
