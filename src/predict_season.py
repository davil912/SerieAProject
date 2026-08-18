"""
Fase 5 - Previsione della stagione 2026/2027: probabilita' 1X2 partita per
partita e classifica finale prevista, via simulazione Monte Carlo.

IDEA GENERALE
-------------
I modelli allenati finora (Logistic Regression + XGBoost, combinati
nell'ensemble di Fase 4 con peso w=0.6/0.4) prevedono l'esito di UNA partita
nota il suo Elo/forma/H2H al momento del calcio d'inizio. Per una stagione
INTERA che deve ancora iniziare, Elo e forma delle squadre cambiano partita
dopo partita in base ai risultati - risultati che non conosciamo. Si simula
quindi l'intera stagione N volte (default 1000): ad ogni simulazione, si
gioca virtualmente ogni giornata in ordine cronologico, si estraggono
probabilita' 1X2 dall'ensemble con lo stato (Elo/forma/H2H) aggiornato fino a
quel punto DI QUELLA SIMULAZIONE, si estrae un esito casuale da quelle
probabilita', si aggiorna lo stato e si passa alla giornata successiva.

Le 1000 traiettorie simulate, mediate, danno:
  - per ogni partita: probabilita' 1X2 (media della probabilita' stimata
    dall'ensemble su tutte le simulazioni, non solo il conteggio degli esiti
    campionati - stima piu' precisa a parita' di N)
  - per ogni squadra: punti medi, posizione media, probabilita' di vincere
    il campionato / qualificarsi in Europa / retrocedere

Feature che NON dipendono dalla simulazione (identiche in ogni traiettoria,
perche' dipendono solo dal calendario o da uno snapshot statico) vengono
calcolate UNA volta sola: giorni di riposo, partite giocate in stagione,
valore rosa (snapshot piu' recente disponibile), vantaggio-casa "di periodo"
(congelato all'ultimo valore storico), rating Poisson attacco/difesa
(rifittati una volta sola su una finestra mobile di 5 stagioni, esattamente
come nel backtest storico - src/build_poisson_features.py).

Feature che DIPENDONO dalla simulazione (diverse traiettoria per traiettoria,
perche' dipendono dai risultati simulati): Elo, forma (ultime 3/5), scontri
diretti (H2H) nella stagione corrente.

LIMITI DICHIARATI
------------------
- Per le 3 neopromosse (Frosinone, Monza, Venezia) l'ultimo dato Elo/forma
  disponibile risale alla loro ultima apparizione in Serie A (2024/25), non
  alla loro stagione di Serie B 2025/26 appena vinta: non abbiamo i risultati
  di Serie B 2025/26 nel dataset. Elo/forma di partenza per queste 3 squadre
  sono quindi meno aggiornati che per le altre 17 (limite noto, non risolvibile
  con i dati attualmente disponibili).
- Valore rosa: Frosinone e Monza non hanno una copertura Transfermarkt
  sufficiente per la stagione 2025/26 (rispettivamente 5 e 14 giocatori
  valutati, sotto la soglia di affidabilita' di 15) - il valore usato per
  loro e' quindi meno affidabile che per le altre squadre (flag esplicito
  nell'output).
- Il modello Poisson (gol attesi) e i rating Elo/forma NON vengono
  ricalibrati durante la stagione reale: lo script va ri-eseguito periodicamente
  (es. ogni settimana) con i risultati reali nel frattempo integrati in
  data/processed/serieA_matches.csv, per aggiornare le previsioni.
- Classifica finale simulata: spareggio per parita' di punti = differenza
  reti poi gol fatti (regola standard), SENZA il criterio degli scontri
  diretti (che nella realta' ha priorita' sulla differenza reti in Serie A) -
  semplificazione dichiarata, impatto minimo su medie/probabilita' aggregate
  su 1000 simulazioni.

Uso:
    python src/predict_season.py
"""

from pathlib import Path
from collections import defaultdict, deque
import math

import numpy as np
import pandas as pd
import joblib

from elo_updater import update_ratings

BASE = Path(__file__).resolve().parent.parent
MATCHES_PATH = BASE / "data" / "processed" / "serieA_matches.csv"
MATCHES_B_PATH = BASE / "data" / "processed" / "serieB_matches.csv"
FEATURES_PATH = BASE / "data" / "processed" / "serieA_features.csv"
CALENDAR_PATH = BASE / "data" / "raw" / "calendario_2026_27.csv"
SQUAD_VALUE_CURRENT_PATH = BASE / "data" / "processed" / "squad_values_current.csv"
SQUAD_VALUE_HISTORY_PATH = BASE / "data" / "processed" / "squad_values_by_season.csv"
MODELS_DIR = BASE / "models"
OUT_DIR = BASE / "data" / "processed"

N_SIMULATIONS = 5000
RNG_SEED = 42
ENSEMBLE_W_LOGREG = 0.6  # scelto in Fase 4c (src/train_ensemble.py) su holdout imparziale
POISSON_ROLLING_WINDOW = 5
POISSON_ALPHA = 0.01
MAX_GOALS = 8

LOGREG_FEATURES = [
    "elo_diff", "home_elo", "away_elo", "form3_diff", "form5_diff",
    "home_advantage_recent", "rest_days_diff", "season_progress",
    "h2h_home_ppg", "h2h_n_precedenti",
]
XGB_FEATURES = LOGREG_FEATURES + [
    "poisson_exp_goals_home", "poisson_exp_goals_away", "poisson_exp_goals_diff",
    "poisson_prob_home", "poisson_prob_draw", "poisson_prob_away",
    "home_squad_value", "away_squad_value", "squad_value_diff",
]
CLASSES = ["A", "D", "H"]


def points(result: str, is_home: bool) -> int:
    if result == "D":
        return 1
    if (result == "H") == is_home:
        return 3
    return 0


# ---------------------------------------------------------------------------
# 1. Stato "attuale" (Elo, forma, H2H, valore rosa) al 18/08/2026
# ---------------------------------------------------------------------------

def compute_current_elo(combined_hist: pd.DataFrame, teams: list) -> dict:
    """Elo POST ultima partita giocata (non pre-partita): si prende il rating
    pre-partita dell'ultimo incontro disponibile e si applica l'aggiornamento
    Elo standard con il risultato reale, per ottenere il rating "oggi"."""
    elo = {}
    for t in teams:
        sub = combined_hist[(combined_hist["home_team"] == t) | (combined_hist["away_team"] == t)]
        sub = sub.sort_values("date")
        last = sub.iloc[-1]
        r_home_post, r_away_post = update_ratings(last["home_elo"], last["away_elo"], last["result"])
        elo[t] = r_home_post if last["home_team"] == t else r_away_post
    return elo


def compute_current_form(combined_hist: pd.DataFrame, teams: list) -> dict:
    """Ultimi (fino a) 5 risultati per squadra, in punti, ordine cronologico
    (piu' vecchio -> piu' recente), includendo l'ultima partita disponibile."""
    form = {}
    for t in teams:
        sub = combined_hist[(combined_hist["home_team"] == t) | (combined_hist["away_team"] == t)]
        sub = sub.sort_values("date").tail(5)
        pts = [points(r.result, r.home_team == t) for r in sub.itertuples(index=False)]
        while len(pts) < 5:  # difensivo, non dovrebbe mai scattare con questo storico
            pts.insert(0, 1.3)
        form[t] = np.array(pts, dtype=float)
    return form


def build_h2h_seed(serieA_hist: pd.DataFrame, teams: list) -> dict:
    """Ultimi (fino a) 5 scontri diretti Serie A per ogni coppia di squadre
    tra le 20 correnti (stesso ambito usato in feature_builder.py: solo Serie A)."""
    team_set = set(teams)
    sub = serieA_hist[serieA_hist["home_team"].isin(team_set) & serieA_hist["away_team"].isin(team_set)]
    sub = sub.sort_values("date")
    h2h = defaultdict(lambda: deque(maxlen=5))
    for r in sub.itertuples(index=False):
        pair = tuple(sorted([r.home_team, r.away_team]))
        h2h[pair].append((r.home_team, r.result))
    return dict(h2h)


def h2h_home_ppg_and_n(deque_hist, current_home_team):
    if len(deque_hist) == 0:
        return np.nan, 0
    pts = []
    for past_home, past_result in deque_hist:
        if past_home == current_home_team:
            pts.append(3 if past_result == "H" else 1 if past_result == "D" else 0)
        else:
            pts.append(3 if past_result == "A" else 1 if past_result == "D" else 0)
    return sum(pts) / len(pts), len(deque_hist)


def build_squad_values(teams: list) -> dict:
    current = pd.read_csv(SQUAD_VALUE_CURRENT_PATH).set_index("team")["valore_rosa_milioni"].to_dict()
    hist = pd.read_csv(SQUAD_VALUE_HISTORY_PATH)
    sv, flagged_unreliable = {}, []
    for t in teams:
        if t in current:
            sv[t] = current[t]
        else:
            # neopromosse: non presenti nello snapshot 2025/26 Serie A (players.csv) ->
            # ripiego sull'ultima stagione disponibile nello storico per-stagione.
            sub = hist[hist["team"] == t].sort_values("season")
            if len(sub) == 0:
                sv[t] = np.nan
                continue
            last = sub.iloc[-1]
            sv[t] = last["valore_rosa_milioni"]
            if not last["affidabile"]:
                flagged_unreliable.append((t, last["season"], int(last["n_giocatori"])))
    return sv, flagged_unreliable


# ---------------------------------------------------------------------------
# 2. Feature statiche (uguali in ogni simulazione) per le 380 partite future
# ---------------------------------------------------------------------------

def build_static_features(fixtures: pd.DataFrame, combined_hist: pd.DataFrame, teams: list,
                            squad_value: dict, home_adv_recent: float, poisson_model) -> pd.DataFrame:
    last_match_date = {}
    for t in teams:
        sub = combined_hist[(combined_hist["home_team"] == t) | (combined_hist["away_team"] == t)]
        last_match_date[t] = sub["date"].max()
    games_played = {t: 0 for t in teams}

    rows = []
    for r in fixtures.sort_values(["matchday", "date"]).itertuples(index=False):
        h, a = r.home_team, r.away_team
        rest_h = (r.date - last_match_date[h]).days
        rest_a = (r.date - last_match_date[a]).days
        gp_h, gp_a = games_played[h], games_played[a]
        lam_h, lam_a, p_h, p_d, p_a = poisson_model.match_probs(h, a)
        rows.append({
            "matchday": r.matchday, "date": r.date, "home_team": h, "away_team": a,
            "rest_days_diff": rest_h - rest_a,
            "season_progress": (gp_h + gp_a) / 2 / 38.0,
            "home_advantage_recent": home_adv_recent,
            "home_squad_value": squad_value.get(h, np.nan),
            "away_squad_value": squad_value.get(a, np.nan),
            "squad_value_diff": squad_value.get(h, np.nan) - squad_value.get(a, np.nan),
            "poisson_exp_goals_home": lam_h, "poisson_exp_goals_away": lam_a,
            "poisson_exp_goals_diff": lam_h - lam_a,
            "poisson_prob_home": p_h, "poisson_prob_draw": p_d, "poisson_prob_away": p_a,
        })
        last_match_date[h] = r.date
        last_match_date[a] = r.date
        games_played[h] += 1
        games_played[a] += 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Simulazione Monte Carlo
# ---------------------------------------------------------------------------

def sample_goals(lam_home, lam_away, results, rng):
    """Campiona un punteggio plausibile COERENTE con l'esito gia' estratto
    dall'ensemble (per la differenza reti in classifica), usando i gol attesi
    del modello Poisson per modulare il margine."""
    n = len(results)
    goals_home = np.zeros(n, dtype=int)
    goals_away = np.zeros(n, dtype=int)

    is_d = results == "D"
    is_h = results == "H"
    is_a = results == "A"

    if is_d.any():
        base = rng.poisson((lam_home[is_d] + lam_away[is_d]) / 2)
        goals_home[is_d] = base
        goals_away[is_d] = base
    if is_h.any():
        away_g = rng.poisson(lam_away[is_h])
        margin = 1 + rng.poisson(np.clip(lam_home[is_h] - lam_away[is_h], 0.4, None))
        goals_away[is_h] = away_g
        goals_home[is_h] = away_g + margin
    if is_a.any():
        home_g = rng.poisson(lam_home[is_a])
        margin = 1 + rng.poisson(np.clip(lam_away[is_a] - lam_home[is_a], 0.4, None))
        goals_home[is_a] = home_g
        goals_away[is_a] = home_g + margin
    return goals_home, goals_away


def main():
    print(f"=== Fase 5 - Simulazione Monte Carlo stagione 2026/2027 (N={N_SIMULATIONS}) ===\n")
    rng = np.random.default_rng(RNG_SEED)

    matches_a = pd.read_csv(MATCHES_PATH, parse_dates=["date"])
    matches_b = pd.read_csv(MATCHES_B_PATH, parse_dates=["date"])
    combined_hist = pd.concat([matches_a, matches_b], ignore_index=True)
    fixtures = pd.read_csv(CALENDAR_PATH, parse_dates=["date"])
    teams = sorted(fixtures["home_team"].unique())
    assert len(teams) == 20, f"attese 20 squadre, trovate {len(teams)}"

    # --- stato iniziale ---
    seed_elo = compute_current_elo(combined_hist, teams)
    seed_form = compute_current_form(combined_hist, teams)
    h2h_seed = build_h2h_seed(matches_a, teams)
    squad_value, unreliable_sv = build_squad_values(teams)
    home_adv_recent = pd.read_csv(FEATURES_PATH, parse_dates=["date"]).sort_values("date").iloc[-1]["home_advantage_recent"]

    print("Elo di partenza (18/08/2026, post ultima partita nota):")
    for t in sorted(teams, key=lambda x: -seed_elo[x]):
        print(f"  {t:12s} {seed_elo[t]:.1f}")
    print(f"\nValore rosa non affidabile (copertura Transfermarkt insufficiente) per: {unreliable_sv}")

    # --- Poisson: rifit una sola volta, finestra mobile ultime 5 stagioni ---
    seasons = sorted(matches_a["season"].unique())
    window = seasons[-POISSON_ROLLING_WINDOW:]
    from poisson_model import fit_poisson_model
    poisson_model = fit_poisson_model(matches_a[matches_a["season"].isin(window)], alpha=POISSON_ALPHA)
    print(f"\nModello Poisson rifittato su: {window}")

    # --- feature statiche (uguali per tutte le simulazioni) ---
    static_df = build_static_features(fixtures, combined_hist, teams, squad_value, home_adv_recent, poisson_model)
    static_df = static_df.sort_values(["matchday", "date"]).reset_index(drop=True)
    n_fixtures = len(static_df)
    print(f"\nPartite totali da prevedere: {n_fixtures}")

    # --- modelli finali (allenati su TUTTA la storia disponibile) ---
    logreg_pipe = joblib.load(MODELS_DIR / "baseline_logreg.pkl")
    xgb_model = joblib.load(MODELS_DIR / "xgboost_final.pkl")

    # --- stato per-simulazione ---
    N = N_SIMULATIONS
    elo = {t: np.full(N, seed_elo[t]) for t in teams}
    form5 = {t: np.tile(seed_form[t], (N, 1)) for t in teams}  # (N, 5), vecchio -> nuovo
    h2h_state = [defaultdict(lambda: deque(maxlen=5)) for _ in range(N)]
    for sim in range(N):
        for pair, dq in h2h_seed.items():
            h2h_state[sim][pair] = deque(dq, maxlen=5)

    pts = {t: np.zeros(N, dtype=int) for t in teams}
    gf = {t: np.zeros(N, dtype=int) for t in teams}
    ga = {t: np.zeros(N, dtype=int) for t in teams}
    wins = {t: np.zeros(N, dtype=int) for t in teams}
    draws = {t: np.zeros(N, dtype=int) for t in teams}
    losses = {t: np.zeros(N, dtype=int) for t in teams}

    prob_sum = np.zeros((n_fixtures, 3))  # accumulatore probabilita' ensemble medie [A,D,H]

    fixture_row_idx = 0
    for md, group in static_df.groupby("matchday", sort=True):
        group = group.reset_index(drop=True)
        n_m = len(group)
        rows_h, rows_a = group["home_team"].values, group["away_team"].values

        feat = {c: np.zeros(n_m * N) for c in set(LOGREG_FEATURES) | set(XGB_FEATURES)}
        for i in range(n_m):
            h, a = rows_h[i], rows_a[i]
            sl = slice(i * N, (i + 1) * N)
            e_h, e_a = elo[h], elo[a]
            f_h, f_a = form5[h], form5[a]
            feat["elo_diff"][sl] = e_h - e_a
            feat["home_elo"][sl] = e_h
            feat["away_elo"][sl] = e_a
            feat["form3_diff"][sl] = f_h[:, -3:].sum(axis=1) - f_a[:, -3:].sum(axis=1)
            feat["form5_diff"][sl] = f_h.sum(axis=1) - f_a.sum(axis=1)
            feat["rest_days_diff"][sl] = group.loc[i, "rest_days_diff"]
            feat["season_progress"][sl] = group.loc[i, "season_progress"]
            feat["home_advantage_recent"][sl] = group.loc[i, "home_advantage_recent"]
            feat["home_squad_value"][sl] = group.loc[i, "home_squad_value"]
            feat["away_squad_value"][sl] = group.loc[i, "away_squad_value"]
            feat["squad_value_diff"][sl] = group.loc[i, "squad_value_diff"]
            feat["poisson_exp_goals_home"][sl] = group.loc[i, "poisson_exp_goals_home"]
            feat["poisson_exp_goals_away"][sl] = group.loc[i, "poisson_exp_goals_away"]
            feat["poisson_exp_goals_diff"][sl] = group.loc[i, "poisson_exp_goals_diff"]
            feat["poisson_prob_home"][sl] = group.loc[i, "poisson_prob_home"]
            feat["poisson_prob_draw"][sl] = group.loc[i, "poisson_prob_draw"]
            feat["poisson_prob_away"][sl] = group.loc[i, "poisson_prob_away"]
            pair = tuple(sorted([h, a]))
            h2h_vals = [h2h_home_ppg_and_n(h2h_state[sim][pair], h) for sim in range(N)]
            feat["h2h_home_ppg"][sl] = [v[0] for v in h2h_vals]
            feat["h2h_n_precedenti"][sl] = [v[1] for v in h2h_vals]

        X_all = pd.DataFrame(feat)

        # --- Logistic Regression (con la stessa imputazione usata in training) ---
        X_lr = X_all[LOGREG_FEATURES].copy()
        X_lr["h2h_home_ppg"] = X_lr["h2h_home_ppg"].fillna(1.5)
        X_lr["home_advantage_recent"] = X_lr["home_advantage_recent"].fillna(0.45)
        X_lr = X_lr.fillna(0)
        proba_lr = pd.DataFrame(logreg_pipe.predict_proba(X_lr), columns=logreg_pipe.classes_)[CLASSES].values

        # --- XGBoost (NaN gestiti nativamente) ---
        X_xgb = X_all[XGB_FEATURES]
        proba_xgb = xgb_model.predict_proba(X_xgb)  # colonne [A, D, H]

        proba_ens = ENSEMBLE_W_LOGREG * proba_lr + (1 - ENSEMBLE_W_LOGREG) * proba_xgb  # (n_m*N, 3)

        # --- campiona un esito per ogni (partita, simulazione) ---
        u = rng.random(n_m * N)
        cum = np.cumsum(proba_ens, axis=1)
        outcome_idx = (u[:, None] < cum).argmax(axis=1)  # 0=A, 1=D, 2=H
        outcome = np.array(CLASSES)[outcome_idx]

        lam_home_arr = X_all["poisson_exp_goals_home"].values
        lam_away_arr = X_all["poisson_exp_goals_away"].values
        goals_h, goals_a = sample_goals(lam_home_arr, lam_away_arr, outcome, rng)

        # --- aggiorna stato e statistiche per ogni partita della giornata ---
        for i in range(n_m):
            h, a = rows_h[i], rows_a[i]
            sl = slice(i * N, (i + 1) * N)
            res = outcome[sl]
            g_h, g_a = goals_h[sl], goals_a[sl]

            # update_ratings (elo_updater.py) e' scritto per un singolo scalare/risultato: qui
            # serve la versione vettoriale (stesse formule, applicate a tutte le N simulazioni
            # in una volta), quindi la reimplementiamo inline invece di chiamare la funzione.
            s_home = np.where(res == "H", 1.0, np.where(res == "D", 0.5, 0.0))
            e_home = 1.0 / (1.0 + 10 ** (-((elo[h] + 100) - elo[a]) / 400))
            elo[h] = elo[h] + 20 * (s_home - e_home)
            elo[a] = elo[a] + 20 * ((1 - s_home) - (1 - e_home))

            pts_h = np.where(res == "H", 3, np.where(res == "D", 1, 0))
            pts_a = np.where(res == "H", 0, np.where(res == "D", 1, 3))
            form5[h] = np.roll(form5[h], -1, axis=1); form5[h][:, -1] = pts_h
            form5[a] = np.roll(form5[a], -1, axis=1); form5[a][:, -1] = pts_a

            pts[h] += pts_h; pts[a] += pts_a
            gf[h] += g_h; ga[h] += g_a
            gf[a] += g_a; ga[a] += g_h
            wins[h] += (res == "H").astype(int); losses[h] += (res == "A").astype(int); draws[h] += (res == "D").astype(int)
            wins[a] += (res == "A").astype(int); losses[a] += (res == "H").astype(int); draws[a] += (res == "D").astype(int)

            pair = tuple(sorted([h, a]))
            for sim in range(N):
                h2h_state[sim][pair].append((h, res[sim]))

            prob_sum[fixture_row_idx] += proba_ens[i * N:(i + 1) * N].mean(axis=0)
            fixture_row_idx += 1

        print(f"  giornata {md:2d}/38 simulata")

    # -----------------------------------------------------------------
    # 4. Output: previsioni partite
    # -----------------------------------------------------------------
    # prob_sum[j] e' gia' la probabilita' media sulle N simulazioni per la partita j
    # (accumulata una sola volta per partita, ciascuna gia' mediata con .mean(axis=0)):
    # NON va divisa di nuovo per N.
    prob_avg = prob_sum  # [A, D, H]
    out_matches = static_df[["matchday", "date", "home_team", "away_team"]].copy()
    out_matches["prob_home"] = (prob_avg[:, 2] * 100).round(1)
    out_matches["prob_draw"] = (prob_avg[:, 1] * 100).round(1)
    out_matches["prob_away"] = (prob_avg[:, 0] * 100).round(1)
    out_matches["esito_piu_probabile"] = out_matches[["prob_away", "prob_draw", "prob_home"]].idxmax(axis=1).map(
        {"prob_away": "Vittoria trasferta", "prob_draw": "Pareggio", "prob_home": "Vittoria casa"})
    out_matches.to_csv(OUT_DIR / "previsioni_partite_2026_27.csv", index=False)
    print(f"\nSalvato: {OUT_DIR / 'previsioni_partite_2026_27.csv'}")

    # -----------------------------------------------------------------
    # 5. Output: classifica prevista (media + probabilita' su N simulazioni)
    # -----------------------------------------------------------------
    rows = []
    # ranking per ogni simulazione (punti desc, DR desc, GF desc - no scontri diretti, vedi limiti)
    dr = {t: gf[t] - ga[t] for t in teams}
    rank_matrix = np.zeros((N, len(teams)), dtype=int)  # posizione (1-20) per squadra x simulazione
    team_order = teams
    for sim in range(N):
        scores = [(pts[t][sim], dr[t][sim], gf[t][sim], t) for t in team_order]
        scores.sort(key=lambda x: (-x[0], -x[1], -x[2]))
        for pos, (_, _, _, t) in enumerate(scores, start=1):
            rank_matrix[sim, team_order.index(t)] = pos

    for idx, t in enumerate(team_order):
        positions = rank_matrix[:, idx]
        rows.append({
            "team": t,
            "punti_medi": round(pts[t].mean(), 1),
            "posizione_media": round(positions.mean(), 1),
            "GF_medi": round(gf[t].mean(), 1),
            "GS_medi": round(ga[t].mean(), 1),
            "DR_medio": round(dr[t].mean(), 1),
            "V_medie": round(wins[t].mean(), 1),
            "N_medie": round(draws[t].mean(), 1),
            "P_medie": round(losses[t].mean(), 1),
            "prob_titolo_%": round((positions == 1).mean() * 100, 1),
            "prob_champions_top4_%": round((positions <= 4).mean() * 100, 1),
            "prob_europa_top6_%": round((positions <= 6).mean() * 100, 1),
            "prob_retrocessione_%": round((positions >= 18).mean() * 100, 1),
        })
    classifica_prevista = pd.DataFrame(rows).sort_values("posizione_media").reset_index(drop=True)
    classifica_prevista.insert(0, "pos", range(1, len(classifica_prevista) + 1))
    classifica_prevista.to_csv(OUT_DIR / "classifica_prevista_2026_27.csv", index=False)
    print(f"Salvato: {OUT_DIR / 'classifica_prevista_2026_27.csv'}\n")

    print("=== Classifica prevista 2026/2027 (media su {} simulazioni) ===".format(N))
    print(classifica_prevista.to_string(index=False))


if __name__ == "__main__":
    main()
