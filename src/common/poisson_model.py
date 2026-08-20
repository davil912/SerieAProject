"""
Fase 3 - Modello Poisson (stile Dixon-Coles semplificato) come generatore di feature.

Stima un rating di attacco e uno di difesa per ogni squadra, con un effetto
fisso di vantaggio-casa, tramite una regressione Poisson:

    E[gol segnati da una squadra] = exp(home_adv * is_casa + attacco_squadra - difesa_avversario)

Il modello viene rifittato una sola volta per stagione, usando SOLO le
partite delle stagioni precedenti (nessun leakage): è la stessa logica
walk-forward già usata per il modello baseline e per il backtest.

Semplificazione dichiarata rispetto al Dixon-Coles originale (1997): non è
incluso il fattore di correlazione rho per i risultati bassi (0-0, 1-0, 0-1,
1-1) - qui il modello è usato come GENERATORE DI FEATURE per un modello a
valle (XGBoost), non come predittore finale, quindi la semplificazione ha
un impatto limitato.

Uso:
    python src/common/poisson_model.py   # esegue un self-test rapido
"""

from pathlib import Path
from dataclasses import dataclass
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

MAX_GOALS = 8  # griglia per il calcolo delle probabilità 1X2 dalla distribuzione di Poisson


@dataclass
class PoissonModel:
    teams: list
    home_adv: float
    attack: dict     # squadra -> rating attacco
    defense: dict     # squadra -> rating difesa
    attack_avg: float
    defense_avg: float

    def expected_goals(self, home_team: str, away_team: str):
        a_home = self.attack.get(home_team, self.attack_avg)
        d_home = self.defense.get(home_team, self.defense_avg)
        a_away = self.attack.get(away_team, self.attack_avg)
        d_away = self.defense.get(away_team, self.defense_avg)
        lam_home = np.exp(self.home_adv + a_home - d_away)
        lam_away = np.exp(a_away - d_home)
        return lam_home, lam_away

    def match_probs(self, home_team: str, away_team: str):
        lam_home, lam_away = self.expected_goals(home_team, away_team)
        g = np.arange(0, MAX_GOALS + 1)
        factorials = np.array([math.factorial(i) for i in g])
        p_home_goals = np.exp(-lam_home) * lam_home ** g / factorials
        p_away_goals = np.exp(-lam_away) * lam_away ** g / factorials
        grid = np.outer(p_home_goals, p_away_goals)  # grid[i,j] = P(home=i, away=j)
        p_home = np.tril(grid, -1).sum()
        p_draw = np.trace(grid)
        p_away = np.triu(grid, 1).sum()
        tot = p_home + p_draw + p_away
        return lam_home, lam_away, p_home / tot, p_draw / tot, p_away / tot


def fit_poisson_model(train_df: pd.DataFrame, alpha: float = 0.01) -> PoissonModel:
    """Fitta il modello attacco/difesa su un set di partite storiche (train_df)."""
    teams = sorted(set(train_df["home_team"]) | set(train_df["away_team"]))
    team_idx = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)

    rows_attack = []
    rows_defense = []
    rows_home_ind = []
    y_goals = []

    for r in train_df.itertuples(index=False):
        # riga per i gol della squadra di casa
        rows_attack.append(team_idx[r.home_team])
        rows_defense.append(team_idx[r.away_team])
        rows_home_ind.append(1)
        y_goals.append(r.home_goals)
        # riga per i gol della squadra ospite
        rows_attack.append(team_idx[r.away_team])
        rows_defense.append(team_idx[r.home_team])
        rows_home_ind.append(0)
        y_goals.append(r.away_goals)

    n_rows = len(y_goals)
    X = np.zeros((n_rows, 2 * n_teams + 1))
    X[np.arange(n_rows), rows_attack] = 1
    X[np.arange(n_rows), n_teams + np.array(rows_defense)] = -1
    X[:, -1] = rows_home_ind
    y = np.array(y_goals, dtype=float)

    # Regolarizzazione L2 (alpha) necessaria: senza vincoli il modello non e'
    # identificabile (attacco/difesa di tutte le squadre potrebbero traslare
    # di una costante). Il ridge tiene i rating centrati e stabili.
    reg = PoissonRegressor(alpha=alpha, max_iter=500)
    reg.fit(X, y)

    coef = reg.coef_
    attack = {t: coef[team_idx[t]] for t in teams}
    defense = {t: coef[n_teams + team_idx[t]] for t in teams}
    home_adv = coef[-1] + reg.intercept_

    return PoissonModel(
        teams=teams, home_adv=home_adv, attack=attack, defense=defense,
        attack_avg=float(np.mean(list(attack.values()))),
        defense_avg=float(np.mean(list(defense.values()))),
    )


if __name__ == "__main__":
    BASE = Path(__file__).resolve().parent.parent.parent
    df = pd.read_csv(BASE / "data" / "processed" / "serieA_matches.csv", parse_dates=["date"])
    train = df[df["season"] < "2023/2024"]
    model = fit_poisson_model(train)
    print("Self-test: gol attesi e probabilita' per alcune partite 2023/2024\n")
    test = df[df["season"] == "2023/2024"].head(5)
    for r in test.itertuples(index=False):
        lam_h, lam_a, ph, pd_, pa = model.match_probs(r.home_team, r.away_team)
        print(f"{r.home_team:12s} vs {r.away_team:12s} | xG {lam_h:.2f}-{lam_a:.2f} | "
              f"P(H)={ph:.2f} P(D)={pd_:.2f} P(A)={pa:.2f} | risultato reale: {r.home_goals:.0f}-{r.away_goals:.0f} ({r.result})")
