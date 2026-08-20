"""
Fase 4 (estensione) - Valore rosa attuale (2025/26) da Transfermarkt (via
dataset Kaggle davidcariboo/player-scores, file players.csv).

E' uno SNAPSHOT attuale (valore di mercato dei giocatori all'ultimo
aggiornamento del dataset), non uno storico stagione per stagione - per
quello serve player_valuations.csv (vedi src/build_squad_value_history.py,
da completare quando disponibile). Usare questa versione come feature
"statica" nel frattempo e' comunque ragionevole: il valore di una rosa
cambia lentamente rispetto alla forza/Elo, quindi un solo snapshot recente
resta informativo anche per confrontare le squadre tra loro.

Uso:
    python src/preprocessing/build_squad_value_current.py
"""

from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent.parent
PLAYERS_PATH = BASE / "data" / "raw" / "transfermarkt_players.csv"
OUT_PATH = BASE / "data" / "processed" / "squad_values_current.csv"

# Nome squadra (come usato nel nostro dataset partite) -> nome club su Transfermarkt
TEAM_NAME_MAP = {
    "Atalanta": "Atalanta BC",
    "Bologna": "Bologna Football Club 1909",
    "Cagliari": "Cagliari Calcio",
    "Como": "Como 1907",
    "Cremonese": "US Cremonese",
    "Fiorentina": "ACF Fiorentina",
    "Genoa": "Genoa CFC",
    "Inter": "Inter Milan",
    "Juventus": "Juventus FC",
    "Lazio": "Società Sportiva Lazio S.p.A.",
    "Lecce": "US Lecce",
    "Milan": "AC Milan",
    "Napoli": "SSC Napoli",
    "Parma": "Parma Calcio 1913",
    "Pisa": "Pisa Sporting Club",
    "Roma": "Associazione Sportiva Roma",
    "Sassuolo": "US Sassuolo",
    "Torino": "Torino FC",
    "Udinese": "Udinese Calcio",
    "Verona": "Hellas Verona",
}


def main():
    players = pd.read_csv(PLAYERS_PATH, low_memory=False)
    players = players[players["current_club_domestic_competition_id"] == "IT1"]

    reverse_map = {v: k for k, v in TEAM_NAME_MAP.items()}
    players = players[players["current_club_name"].isin(reverse_map)]
    players["team"] = players["current_club_name"].map(reverse_map)

    missing = players["market_value_in_eur"].isna().sum()
    print(f"Giocatori Serie A trovati: {len(players)} (di cui {missing} senza valore di mercato, esclusi dalla somma)")

    squad_value = players.groupby("team")["market_value_in_eur"].agg(
        valore_rosa_eur="sum", n_giocatori_valutati="count"
    ).reset_index()
    squad_value["valore_rosa_milioni"] = (squad_value["valore_rosa_eur"] / 1e6).round(1)
    squad_value = squad_value.sort_values("valore_rosa_eur", ascending=False)

    # verifica che tutte le 20 squadre attuali siano state trovate
    found = set(squad_value["team"])
    missing_teams = set(TEAM_NAME_MAP) - found
    if missing_teams:
        print(f"[Attenzione] squadre non trovate nel file giocatori: {missing_teams}")

    squad_value.to_csv(OUT_PATH, index=False)
    print(f"\nValore rosa Serie A 2025/26 (snapshot attuale):")
    print(squad_value[["team", "valore_rosa_milioni", "n_giocatori_valutati"]].to_string(index=False))
    print(f"\nSalvato: {OUT_PATH}")


if __name__ == "__main__":
    main()
