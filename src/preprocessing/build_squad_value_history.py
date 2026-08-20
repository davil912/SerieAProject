"""
Fase 4 (estensione) - Valore rosa STORICO per stagione, da player_valuations.csv
(Transfermarkt, via dataset Kaggle davidcariboo/player-scores).

Nota metodologica importante: il file NON contiene un "club al momento della
valutazione" ovvio a colpo d'occhio, ma verificando a mano (vedi conversazione)
il campo `current_club_name` è in realtà point-in-time (cambia correttamente
nel tempo seguendo i trasferimenti del giocatore) - PERÒ i nomi club su
Transfermarkt cambiano nel tempo (rifondazioni societarie, es. Parma, Salernitana)
e un semplice text-matching per sottostringa produce falsi positivi pericolosi
(es. "Feralpisalò" contiene "pisa" ma è un club diverso da Pisa SC). Per questo
la mappatura qui sotto è stata curata a mano squadra per squadra, guardando i
nomi effettivamente presenti nel file, invece che generata automaticamente.

SCOPE DICHIARATO: la mappatura copre le squadre attive in Serie A dalla
stagione 2018/19 in poi (32 squadre) - le stagioni piu' vecchie del nostro
storico (2005/06-2017/18) NON hanno il valore rosa disponibile, perche'
estendere la mappatura ai nomi storici di quell'epoca (con ulteriori
rifondazioni societarie da verificare una per una) avrebbe un rischio di
errore silenzioso troppo alto per il beneficio - meglio lasciare il dato
mancante che sbagliarlo.

Uso:
    python src/preprocessing/build_squad_value_history.py
"""

from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent.parent
# Percorso relativo al progetto (portabile in locale) - il file va scaricato a parte
# dal dataset Kaggle davidcariboo/player-scores (player_valuations.csv, non incluso
# nel repo per dimensione) e messo qui prima di lanciare questo script.
VALUATIONS_PATH = BASE / "data" / "raw" / "player_valuations.csv"
OUT_PATH = BASE / "data" / "processed" / "squad_values_by_season.csv"

# squadra (nome usato nel nostro dataset) -> lista di alias esatti su Transfermarkt
# (curata a mano; SOLO nomi verificati della prima squadra, escluse Primavera/Under/Youth)
TEAM_ALIASES = {
    "Atalanta": ["Atalanta BC"],
    "Benevento": ["Benevento Calcio"],
    "Bologna": ["Bologna FC 1909"],
    "Brescia": ["Brescia Calcio", "Brescia Calcio (- 2025)"],
    "Cagliari": ["Cagliari Calcio"],
    "Chievo": ["Chievo Verona"],
    "Como": ["Como 1907", "Calcio Como"],
    "Cremonese": ["US Cremonese"],
    "Crotone": ["FC Crotone"],
    "Empoli": ["FC Empoli", "Empoli FC "],
    "Fiorentina": ["ACF Fiorentina"],
    "Frosinone": ["Frosinone Calcio"],
    "Genoa": ["Genoa CFC"],
    "Inter": ["Inter Milan"],
    "Juventus": ["Juventus FC"],
    "Lazio": ["SS Lazio"],
    "Lecce": ["US Lecce"],
    "Milan": ["AC Milan"],
    "Monza": ["AC Monza"],
    "Napoli": ["SSC Napoli", "Napoli Soccer"],
    "Parma": ["Parma Calcio 1913", "Parma FC"],
    "Pisa": ["Pisa Sporting Club", "AC Pisa 1909"],
    "Roma": ["AS Roma"],
    "Salernitana": ["US Salernitana 1919", "US Salernitana", "Salernitana Calcio 1919"],
    "Sampdoria": ["UC Sampdoria"],
    "Sassuolo": ["US Sassuolo"],
    "Spal": ["SPAL 1907", "SPAL 2013", "SPAL"],
    "Spezia": ["Spezia Calcio", "Spezia Calcio 1906"],
    "Torino": ["Torino FC"],
    "Udinese": ["Udinese Calcio"],
    "Venezia": ["Venezia FC", "SSC Venezia"],
    "Verona": ["Hellas Verona"],
}


def assign_season(date: pd.Timestamp) -> str:
    if date.month >= 7:
        return f"{date.year}/{date.year + 1}"
    return f"{date.year - 1}/{date.year}"


def main():
    if not VALUATIONS_PATH.exists():
        print(f"[Errore] file non trovato: {VALUATIONS_PATH}. Verifica di averlo staged dalla cartella connessa.")
        return

    df = pd.read_csv(VALUATIONS_PATH, parse_dates=["date"], usecols=[
        "player_id", "date", "market_value_in_eur", "current_club_name", "player_club_domestic_competition_id"
    ])
    df = df[df["player_club_domestic_competition_id"].isin(["IT1", "IT2"])]

    alias_to_team = {alias: team for team, aliases in TEAM_ALIASES.items() for alias in aliases}
    df = df[df["current_club_name"].isin(alias_to_team)]
    df["team"] = df["current_club_name"].map(alias_to_team)
    df["season"] = df["date"].apply(assign_season)

    # per ogni giocatore, in ogni stagione, tengo solo la valutazione piu' recente
    # (puo' capitare piu' di una rilevazione a stagione, es. gennaio e giugno)
    df = df.sort_values("date")
    latest_per_player_season = df.groupby(["team", "season", "player_id"]).tail(1)

    squad_value = latest_per_player_season.groupby(["team", "season"])["market_value_in_eur"].agg(
        valore_rosa_eur="sum", n_giocatori="count"
    ).reset_index()
    squad_value["valore_rosa_milioni"] = (squad_value["valore_rosa_eur"] / 1e6).round(1)

    # Copertura del dataset molto scarsa nelle stagioni piu' vecchie (poche decine
    # di giocatori valutati su Transfermarkt a meta' anni 2000): sotto una soglia
    # minima di giocatori censiti il numero non e' rappresentativo del valore
    # reale della rosa, quindi lo marchiamo come non affidabile invece di tenerlo.
    MIN_PLAYERS_RELIABLE = 15
    squad_value["affidabile"] = squad_value["n_giocatori"] >= MIN_PLAYERS_RELIABLE
    n_unreliable = (~squad_value["affidabile"]).sum()
    print(f"Righe sotto la soglia di affidabilita' ({MIN_PLAYERS_RELIABLE} giocatori valutati): {n_unreliable} su {len(squad_value)} (marcate affidabile=False, non eliminate)")

    squad_value.to_csv(OUT_PATH, index=False)
    print(f"Righe prodotte: {len(squad_value)} (squadra x stagione)")
    print(f"Squadre coperte: {sorted(squad_value['team'].unique())}")
    print(f"Stagioni coperte: {sorted(squad_value['season'].unique())}")

    print("\nEsempio - Inter nel tempo:")
    print(squad_value[squad_value.team == "Inter"].sort_values("season")[["season", "valore_rosa_milioni", "n_giocatori"]].to_string(index=False))

    print(f"\nSalvato: {OUT_PATH}")


if __name__ == "__main__":
    main()
