# Serie A Predictor

Progetto Python per addestrare un modello in grado di prevedere gli esiti (1X2)
delle partite di Serie A 2026/2027, con classifica aggiornata automaticamente
ad ogni giornata giocata.

## Struttura

```
serieA_predictor/
├── data/
│   ├── raw/          # dati grezzi scaricati (storico partite Serie A e Serie B)
│   └── processed/     # dati puliti, pronti per analisi e training
├── src/
│   ├── prepare_data.py   # pulizia e unificazione dati grezzi
│   └── classifica.py     # calcolo classifica (anche "congelata" a una data)
├── notebooks/         # script/notebook di analisi esplorativa e relativi grafici
├── models/             # modelli addestrati (non versionati)
├── dashboard/           # report/visualizzazioni
└── requirements.txt
```

## Stato del progetto

- **Fase 1 — Raccolta dati ed esplorazione**: completata. Vedi `FASE1_REPORT.md`
  per il dettaglio (9.012 partite Serie A 2000/01-2024/25, classifica verificata
  contro dati ufficiali, grafici esplorativi).
- **Fase 2 — Feature engineering e modello baseline**: in corso.

## Setup

```bash
pip install -r requirements.txt
python src/prepare_data.py     # rigenera i dati puliti in data/processed/
python src/classifica.py --season 2023/2024   # esempio: calcola una classifica
```

## Fonte dati

Storico partite da [Football-Data.co.uk](https://www.football-data.co.uk/),
tramite il mirror GitHub
[xgabora/Club-Football-Match-Data-2000-2025](https://github.com/xgabora/Club-Football-Match-Data-2000-2025).
