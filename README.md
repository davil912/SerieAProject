# Serie A Predictor

Progetto Python per addestrare un modello in grado di prevedere gli esiti (1X2)
delle partite di Serie A 2026/2027, con classifica aggiornata automaticamente
ad ogni giornata giocata.

## Struttura

```
serieA_predictor/
├── data/
│   ├── raw/            # dati grezzi scaricati (storico partite Serie A e Serie B)
│   ├── processed/      # dati puliti, pronti per analisi e training
│   └── incoming/        # CSV settimanale scaricato a mano durante la stagione
├── src/
│   ├── common/          # moduli condivisi: elo_updater.py, classifica.py, poisson_model.py
│   ├── preprocessing/   # pulizia dati e feature engineering
│   ├── training/         # addestramento dei modelli
│   ├── simulation/       # simulazione Monte Carlo della stagione
│   └── live_update/      # integrazione risultati reali durante la stagione
├── notebooks/           # script/notebook di analisi esplorativa e relativi grafici
├── models/               # modelli addestrati (non versionati)
├── dashboard/             # dashboard HTML autonoma (classifica, partite, cronologia, evoluzione previsione)
└── requirements.txt
```

Vedi `GUIDA_ESECUZIONE_LOCALE.md` per l'elenco completo dei comandi, nell'ordine
in cui vanno eseguiti.

## Stato del progetto

Fasi 1-6 completate: raccolta dati ed esplorazione, feature engineering e
modello baseline, feature Poisson e XGBoost, ensemble e classifica live,
simulazione Monte Carlo della stagione 2026/2027, aggiornamento ibrido
reale/simulato durante il campionato. Dettaglio di ciascuna fase nei rispettivi
`FASE*_REPORT.md`.

## Setup

```bash
pip install -r requirements.txt
python src/preprocessing/prepare_data.py     # rigenera i dati puliti in data/processed/
python src/common/classifica.py --season 2023/2024   # esempio: calcola una classifica
```

## Fonte dati

Storico partite da [Football-Data.co.uk](https://www.football-data.co.uk/),
tramite il mirror GitHub
[xgabora/Club-Football-Match-Data-2000-2025](https://github.com/xgabora/Club-Football-Match-Data-2000-2025).
