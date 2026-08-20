# data/incoming/ — aggiornamento settimanale (Fase 6)

Cartella "buca delle lettere": ci va appoggiato il CSV formato football-data.co.uk
con i risultati della stagione 2026/2027 **aggiornati ad oggi** (il file che
football-data.co.uk aggiorna settimana dopo settimana con TUTTE le partite
giocate finora, non solo le ultime).

Nome atteso: `serieA_2026_27.csv` (sovrascrivibile ogni settimana con la
versione piu' recente scaricata).

## Perche' manuale

Il progetto ha provato ad automatizzare il download (`src/update_pipeline.py`),
ma le due fonti disponibili si sono rivelate inaffidabili per la stagione in
corso: il mirror GitHub storico usato finora si e' fermato al 2025 e non
arriva alla stagione 2026/2027, mentre football-data.co.uk blocca le
richieste dirette dall'ambiente cloud di questa sessione (risposta 403).
Il download manuale (dal sito, nel browser) resta quindi il modo piu'
affidabile per procurarsi i dati aggiornati.

## Cosa succede quando arriva un file qui

Lanciare (a mano, o via il task pianificato settimanale):

```
python src/integrate_new_season.py data/incoming/serieA_2026_27.csv 2026/2027
python src/prepare_data.py
python src/feature_builder.py
python src/build_poisson_features.py
python src/predict_season.py
python dashboard/build_dashboard.py
```

`integrate_new_season.py` e' IDEMPOTENTE (Fase 6): si puo' rilanciare ogni
settimana con il file scaricato di nuovo (che contiene sempre tutte le
partite dalla giornata 1 ad oggi) - vengono integrate solo le partite non
ancora presenti, senza duplicati e senza sballare l'Elo.

`predict_season.py` si accorge da solo delle partite 2026/2027 gia'
integrate: le usa come classifica reale di partenza e simula solo le
giornate rimanenti (vedi FASE6_AGGIORNAMENTO_LIVE.md).
