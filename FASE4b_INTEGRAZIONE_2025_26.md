# Integrazione stagione 2025/2026 — Report

## Cosa è stato fatto

Hai fornito il file `I1.csv` (formato originale Football-Data.co.uk) con le 380 partite complete della stagione 2025/2026. A differenza delle stagioni precedenti — arrivate già complete di Elo e forma dal mirror GitHub usato in Fase 1 — questa fonte non include quei valori pre-calcolati, quindi ho costruito un piccolo motore Elo proprio:

- **`src/elo_updater.py`**: formula Elo standard (K=20, vantaggio-casa=100 punti Elo — gli stessi parametri storicamente usati dal World Football Elo Ratings). Semplificata rispetto alla metodologia esatta di ClubElo (che pesa anche il margine di vittoria), ma coerente per continuare i rating.
- **`src/integrate_new_season.py`**: recupera l'ultimo Elo noto per ciascuna squadra (da Serie A o, se più recente — utile per le neopromosse — da Serie B), lo aggiorna partita per partita attraverso la nuova stagione, e calcola forma (ultimi 3/5 risultati) allo stesso modo. Le tre neopromosse (Cremonese, Pisa, Sassuolo) sono partite con l'Elo della loro ultima partita di Serie B (maggio 2025), molto più informativo di un valore "a caso".

## Verifica

Ho calcolato la classifica finale 2025/2026 dai dati integrati e l'ho confrontata con fonti esterne (Wikipedia, calcio.com): **coincide esattamente**, fino ai dettagli di vittorie/pareggi/sconfitte e gol fatti/subiti — Inter campione con 87 punti (27V-6N-5P, 89-35), fino alle retrocesse Cremonese/Verona/Pisa. Il file che hai caricato è affidabile al 100%.

## Impatto sui modelli: una vera validazione fuori-campione

Con la nuova stagione, il backtest walk-forward passa da 15 a **16 stagioni** (fino al 2025/26, mai vista durante tutto lo sviluppo del progetto finora — la validazione più genuina possibile). I risultati confermano quanto visto finora:

| Modello | Accuracy (16 stagioni) | Log-loss |
|---|---|---|
| Quote di mercato | 54,6% | 0,961 |
| **Logistic Regression** | 53,8% | 0,977 |
| **Ensemble (w=0,8)** | 53,7% | 0,976 |
| XGBoost | 53,2% | 0,981 |
| Baseline "vince sempre casa" | 43,3% | 1,745 |

L'ensemble si conferma con il miglior log-loss sull'holdout imparziale (0,973 sulle ultime 6 stagioni, incluso il 2025/26) tra i modelli propri — stesso peso w=0,8 già scelto in precedenza, quindi la scelta non era casuale.

## File aggiornati

```
serieA_predictor/
├── data/raw/serieA_2025_26_football-data.csv   # file originale caricato (copia per tracciabilità)
├── data/processed/serieA_matches.csv            # ora include 380 partite 2025/26 (7980 totali, 26 stagioni)
├── data/processed/serieA_features.csv           # feature ricalcolate su tutto lo storico aggiornato
├── src/elo_updater.py                            # motore Elo proprio
├── src/integrate_new_season.py                  # script di integrazione (riusabile per stagioni future)
├── models/baseline_logreg.pkl, xgboost_final.pkl # ri-addestrati sui dati aggiornati
└── notebooks/*.png, *.csv                        # grafici e risultati backtest aggiornati (16 stagioni)
```

## Nota per il futuro

`src/integrate_new_season.py` è riusabile: quando inizierà il campionato 2026/2027 potrai scaricare via browser i risultati via via disponibili (stesso formato Football-Data.co.uk) e rilanciare lo script per tenere tutto aggiornato, oppure aspettare che il mirror GitHub della Fase 1 pubblichi la stagione e usare `update_pipeline.py` come fatto finora.
