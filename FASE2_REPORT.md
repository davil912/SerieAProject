# Fase 2 — Feature engineering e modello baseline — Report

## Cosa è stato fatto

1. **Verifica anti-leakage**: controllate a mano le colonne Elo e forma (Form3/Form5) già presenti nel dataset — confermato che riflettono sempre lo stato **prima** della partita (nessuna fuga di informazioni dal futuro).
2. **Feature engineering** (`src/feature_builder.py`), costruite scorrendo le partite in ordine cronologico:
   - `elo_diff`, `home_elo`, `away_elo` — forza delle squadre (rating Elo da ClubElo).
   - `form3_diff`, `form5_diff` — differenza di forma nelle ultime 3/5 partite.
   - `home_advantage_recent` — % di vittorie casalinghe nelle 200 partite di campionato precedenti (cattura il trend calante del vantaggio-casa individuato in Fase 1), calcolata **senza** includere la partita corrente.
   - `rest_days_home/away/diff` — giorni di riposo dall'ultima partita di campionato.
   - `games_played_home/away`, `season_progress` — avanzamento nella stagione.
   - `h2h_home_ppg`, `h2h_n_precedenti` — punti-a-partita della squadra di casa negli ultimi 5 scontri diretti.
   - Probabilità implicite di mercato (`market_prob_home/draw/away`), calcolate dalle quote bookmaker ma **tenute fuori dal training** e usate solo come benchmark esterno.
   - Nota: il **valore di mercato delle rose (Transfermarkt)** non è stato incluso in questa fase — il sito non è raggiungibile dall'ambiente di elaborazione e le fonti alternative trovate (dataset GitHub) non erano scaricabili direttamente. Rimane un'estensione possibile per iterazioni future.
3. **Modello baseline** (`src/train_baseline.py`): Logistic Regression multinomiale, con **validazione walk-forward** stagione per stagione (mai split casuale) su 15 stagioni di backtest (2010/11 → 2024/25): si allena solo su stagioni passate, si valuta sulla stagione successiva, si avanza.

## Risultati del backtest

| Modello | Accuracy | Log-loss | Brier |
|---|---|---|---|
| **Quote di mercato** (benchmark) | 54,6% | 0,960 | 0,570 |
| **Logistic Regression (nostro modello)** | 54,0% | 0,975 | 0,580 |
| Baseline "vince sempre la casa" | 43,6% | 1,737 | 0,975 |

Il modello baseline, con sole 10 feature semplici, **si avvicina molto alle quote di mercato dei bookmaker** (che incorporano informazioni non disponibili al modello, come notizie su formazioni e infortuni) e batte nettamente la baseline banale. È un risultato solido per un primo modello: conferma che l'intera pipeline — dati, feature, training, validazione — funziona correttamente end-to-end, com'era l'obiettivo della Fase 2. Nel dettaglio, in 6 delle 15 stagioni testate il modello ha eguagliato o superato la precisione delle quote di mercato (grafico allegato).

**Coefficienti del modello finale** (feature standardizzate, quindi confrontabili in grandezza): `elo_diff` è di gran lunga la feature più importante (coefficiente più alto in valore assoluto sia per la classe H che per la A), seguita da `home_elo`/`away_elo` individualmente e da `home_advantage_recent`. Forma recente e scontri diretti pesano meno ma nella direzione attesa.

## File prodotti

```
serieA_predictor/
├── data/processed/serieA_features.csv       # dataset con tutte le feature
├── src/feature_builder.py                    # costruzione feature
├── src/train_baseline.py                     # training + backtest walk-forward
├── models/baseline_logreg.pkl                 # modello addestrato (non versionato su git)
├── notebooks/backtest_chart.py + backtest_logloss.png
└── notebooks/backtest_results.csv             # metriche dettagliate per stagione/modello
```

## Prossimi passi (Fase 3)

Passare a un modello Gradient Boosting (XGBoost/LightGBM), che dovrebbe catturare meglio le interazioni non lineari tra le feature; valutare l'aggiunta del modello Poisson/Dixon-Coles come generatore di feature aggiuntive (gol attesi); calibrazione esplicita delle probabilità; eventualmente recuperare la stagione 2025/26 mancante e i dati di valore di mercato per arricchire ulteriormente le feature.
