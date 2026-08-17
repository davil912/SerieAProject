# Fase 4 — Ensemble, pipeline di aggiornamento e classifica live — Report

## Cosa è stato fatto

### 1. Ensemble Logistic Regression + XGBoost (`src/train_ensemble.py`)

Anziché scegliere un vincitore netto tra i due modelli della Fase 3 (che si erano rivelati molto vicini), le probabilità vengono combinate con una media pesata: `p_ensemble = w · p_logreg + (1-w) · p_xgboost`.

Per scegliere il peso `w` in modo onesto, le 15 stagioni di backtest sono state divise in due blocchi cronologici: le prime 10 (2010/11-2019/20) usate per **scegliere** il peso, le ultime 5 (2020/21-2024/25) tenute da parte come **holdout** mai visto durante la scelta — così la valutazione finale non è "truccata".

**Peso scelto**: w = 0,8 (80% Logistic Regression, 20% XGBoost) — log-loss minimo sulle stagioni di selezione (grafico allegato).

**Risultato sull'holdout imparziale** (5 stagioni mai usate per scegliere il peso):

| Modello | Accuracy | Log-loss |
|---|---|---|
| **Ensemble (w=0,8)** | 53,6% | **0,9681** |
| Logistic Regression | 54,0% | 0,9685 |
| XGBoost | 53,2% | 0,9720 |

L'ensemble ottiene il **miglior log-loss** dei tre (probabilità leggermente più affidabili), pur con un'accuracy secca un filo sotto la sola Logistic Regression — coerente con l'idea che l'ensemble "ammorbidisce" le previsioni più estreme di un singolo modello, il che aiuta la log-loss (che penalizza gli errori sicuri) più che l'accuracy pura. Risultato analogo sulla media delle 15 stagioni totali: log-loss 0,9745 per l'ensemble contro 0,9751 della sola Logistic Regression.

### 2. Pipeline di aggiornamento automatico + classifica live (`src/update_pipeline.py`)

Script pensato per essere eseguito periodicamente (es. ogni lunedì dopo il weekend di campionato):
1. Riscarica la fonte dati.
2. Individua le partite non ancora presenti nello storico locale (confronto su data+squadre) e le accoda, senza duplicati.
3. Ricalcola e stampa la **classifica della stagione più recente disponibile** ("classifica live").
4. Se vengono trovate partite nuove, segnala che vanno rilanciati `feature_builder.py`, `build_poisson_features.py` ed eventualmente i training, per aggiornare feature e modelli — deliberatamente **non automatico**, per tenere separato l'aggiornamento dei dati dal retraining (che non ha senso rifare ad ogni singola partita, come indicato nel piano d'azione originale).

Test eseguito con successo: nessuna partita nuova trovata (la fonte non ha ancora pubblicato aggiornamenti oltre al 25 maggio 2025), classifica live stampata correttamente per la stagione 2024/2025 (Napoli campione con 82 punti — coerente con l'esito reale).

### 3. Stagione 2025/26 mancante — ancora non recuperata

Ho ritentato con una fonte diversa (fixturedownload.com), che risulta avere la stagione 2025/26 completa (380 partite, dal 23 agosto 2025 al 24 maggio 2026). Il download diretto del CSV, però, è esplicitamente vietato dal `robots.txt` del sito: non l'ho quindi scaricato, per rispetto delle condizioni d'uso dichiarate dalla fonte. La stagione 2025/26 resta un aggiornamento da fare con un'altra fonte (o inserendo manualmente i dati) quando disponibile.

## File prodotti

```
serieA_predictor/
├── src/train_ensemble.py                        # ensemble + scelta onesta del peso
├── src/update_pipeline.py                         # aggiornamento dati + classifica live
├── notebooks/ensemble_chart.py + ensemble_weight_search.png
├── notebooks/ensemble_results.csv, ensemble_weight_search.csv
```

## Stato dei modelli: l'ensemble diventa il riferimento

Con questa fase, il **modello di riferimento del progetto passa dall'ensemble** (Logistic Regression + XGBoost, w=0,8), che sull'holdout imparziale ha il miglior log-loss tra tutte le opzioni provate finora (Fase 2, Fase 3, Fase 4) — pur restando, come tutti i modelli testati, leggermente sotto le quote di mercato dei bookmaker (log-loss 0,960 sulle 15 stagioni).

## Prossimi passi

- Recuperare la stagione 2025/26 (e, appena disponibile, l'inizio della 2026/27) da una fonte che ne consenta legittimamente lo scaricamento.
- Automatizzare l'esecuzione di `update_pipeline.py` (es. tramite un'attività pianificata) una volta iniziato il campionato 2026/27.
- Rivalutare il valore di mercato delle rose come feature, se si trova una fonte accessibile.
- Estendere l'ensemble con il modello Poisson puro come terzo componente (attualmente usato solo come generatore di feature per XGBoost).
