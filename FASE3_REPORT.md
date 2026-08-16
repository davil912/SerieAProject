# Fase 3 — Modello Poisson, XGBoost e calibrazione — Report

## Cosa è stato fatto

1. **Modello Poisson/Dixon-Coles semplificato** (`src/poisson_model.py`): rating di attacco e difesa per squadra stimati con una regressione Poisson regolarizzata (ridge), rifittata **una volta per stagione** su una finestra mobile delle **ultime 5 stagioni** precedenti (mai tutta la storia: la forza di una squadra di 15 anni fa non è indicativa di quella attuale). Semplificazione dichiarata rispetto al Dixon-Coles originale: non è incluso il fattore di correlazione ρ per i risultati bassi (0-0, 1-0, ecc.) — accettabile perché qui il modello è usato come *generatore di feature*, non come predittore finale.
2. **Feature aggiuntive** (`src/build_poisson_features.py`): gol attesi casa/trasferta, loro differenza, e probabilità 1X2 implicite dal modello Poisson — aggiunte al dataset di feature della Fase 2 (ora 16 feature totali).
3. **Modello XGBoost** (`src/train_xgboost.py`), stesso schema di validazione walk-forward della Fase 2 (train solo su stagioni passate). Micro-confronto manuale di iperparametri (non una grid search esaustiva) ha mostrato che alberi poco profondi e molta regolarizzazione danno risultati più stabili — coerente con un dataset di poche migliaia di partite e un target intrinsecamente rumoroso.
4. **Calibrazione delle probabilità**: tenuta da parte l'ultima stagione di ogni finestra di training come set di calibrazione (mai la stagione di test), poi calibrazione **sigmoid** (Platt scaling). Nota: il primo tentativo con calibrazione **isotonic** ha peggiorato nettamente il log-loss (overfitting sulla calibrazione con appena ~380 partite per 3 classi) — un promemoria pratico che l'isotonic richiede più dati di quanti ne avessimo disponibili per singola stagione di calibrazione.

## Risultati del backtest (stesse 15 stagioni, 2010/11 → 2024/25)

| Modello | Accuracy | Log-loss | Brier |
|---|---|---|---|
| Quote di mercato (benchmark) | 54,6% | 0,960 | 0,570 |
| **Logistic Regression (Fase 2)** | **54,0%** | **0,975** | **0,580** |
| XGBoost (Fase 3) | 53,5% | 0,979 | 0,583 |
| XGBoost calibrato (Fase 3) | 53,4% | 0,981 | 0,584 |
| Baseline "vince sempre la casa" | 43,6% | 1,737 | 0,975 |

## Risultato onesto: XGBoost non ha battuto la Logistic Regression

Va detto chiaramente: **il modello XGBoost, pur con 6 feature aggiuntive dal modulo Poisson, non ha superato la semplice Logistic Regression della Fase 2** sul backtest — è leggermente peggiore sia in accuracy che in log-loss, anche se molto vicino. Non è un errore né un fallimento della pipeline: è un risultato plausibile e non raro nella letteratura sulle previsioni calcistiche, per due motivi concreti:

- Il dataset di training è relativamente piccolo (poche migliaia di partite per finestra), e gli esiti delle partite di calcio hanno una componente di rumore molto alta — condizioni in cui un modello lineare ben regolarizzato spesso generalizza meglio di un modello più flessibile come il gradient boosting, che tende a modellare rumore invece di segnale reale.
- Le feature più informative (differenza Elo, probabilità Poisson) hanno relazioni abbastanza lineari/monotone con l'esito: in questi casi il vantaggio tipico degli alberi (catturare interazioni non lineari complesse) pesa poco.

**Nota positiva**: la feature importance di XGBoost conferma che le nuove feature Poisson sono tra le più informative in assoluto (`poisson_prob_home`, `poisson_exp_goals_diff` e `poisson_prob_away` sono nella top-4 insieme a `elo_diff`) — il modulo Poisson aggiunge segnale utile, anche se in questa configurazione non basta a far vincere XGBoost sulla Logistic Regression.

## File prodotti

```
serieA_predictor/
├── src/poisson_model.py               # modello Poisson/Dixon-Coles semplificato
├── src/build_poisson_features.py      # aggiunge le feature Poisson al dataset
├── src/train_xgboost.py                # training XGBoost + calibrazione + backtest
├── models/xgboost_final.pkl            # modello finale (non versionato su git)
├── notebooks/compare_all_models.py + final_comparison.png
├── notebooks/backtest_results_fase3.csv
└── notebooks/final_comparison_summary.csv
```

## Prossimi passi (Fase 4)

Dato che al momento **la Logistic Regression resta il modello di riferimento**, ha senso: (a) provare un ensemble Logistic Regression + XGBoost (media pesata) piuttosto che scegliere un vincitore netto; (b) recuperare la stagione 2025/26 mancante per aggiornare classifica e feature all'attualità; (c) costruire la pipeline di aggiornamento automatico e il modulo classifica "live"; (d) valutare il valore di mercato delle rose se si trova una fonte accessibile, come feature aggiuntiva.
