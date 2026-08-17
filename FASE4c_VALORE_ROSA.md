# Valore rosa (Transfermarkt) — Report

## Cosa è stato fatto

Grazie ai due file che hai caricato (`players.csv` e `player_valuations.csv`, dataset Kaggle "Football Data from Transfermarkt" di davidcariboo) è stata costruita una feature "valore rosa" in due versioni:

### 1. Snapshot attuale (`src/build_squad_value_current.py`)
Somma del valore di mercato attuale di tutti i giocatori delle 20 squadre di Serie A 2025/26 (da `players.csv`). Numeri molto plausibili: Inter (644M€), Juventus (566M€), Roma (504M€), Milan (487M€), Napoli (477M€) in testa.

### 2. Storico per stagione (`src/build_squad_value_history.py`)
Da `player_valuations.csv` (storico dei valori nel tempo). **Nota tecnica importante**: il campo che indica il club di un giocatore in questo file cambia correttamente nel tempo seguendo i suoi trasferimenti — l'ho verificato tracciando un giocatore campione su 20 anni di carriera — ma i nomi dei club su Transfermarkt cambiano nel tempo (rifondazioni societarie: Parma, Salernitana, ecc.), e un semplice confronto per sottostringa produce falsi positivi pericolosi (es. "Feralpisalò" contiene "pisa" ma è tutt'altro club). Per questo motivo **ho curato a mano la mappatura nome-per-nome**, invece di generarla automaticamente — più lento ma senza rischio di inquinare i dati con abbinamenti sbagliati.

**Scope dichiarato**: la mappatura copre le 32 squadre attive in Serie A dal 2018/19 in poi. Le stagioni precedenti (2005/06-2017/18) non hanno il dato — estendere la mappatura a quell'epoca (con ulteriori rifondazioni societarie da verificare una per una) avrebbe un rischio di errore troppo alto rispetto al beneficio.

Inoltre, le stagioni con meno di 15 giocatori valutati su Transfermarkt (tipico di dati molto vecchi, quando il sito copriva meno giocatori) sono state marcate esplicitamente come "non affidabili" e trattate come dato mancante, non come uno zero o un numero falsato.

## Verifica

Ho confrontato, per la stagione 2025/26, il valore storico appena calcolato con lo snapshot attuale indipendente (metodo 1): la maggior parte delle squadre è entro il 10% di differenza (fisiologico, sono due istantanee prese in momenti diversi, con il mercato estivo di mezzo) — buona conferma incrociata che il metodo funziona correttamente.

## Impatto sui modelli: miglioramento concreto

Ho aggiunto `home_squad_value`, `away_squad_value` e `squad_value_diff` come feature a XGBoost e all'ensemble, poi rilanciato tutti i backtest.

**Test mirato** (le 8 stagioni dal 2018/19, dove il dato è disponibile al 100%): XGBoost passa da 52,9% a 53,6% di accuracy, e da 0,980 a 0,974 di log-loss — un miglioramento vero, non rumore.

**Backtest completo a 16 stagioni**, dopo l'integrazione:

| Modello | Accuracy | Log-loss |
|---|---|---|
| Quote di mercato | 54,6% | 0,961 |
| Logistic Regression | 53,8% | 0,977 |
| **XGBoost (ora con valore rosa)** | **53,7%** | **0,977** |
| Ensemble | 53,7% | 0,975 |

La differenza più interessante è sull'**holdout imparziale** (ultime 6 stagioni, 2020/21-2025/26): per la prima volta **XGBoost da solo supera la Logistic Regression** (log-loss 0,972 contro 0,974) — prima della feature valore-rosa era sempre stato leggermente indietro. Il peso ottimale dell'ensemble si è spostato da w=0,8 a **w=0,6** (60% Logistic Regression, 40% XGBoost, invece di 80/20): coerente col fatto che XGBoost ora "pesa" di più perché è diventato più forte. L'ensemble resta comunque la scelta migliore in assoluto (log-loss 0,971 sull'holdout).

Nella feature importance di XGBoost, `squad_value_diff` è ora la **5ª feature più importante** in assoluto (dopo le probabilità Poisson ed Elo) — conferma che il valore di mercato porta segnale che le altre feature (basate solo sui risultati delle partite) non catturavano.

## File prodotti/aggiornati

```
serieA_predictor/
├── data/raw/transfermarkt_players.csv              # players.csv caricato (copia)
├── data/processed/squad_values_current.csv          # snapshot attuale (20 squadre)
├── data/processed/squad_values_by_season.csv         # storico 2004/05-2025/26 (32 squadre, con flag affidabilità)
├── data/processed/serieA_features.csv                # ora include home/away/diff squad value
├── src/build_squad_value_current.py
├── src/build_squad_value_history.py
├── src/merge_squad_value_feature.py
├── src/train_xgboost.py, src/train_ensemble.py        # aggiornati con la nuova feature
└── notebooks/final_comparison.png, ensemble_weight_search.png   # aggiornati
```

(Nota: `player_valuations.csv`, 30MB, non è stato copiato nel repository per restare leggero — resta nella tua cartella locale `data/raw/`; se vuoi rigenerare `squad_values_by_season.csv` da zero, il file deve essere presente lì.)

## Limiti dichiarati

- Copertura storica solo dal 2018/19 (le stagioni precedenti hanno la feature mancante/NaN, gestita nativamente da XGBoost e via imputazione nella Logistic Regression).
- Lo snapshot "attuale" e quello "storico per stagione" sono due fonti/metodi leggermente diversi — normale che non coincidano al 100%.
- Non è stato incluso un controllo di qualità squadra per squadra su tutte le 32 mappature manuali: sono state verificate a campione (Inter, e il confronto aggregato 2025/26), non ogni singola stagione di ogni squadra.
