# Guida — Eseguire l'intera pipeline in locale

Tutti i comandi vanno lanciati dalla cartella principale del progetto
(`SerieAProject/`), con l'ambiente virtuale attivato. Sono elencati
nell'ordine esatto in cui vanno eseguiti: ogni passaggio produce un file che
il passaggio successivo si aspetta di trovare già pronto.

## 0. Prerequisiti (una tantum)

Richiede Python 3.10+ (sviluppato/testato con 3.11). Le librerie installate
sono: pandas, numpy, matplotlib, scikit-learn, xgboost (vedi `requirements.txt`).

### Windows (PowerShell)

Prima verifica se Python è già installato (su Windows il comando è `python`
o `py`, **non** `python3` — se lanci `python3` e non hai installato nulla,
Windows apre lo stub del Microsoft Store che stampa un messaggio d'errore
invece di dire chiaramente "comando non trovato"):

```powershell
py --version
```

Se stampa un numero di versione (es. `Python 3.12.4`), Python è già
installato: salta al blocco venv qui sotto. Se dà errore, scaricalo da
[python.org/downloads](https://www.python.org/downloads/) (non dal Microsoft
Store, dà meno problemi con i PATH) — durante l'installazione **spunta la
casella "Add python.exe to PATH"** nella prima schermata, poi riapri
PowerShell perché il PATH venga ricaricato.

```powershell
cd SerieAProject
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Se `Activate.ps1` dà un errore di "execution policy" (PowerShell blocca gli
script per default), esegui una volta sola:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

e poi rilancia `.venv\Scripts\Activate.ps1`. In alternativa, dal Prompt dei
comandi (cmd.exe, non PowerShell) l'attivazione è `.venv\Scripts\activate.bat`,
senza problemi di execution policy.

Una volta attivato il venv (il prompt mostra `(.venv)` davanti al percorso),
tutti i comandi della guida sotto (`python src/...`) funzionano identici a
macOS/Linux — l'attivazione fa sì che `python` punti automaticamente
all'interprete giusto dentro `.venv`.

### macOS / Linux

```bash
cd SerieAProject
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1. Dati grezzi (già inclusi nel progetto)

Questi file vivono in `data/raw/` e sono già presenti così come consegnati -
non servono download per un primo giro completo:

- `serieA_raw.csv`, `serieB_raw.csv` — storico partite 2000/01-2024/25 (mirror
  GitHub di Football-Data.co.uk, `xgabora/Club-Football-Match-Data-2000-2025`).
- `serieA_2025_26_football-data.csv` — stagione 2025/26 completa, scaricata a
  mano da football-data.co.uk (il mirror GitHub non arriva così avanti).
- `transfermarkt_players.csv` — snapshot rose attuali (dataset Kaggle
  `davidcariboo/player-scores`, file `players.csv`).
- `calendario_2026_27.csv` — calendario ufficiale 2026/2027, 380 partite.

**Unico file mancante nel repo** (escluso per dimensione): `player_valuations.csv`,
sempre dal dataset Kaggle `davidcariboo/player-scores` — serve SOLO se vuoi
rigenerare `squad_values_by_season.csv` da zero (passaggio 5 più sotto).
Scaricalo da Kaggle e mettilo in `data/raw/player_valuations.csv`. Se non ti
serve rigenerare quel file, puoi saltare il passaggio 5 e usare
`data/processed/squad_values_by_season.csv` così com'è (già incluso).

## 2. Pulizia e unificazione dati

```bash
python src/prepare_data.py
```

Legge i CSV grezzi di Serie A/B e produce `data/processed/serieA_matches.csv`
e `serieB_matches.csv` (dati puliti, uniti, pronti per il resto della pipeline).

## 3. Feature engineering di base

```bash
python src/feature_builder.py
```

Crea `data/processed/serieA_features.csv`: differenza Elo/forma, scontri
diretti (H2H), giorni di riposo, avanzamento stagione, vantaggio-casa "di
periodo". Nessuna informazione futura entra nel calcolo (niente leakage
temporale).

## 4. Feature del modello Poisson (gol attesi)

```bash
python src/build_poisson_features.py
```

Aggiunge a `serieA_features.csv` le colonne `poisson_exp_goals_*` e
`poisson_prob_*` (gol attesi/probabilità 1X2 stimati da un modello
Dixon-Coles semplificato, rifittato su una finestra mobile delle ultime 5
stagioni per ogni punto del backtest).

## 5. Valore rosa (opzionale se non hai `player_valuations.csv`)

```bash
python src/build_squad_value_current.py     # snapshot attuale -> squad_values_current.csv
python src/build_squad_value_history.py     # storico per stagione -> squad_values_by_season.csv (richiede player_valuations.csv in data/raw/)
python src/merge_squad_value_feature.py     # unisce il valore rosa a serieA_features.csv
```

Se salti `build_squad_value_history.py` (perché non hai scaricato
`player_valuations.csv`), usa direttamente il file già incluso in
`data/processed/squad_values_by_season.csv` e lancia solo
`merge_squad_value_feature.py`.

## 6. Training dei modelli

```bash
python src/train_baseline.py     # Logistic Regression -> models/baseline_logreg.pkl
python src/train_xgboost.py      # XGBoost -> models/xgboost_final.pkl
```

Entrambi fanno un backtest walk-forward (mai split casuale) su 15 stagioni e
stampano le metriche (accuracy/log-loss/brier) a confronto con le quote di
mercato e la baseline "vince sempre la casa". Alla fine salvano il modello
allenato su TUTTA la storia disponibile, pronto per la simulazione.

```bash
python src/train_ensemble.py     # opzionale: ri-deriva w (peso ensemble) e half-life
```

Questo script è **esplorativo/di validazione**, non produce un modello da
usare direttamente: ricerca — su un blocco di stagioni di sola selezione, mai
sull'holdout — il peso ottimale tra i due modelli e l'half-life di
ponderazione delle stagioni recenti, e li stampa a schermo. I valori scelti
sono già scritti come costanti in `train_baseline.py`, `train_xgboost.py` e
`predict_season.py` (`ENSEMBLE_W_LOGREG = 0.5`, `SEASON_HALF_LIFE = 10`): se
rilanci questo script otterrai la stessa conferma già documentata in
`FASE4_REPORT.md`/`FASE4e_RECENCY.md`, a meno che i dati sottostanti siano
cambiati (es. dopo aver integrato molte nuove partite reali).

## 7. Simulazione della stagione 2026/2027

```bash
python src/predict_season.py
```

Simula 5000 volte l'intera stagione (Monte Carlo), giornata per giornata,
usando l'ensemble Logistic Regression + XGBoost. Se ci sono partite
2026/2027 già giocate nello storico (vedi Fase 6 più sotto), le usa come
punto di partenza reale e simula solo il resto. Produce in
`data/processed/`:

- `previsioni_partite_2026_27.csv` — probabilità 1X2 partita per partita
- `classifica_prevista_2026_27.csv` — classifica finale prevista
- `classifica_storia_2026_27.csv` — cronologia giornata per giornata (reale +
  proiettata) di QUESTA esecuzione, sovrascritto ogni volta
- `previsioni_storia.csv` — **archivio accumulativo** (non sovrascritto): ad
  ogni esecuzione aggiunge uno "scatto" della classifica finale prevista in
  quel momento, taggato con la giornata reale di riferimento. È il file che
  permette di vedere, a fine stagione, come è cambiata la previsione dalla
  prima giornata all'ultima (vedi Fase 6 più sotto)

Impiega circa 30 secondi su una macchina normale.

## 8. Dashboard

```bash
python dashboard/build_dashboard.py
```

Genera `dashboard/previsioni_2026_27.html`, un unico file autonomo (nessuna
chiamata di rete) apribile in qualunque browser, con queste sezioni:

- **Classifica** e **Partite**: stato/previsioni correnti.
- **Cronologia**: come si sviluppa una singola previsione lungo le 38
  giornate della stagione (linea piena per le giornate già giocate, tratteggio
  per la proiezione, marcatore "Oggi").
- **Evoluzione previsione**: come cambia LA PREVISIONE STESSA da
  un'esecuzione all'altra nel corso della stagione (dati da
  `previsioni_storia.csv`, sopra) — oggi, con un solo scatto disponibile
  (pre-campionato), ogni squadra è un punto singolo; diventerà una linea vera
  e propria man mano che accumuli aggiornamenti settimanali. Ha un selettore a
  tre metriche: **Piazzamento** (la posizione 1-20 reale in classifica,
  default), **Posizione media** (la media grezza sulle 5000 simulazioni
  Monte Carlo — un numero decimale, es. 1.8, che può differire dal
  piazzamento vero e proprio) e **Punti**.

## 9. Durante la stagione: integrare risultati reali e cadenza consigliata (Fase 6)

Quando le partite iniziano a essere giocate ci sono due tipi diversi di
aggiornamento, con frequenza diversa: uno **leggero** (ogni settimana, dopo
ogni giornata) e uno **completo con retraining** (ogni 3-5 giornate). Non
serve mai fare il retraining dopo ogni singola giornata: Elo e forma si
aggiornano già da soli (calcolo diretto, non machine learning), mentre i
modelli veri e propri (Logistic Regression, XGBoost) traggono beneficio da
un retraining solo quando c'è un numero di partite nuove sufficiente a
"contare" qualcosa nell'apprendimento — farlo ogni giornata sarebbe uno
sforzo computazionale inutile per un guadagno pressoché nullo.

### Ogni settimana (leggero — solo classifica/previsioni aggiornate)

1. Scarica da football-data.co.uk (Italy → Serie A) il CSV "stagione
   2026/2027 ad oggi" e mettilo in `data/incoming/serieA_2026_27.csv` (vedi
   `data/incoming/README.md`).
2. Rilancia in ordine:

```bash
python src/integrate_new_season.py data/incoming/serieA_2026_27.csv 2026/2027
python src/prepare_data.py
python src/feature_builder.py
python src/build_poisson_features.py
python src/predict_season.py
python dashboard/build_dashboard.py
```

`integrate_new_season.py` è idempotente: puoi rilanciarlo ogni settimana con
il file scaricato di nuovo (che contiene sempre tutte le partite dalla
giornata 1 ad oggi) senza creare duplicati. Questo blocco aggiorna la
classifica reale, ricalcola le feature con Elo/forma/H2H ora aggiornati, e
rigenera previsioni + dashboard con i modelli **già allenati** (nessun
retraining) — è il blocco che uso anche nello scheduled task automatico del
martedì sera.

### Ogni 3-5 giornate (completo — con retraining dei modelli)

Stesso blocco di sopra, con in più il retraining vero e proprio (passaggio 6
della guida) subito dopo aver aggiornato i dati e PRIMA di rilanciare la
simulazione:

```bash
python src/integrate_new_season.py data/incoming/serieA_2026_27.csv 2026/2027
python src/prepare_data.py
python src/feature_builder.py
python src/build_poisson_features.py
python src/build_squad_value_current.py      # opzionale: solo se hai un export Transfermarkt piu' recente
python src/merge_squad_value_feature.py
python src/train_baseline.py
python src/train_xgboost.py
python src/predict_season.py
python dashboard/build_dashboard.py
```

Qui i modelli vengono ri-addestrati includendo anche le partite 2026/2027
giocate finora nel proprio storico di training (non solo nella simulazione),
quindi "imparano" davvero dai risultati recenti — coerente con la
ponderazione che dà più peso alle stagioni recenti (`SEASON_HALF_LIFE = 10`,
vedi `FASE4e_RECENCY.md`). Un buon ritmo è ogni 3-5 giornate (circa ogni
2-3 settimane di campionato); farlo più spesso non porta benefici pratici
apprezzabili.

### Riepilogo cadenza

| Quando | Cosa | Comandi |
|---|---|---|
| Ogni settimana | Integra risultati + aggiorna classifica/previsioni/dashboard | blocco "Ogni settimana" sopra |
| Ogni 3-5 giornate | Come sopra + re-training dei modelli | blocco "Ogni 3-5 giornate" sopra |

## Utilità: classifica di una stagione qualsiasi

```bash
python src/classifica.py --season 2023/2024
python src/classifica.py --season 2023/2024 --upto-date 2024-01-15   # classifica "congelata" a una data
```

## Riepilogo — pipeline completa da zero in un unico blocco

```bash
python src/prepare_data.py
python src/feature_builder.py
python src/build_poisson_features.py
python src/build_squad_value_current.py
python src/build_squad_value_history.py   # salta se non hai player_valuations.csv
python src/merge_squad_value_feature.py
python src/train_baseline.py
python src/train_xgboost.py
python src/predict_season.py
python dashboard/build_dashboard.py
```
