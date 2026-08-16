# Fase 1 — Raccolta dati ed esplorazione — Report

## Cosa è stato fatto

1. **Struttura progetto** creata (`data/raw`, `data/processed`, `src`, `models`, `notebooks`, `dashboard`).
2. **Dati storici scaricati**: 9.012 partite di Serie A (stagioni 2000/01 → 2024/25) e 9.921 partite di Serie B (stessa finestra, utile in futuro per stimare la forza delle squadre neopromosse), da un mirror GitHub del noto dataset Football-Data.co.uk (`xgabora/Club-Football-Match-Data-2000-2025`), che include già rating Elo e forma recente pre-calcolati.
3. **Pulizia dati**: normalizzazione colonne, calcolo stagione a partire dai blocchi temporali reali delle partite (non dal semplice mese di calendario — necessario per gestire correttamente la stagione 2019/20, prolungata fino ad agosto 2020 per il COVID), correzione di 0 incoerenze esito/gol trovate.
4. **Analisi esplorativa**: statistiche su copertura, distribuzione esiti, gol medi, trend vantaggio-casa nel tempo (grafici allegati).
5. **Modulo classifica** (`src/classifica.py`): calcola la classifica di qualunque stagione, anche "congelata" a una data specifica (propedeutico al feature engineering della Fase 2, dove serve sapere la classifica *prima* di ogni partita).
6. **Verifica**: la classifica calcolata per la stagione 2023/2024 è stata confrontata con i dati ufficiali — il primo posto (Inter, 94 punti, 29V-7N-2P, 89 gol fatti/22 subiti) coincide esattamente; punti finali delle prime 6 squadre (94-75-71-69-68-63) coincidono tutti. Sono stati inoltre eseguiti controlli automatici di coerenza su tutte le 20 stagioni "complete" (2005/06→2024/25): gol fatti totali = gol subiti totali, 20 squadre, 38 partite ciascuna — **nessuna anomalia trovata**.

## Risultati chiave dell'esplorazione

- **9.012 partite**, 49 squadre distinte nell'arco delle 25 stagioni.
- **Distribuzione esiti 1X2** (stagioni complete 2005/06-2024/25): **44,5% vittoria casa, 26,5% pareggio, 29,0% vittoria trasferta**. Confermano il vantaggio-casa tipico del calcio italiano, anche se in calo strutturale negli anni (grafico allegato: si passa da picchi del 48-50% a fine anni 2000 a valori sotto il 40% nelle ultime stagioni).
- **Media gol/partita**: 2,68.
- **Qualità dati**: 0% valori mancanti su Elo; ~1,3% mancanti sulle quote bookmaker; ~15,8% mancanti sulle statistiche di tiri (shots), presenti solo a partire da un certo anno — da tenere presente in Fase 2 per decidere se includerle solo per le stagioni recenti.
- **Nota importante**: le prime 4-5 stagioni (2000/01–2004/05) hanno dati grezzi incompleti (es. 2003/04 con solo 194 partite invece di 306) — probabile lacuna della fonte per gli anni più vecchi. Per il training andranno **escluse o trattate con cautela**; il training set "pulito" di riferimento parte dal 2005/06 (20 stagioni complete).
- **Copertura temporale**: dati fino al 25 maggio 2025 (fine stagione 2024/25). **Manca la stagione 2025/26**, appena conclusa — da recuperare in una fase successiva (l'accesso diretto a football-data.co.uk non è raggiungibile da questo ambiente; verrà affrontato quando costruiremo la pipeline di aggiornamento automatico).

## File prodotti

```
serieA_predictor/
├── data/raw/serieA_raw.csv, serieB_raw.csv      # dati grezzi
├── data/processed/serieA_matches.csv, serieB_matches.csv  # dati puliti pronti all'uso
├── notebooks/eda_charts.py + 2 PNG               # script e grafici esplorativi
├── src/prepare_data.py                           # pulizia/unificazione dati
├── src/classifica.py                              # calcolo classifica (anche "congelata" a una data)
└── requirements.txt
```

## Prossimi passi (Fase 2)

Feature engineering: Elo dinamico incrementale (già presente come base nel dataset, da estendere), forma recente pesata, scontri diretti, valore di mercato squadre (Transfermarkt), variabili di contesto (giorni di riposo, fase stagionale). Poi baseline model (Logistic Regression) per validare l'intera pipeline prima di passare a XGBoost.
