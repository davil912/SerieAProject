# Fase 4e — Cronologia classifica, peso alle stagioni recenti, aggiornamento rose

Richiesta originale dell'utente:

> "Vorrei che venisse mostrata la cronologia dell'aggiornamento della classifica (per vedere ogni giornata quanto cambia a seconda delle partite disputate). In piu, vorrei dare un peso maggiore alle ultime stagioni. Le rose delle squadre sono aggiornate al giorno di oggi? Perche nel frattempo stanno comprando altri giocatori"

Tre richieste distinte, trattate una per una qui sotto.

## 1. Cronologia della classifica (giornata per giornata)

Finora `predict_season.py` restituiva solo la classifica prevista **finale**, dopo tutte le 38 giornate simulate. Ora, durante ciascuna delle 5000 simulazioni Monte Carlo, viene registrato un "fotogramma" della classifica media **dopo ogni giornata**, non solo alla fine.

Tecnicamente: la logica di calcolo classifica (ordinamento punti/differenza reti/gol fatti per ogni singola simulazione, poi media delle posizioni sulle 5000 simulazioni) è stata estratta in una funzione riutilizzabile `classifica_snapshot()`, usata sia per la classifica finale sia per ogni fotogramma intermedio — quindi nessuna logica duplicata, e la classifica finale resta identica a prima (stesso identico calcolo).

Output: `data/processed/classifica_storia_2026_27.csv`, 760 righe (38 giornate x 20 squadre), colonne `matchday, team, punti_medi, posizione_media, GF_medi, GS_medi, DR_medio`.

Costo aggiuntivo: trascurabile, la simulazione completa passa da un tempo equivalente a ~28.5s totali (N=5000 simulazioni, incluso il calcolo dei 38 fotogrammi).

**Nella dashboard**, nuova sezione "Cronologia" con un grafico a linee (posizione o punti sull'asse verticale, giornata 1-38 sull'asse orizzontale). Con 20 squadre da mostrare contemporaneamente, evidenziare tutte le linee a colore pieno sarebbe illeggibile: di default sono evidenziate le squadre previste nelle prime 4 posizioni e nelle ultime 3 (zona retrocessione), le altre restano in grigio chiaro sullo sfondo; cliccando sul "chip" di qualunque squadra la si accende/spegne singolarmente. C'è anche uno slider per giornata, che mostra una mini-classifica ordinata in quel preciso momento della stagione, e un tooltip al passaggio del mouse sulle linee.

## 2. Peso maggiore alle stagioni recenti

Implementato tramite pesi campione esponenziali (`sample_weight`, nativamente supportato sia da Logistic Regression sia da XGBoost in fase di training): la stagione più recente disponibile in ciascun blocco di training pesa 1.0, e il peso si dimezza ogni `half_life` stagioni indietro. Questo si applica al training walk-forward (ogni fold vede solo il passato, mai il futuro, come sempre) — semplicemente le stagioni più vicine alla stagione da prevedere contano di più nell'addestramento.

Come per il peso dell'ensemble (Fase 4) e il moltiplicatore del valore rosa (Fase 4d), l'half-life è stato scelto **onestamente**, senza spiare i dati di verifica: confrontando i candidati `[nessuno, 10, 6, 3]` (in stagioni) solo sulle 10 stagioni di SELEZIONE (2010/11-2019/20), poi verificando la scelta sulle 6 stagioni di HOLDOUT (2020/21-2025/26) mai toccate durante la scelta.

**Scelto: half_life = 10 stagioni** (decadimento lieve — anche stagioni piuttosto lontane pesano ancora in modo significativo).

Risultati sull'holdout (mai visto durante la scelta):

| Modello | Accuracy | Log-loss | Brier |
|---|---|---|---|
| XGBoost (nessuna ponderazione) | 53.46% | 0.9706 | 0.5770 |
| XGBoost (half-life=10) | 53.64% | 0.9701 | 0.5764 |
| Ensemble (nessuna ponderazione, baseline Fase 4d) | 53.77% | 0.9680 | 0.5762 |
| **Ensemble (w=0.5, half-life=10 — scelto)** | **53.90%** | **0.9675** | **0.5757** |

A differenza dell'esperimento sul valore rosa (Fase 4d), dove XGBoost da solo peggiorava leggermente, qui **sia XGBoost da solo sia l'ensemble migliorano** in modo consistente sull'holdout — un segnale più convincente, anche se comunque modesto in valore assoluto (frazioni di punto percentuale). Non è un salto netto, ma va nella direzione giusta e non ha controindicazioni, quindi è stato adottato come impostazione definitiva.

Effetto pratico sulla classifica prevista 2026/27: Juventus e Napoli, che nella versione precedente (Fase 4d) erano separate, ora sono sostanzialmente appaiate per il 2°/3° posto (70.6 punti medi, posizione media 4.5 entrambe) — un cambiamento coerente con il fatto che le stagioni più recenti contano ora leggermente di più nel training.

Grafici aggiornati: `notebooks/ensemble_weight_search.png` (log-loss vs peso ensemble, per l'half-life scelto) e `notebooks/final_comparison.png`.

## 3. Le rose sono aggiornate a oggi?

**No.** Verificato direttamente sui dati sorgente, non per supposizione:

- `data/raw/player_valuations.csv` (serie storica delle valutazioni giocatore per giocatore): la data più recente presente è **12 giugno 2026**.
- `data/raw/transfermarkt_players.csv` (snapshot "corrente" delle rose): il campo `last_season` per le squadre di Serie A arriva al massimo alla stagione **2025** (cioè 2025/26) — nessuna squadra ha dati aggiornati alla stagione 2026/27 in corso di preparazione.

Il dataset sorgente (Kaggle, `davidcariboo/player-scores`) non è aggiornato in tempo reale: è uno snapshot statico che l'utente ha fornito, non uno scraping live di Transfermarkt. Questo significa che **mancano circa gli ultimi due mesi di mercato** (metà giugno - metà agosto 2026), che sono storicamente il periodo più attivo della sessione estiva, proprio a ridosso dell'inizio del campionato (22-23 agosto 2026). Colpi di mercato recenti — acquisti, cessioni, rivalutazioni dei cartellini — non sono catturati nelle feature `home_squad_value` / `away_squad_value` / `squad_value_diff` / `squad_value_log_ratio` usate dal modello.

Cosa significa in pratica per le previsioni: il valore rosa pesa solo come "istantanea di partenza" pre-stagione nel modello (Fase 4d), mentre Elo, forma recente e H2H si aggiornano partita per partita via via che la stagione si gioca — quindi questo limite pesa di più sulle primissime giornate e si attenua rapidamente man mano che il campionato prende forma. Se in futuro sarà disponibile una fonte di valutazioni più aggiornata (es. un nuovo export Kaggle/Transfermarkt più recente), si può semplicemente sostituire `player_valuations.csv` senza toccare il codice — la pipeline di merge è già generica.

## File modificati/aggiunti in questa fase

- `src/train_baseline.py`, `src/train_xgboost.py`, `src/train_ensemble.py`: aggiunta ponderazione stagionale (`season_sample_weights`, half_life=10).
- `src/predict_season.py`: nuova funzione `classifica_snapshot()`, nuova sezione output `classifica_storia_2026_27.csv`.
- `models/baseline_logreg.pkl`, `models/xgboost_final.pkl`: retrained con la nuova ponderazione.
- `data/processed/classifica_storia_2026_27.csv` (nuovo), `previsioni_partite_2026_27.csv`, `classifica_prevista_2026_27.csv` (rigenerati).
- `dashboard/build_dashboard.py`, `dashboard/previsioni_2026_27.html`: nuova sezione "Cronologia" con grafico a linee.
- `notebooks/ensemble_chart.py`: adattato alla nuova griglia (colonna `season_half_life` al posto di `squad_value_multiplier`).
- `notebooks/ensemble_weight_search.png`, `notebooks/final_comparison.png`, `notebooks/final_comparison_summary.csv`, `notebooks/ensemble_results.csv`, `notebooks/ensemble_weight_search.csv`, `notebooks/backtest_results.csv`, `notebooks/backtest_results_fase3.csv`: rigenerati.
