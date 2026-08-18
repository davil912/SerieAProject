# Fase 5 — Previsioni stagione 2026/2027: report

## Obiettivo

Generare, per la stagione 2026/2027 (non ancora iniziata: si parte il 22-23 agosto 2026), due output concreti richiesti:

1. **Previsioni partita per partita**: probabilità 1X2 per tutte le 380 partite del calendario completo.
2. **Classifica finale prevista**: posizione attesa di ogni squadra a fine stagione, con probabilità di titolo, qualificazione europea e retrocessione.

Finora il progetto aveva solo script di **backtest storico** (`train_baseline.py`, `train_xgboost.py`, `train_ensemble.py`): allenano e valutano i modelli su stagioni già giocate, non generano previsioni per partite future. Questa fase aggiunge il pezzo mancante: `src/predict_season.py`.

## Il problema da risolvere: una stagione intera non è ancora giocata

I modelli sanno prevedere UNA partita, dato lo stato (Elo, forma, scontri diretti) delle due squadre in quel momento. Ma per un'intera stagione futura, quello stato **cambia partita dopo partita** in base ai risultati — risultati che ovviamente non conosciamo oggi. Prevedere "staticamente" tutte le 380 partite con lo stato di oggi (18 agosto 2026) sarebbe sempre meno realistico man mano che ci si allontana nel calendario (una previsione di aprile 2027 con l'Elo di agosto 2026 ignorerebbe 7 mesi di partite).

**Soluzione: simulazione Monte Carlo.** La stagione viene simulata per intero **5.000 volte**. In ogni simulazione, si gioca ogni giornata in ordine cronologico: si calcolano le probabilità 1X2 con l'ensemble usando lo stato aggiornato *di quella specifica simulazione*, si estrae un esito casuale da quelle probabilità, si aggiornano Elo/forma/scontri diretti, e si passa alla giornata successiva. Mediando le 5.000 traiettorie si ottiene:

- per ogni partita: la probabilità 1X2 media (tiene conto dell'incertezza su come si evolverà la forma delle squadre, non solo dello stato di oggi)
- per ogni squadra: punti medi, posizione media a fine stagione, probabilità di vincere il titolo / qualificarsi in Champions (primi 4) / andare in Europa (primi 6) / retrocedere (ultimi 3)

## Cosa varia e cosa no tra le simulazioni

Non tutto ha bisogno di essere ri-simulato 5.000 volte — solo le feature che dipendono davvero dai risultati:

| Feature | Varia per simulazione? | Perché |
|---|---|---|
| Elo, forma (ultime 3/5), scontri diretti stagionali | **Sì** | dipendono dai risultati simulati |
| Giorni di riposo, partite giocate, avanzamento stagione | No | dipendono solo dal calendario (date reali) |
| Valore rosa | No | snapshot statico più recente disponibile |
| Vantaggio-casa "di periodo" | No | congelato all'ultimo valore storico (0,395) |
| Rating Poisson attacco/difesa | No | rifittati una sola volta (finestra mobile 5 stagioni: 2021/22-2025/26), stessa logica già usata nel backtest storico |

Questo ha anche reso la simulazione molto più veloce del previsto: ~20 secondi per 5.000 stagioni simulate complete, grazie a calcolare le parti "statiche" una volta sola e a vettorizzare Elo/forma su tutte le simulazioni con NumPy.

## Stato di partenza (18 agosto 2026)

Elo attuale per squadra, calcolato prendendo il rating pre-partita dell'ultimo incontro disponibile per ciascuna squadra e applicando l'aggiornamento Elo standard con il risultato reale (per ottenere il rating **dopo** l'ultima partita, non prima):

| Squadra | Elo | Squadra | Elo | Squadra | Elo | Squadra | Elo |
|---|---|---|---|---|---|---|---|
| Inter 1959 | Como 1776 | Sassuolo 1656 | Lecce 1595 |
| Napoli 1849 | Lazio 1751 | Genoa 1645 | Venezia 1571 |
| Roma 1825 | Bologna 1746 | Parma 1642 | Monza 1537 |
| Juventus 1817 | Fiorentina 1706 | Cagliari 1632 | Frosinone 1500 |
| Atalanta 1802 | Torino 1670 | Udinese 1667 | |
| Milan 1799 | | | |

## Classifica prevista 2026/2027 (media su 5.000 simulazioni)

| Pos | Squadra | Punti medi | GF | GS | DR | Titolo | Champions (top 4) | Europa (top 6) | Retrocessione |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Inter | 82,9 | 93,9 | 48,6 | +45,3 | 60,5% | 93,5% | 98,1% | 0,0% |
| 2 | Napoli | 71,2 | 76,8 | 51,4 | +25,3 | 11,5% | 61,0% | 81,0% | 0,0% |
| 3 | Roma | 68,6 | 72,8 | 53,3 | +19,5 | 8,4% | 49,8% | 72,6% | 0,1% |
| 4 | Juventus | 68,2 | 71,0 | 51,0 | +20,0 | 6,6% | 49,5% | 71,4% | 0,0% |
| 5 | Milan | 65,9 | 75,9 | 57,2 | +18,6 | 4,3% | 40,4% | 64,0% | 0,2% |
| 6 | Atalanta | 65,1 | 76,9 | 59,1 | +17,9 | 4,2% | 37,0% | 60,2% | 0,1% |
| 7 | Como | 62,4 | 70,1 | 57,7 | +12,4 | 2,4% | 27,1% | 49,4% | 0,2% |
| 8 | Lazio | 57,8 | 68,9 | 61,0 | +7,9 | 0,9% | 15,2% | 32,7% | 1,0% |
| 9 | Bologna | 56,9 | 66,2 | 61,2 | +5,1 | 0,9% | 12,8% | 29,2% | 0,9% |
| 10 | Fiorentina | 52,2 | 65,7 | 64,5 | +1,2 | 0,2% | 6,2% | 15,4% | 2,7% |
| 11 | Torino | 47,0 | 56,0 | 62,4 | -6,4 | 0,1% | 2,2% | 6,8% | 8,0% |
| 12 | Udinese | 46,4 | 60,3 | 68,3 | -7,9 | 0,0% | 2,0% | 5,9% | 7,5% |
| 13 | Sassuolo | 45,3 | 64,2 | 75,3 | -11,1 | 0,0% | 1,3% | 4,8% | 10,3% |
| 14 | Genoa | 42,6 | 53,7 | 66,6 | -12,9 | 0,0% | 0,6% | 2,7% | 15,4% |
| 15 | Parma | 42,5 | 54,1 | 67,1 | -13,0 | 0,0% | 0,6% | 2,9% | 15,7% |
| 16 | Cagliari | 41,4 | 56,6 | 73,2 | -16,6 | 0,0% | 0,6% | 2,0% | 18,9% |
| 17 | Lecce | 36,9 | 47,7 | 67,8 | -20,1 | 0,0% | 0,2% | 0,7% | 32,4% |
| 18 | Venezia | 33,1 | 50,0 | 76,5 | -26,5 | 0,0% | 0,0% | 0,1% | 49,3% |
| 19 | Monza | 30,4 | 49,6 | 76,6 | -27,0 | 0,0% | 0,0% | 0,1% | 60,9% |
| 20 | Frosinone | 26,8 | 51,7 | 83,5 | -31,9 | 0,0% | 0,0% | 0,0% | 76,6% |

Da notare: la somma delle probabilità di retrocessione delle ultime squadre supera ampiamente 300% solo apparentemente — sono probabilità individuali (ogni squadra può retrocedere o no), e sommate sulle 20 squadre danno esattamente 300% (3 posti di retrocessione), come verificato.

## Esempi di previsioni partita per partita (1ª giornata, 22-24 agosto 2026)

| Casa | Ospite | Casa % | Pareggio % | Ospite % | Esito più probabile |
|---|---|---|---|---|---|
| Inter | Monza | 83,1% | 11,4% | 5,5% | Vittoria casa |
| Udinese | Como | 27,4% | 27,9% | 44,7% | Vittoria trasferta |
| Genoa | Napoli | 19,2% | 24,1% | 56,6% | Vittoria trasferta |
| Parma | Cagliari | 43,5% | 31,9% | 24,7% | Vittoria casa |
| Venezia | Lecce | 36,6% | 36,4% | 26,9% | Vittoria casa |
| Frosinone | Juventus | 11,2% | 23,3% | 65,5% | Vittoria trasferta |
| Atalanta | Sassuolo | 61,8% | 24,2% | 14,0% | Vittoria casa |
| Torino | Milan | 26,0% | 25,9% | 48,1% | Vittoria trasferta |
| Bologna | Lazio | 42,2% | 29,0% | 28,8% | Vittoria casa |
| Roma | Fiorentina | 58,5% | 24,4% | 17,1% | Vittoria casa |

Tutte le 380 righe sono in `data/processed/previsioni_partite_2026_27.csv` (verificato: somma delle tre probabilità sempre ~100% ± 0,1 di arrotondamento).

## Limiti dichiarati

- **Neopromosse (Frosinone, Monza, Venezia)**: l'ultimo dato Elo/forma disponibile nel nostro storico risale alla loro ultima apparizione in Serie A (2024/25), non alla stagione di Serie B 2025/26 appena vinta — non abbiamo i risultati di quella stagione di Serie B nel dataset (`serieB_matches.csv` si ferma al 2024/25). Il rating di partenza per queste 3 squadre è quindi meno aggiornato che per le altre 17, e probabilmente le sottostima leggermente (hanno appena vinto/ottenuto la promozione, quindi sono presumibilmente più in forma di quanto risulti dal loro ultimo dato disponibile).
- **Valore rosa**: Frosinone (5 giocatori valutati) e Monza (14) restano sotto la soglia di affidabilità di 15 giocatori valutati anche nell'ultima stagione disponibile — il loro valore rosa usato nel modello è quindi da trattare con cautela.
- **Nessun ricalcolo durante la simulazione** di valore rosa, vantaggio-casa "di periodo" e rating Poisson attacco/difesa: restano fissi allo snapshot pre-stagione per tutte le 38 giornate simulate, esattamente come già avveniva nel backtest storico (rifit una volta per stagione, non partita per partita).
- **Spareggio classifica**: a parità di punti la simulazione ordina per differenza reti poi gol fatti, SENZA il criterio degli scontri diretti (che in Serie A ha priorità sulla differenza reti) — semplificazione che ha impatto trascurabile su medie e probabilità aggregate su 5.000 simulazioni, ma va tenuta presente se si guarda una singola simulazione isolata.
- **Va ri-eseguito periodicamente**: appena si giocano partite reali, vanno integrate in `data/processed/serieA_matches.csv` (stesso procedimento già usato per il 2025/26, vedi `src/integrate_new_season.py`) e lo script va ri-lanciato per aggiornare le previsioni con i risultati reali nel frattempo.

## File prodotti

```
serieA_predictor/
├── data/raw/calendario_2026_27.csv                   # calendario completo 380 partite (fonte: worldfootball.net)
├── data/processed/previsioni_partite_2026_27.csv       # probabilità 1X2 per tutte le 380 partite
├── data/processed/classifica_prevista_2026_27.csv      # classifica finale prevista + probabilità titolo/Europa/retrocessione
├── src/predict_season.py                                # motore di simulazione Monte Carlo
└── dashboard/previsioni_2026_27.html                     # dashboard visuale (classifica + partite filtrabili)
```

## Come aggiornare le previsioni durante la stagione

1. Integrare i risultati reali giocati in `data/processed/serieA_matches.csv` (formato football-data.co.uk, stesso procedimento di `src/integrate_new_season.py` usato per il 2025/26).
2. Ri-lanciare `python src/predict_season.py` — ricalcola automaticamente Elo/forma/scontri diretti aggiornati e ri-simula tutte le partite rimanenti.
3. (Opzionale, periodicamente) ri-lanciare `src/train_xgboost.py` e `src/train_ensemble.py` per includere le partite appena giocate anche nel training dei modelli, non solo nello stato Elo/forma.
