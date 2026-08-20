# Fase 6 — Classifica ibrida reale/simulata, aggiornamento durante la stagione

Richiesta originale dell'utente:

> "Man mano che verranno giocate le partite verrà aggiornata la classifica che ad ora è solo fatta di previsioni? Ci sarà un nuovo training ad ogni nuova partita?" → "Implementala"

## 1. Cosa cambia

Finora `predict_season.py` (src/simulation/) simulava SEMPRE l'intera stagione da zero, ogni volta che veniva lanciato, ignorando qualunque partita 2026/2027 eventualmente già giocata. Ora lo script si accorge da solo se ci sono partite reali della stagione 2026/2027 già integrate nello storico (`data/processed/serieA_matches.csv`):

- se sì, usa la classifica reale (punti, GF, GS, V/N/P) come punto di partenza, ed Elo/forma/scontri diretti ripartono dallo stato "post ultima partita reale" (non da fine 2025/26) — poi la simulazione Monte Carlo copre SOLO le giornate rimanenti del calendario;
- se no (situazione di oggi, il campionato non è ancora iniziato), il comportamento è identico a prima: l'intera stagione viene simulata da capo.

Non serve alcun parametro o flag: lo script legge lo storico e si comporta di conseguenza. Ho verificato questo con un test (dati sintetici, poi scartati, il progetto reale non è stato toccato): iniettando 10 partite fittizie della giornata 1, lo script ha correttamente riconosciuto le partite già giocate, calcolato la classifica reale in modo esatto (verificata a mano, risultato per risultato), simulato solo le 370 partite restanti, e prodotto una classifica finale coerente con quel punto di partenza.

La **cronologia** (`classifica_storia_2026_27.csv`, Fase 4e) ora ha una colonna in più, `tipo`: `"reale"` per le giornate già giocate (valori esatti, non medie) e `"simulata"` per quelle ancora da giocare (media su 5000 simulazioni, come prima). Nella dashboard, il grafico "Cronologia" disegna la parte reale con una linea piena e quella proiettata con tratteggio più leggero, con una linea verticale "Oggi (giornata N)" a separarle — visibile solo quando esistono davvero giornate reali (oggi, con zero partite giocate, il grafico è identico a prima).

## 2. Come arrivano i risultati reali

Ho verificato direttamente (non per supposizione) le due fonti automatiche disponibili nell'ambiente cloud di questa sessione:

- il mirror GitHub finora usato da `update_pipeline.py` si è fermato al 2025 e non copre la stagione 2026/2027;
- football-data.co.uk, l'alternativa più ovvia, blocca le richieste dirette da questo ambiente (risposta 403 Forbidden).

Il download automatico affidabile non è quindi disponibile da qui. La via pratica resta manuale ma è stata resa il più semplice possibile: scarichi da football-data.co.uk (sezione Italy → Serie A) il CSV "stagione 2026/2027 ad oggi" (un file che si aggiorna e cresce ogni settimana con tutte le partite giocate fino a quel momento) e lo appoggi in `data/incoming/serieA_2026_27.csv` (cartella nuova, con un README che spiega la convenzione). Da lì:

```
python src/live_update/integrate_new_season.py data/incoming/serieA_2026_27.csv 2026/2027
python src/preprocessing/prepare_data.py
python src/preprocessing/feature_builder.py
python src/preprocessing/build_poisson_features.py
python src/simulation/predict_season.py
python dashboard/build_dashboard.py
```

`integrate_new_season.py` è ora **idempotente**: lo si può rilanciare ogni settimana con il file scaricato di nuovo (che contiene sempre tutte le partite dalla giornata 1 ad oggi, non solo le ultime) — riconosce le partite già integrate e aggiunge solo quelle nuove, senza duplicati e senza sballare l'Elo. Ho verificato questo lanciandolo due volte di fila sullo stesso file: la seconda volta ha correttamente riconosciuto "10 partite già presenti... 0 nuove" e non ha toccato nulla.

Approfittando di questo lavoro ho anche corretto un piccolo bug preesistente nel seed dell'Elo (`src/common/elo_updater.py`, `seed_ratings`): prendeva il rating Elo PRE-partita dell'ultima partita nota invece di quello POST-partita (cioè "di oggi"), un disallineamento di una partita che si sarebbe ripresentato ad ogni integrazione di nuovi dati — ora corretto per essere coerente con la stessa logica già usata in `predict_season.py` (src/simulation/).

## 3. Automazione: scheduled task settimanale

Ho creato uno scheduled task ("SerieAProject - aggiornamento settimanale classifica", ogni lunedì mattina) che, quando l'app desktop è connessa: controlla se hai messo un nuovo file in `data/incoming/`, e se sì lancia l'intera pipeline sopra, sincronizza i risultati sul tuo computer e aggiorna la dashboard salvata. Se non c'è nessun file nuovo, o se il computer non è raggiungibile in quel momento, il task lo segnala semplicemente e non fa nulla di distruttivo — è sicuro che giri anche se dimentichi di scaricare il CSV per qualche settimana.

**Nota onesta sui limiti**: questa parte automatica dipende dal fatto che tu scarichi manualmente il CSV ogni tanto (settimanalmente è un buon ritmo, in linea con il calendario di Serie A) e dal fatto che l'app desktop sia aperta quando il task lunedì mattina parte. Se preferisci un ritmo diverso, o vuoi che te lo ricordi in altro modo, fammelo sapere.

## 4. Retraining: confermato "non ad ogni partita"

Come discusso, il retraining dei modelli (`models/*.pkl`) resta volutamente ESCLUSO da questo aggiornamento automatico. Elo e forma si aggiornano già incrementalmente (aritmetica, non serve riaddestrare nulla), mentre `train_baseline.py`/`train_xgboost.py`/`train_ensemble.py` vanno rilanciati a mano ogni tanto (ogni 3-5 giornate circa, non più spesso) quando vorrai incorporare i nuovi risultati anche nell'apprendimento dei modelli stessi, non solo nella simulazione — esattamente come previsto dal piano d'azione originale.

## 5b. Addendum — Archivio delle previsioni (come cambia la previsione nel tempo)

Richiesta successiva dell'utente:

> "In questo progetto è inclusa anche la cronologia delle classifiche predette? [...] a fine anno vorrei vedere come è cambiata la predizione dalla prima partita fino all'ultima" → "si"

Da non confondere con `classifica_storia_2026_27.csv` (Fase 4e/6), che descrive come si sviluppa UNA previsione lungo le 38 giornate all'interno di una singola esecuzione, e viene sovrascritto ad ogni run. Qui si tiene traccia di come cambia LA PREVISIONE STESSA da un'esecuzione all'altra, mano a mano che nuovi risultati reali vengono integrati durante la stagione.

`predict_season.py` (src/simulation/) ora produce anche `data/processed/previsioni_storia.csv`: un file **accumulativo** (non sovrascritto) a cui, ad ogni esecuzione, viene aggiunto uno "scatto" della classifica finale prevista in quel momento, taggato con `giornata_riferimento` (l'ultima giornata reale integrata al momento del run — 0 significa "prima dell'inizio del campionato") e `data_previsione` (la data del run). Se lo script viene rilanciato più volte per la stessa giornata (es. dopo un retraining), lo scatto di quella giornata viene sostituito, non duplicato — verificato lanciando lo script due volte di fila sugli stessi dati (rimane un solo scatto, non due) e poi simulando l'arrivo di una nuova giornata reale (secondo scatto aggiunto correttamente, il primo resta intatto).

Nella dashboard, nuova sezione "Evoluzione previsione": stesso stile della Cronologia (linee per squadra, chip per evidenziare, tooltip), ma con l'asse orizzontale sulla giornata di riferimento invece che sulla giornata simulata. Oggi, con un solo scatto disponibile (pre-campionato), ogni squadra è un punto singolo — diventerà una linea vera e propria man mano che si accumulano gli scatti settimanali durante la stagione (validato anche con dati sintetici multi-scatto, poi scartati).

**Correzione successiva** — segnalazione dell'utente: "vedo Inter 2° ma teoricamente è prima". Il grafico inizialmente esponeva solo `posizione_media` (la media grezza della posizione di una squadra sulle 5000 simulazioni Monte Carlo, un numero decimale che può benissimo essere 1.8 anche per la squadra prevista prima in assoluto, se non vince sempre esattamente il primo posto in ogni simulazione), non il piazzamento vero e proprio. Il toggle della sezione ora ha **tre** metriche invece di due: "Piazzamento" (la posizione reale 1-20 in classifica, cioè come le squadre sono effettivamente ordinate — nuovo default), "Posizione media" (la statistica grezza di prima, ancora disponibile), "Punti". Verificato con dati sintetici multi-scatto costruiti apposta per riprodurre il caso segnalato (una squadra con la media migliore ma non esattamente 1.0): con "Piazzamento" risulta correttamente 1°, con "Posizione media" mostra il valore medio decimale.

## 5c. Addendum — Riorganizzazione degli script in cartelle per fase

Richiesta successiva dell'utente:

> "Vorrei che tutti gli import in locale siano posizionati in alto nel file e che il progetto sia organizzato in ulteriori cartelle in base alla fase dell'architettura (ad esempio: preprocessing, training, ecc.)"

Due interventi distinti:

1. **Import in alto**: in due file (`predict_season.py`, `update_pipeline.py`) un import locale (rispettivamente `poisson_model` e `urllib.request`) era dentro una funzione invece che in cima al file — spostati entrambi in cima, insieme agli altri import.
2. **Riorganizzazione di `src/`**: gli script, prima tutti allo stesso livello, sono stati spostati in sottocartelle per fase: `src/common/` (moduli condivisi — `elo_updater.py`, `classifica.py`, `poisson_model.py`), `src/preprocessing/`, `src/training/`, `src/simulation/` (`predict_season.py`), `src/live_update/` (`integrate_new_season.py`, `update_pipeline.py`). Gli script che dipendono dai moduli condivisi (`predict_season.py`, `build_poisson_features.py`, `integrate_new_season.py`, `update_pipeline.py`) aggiungono `src/common/` al path di ricerca dei moduli Python subito dopo aver calcolato la cartella del progetto, prima del relativo import — stesso pattern già usato da `update_pipeline.py` in precedenza, solo esteso agli altri script.

Verificato eseguendo l'intera pipeline da zero con la nuova struttura, prima in una copia di prova e poi nel progetto reale: stessi identici risultati di prima (stessa classifica prevista, stesse probabilità) — la riorganizzazione non cambia alcun calcolo, solo la posizione dei file. Aggiornati di conseguenza tutti i comandi in `GUIDA_ESECUZIONE_LOCALE.md`, `README.md`, i riferimenti nei report e lo scheduled task settimanale.

## 5. File modificati/aggiunti in questa fase

- `src/simulation/predict_season.py`: nuova Sezione 0 (rilevamento partite reali 2026/2027), seed di punti/GF/GS/V/N/P/Elo/forma/H2H dallo stato reale, filtro del calendario alle sole partite rimanenti, cronologia con colonna `tipo`.
- `src/live_update/integrate_new_season.py`: reso idempotente (dedup su stagione+home_team+away_team).
- `src/common/elo_updater.py`: `seed_ratings` corretto per restituire l'Elo POST ultima partita (non pre).
- `src/live_update/update_pipeline.py`: messaggi aggiornati per riflettere i limiti reali del download automatico e indirizzare al flusso manuale.
- `data/incoming/README.md` (nuovo): convenzione per l'aggiornamento manuale settimanale.
- `dashboard/build_dashboard.py`, `dashboard/previsioni_2026_27.html`: cronologia con linea piena (reale) / tratteggiata (proiezione) e marcatore "Oggi"; nuova sezione "Evoluzione previsione".
- `data/processed/previsioni_storia.csv` (nuovo): archivio accumulativo delle previsioni finali, uno scatto per ogni esecuzione di `predict_season.py` (src/simulation/).
- Scheduled task "SerieAProject - aggiornamento settimanale classifica" (ogni martedì sera).
