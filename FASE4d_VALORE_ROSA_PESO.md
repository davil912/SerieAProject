# Fase 4d — Più peso al valore rosa: report

## Richiesta

> "Vorrei dare più importanza al valore della rosa attuale (che secondo me incide)"

Fino a Fase 4c il valore rosa era usato SOLO da XGBoost (una fra 19 feature, senza alcun trattamento preferenziale), e non entrava affatto nella Logistic Regression. Tre modifiche concrete per dargli davvero più peso:

1. **Aggiunto alla Logistic Regression** (`squad_value_diff`, prima assente).
2. **Nuova feature: rapporto in scala logaritmica** (`squad_value_log_ratio = log((valore_casa+0.1)/(valore_trasferta+0.1))`), oltre alla semplice differenza assoluta. Motivo: 50M€ di differenza pesano molto di più tra due squadre da 60M€ che tra due da 600M€ — la differenza assoluta da sola non cattura questo effetto moltiplicativo, il rapporto sì.
3. **XGBoost "spinto" a considerare di più il valore rosa**: tramite `feature_weights`, un parametro nativo di XGBoost che aumenta la probabilità che una feature venga considerata ad ogni split durante il campionamento delle colonne (`colsample_bytree=0.7`, già in uso). Non forza il modello a usarla, ma gliene aumenta le occasioni.

## Come è stato scelto quanto "spingere" (metodologia)

Stessa logica onesta già usata per scegliere il peso dell'ensemble in Fase 4: **mai guardare l'holdout mentre si sceglie un iperparametro**. Ho testato 4 moltiplicatori per XGBoost (1.0=nessuna spinta, 2.0, 3.0, 5.0) insieme al peso ensemble w, tutti valutati SOLO sulle 10 stagioni di selezione (2010/11-2019/20), poi validati sulle 6 di holdout (2020/21-2025/26) mai toccate nella scelta.

**Risultato onesto**: sulle stagioni di selezione i 4 moltiplicatori sono sostanzialmente indistinguibili (log-loss tra 0,9767 e 0,9769 — differenze nella terza cifra decimale, rumore). Il moltiplicatore 2.0 vince per un margine minimo ed è stato scelto.

## L'esito: un miglioramento reale ma marginale, non un salto

| Modello | Prima (Fase 4c) | Dopo (Fase 4d) |
|---|---|---|
| LogisticRegression | 53,8% / 0,977 | 53,8% / 0,979 |
| XGBoost | 53,7% / 0,977 | 53,6% / 0,977 |
| Ensemble | 53,7% / 0,975 | **53,9% / 0,973** |

Sull'holdout imparziale (6 stagioni mai viste nella scelta): l'ensemble con più peso al valore rosa fa leggermente meglio in accuracy (53,64%→53,77%) ma il log-loss resta identico (0,9680). Da notare che XGBoost **da solo** peggiora leggermente con la spinta extra (53,60%→53,46% accuracy) — segno che "forzare" un modello di per sé già efficiente a guardare di più una feature non sempre aiuta: XGBoost sceglie già gli split migliori automaticamente quando una feature è informativa, e "spingerlo" oltre può fargli sprecare qualche split su una scelta localmente meno ottimale.

**Perché il guadagno è piccolo e non un salto netto**: il valore rosa è fortemente correlato con l'Elo e con i rating Poisson attacco/difesa già presenti nel modello — una squadra che ha comprato bene di solito ha già un Elo alto e segna/subisce in modo coerente con quello. Il modello estraeva già gran parte del segnale "forza economica" indirettamente attraverso queste altre feature; il valore rosa aggiunge principalmente il pezzo che Elo/forma NON catturano ancora (es. un mercato estivo appena concluso, prima che si traduca in risultati) — un contributo reale ma per natura limitato.

**Cosa è comunque cambiato concretamente**: nella feature importance del modello finale, il "gruppo valore rosa" (le 4 feature collegate: differenza, rapporto, valore casa, valore trasferta) ora pesa complessivamente **~18%** dell'importanza totale — quarto fattore più importante dopo Elo e le due probabilità Poisson, e più prominente di prima (dove `squad_value_diff` da sola era il 5° fattore). Il peso dell'ensemble si è inoltre riequilibrato da w=0,6 (60% Logistic Regression) a **w=0,5** (50/50): con il valore rosa ora presente anche nella Logistic Regression, i due modelli si sono avvicinati ulteriormente in capacità predittiva.

## Effetto concreto sulle previsioni 2026/2027

Ho rigenerato la simulazione Monte Carlo (Fase 5) con i modelli aggiornati. Il cambiamento più visibile riguarda le **neopromosse** (Frosinone, Monza, Venezia), che hanno i valori rosa più bassi del campionato: dando più peso a questa feature, le loro probabilità di retrocessione sono salite sensibilmente.

| Squadra | Retrocessione (prima) | Retrocessione (dopo) |
|---|---|---|
| Frosinone | 76,6% | 91,6% |
| Monza | 60,9% | 77,2% |
| Venezia | 49,3% | 70,3% |

In testa, il titolo di Inter scende leggermente (60,5%→55,0%) perché Juventus e Roma — squadre con rose di valore più alto rispetto al loro Elo attuale — guadagnano terreno (entrambe ora al 4° posto previsto a pari merito circa, davanti a Milan e Atalanta). La classifica prevista completa aggiornata è nei file allegati.

## Conclusione onesta

Il tuo istinto era in parte corretto: il valore rosa contribuisce in modo reale, e dandogli più spazio nel modello lo si vede riflesso in modo sensato nelle previsioni (soprattutto per le squadre economicamente più deboli). Ma il backtest dice chiaramente che **spingere il moltiplicatore oltre 2.0 (fino a 5.0) non porta benefici aggiuntivi** — il segnale aggiuntivo che il valore rosa può dare, al netto di quello che Elo/forma/Poisson già catturano, è limitato e il modello lo sta già usando in modo vicino all'ottimo con moltiplicatore 2.0. Se preferisci comunque forzare un peso più aggressivo nonostante l'evidenza (es. per un'intuizione che i dati storici non catturano pienamente, come l'effetto di un mercato estivo particolarmente forte), è una scelta legittima — basta cambiare `SQUAD_VALUE_WEIGHT_MULTIPLIER` in `src/train_xgboost.py` e rilanciare la pipeline; ma non è quello che il backtest onesto consiglia.

## File modificati/prodotti

```
serieA_predictor/
├── src/merge_squad_value_feature.py    # + squad_value_log_ratio
├── src/train_baseline.py                # + squad_value_diff/log_ratio in LogReg
├── src/train_xgboost.py                 # + squad_value_log_ratio, feature_weights (moltiplicatore 2.0)
├── src/train_ensemble.py                # grid search congiunta w + moltiplicatore, su selezione/holdout
├── src/predict_season.py                # aggiornato con le nuove feature e w=0.5
├── data/processed/serieA_features.csv    # + colonna squad_value_log_ratio
├── data/processed/previsioni_partite_2026_27.csv   # rigenerato
├── data/processed/classifica_prevista_2026_27.csv   # rigenerato
├── dashboard/previsioni_2026_27.html      # rigenerata
├── models/baseline_logreg.pkl, models/xgboost_final.pkl   # riallenati
└── notebooks/ensemble_results.csv, ensemble_weight_search.csv   # rigenerati (con colonna moltiplicatore)
```
