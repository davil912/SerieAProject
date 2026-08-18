#!/usr/bin/env python3
"""Builds the single-file Serie A 2026/27 predictions dashboard.
Reads the two source CSVs (unmodified) and embeds their data as JS constants
inside a self-contained HTML file. No external network calls, no localStorage.
"""
import csv
import json
import html as htmlmod

BASE = "/root/serieA_predictor/data/processed"
CLASSIFICA_CSV = f"{BASE}/classifica_prevista_2026_27.csv"
MATCHES_CSV = f"{BASE}/previsioni_partite_2026_27.csv"
OUT_PATH = "/root/serieA_predictor/dashboard/previsioni_2026_27.html"


def read_classifica():
    rows = []
    with open(CLASSIFICA_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "pos": int(r["pos"]),
                "team": r["team"],
                "pts": float(r["punti_medi"]),
                "pos_media": float(r["posizione_media"]),
                "gf": float(r["GF_medi"]),
                "gs": float(r["GS_medi"]),
                "dr": float(r["DR_medio"]),
                "w": float(r["V_medie"]),
                "d": float(r["N_medie"]),
                "l": float(r["P_medie"]),
                "p_title": float(r["prob_titolo_%"]),
                "p_top4": float(r["prob_champions_top4_%"]),
                "p_top6": float(r["prob_europa_top6_%"]),
                "p_releg": float(r["prob_retrocessione_%"]),
            })
    rows.sort(key=lambda x: x["pos"])
    return rows


def read_matches():
    rows = []
    with open(MATCHES_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append([
                int(r["matchday"]),
                r["date"],
                r["home_team"],
                r["away_team"],
                round(float(r["prob_home"]), 1),
                round(float(r["prob_draw"]), 1),
                round(float(r["prob_away"]), 1),
            ])
    rows.sort(key=lambda m: (m[0], m[1]))
    return rows


def esc(s):
    return htmlmod.escape(str(s), quote=True)


def zone_class(pos):
    if pos <= 4:
        return "zone-cl"
    if pos <= 6:
        return "zone-europa"
    if pos >= 18:
        return "zone-releg"
    return ""


def fmt1(x):
    return f"{x:.1f}"


def bar_cell(value, kind):
    # kind: 'pos' (blue, positive outcome) or 'neg' (red, relegation risk)
    pct = max(0.0, min(100.0, value))
    return (
        f'<td class="prob-cell"><div class="bar-track">'
        f'<div class="bar-fill {kind}" style="width:{pct:.1f}%"></div></div>'
        f'<span class="prob-value">{fmt1(value)}%</span></td>'
    )


def build_classifica_rows(teams):
    out = []
    for t in teams:
        zc = zone_class(t["pos"])
        cls = f' class="{zc}"' if zc else ""
        wdl = f'{t["w"]:.1f}-{t["d"]:.1f}-{t["l"]:.1f}'
        dr_sign = "+" if t["dr"] > 0 else ""
        row = (
            f'<tr{cls}>'
            f'<td class="col-pos">{t["pos"]}</td>'
            f'<td class="col-team">{esc(t["team"])}</td>'
            f'<td class="num">{fmt1(t["pts"])}</td>'
            f'<td class="num">{fmt1(t["gf"])}</td>'
            f'<td class="num">{fmt1(t["gs"])}</td>'
            f'<td class="num">{dr_sign}{fmt1(t["dr"])}</td>'
            f'<td class="num wdl">{wdl}</td>'
            f'{bar_cell(t["p_title"], "pos")}'
            f'{bar_cell(t["p_top4"], "pos")}'
            f'{bar_cell(t["p_top6"], "pos")}'
            f'{bar_cell(t["p_releg"], "neg")}'
            f'</tr>'
        )
        out.append(row)
    return "\n".join(out)


def build_matchday_options(max_md):
    opts = []
    for i in range(1, max_md + 1):
        opts.append(f'<option value="{i}">Giornata {i}</option>')
    return "\n".join(opts)


def main():
    teams = read_classifica()
    matches = read_matches()
    max_md = max(m[0] for m in matches)
    assert len(teams) == 20, f"expected 20 teams, got {len(teams)}"
    assert len(matches) == 380, f"expected 380 matches, got {len(matches)}"

    teams_json = json.dumps(
        [[t["pos"], t["team"]] for t in teams], ensure_ascii=False, separators=(",", ":")
    )
    matches_json = json.dumps(matches, ensure_ascii=False, separators=(",", ":"))

    classifica_rows_html = build_classifica_rows(teams)
    matchday_options_html = build_matchday_options(max_md)

    html_out = HTML_TEMPLATE.format(
        classifica_rows=classifica_rows_html,
        matchday_options=matchday_options_html,
        max_md=max_md,
        teams_json=teams_json,
        matches_json=matches_json,
        n_teams=len(teams),
        n_matches=len(matches),
    )

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(html_out)

    print(f"Wrote {OUT_PATH} ({len(html_out):,} bytes)")
    print(f"Teams embedded: {len(teams)}, Matches embedded: {len(matches)}, Matchdays: {max_md}")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Serie A 2026/2027 — Previsioni</title>
<style>
  :root {{
    color-scheme: light;
    --surface-1:      #fcfcfb;
    --page-plane:     #f9f9f7;
    --text-primary:   #0b0b0b;
    --text-secondary: #52514e;
    --text-muted:     #898781;
    --gridline:       #e1e0d9;
    --baseline:       #c3c2b7;
    --border:         rgba(11,11,11,0.10);
    --success-text:   #006300;
    --series-blue:    #2a78d6;
    --series-blue-soft: #cde2fb;
    --series-orange:  #eb6834;
    --series-red:     #e34948;
    --draw-fill:      #c3c2b7;
    --tint-cl:        rgba(42,120,214,0.09);
    --tint-europa:    rgba(42,120,214,0.045);
    --tint-releg:     rgba(227,73,72,0.09);
    --shadow: 0 1px 2px rgba(11,11,11,0.06), 0 1px 1px rgba(11,11,11,0.04);
  }}
  @media (prefers-color-scheme: dark) {{
    :root:where(:not([data-theme="light"])) {{
      color-scheme: dark;
      --surface-1:      #1a1a19;
      --page-plane:     #0d0d0d;
      --text-primary:   #ffffff;
      --text-secondary: #c3c2b7;
      --text-muted:     #898781;
      --gridline:       #2c2c2a;
      --baseline:       #383835;
      --border:         rgba(255,255,255,0.10);
      --success-text:   #0ca30c;
      --series-blue:    #3987e5;
      --series-blue-soft: #184f95;
      --series-orange:  #d95926;
      --series-red:     #e66767;
      --draw-fill:      #52514e;
      --tint-cl:        rgba(57,135,229,0.14);
      --tint-europa:    rgba(57,135,229,0.07);
      --tint-releg:     rgba(230,103,103,0.14);
      --shadow: 0 1px 2px rgba(0,0,0,0.4), 0 1px 1px rgba(0,0,0,0.3);
    }}
  }}
  :root[data-theme="dark"] {{
    color-scheme: dark;
    --surface-1:      #1a1a19;
    --page-plane:     #0d0d0d;
    --text-primary:   #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted:     #898781;
    --gridline:       #2c2c2a;
    --baseline:       #383835;
    --border:         rgba(255,255,255,0.10);
    --success-text:   #0ca30c;
    --series-blue:    #3987e5;
    --series-blue-soft: #184f95;
    --series-orange:  #d95926;
    --series-red:     #e66767;
    --draw-fill:      #52514e;
    --tint-cl:        rgba(57,135,229,0.14);
    --tint-europa:    rgba(57,135,229,0.07);
    --tint-releg:     rgba(230,103,103,0.14);
    --shadow: 0 1px 2px rgba(0,0,0,0.4), 0 1px 1px rgba(0,0,0,0.3);
  }}

  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; }}
  body {{
    background: var(--page-plane);
    color: var(--text-primary);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    line-height: 1.45;
  }}
  .wrap {{
    max-width: 1180px;
    margin: 0 auto;
    padding: 0 20px 64px;
  }}

  /* ---------- Header / banner ---------- */
  header.banner {{
    background: linear-gradient(135deg, var(--series-blue) 0%, #184f95 100%);
    color: #ffffff;
    padding: 36px 20px 30px;
    margin-bottom: 28px;
  }}
  .banner-inner {{
    max-width: 1180px;
    margin: 0 auto;
  }}
  header.banner h1 {{
    margin: 0 0 8px;
    font-size: clamp(24px, 3.4vw, 34px);
    font-weight: 700;
    letter-spacing: -0.01em;
  }}
  header.banner .subtitle {{
    margin: 0 0 16px;
    max-width: 860px;
    font-size: 14.5px;
    color: rgba(255,255,255,0.88);
  }}
  .meta-chips {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 16px;
  }}
  .meta-chip {{
    background: rgba(255,255,255,0.14);
    border: 1px solid rgba(255,255,255,0.22);
    border-radius: 999px;
    padding: 4px 12px;
    font-size: 12.5px;
    color: #fff;
  }}
  details.limitations {{
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.22);
    border-radius: 10px;
    padding: 10px 14px;
    max-width: 860px;
  }}
  details.limitations summary {{
    cursor: pointer;
    font-weight: 600;
    font-size: 13.5px;
    color: #fff;
    list-style: none;
    display: flex;
    align-items: center;
    gap: 6px;
  }}
  details.limitations summary::-webkit-details-marker {{ display: none; }}
  details.limitations summary::before {{
    content: "ⓘ";
    font-weight: 700;
  }}
  details.limitations[open] summary {{ margin-bottom: 8px; }}
  details.limitations ul {{
    margin: 0;
    padding-left: 18px;
    font-size: 13px;
    color: rgba(255,255,255,0.92);
  }}
  details.limitations li {{ margin-bottom: 6px; }}
  details.limitations li:last-child {{ margin-bottom: 0; }}

  /* ---------- Sections ---------- */
  section {{ margin-bottom: 40px; }}
  .section-head {{
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 10px;
    margin-bottom: 14px;
  }}
  .section-head h2 {{
    margin: 0;
    font-size: 20px;
    font-weight: 700;
  }}
  .section-note {{
    font-size: 12.5px;
    color: var(--text-muted);
    margin: -6px 0 14px;
  }}

  .legend {{
    display: flex;
    flex-wrap: wrap;
    gap: 14px;
  }}
  .legend-item {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 12.5px;
    color: var(--text-secondary);
  }}
  .swatch {{
    width: 12px;
    height: 12px;
    border-radius: 3px;
    display: inline-block;
    flex: none;
  }}
  .swatch.cl {{ background: var(--series-blue); }}
  .swatch.europa {{ background: var(--series-blue-soft); }}
  .swatch.releg {{ background: var(--series-red); }}
  .swatch.home {{ background: var(--series-blue); }}
  .swatch.draw {{ background: var(--draw-fill); }}
  .swatch.away {{ background: var(--series-orange); }}

  /* ---------- Classifica table ---------- */
  .table-wrap {{
    background: var(--surface-1);
    border: 1px solid var(--border);
    border-radius: 12px;
    box-shadow: var(--shadow);
    overflow-x: auto;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13.5px;
    min-width: 880px;
  }}
  thead th {{
    text-align: left;
    font-size: 11.5px;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    color: var(--text-muted);
    font-weight: 600;
    padding: 10px 10px;
    border-bottom: 1px solid var(--gridline);
    white-space: nowrap;
  }}
  thead th.num, thead th.prob-cell {{ text-align: right; }}
  tbody td {{
    padding: 8px 10px;
    border-bottom: 1px solid var(--gridline);
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
  }}
  tbody tr:last-child td {{ border-bottom: none; }}
  td.col-pos {{
    color: var(--text-muted);
    font-weight: 600;
    width: 30px;
  }}
  td.col-team {{
    font-weight: 600;
    color: var(--text-primary);
    font-variant-numeric: normal;
  }}
  td.num {{ text-align: right; color: var(--text-secondary); }}
  td.wdl {{ color: var(--text-muted); font-size: 12.5px; }}
  tr.zone-cl {{ background: var(--tint-cl); }}
  tr.zone-europa {{ background: var(--tint-europa); }}
  tr.zone-releg {{ background: var(--tint-releg); }}

  td.prob-cell {{
    text-align: right;
  }}
  .bar-track {{
    display: inline-block;
    vertical-align: middle;
    width: 56px;
    height: 6px;
    border-radius: 3px;
    background: var(--gridline);
    overflow: hidden;
    margin-right: 8px;
  }}
  .bar-fill {{
    height: 100%;
    border-radius: 3px 0 0 3px;
  }}
  .bar-fill.pos {{ background: var(--series-blue); }}
  .bar-fill.neg {{ background: var(--series-red); }}
  .prob-value {{
    display: inline-block;
    min-width: 40px;
    text-align: right;
    color: var(--text-secondary);
    font-weight: 600;
  }}

  /* ---------- Controls row ---------- */
  .controls-row {{
    display: flex;
    flex-wrap: wrap;
    align-items: end;
    gap: 16px;
    background: var(--surface-1);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 14px;
  }}
  .controls-row label {{
    display: flex;
    flex-direction: column;
    gap: 4px;
    font-size: 11.5px;
    color: var(--text-muted);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.02em;
  }}
  .controls-row select,
  .controls-row input[type="text"] {{
    font: inherit;
    font-size: 14px;
    padding: 7px 10px;
    border-radius: 8px;
    border: 1px solid var(--baseline);
    background: var(--page-plane);
    color: var(--text-primary);
    min-width: 150px;
  }}
  .controls-row select:focus,
  .controls-row input[type="text"]:focus {{
    outline: 2px solid var(--series-blue);
    outline-offset: 1px;
  }}
  .btn-group {{ display: flex; gap: 4px; }}
  .btn-group button, .icon-btn {{
    font: inherit;
    font-size: 15px;
    font-weight: 700;
    width: 34px;
    height: 34px;
    border-radius: 8px;
    border: 1px solid var(--baseline);
    background: var(--page-plane);
    color: var(--text-primary);
    cursor: pointer;
  }}
  .btn-group button:hover, .icon-btn:hover {{ background: var(--tint-cl); }}
  .checkbox-label {{
    flex-direction: row !important;
    align-items: center;
    gap: 6px !important;
    text-transform: none !important;
    font-size: 13px !important;
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
  }}
  .checkbox-label input {{ width: 16px; height: 16px; accent-color: var(--series-blue); }}
  .results-info {{
    margin-left: auto;
    font-size: 12.5px;
    color: var(--text-muted);
    align-self: center;
  }}
  .outcome-legend {{ margin-bottom: 14px; }}

  /* ---------- Match cards ---------- */
  .matches-list {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 12px;
  }}
  .match-card {{
    background: var(--surface-1);
    border: 1px solid var(--border);
    border-radius: 12px;
    box-shadow: var(--shadow);
    padding: 12px 14px;
  }}
  .match-meta {{
    display: flex;
    justify-content: space-between;
    font-size: 11.5px;
    color: var(--text-muted);
    margin-bottom: 8px;
  }}
  .md-badge {{
    font-weight: 700;
    color: var(--text-secondary);
  }}
  .match-teams {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    font-size: 14.5px;
    font-weight: 600;
    margin-bottom: 10px;
  }}
  .match-teams .team {{
    flex: 1;
    color: var(--text-secondary);
  }}
  .match-teams .team.away {{ text-align: right; }}
  .match-teams .team.winner {{ color: var(--text-primary); }}
  .match-teams .vs {{
    color: var(--text-muted);
    font-weight: 400;
    font-size: 12px;
    flex: none;
  }}
  .prob-bar {{
    display: flex;
    gap: 2px;
    height: 8px;
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 8px;
    background: var(--gridline);
  }}
  .prob-bar .seg {{ height: 100%; }}
  .prob-bar .seg.home {{ background: var(--series-blue); border-radius: 4px 0 0 4px; }}
  .prob-bar .seg.draw {{ background: var(--draw-fill); }}
  .prob-bar .seg.away {{ background: var(--series-orange); border-radius: 0 4px 4px 0; }}
  .prob-badges {{
    display: flex;
    gap: 6px;
  }}
  .badge {{
    flex: 1;
    text-align: center;
    padding: 4px 4px;
    border-radius: 7px;
    font-size: 12px;
    font-variant-numeric: tabular-nums;
    color: var(--text-secondary);
    background: var(--page-plane);
    border: 1px solid var(--gridline);
  }}
  .badge.emph {{
    font-weight: 700;
    color: #fff;
  }}
  .badge.home.emph {{ background: var(--series-blue); border-color: var(--series-blue); }}
  .badge.draw.emph {{ background: var(--draw-fill); border-color: var(--draw-fill); color: var(--text-primary); }}
  .badge.away.emph {{ background: var(--series-orange); border-color: var(--series-orange); }}

  .empty-state {{
    grid-column: 1 / -1;
    text-align: center;
    padding: 40px 20px;
    color: var(--text-muted);
    font-size: 14px;
  }}

  footer {{
    max-width: 1180px;
    margin: 0 auto;
    padding: 0 20px 30px;
    font-size: 12px;
    color: var(--text-muted);
  }}

  @media (max-width: 720px) {{
    header.banner {{ padding: 26px 16px 22px; }}
    .controls-row {{ flex-direction: column; align-items: stretch; }}
    .controls-row select, .controls-row input[type="text"] {{ min-width: 0; width: 100%; }}
    .results-info {{ margin-left: 0; }}
    .matches-list {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>

<header class="banner">
  <div class="banner-inner">
    <h1>Serie A 2026/2027 — Previsioni</h1>
    <p class="subtitle">
      Proiezioni pre-stagione basate su 5.000 simulazioni Monte Carlo dell'intero campionato,
      generate il 18 agosto 2026. Il modello combina un ensemble Logistic Regression + XGBoost
      con una componente Poisson per i gol attesi (Elo, forma recente e scontri diretti evolvono
      dentro ogni simulazione man mano che si "giocano" i risultati), validato con backtest
      walk-forward su 16 stagioni storiche (~53.7% di accuratezza sull'esito 1X2, a fronte di un
      benchmark quote di mercato ~54.6%).
    </p>
    <div class="meta-chips">
      <span class="meta-chip">5.000 simulazioni Monte Carlo</span>
      <span class="meta-chip">Ensemble LogReg + XGBoost + Poisson xG</span>
      <span class="meta-chip">Backtest 16 stagioni · ~53.7% accuratezza</span>
      <span class="meta-chip">Generato il 18/08/2026</span>
    </div>
    <details class="limitations">
      <summary>Limiti e avvertenze del modello</summary>
      <ul>
        <li><strong>Neopromosse (Frosinone, Monza, Venezia):</strong> per queste 3 squadre l'ultimo
          dato Elo/forma disponibile risale alla loro ultima presenza in Serie A (2024/25), non alla
          stagione di Serie B 2025/26 appena vinta sul campo — il punto di partenza del modello è
          quindi meno aggiornato per loro rispetto alle altre 17 squadre.</li>
        <li><strong>Valore rosa:</strong> per Frosinone e Monza la copertura dei dati Transfermarkt è
          sotto la soglia di affidabilità considerata accettabile, per cui la stima del valore di
          rosa per queste due squadre va trattata con cautela.</li>
        <li><strong>Incertezza crescente nel tempo:</strong> le previsioni sulle giornate più lontane
          sono naturalmente più incerte di quelle sulle prime giornate e andranno riviste quando il
          modello verrà ri-eseguito incorporando i risultati reali via via disponibili durante la
          stagione.</li>
      </ul>
    </details>
  </div>
</header>

<div class="wrap">

  <section id="classifica">
    <div class="section-head">
      <h2>Classifica prevista</h2>
      <div class="legend">
        <span class="legend-item"><span class="swatch cl"></span>Zona Champions (1°–4°)</span>
        <span class="legend-item"><span class="swatch europa"></span>Zona Europa (5°–6°)</span>
        <span class="legend-item"><span class="swatch releg"></span>Zona retrocessione (18°–20°)</span>
      </div>
    </div>
    <p class="section-note">
      Valori medi sulle 5.000 simulazioni della stagione. Le barre nelle colonne di probabilità sono
      su scala 0–100%.
    </p>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Squadra</th>
            <th class="num">Pt</th>
            <th class="num">GF</th>
            <th class="num">GS</th>
            <th class="num">DR</th>
            <th class="num">V-N-P</th>
            <th class="prob-cell">Titolo %</th>
            <th class="prob-cell">Champions %</th>
            <th class="prob-cell">Europa %</th>
            <th class="prob-cell">Retro. %</th>
          </tr>
        </thead>
        <tbody>
{classifica_rows}
        </tbody>
      </table>
    </div>
  </section>

  <section id="partite">
    <div class="section-head">
      <h2>Partite</h2>
    </div>
    <p class="section-note">
      380 partite previste (38 giornate). Filtra per giornata o cerca una squadra per vederne tutte
      le partite della stagione.
    </p>

    <div class="controls-row">
      <label>Giornata
        <select id="matchday-select">
{matchday_options}
        </select>
      </label>
      <div class="btn-group">
        <button type="button" id="md-prev" aria-label="Giornata precedente">‹</button>
        <button type="button" id="md-next" aria-label="Giornata successiva">›</button>
      </div>
      <label>Cerca squadra
        <input type="text" id="team-search" placeholder="Es. Inter, Napoli…" autocomplete="off">
      </label>
      <label class="checkbox-label">
        <input type="checkbox" id="big-match-toggle">
        Solo big match (Top 6)
      </label>
      <span class="results-info" id="results-info"></span>
    </div>

    <div class="legend outcome-legend">
      <span class="legend-item"><span class="swatch home"></span>Vittoria casa</span>
      <span class="legend-item"><span class="swatch draw"></span>Pareggio</span>
      <span class="legend-item"><span class="swatch away"></span>Vittoria trasferta</span>
    </div>

    <div class="matches-list" id="matches-list"></div>
  </section>

</div>

<footer>
  Fonte dati: simulazione interna (classifica_prevista_2026_27.csv, {n_teams} squadre — previsioni_partite_2026_27.csv, {n_matches} partite su {max_md} giornate). Nessuna chiamata di rete esterna: tutti i dati sono incorporati nel file al momento della generazione.
</footer>

<script>
(function () {{
  "use strict";

  // ---- Embedded data (build-time constants; no network calls) ----
  // TEAMS: [pos, teamName]
  var TEAMS = {teams_json};
  // MATCHES: [matchday, date, homeTeam, awayTeam, probHome, probDraw, probAway]
  var MATCHES = {matches_json};
  var MAX_MD = {max_md};

  var BIG_TEAMS = new Set(TEAMS.filter(function (t) {{ return t[0] <= 6; }}).map(function (t) {{ return t[1]; }}));

  // ---- Plain-JS UI state (no localStorage/sessionStorage) ----
  var state = {{
    matchday: 1,
    search: "",
    bigOnly: false
  }};

  var mdSelect = document.getElementById("matchday-select");
  var mdPrev = document.getElementById("md-prev");
  var mdNext = document.getElementById("md-next");
  var searchInput = document.getElementById("team-search");
  var bigToggle = document.getElementById("big-match-toggle");
  var resultsInfo = document.getElementById("results-info");
  var listEl = document.getElementById("matches-list");

  var dateFmt;
  try {{
    dateFmt = new Intl.DateTimeFormat("it-IT", {{ weekday: "short", day: "2-digit", month: "short" }});
  }} catch (e) {{
    dateFmt = null;
  }}

  function formatDate(iso) {{
    if (!dateFmt) return iso;
    var parts = iso.split("-");
    var d = new Date(Date.UTC(parseInt(parts[0], 10), parseInt(parts[1], 10) - 1, parseInt(parts[2], 10)));
    var s = dateFmt.format(d);
    return s.charAt(0).toUpperCase() + s.slice(1);
  }}

  function normalize(s) {{
    return s.toLowerCase();
  }}

  function outcomeIndex(pH, pD, pA) {{
    if (pH >= pD && pH >= pA) return 0;
    if (pA >= pD && pA >= pH) return 2;
    return 1;
  }}

  function getFiltered() {{
    var list = MATCHES;
    if (state.bigOnly) {{
      list = list.filter(function (m) {{ return BIG_TEAMS.has(m[2]) && BIG_TEAMS.has(m[3]); }});
    }}
    var q = state.search.trim();
    if (q) {{
      var nq = normalize(q);
      list = list.filter(function (m) {{
        return normalize(m[2]).indexOf(nq) !== -1 || normalize(m[3]).indexOf(nq) !== -1;
      }});
    }} else {{
      list = list.filter(function (m) {{ return m[0] === state.matchday; }});
    }}
    return list;
  }}

  function makeCard(m) {{
    var md = m[0], date = m[1], home = m[2], away = m[3], pH = m[4], pD = m[5], pA = m[6];
    var idx = outcomeIndex(pH, pD, pA);

    var card = document.createElement("article");
    card.className = "match-card";

    var meta = document.createElement("div");
    meta.className = "match-meta";
    var mdBadge = document.createElement("span");
    mdBadge.className = "md-badge";
    mdBadge.textContent = "Giornata " + md;
    var dateSpan = document.createElement("span");
    dateSpan.className = "match-date";
    dateSpan.textContent = formatDate(date);
    meta.appendChild(mdBadge);
    meta.appendChild(dateSpan);
    card.appendChild(meta);

    var teams = document.createElement("div");
    teams.className = "match-teams";
    var homeSpan = document.createElement("span");
    homeSpan.className = "team home" + (idx === 0 ? " winner" : "");
    homeSpan.textContent = home;
    var vsSpan = document.createElement("span");
    vsSpan.className = "vs";
    vsSpan.textContent = "vs";
    var awaySpan = document.createElement("span");
    awaySpan.className = "team away" + (idx === 2 ? " winner" : "");
    awaySpan.textContent = away;
    teams.appendChild(homeSpan);
    teams.appendChild(vsSpan);
    teams.appendChild(awaySpan);
    card.appendChild(teams);

    var bar = document.createElement("div");
    bar.className = "prob-bar";
    bar.title = home + " " + pH + "% · Pareggio " + pD + "% · " + away + " " + pA + "%";
    var segHome = document.createElement("div");
    segHome.className = "seg home";
    segHome.style.width = pH + "%";
    var segDraw = document.createElement("div");
    segDraw.className = "seg draw";
    segDraw.style.width = pD + "%";
    var segAway = document.createElement("div");
    segAway.className = "seg away";
    segAway.style.width = pA + "%";
    bar.appendChild(segHome);
    bar.appendChild(segDraw);
    bar.appendChild(segAway);
    card.appendChild(bar);

    var badges = document.createElement("div");
    badges.className = "prob-badges";
    var labels = [["home", "1", pH], ["draw", "X", pD], ["away", "2", pA]];
    labels.forEach(function (l, i) {{
      var b = document.createElement("span");
      b.className = "badge " + l[0] + (i === idx ? " emph" : "");
      b.textContent = l[1] + " · " + l[2] + "%";
      badges.appendChild(b);
    }});
    card.appendChild(badges);

    return card;
  }}

  function render() {{
    var filtered = getFiltered().slice().sort(function (a, b) {{
      if (a[0] !== b[0]) return a[0] - b[0];
      if (a[1] !== b[1]) return a[1] < b[1] ? -1 : 1;
      return 0;
    }});

    listEl.textContent = "";
    if (filtered.length === 0) {{
      var empty = document.createElement("div");
      empty.className = "empty-state";
      empty.textContent = "Nessuna partita trovata con questi filtri.";
      listEl.appendChild(empty);
    }} else {{
      var frag = document.createDocumentFragment();
      filtered.forEach(function (m) {{ frag.appendChild(makeCard(m)); }});
      listEl.appendChild(frag);
    }}

    var q = state.search.trim();
    if (q) {{
      resultsInfo.textContent = filtered.length + " partite trovate per \"" + q + "\" (tutte le giornate)";
      mdSelect.disabled = true;
      mdPrev.disabled = true;
      mdNext.disabled = true;
    }} else {{
      resultsInfo.textContent = filtered.length + " partite — Giornata " + state.matchday;
      mdSelect.disabled = false;
      mdPrev.disabled = state.matchday <= 1;
      mdNext.disabled = state.matchday >= MAX_MD;
    }}
  }}

  mdSelect.addEventListener("change", function () {{
    state.matchday = parseInt(mdSelect.value, 10);
    render();
  }});
  mdPrev.addEventListener("click", function () {{
    if (state.matchday > 1) {{
      state.matchday -= 1;
      mdSelect.value = String(state.matchday);
      render();
    }}
  }});
  mdNext.addEventListener("click", function () {{
    if (state.matchday < MAX_MD) {{
      state.matchday += 1;
      mdSelect.value = String(state.matchday);
      render();
    }}
  }});
  searchInput.addEventListener("input", function () {{
    state.search = searchInput.value;
    render();
  }});
  bigToggle.addEventListener("change", function () {{
    state.bigOnly = bigToggle.checked;
    render();
  }});

  mdSelect.value = "1";
  render();
}})();
</script>

</body>
</html>
"""

if __name__ == "__main__":
    main()
