#!/usr/bin/env python3
"""Builds the single-file Serie A 2026/27 predictions dashboard.
Reads the two source CSVs (unmodified) and embeds their data as JS constants
inside a self-contained HTML file. No external network calls, no localStorage.
"""
import csv
import json
import html as htmlmod
from pathlib import Path

# Percorsi relativi alla cartella del progetto (portabile: funziona a prescindere
# da dove il progetto viene clonato/copiato, non solo su questa macchina).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE = PROJECT_ROOT / "data" / "processed"
CLASSIFICA_CSV = BASE / "classifica_prevista_2026_27.csv"
MATCHES_CSV = BASE / "previsioni_partite_2026_27.csv"
STORIA_CSV = BASE / "classifica_storia_2026_27.csv"
PREV_STORIA_CSV = BASE / "previsioni_storia.csv"
OUT_PATH = Path(__file__).resolve().parent / "previsioni_2026_27.html"


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


def read_classifica_storia():
    rows = []
    with open(STORIA_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append([
                int(r["matchday"]),
                r["team"],
                round(float(r["punti_medi"]), 1),
                round(float(r["posizione_media"]), 1),
                round(float(r["GF_medi"]), 1),
                round(float(r["GS_medi"]), 1),
                round(float(r["DR_medio"]), 1),
                r["tipo"],
            ])
    rows.sort(key=lambda x: (x[0], x[1]))
    return rows


def read_previsioni_storia():
    # tracks how the FINAL predicted table itself changes across successive runs of
    # predict_season.py over the season (one full 20-team snapshot per run), as opposed to
    # STORIA_CSV which tracks the giornata-by-giornata progression within a single run.
    rows = []
    with open(PREV_STORIA_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append([
                int(r["giornata_riferimento"]),
                r["data_previsione"],
                r["team"],
                round(float(r["punti_medi"]), 1),
                round(float(r["posizione_media"]), 1),
                round(float(r["DR_medio"]), 1),
                round(float(r["prob_titolo_%"]), 1),
                round(float(r["prob_champions_top4_%"]), 1),
                round(float(r["prob_europa_top6_%"]), 1),
                round(float(r["prob_retrocessione_%"]), 1),
                int(r["pos"]),  # piazzamento vero e proprio (rank 1-20), non la media grezza delle simulazioni
            ])
    rows.sort(key=lambda x: (x[0], x[2]))
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
    history = read_classifica_storia()
    prev_storia = read_previsioni_storia()
    max_md = max(m[0] for m in matches)
    assert len(teams) == 20, f"expected 20 teams, got {len(teams)}"
    assert len(matches) == 380, f"expected 380 matches, got {len(matches)}"
    assert len(history) == 760, f"expected 760 history rows, got {len(history)}"
    assert len(prev_storia) % 20 == 0, (
        f"expected previsioni_storia rows to be a multiple of 20 (one row per team per "
        f"snapshot), got {len(prev_storia)}"
    )

    teams_json = json.dumps(
        [[t["pos"], t["team"]] for t in teams], ensure_ascii=False, separators=(",", ":")
    )
    matches_json = json.dumps(matches, ensure_ascii=False, separators=(",", ":"))
    history_json = json.dumps(history, ensure_ascii=False, separators=(",", ":"))
    prev_storia_json = json.dumps(prev_storia, ensure_ascii=False, separators=(",", ":"))
    n_prev_snapshots = len(set(r[0] for r in prev_storia))

    classifica_rows_html = build_classifica_rows(teams)
    matchday_options_html = build_matchday_options(max_md)

    html_out = HTML_TEMPLATE.format(
        classifica_rows=classifica_rows_html,
        matchday_options=matchday_options_html,
        max_md=max_md,
        teams_json=teams_json,
        matches_json=matches_json,
        history_json=history_json,
        prev_storia_json=prev_storia_json,
        n_teams=len(teams),
        n_matches=len(matches),
        n_history=len(history),
        n_prev_storia=len(prev_storia),
        n_prev_snapshots=n_prev_snapshots,
    )

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(html_out)

    print(f"Wrote {OUT_PATH} ({len(html_out):,} bytes)")
    print(
        f"Teams embedded: {len(teams)}, Matches embedded: {len(matches)}, "
        f"History rows embedded: {len(history)}, Matchdays: {max_md}, "
        f"Previsioni storia rows embedded: {len(prev_storia)} ({n_prev_snapshots} snapshot/s)"
    )


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
    --series-aqua:    #1baf7a;
    --series-yellow:  #eda100;
    --series-magenta: #e87ba4;
    --series-green:   #008300;
    --series-violet:  #4a3aa7;
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
      --series-aqua:    #199e70;
      --series-yellow:  #c98500;
      --series-magenta: #d55181;
      --series-green:   #008300;
      --series-violet:  #9085e9;
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
    --series-aqua:    #199e70;
    --series-yellow:  #c98500;
    --series-magenta: #d55181;
    --series-green:   #008300;
    --series-violet:  #9085e9;
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

  /* ---------- Section nav ---------- */
  .section-nav {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 0 0 24px;
  }}
  .section-nav a {{
    font-size: 13px;
    font-weight: 600;
    color: var(--text-secondary);
    text-decoration: none;
    padding: 6px 14px;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: var(--surface-1);
    box-shadow: var(--shadow);
  }}
  .section-nav a:hover {{ color: var(--text-primary); background: var(--tint-cl); }}
  .section-nav a:focus-visible {{ outline: 2px solid var(--series-blue); outline-offset: 1px; }}

  /* ---------- Cronologia chart ---------- */
  .btn-group button.metric-btn.active {{
    background: var(--series-blue);
    border-color: var(--series-blue);
    color: #fff;
  }}
  .chart-card {{
    background: var(--surface-1);
    border: 1px solid var(--border);
    border-radius: 12px;
    box-shadow: var(--shadow);
    padding: 16px 16px 6px;
  }}
  .chart-wrap {{
    position: relative;
  }}
  .history-svg {{
    display: block;
    width: 100%;
    height: auto;
  }}
  .hx-gridline {{ stroke: var(--gridline); stroke-width: 1; }}
  .hx-axis-text {{ fill: var(--text-muted); font-size: 11px; font-variant-numeric: tabular-nums; }}
  .hx-line {{ fill: none; stroke: var(--text-muted); stroke-width: 1.25; opacity: 0.32; }}
  .hx-line.active {{ stroke-width: 2.25; opacity: 1; stroke: var(--slot-color); }}
  .hx-line-projected {{ stroke-dasharray: 5 4; opacity: 0.55; }}
  .hx-line.active.hx-line-projected {{ opacity: 0.6; }}
  .hx-today-line {{ stroke: var(--text-muted); stroke-width: 1; stroke-dasharray: 3 3; opacity: 0.7; }}
  .hx-today-label {{ font-size: 10.5px; fill: var(--text-muted); }}
  .hx-dot {{ fill: var(--text-muted); stroke: var(--surface-1); stroke-width: 2; }}
  .hx-dot.active {{ fill: var(--slot-color); }}
  .hx-endlabel {{ font-size: 11.5px; font-weight: 600; fill: var(--text-primary); }}
  .hx-leader {{ stroke: var(--text-muted); stroke-width: 1; opacity: 0.55; }}
  .hx-crosshair {{ stroke: var(--baseline); stroke-width: 1; }}
  .hx-crosshair.hovering {{ stroke: var(--text-secondary); }}
  .slot-0 {{ --slot-color: var(--series-blue); }}
  .slot-1 {{ --slot-color: var(--series-orange); }}
  .slot-2 {{ --slot-color: var(--series-aqua); }}
  .slot-3 {{ --slot-color: var(--series-yellow); }}
  .slot-4 {{ --slot-color: var(--series-magenta); }}
  .slot-5 {{ --slot-color: var(--series-green); }}
  .slot-6 {{ --slot-color: var(--series-violet); }}
  .slot-7 {{ --slot-color: var(--series-red); }}

  .hx-tooltip {{
    position: absolute;
    top: 8px;
    background: var(--surface-1);
    border: 1px solid var(--border);
    border-radius: 10px;
    box-shadow: var(--shadow);
    padding: 8px 10px;
    font-size: 12px;
    min-width: 158px;
    max-width: 220px;
    pointer-events: none;
    display: none;
    z-index: 5;
  }}
  .hx-tooltip.visible {{ display: block; }}
  .hx-tt-title {{
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 4px;
    font-size: 12px;
  }}
  .hx-tt-row {{ display: flex; align-items: center; gap: 6px; padding: 2px 0; }}
  .hx-tt-key {{ width: 11px; height: 2px; border-radius: 1px; flex: none; background: var(--slot-color); }}
  .hx-tt-name {{ color: var(--text-secondary); flex: 1; }}
  .hx-tt-val {{ font-weight: 700; color: var(--text-primary); font-variant-numeric: tabular-nums; }}

  .chip-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin: 14px 0 12px;
  }}
  .team-chip {{
    font: inherit;
    font-size: 12px;
    padding: 4px 10px 4px 8px;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: var(--page-plane);
    color: var(--text-secondary);
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    line-height: 1;
  }}
  .team-chip:hover {{ background: var(--tint-cl); }}
  .team-chip:focus-visible {{ outline: 2px solid var(--series-blue); outline-offset: 1px; }}
  .team-chip .dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--slot-color, var(--baseline));
    flex: none;
  }}
  .team-chip.active {{
    color: var(--text-primary);
    font-weight: 600;
    background: var(--surface-1);
    box-shadow: var(--shadow);
  }}

  .table-wrap.mini {{ margin-top: 14px; }}
  .table-wrap.mini table {{ min-width: 520px; font-size: 12.5px; }}

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
    .table-wrap.mini table {{ min-width: 420px; }}
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

  <nav class="section-nav" aria-label="Sezioni">
    <a href="#classifica">Classifica</a>
    <a href="#partite">Partite</a>
    <a href="#cronologia">Cronologia</a>
    <a href="#evoluzione">Evoluzione previsione</a>
  </nav>

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

  <section id="cronologia">
    <div class="section-head">
      <h2>Cronologia classifica</h2>
    </div>
    <p class="section-note">
      Posizione media prevista di ogni squadra dopo ciascuna delle 38 giornate, calcolata sulle
      stesse 5.000 simulazioni Monte Carlo della classifica qui sopra (che è semplicemente lo
      "scatto" alla 38ª giornata). Nelle prime giornate le squadre sono ancora poco separate — più
      incertezza — mentre verso fine stagione le linee convergono verso la classifica finale.
      Clicca sui nomi delle squadre per evidenziarle; passa il mouse sul grafico o usa il cursore
      della giornata per leggere i valori esatti.
    </p>

    <div class="controls-row">
      <div class="btn-group" id="metric-toggle" role="group" aria-label="Metrica mostrata">
        <button type="button" class="metric-btn active" data-metric="pos">Posizione</button>
        <button type="button" class="metric-btn" data-metric="pts">Punti</button>
      </div>
      <label style="flex:1 1 220px; min-width:220px;">
        Giornata <span id="scrub-value">38</span> / {max_md}
        <input type="range" id="md-scrub" min="1" max="{max_md}" value="{max_md}" step="1">
      </label>
      <span class="results-info" id="history-info"></span>
    </div>

    <div class="chart-card">
      <div class="chart-wrap" id="history-chart-wrap">
        <svg class="history-svg" id="history-svg" viewBox="0 0 960 460" role="img" aria-label="Andamento della posizione media in classifica per giornata">
          <g id="hx-axes"></g>
          <g id="hx-lines"></g>
          <g id="hx-endlabels"></g>
          <line id="hx-crosshair" class="hx-crosshair" x1="0" y1="16" x2="0" y2="412" style="display:none"></line>
        </svg>
        <div class="hx-tooltip" id="hx-tooltip">
          <div class="hx-tt-title" id="hx-tt-title"></div>
          <div id="hx-tt-rows"></div>
        </div>
      </div>
      <div class="chip-row" id="team-chips"></div>
    </div>

    <div class="table-wrap mini">
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Squadra</th>
            <th class="num">Pos. media</th>
            <th class="num">Pt medi</th>
            <th class="num">DR</th>
          </tr>
        </thead>
        <tbody id="history-mini-tbody"></tbody>
      </table>
    </div>
  </section>

  <section id="evoluzione">
    <div class="section-head">
      <h2>Evoluzione previsione</h2>
    </div>
    <p class="section-note">
      Come è cambiata nel tempo la classifica finale prevista dal modello, confrontando le
      previsioni prodotte a ogni nuova esecuzione durante la stagione (ogni volta che vengono
      incorporati i risultati reali delle giornate nel frattempo giocate). "Giornata di
      riferimento 0" indica la previsione pre-stagione, senza ancora partite reali giocate.
      "Piazzamento" è la posizione vera e propria in classifica (1°, 2°, ...); "Posizione media"
      è invece la posizione media grezza calcolata sulle 5000 simulazioni Monte Carlo — le due
      possono differire leggermente (una squadra può avere il piazzamento migliore pur con una
      posizione media superiore a 1, se finisce spesso 2ª o 3ª in singole simulazioni).
      Clicca sui nomi delle squadre per evidenziarle; passa il mouse sul grafico per leggere i
      valori esatti.
    </p>

    <div class="controls-row">
      <div class="btn-group" id="evo-metric-toggle" role="group" aria-label="Metrica mostrata">
        <button type="button" class="metric-btn active" data-metric="rank">Piazzamento</button>
        <button type="button" class="metric-btn" data-metric="pos">Posizione media</button>
        <button type="button" class="metric-btn" data-metric="pts">Punti</button>
      </div>
      <span class="results-info" id="evo-info"></span>
    </div>

    <div class="chart-card">
      <div class="chart-wrap" id="evo-chart-wrap">
        <svg class="history-svg" id="evo-svg" viewBox="0 0 960 460" role="img" aria-label="Evoluzione del piazzamento o dei punti finali previsti, per giornata di riferimento">
          <g id="evo-axes"></g>
          <g id="evo-lines"></g>
          <g id="evo-endlabels"></g>
          <line id="evo-crosshair" class="hx-crosshair" x1="0" y1="16" x2="0" y2="412" style="display:none"></line>
        </svg>
        <div class="hx-tooltip" id="evo-tooltip">
          <div class="hx-tt-title" id="evo-tt-title"></div>
          <div id="evo-tt-rows"></div>
        </div>
      </div>
      <div class="chip-row" id="evo-chips"></div>
    </div>
  </section>

</div>

<footer>
  Fonte dati: simulazione interna (classifica_prevista_2026_27.csv, {n_teams} squadre — previsioni_partite_2026_27.csv, {n_matches} partite su {max_md} giornate — classifica_storia_2026_27.csv, {n_history} righe su {max_md} giornate x {n_teams} squadre — previsioni_storia.csv, {n_prev_storia} righe su {n_prev_snapshots} rilevazioni x {n_teams} squadre). Nessuna chiamata di rete esterna: tutti i dati sono incorporati nel file al momento della generazione.
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

(function () {{
  "use strict";

  // ---- Embedded data (build-time constant; no network calls) ----
  // HISTORY: [matchday, team, punti_medi, posizione_media, GF_medi, GS_medi, DR_medio, tipo]
  // tipo: "reale" (giornata gia' giocata, dato deterministico) o "simulata" (proiezione Monte Carlo)
  var HISTORY = {history_json};
  var MAX_MD = {max_md};
  var SVGNS = "http://www.w3.org/2000/svg";

  var svg = document.getElementById("history-svg");
  if (!svg) return;
  var axesG = document.getElementById("hx-axes");
  var linesG = document.getElementById("hx-lines");
  var labelsG = document.getElementById("hx-endlabels");
  var crosshair = document.getElementById("hx-crosshair");
  var chartWrap = document.getElementById("history-chart-wrap");
  var tooltip = document.getElementById("hx-tooltip");
  var ttTitle = document.getElementById("hx-tt-title");
  var ttRows = document.getElementById("hx-tt-rows");
  var chipRow = document.getElementById("team-chips");
  var scrub = document.getElementById("md-scrub");
  var scrubValue = document.getElementById("scrub-value");
  var historyInfo = document.getElementById("history-info");
  var miniTbody = document.getElementById("history-mini-tbody");
  var metricToggle = document.getElementById("metric-toggle");

  var PLOT_X0 = 46, PLOT_X1 = 860, PLOT_Y0 = 16, PLOT_Y1 = 412, VB_W = 960;

  // ---- group HISTORY rows by team (build-time data is already sorted by matchday,team) ----
  var byTeam = {{}};
  var teamOrderSeen = [];
  HISTORY.forEach(function (r) {{
    var team = r[1];
    if (!byTeam[team]) {{ byTeam[team] = []; teamOrderSeen.push(team); }}
    byTeam[team].push({{ md: r[0], pts: r[2], pos: r[3], gf: r[4], gs: r[5], dr: r[6], tipo: r[7] }});
  }});
  var TEAM_NAMES = teamOrderSeen.slice().sort();

  var PTS_MAX = 0;
  HISTORY.forEach(function (r) {{ if (r[2] > PTS_MAX) PTS_MAX = r[2]; }});
  PTS_MAX = Math.ceil(PTS_MAX / 10) * 10;

  // boundary matchday between "reale" (already played) and "simulata" (projected) rows,
  // i.e. the last played giornata league-wide; null when no real rows exist yet (pre-season)
  var REAL_BOUNDARY_MD = null;
  HISTORY.forEach(function (r) {{
    if (r[7] === "reale" && (REAL_BOUNDARY_MD === null || r[0] > REAL_BOUNDARY_MD)) REAL_BOUNDARY_MD = r[0];
  }});

  // final-matchday standings (used for default highlight + zone shading)
  var finalRows = HISTORY.filter(function (r) {{ return r[0] === MAX_MD; }})
    .slice()
    .sort(function (a, b) {{ return a[3] - b[3]; }});

  // ---- plain-JS UI state (no localStorage/sessionStorage) ----
  var state = {{ metric: "pos", md: MAX_MD, hoverMd: null }};

  var SLOT_COUNT = 8;
  var teamSlot = {{}};
  var usedSlots = [false, false, false, false, false, false, false, false];
  var highlighted = new Set();

  function hashSlot(team) {{
    var h = 0;
    for (var i = 0; i < team.length; i++) h = (h * 31 + team.charCodeAt(i)) >>> 0;
    return h % SLOT_COUNT;
  }}
  function acquireSlot(team) {{
    for (var i = 0; i < SLOT_COUNT; i++) {{
      if (!usedSlots[i]) {{ usedSlots[i] = true; teamSlot[team] = i; return; }}
    }}
    teamSlot[team] = hashSlot(team);
  }}
  function releaseSlot(team) {{
    var s = teamSlot[team];
    if (s !== undefined && usedSlots[s] !== undefined) usedSlots[s] = false;
    delete teamSlot[team];
  }}
  function highlightTeam(team) {{
    if (highlighted.has(team)) return;
    highlighted.add(team);
    acquireSlot(team);
  }}
  function unhighlightTeam(team) {{
    if (!highlighted.has(team)) return;
    highlighted.delete(team);
    releaseSlot(team);
  }}
  function toggleTeam(team) {{
    if (highlighted.has(team)) unhighlightTeam(team);
    else highlightTeam(team);
    renderChart();
    renderChips();
  }}

  // default: predicted top 4 (Champions zone) + bottom 3 (relegation zone) — the two
  // storylines a reader most wants to trace across the season, kept under the 8-slot cap
  var defaultTop = finalRows.slice(0, 4).map(function (r) {{ return r[1]; }});
  var defaultBottom = finalRows.slice(-3).map(function (r) {{ return r[1]; }});
  defaultTop.concat(defaultBottom).forEach(highlightTeam);

  function svgEl(tag, attrs) {{
    var el = document.createElementNS(SVGNS, tag);
    if (attrs) {{
      for (var k in attrs) {{ if (attrs.hasOwnProperty(k)) el.setAttribute(k, attrs[k]); }}
    }}
    return el;
  }}

  function xForMd(md) {{
    return PLOT_X0 + (md - 1) / (MAX_MD - 1) * (PLOT_X1 - PLOT_X0);
  }}
  function yForValue(v, metric) {{
    if (metric === "pos") {{
      var vv = Math.min(20, Math.max(1, v));
      return PLOT_Y0 + (vv - 1) / (20 - 1) * (PLOT_Y1 - PLOT_Y0);
    }}
    var t = Math.min(1, Math.max(0, v / PTS_MAX));
    return PLOT_Y1 - t * (PLOT_Y1 - PLOT_Y0);
  }}
  function mdForX(px) {{
    var t = (px - PLOT_X0) / (PLOT_X1 - PLOT_X0);
    var md = Math.round(1 + t * (MAX_MD - 1));
    return Math.min(MAX_MD, Math.max(1, md));
  }}
  function fmtVal(v, metric) {{
    var s = (Math.round(v * 10) / 10).toFixed(1);
    return metric === "pos" ? s + "°" : s;
  }}

  function renderAxes() {{
    axesG.textContent = "";
    var metric = state.metric;
    var ticks;
    if (metric === "pos") {{
      ticks = [1, 5, 10, 15, 20];
    }} else {{
      var step = PTS_MAX <= 60 ? 10 : 20;
      ticks = [];
      for (var v = 0; v <= PTS_MAX; v += step) ticks.push(v);
    }}
    ticks.forEach(function (t) {{
      var y = yForValue(t, metric);
      axesG.appendChild(svgEl("line", {{ x1: PLOT_X0, x2: PLOT_X1, y1: y, y2: y, "class": "hx-gridline" }}));
      var lbl = svgEl("text", {{ x: PLOT_X0 - 8, y: y + 3.5, "class": "hx-axis-text", "text-anchor": "end" }});
      lbl.textContent = String(t);
      axesG.appendChild(lbl);
    }});
    var mdTicks = [1, 5, 10, 15, 20, 25, 30, 35, MAX_MD];
    mdTicks.forEach(function (md) {{
      var x = xForMd(md);
      var lbl = svgEl("text", {{ x: x, y: PLOT_Y1 + 20, "class": "hx-axis-text", "text-anchor": "middle" }});
      lbl.textContent = String(md);
      axesG.appendChild(lbl);
    }});
    var xTitle = svgEl("text", {{ x: PLOT_X0, y: PLOT_Y1 + 36, "class": "hx-axis-text" }});
    xTitle.textContent = "Giornata";
    axesG.appendChild(xTitle);
    var yTitle = svgEl("text", {{ x: PLOT_X0, y: 10, "class": "hx-axis-text" }});
    yTitle.textContent = metric === "pos" ? "Posizione (1° in alto)" : "Punti medi";
    axesG.appendChild(yTitle);

    // "oggi" / ultima giornata giocata marker — only once real ("reale") rows exist
    if (REAL_BOUNDARY_MD !== null) {{
      var todayX = xForMd(REAL_BOUNDARY_MD);
      axesG.appendChild(svgEl("line", {{
        x1: todayX, x2: todayX, y1: PLOT_Y0, y2: PLOT_Y1, "class": "hx-today-line"
      }}));
      var todayLbl = svgEl("text", {{
        x: todayX + 4, y: PLOT_Y0 + 10, "class": "hx-today-label"
      }});
      todayLbl.textContent = "Oggi (g." + REAL_BOUNDARY_MD + ")";
      axesG.appendChild(todayLbl);
    }}
  }}

  // Returns an object with solid/projected path strings: "solid" is the already-played ("reale") portion of the
  // line up to and including REAL_BOUNDARY_MD, "projected" is the rest (simulated/future),
  // sharing the boundary point so the two segments join without a visual gap. When there is
  // no real data yet (REAL_BOUNDARY_MD === null, today's actual state), "projected" is empty
  // and "solid" is the full line, i.e. rendering is unchanged from before.
  function pathFor(team, metric) {{
    var rows = byTeam[team];
    var solid = "", projected = "";
    for (var i = 0; i < rows.length; i++) {{
      var x = xForMd(rows[i].md);
      var y = yForValue(metric === "pos" ? rows[i].pos : rows[i].pts, metric);
      var pt = x.toFixed(1) + "," + y.toFixed(1) + " ";
      var isReal = REAL_BOUNDARY_MD !== null && rows[i].md <= REAL_BOUNDARY_MD;
      if (isReal) {{
        solid += (solid === "" ? "M" : "L") + pt;
      }} else {{
        if (projected === "" && solid !== "") {{
          // start the projected segment from the last real point so the lines join
          var prev = rows[i - 1];
          var px = xForMd(prev.md).toFixed(1) + "," + yForValue(metric === "pos" ? prev.pos : prev.pts, metric).toFixed(1);
          projected += "M" + px + " ";
        }}
        projected += (projected === "" ? "M" : "L") + pt;
      }}
    }}
    return {{ solid: solid, projected: projected }};
  }}

  function renderLines() {{
    linesG.textContent = "";
    labelsG.textContent = "";
    var metric = state.metric;

    // muted lines first (background layer, all 20 teams)
    TEAM_NAMES.forEach(function (team) {{
      if (highlighted.has(team)) return;
      var d = pathFor(team, metric);
      linesG.appendChild(svgEl("path", {{ d: d.solid, "class": "hx-line" }}));
      if (d.projected) linesG.appendChild(svgEl("path", {{ d: d.projected, "class": "hx-line hx-line-projected" }}));
    }});

    // highlighted lines on top, full color, thicker stroke, end marker
    var endPoints = [];
    TEAM_NAMES.forEach(function (team) {{
      if (!highlighted.has(team)) return;
      var slot = teamSlot[team];
      var d = pathFor(team, metric);
      linesG.appendChild(svgEl("path", {{ d: d.solid, "class": "hx-line active slot-" + slot }}));
      if (d.projected) linesG.appendChild(svgEl("path", {{ d: d.projected, "class": "hx-line active hx-line-projected slot-" + slot }}));
      var rows = byTeam[team];
      var last = rows[rows.length - 1];
      var val = metric === "pos" ? last.pos : last.pts;
      var y = yForValue(val, metric);
      linesG.appendChild(svgEl("circle", {{ cx: xForMd(last.md), cy: y, r: 4, "class": "hx-dot active slot-" + slot }}));
      endPoints.push({{ team: team, slot: slot, trueY: y, y: y }});
    }});

    // direct end-of-line labels; nudge apart + leader line when they'd collide
    endPoints.sort(function (a, b) {{ return a.trueY - b.trueY; }});
    var minGap = 14;
    for (var i = 1; i < endPoints.length; i++) {{
      if (endPoints[i].y - endPoints[i - 1].y < minGap) endPoints[i].y = endPoints[i - 1].y + minGap;
    }}
    var lastX = xForMd(MAX_MD);
    endPoints.forEach(function (ep) {{
      if (Math.abs(ep.y - ep.trueY) > 1) {{
        labelsG.appendChild(svgEl("line", {{ x1: lastX + 5, y1: ep.trueY, x2: lastX + 14, y2: ep.y, "class": "hx-leader" }}));
      }}
      var t = svgEl("text", {{ x: lastX + 16, y: ep.y + 4, "class": "hx-endlabel" }});
      t.textContent = ep.team;
      labelsG.appendChild(t);
    }});
  }}

  function renderChart() {{
    renderAxes();
    renderLines();
    updateCrosshair(state.hoverMd !== null ? state.hoverMd : state.md, state.hoverMd !== null);
  }}

  function renderChips() {{
    chipRow.textContent = "";
    TEAM_NAMES.forEach(function (team) {{
      var active = highlighted.has(team);
      var chip = document.createElement("button");
      chip.type = "button";
      chip.className = "team-chip" + (active ? " active" : "");
      chip.setAttribute("aria-pressed", active ? "true" : "false");
      var dot = document.createElement("span");
      dot.className = "dot" + (active ? " slot-" + teamSlot[team] : "");
      var label = document.createElement("span");
      label.textContent = team;
      chip.appendChild(dot);
      chip.appendChild(label);
      chip.addEventListener("click", (function (teamName) {{
        return function () {{ toggleTeam(teamName); }};
      }})(team));
      chipRow.appendChild(chip);
    }});
  }}

  function updateCrosshair(md, isHover) {{
    var x = xForMd(md);
    crosshair.setAttribute("x1", x);
    crosshair.setAttribute("x2", x);
    crosshair.style.display = "block";
    crosshair.setAttribute("class", "hx-crosshair" + (isHover ? " hovering" : ""));

    var rows = [];
    TEAM_NAMES.forEach(function (team) {{
      if (!highlighted.has(team)) return;
      var r = byTeam[team][md - 1];
      if (!r) return;
      rows.push({{ team: team, slot: teamSlot[team], pos: r.pos, pts: r.pts }});
    }});
    rows.sort(function (a, b) {{ return a.pos - b.pos; }});

    ttTitle.textContent = "Giornata " + md;
    ttRows.textContent = "";
    rows.forEach(function (r) {{
      var row = document.createElement("div");
      row.className = "hx-tt-row";
      var key = document.createElement("span");
      key.className = "hx-tt-key slot-" + r.slot;
      var name = document.createElement("span");
      name.className = "hx-tt-name";
      name.textContent = r.team;
      var val = document.createElement("span");
      val.className = "hx-tt-val";
      val.textContent = state.metric === "pos" ? fmtVal(r.pos, "pos") : fmtVal(r.pts, "pts") + " pt";
      row.appendChild(key);
      row.appendChild(name);
      row.appendChild(val);
      ttRows.appendChild(row);
    }});

    tooltip.classList.toggle("visible", isHover && rows.length > 0);
    if (isHover) {{
      var wrapRect = chartWrap.getBoundingClientRect();
      var relX = x / VB_W * wrapRect.width;
      var left = Math.min(Math.max(relX + 10, 4), Math.max(4, wrapRect.width - 172));
      tooltip.style.left = left + "px";
    }}
  }}

  function zoneClassFor(rank) {{
    if (rank <= 4) return "zone-cl";
    if (rank <= 6) return "zone-europa";
    if (rank >= 18) return "zone-releg";
    return "";
  }}

  function renderMiniTable(md) {{
    var rows = HISTORY.filter(function (r) {{ return r[0] === md; }})
      .slice()
      .sort(function (a, b) {{ return a[3] - b[3]; }});
    miniTbody.textContent = "";
    rows.forEach(function (r, idx) {{
      var rank = idx + 1;
      var tr = document.createElement("tr");
      var zc = zoneClassFor(rank);
      if (zc) tr.className = zc;
      var tdRank = document.createElement("td");
      tdRank.className = "col-pos";
      tdRank.textContent = String(rank);
      var tdTeam = document.createElement("td");
      tdTeam.className = "col-team";
      tdTeam.textContent = r[1];
      var tdPos = document.createElement("td");
      tdPos.className = "num";
      tdPos.textContent = r[3].toFixed(1);
      var tdPts = document.createElement("td");
      tdPts.className = "num";
      tdPts.textContent = r[2].toFixed(1);
      var tdDr = document.createElement("td");
      tdDr.className = "num";
      tdDr.textContent = (r[6] > 0 ? "+" : "") + r[6].toFixed(1);
      tr.appendChild(tdRank);
      tr.appendChild(tdTeam);
      tr.appendChild(tdPos);
      tr.appendChild(tdPts);
      tr.appendChild(tdDr);
      miniTbody.appendChild(tr);
    }});
    historyInfo.textContent = rows.length + " squadre — classifica media prevista dopo la giornata " + md;
  }}

  metricToggle.addEventListener("click", function (e) {{
    var btn = e.target.closest ? e.target.closest(".metric-btn") : null;
    if (!btn) return;
    var metric = btn.getAttribute("data-metric");
    if (metric === state.metric) return;
    state.metric = metric;
    Array.prototype.forEach.call(metricToggle.querySelectorAll(".metric-btn"), function (b) {{
      b.classList.toggle("active", b === btn);
    }});
    renderChart();
    if (state.hoverMd === null) updateCrosshair(state.md, false);
  }});

  scrub.addEventListener("input", function () {{
    state.md = parseInt(scrub.value, 10);
    scrubValue.textContent = String(state.md);
    renderMiniTable(state.md);
    if (state.hoverMd === null) updateCrosshair(state.md, false);
  }});

  svg.addEventListener("pointermove", function (evt) {{
    var rect = svg.getBoundingClientRect();
    if (!rect.width) return;
    var scaleX = VB_W / rect.width;
    var svgX = (evt.clientX - rect.left) * scaleX;
    state.hoverMd = mdForX(svgX);
    updateCrosshair(state.hoverMd, true);
  }});
  svg.addEventListener("pointerleave", function () {{
    state.hoverMd = null;
    updateCrosshair(state.md, false);
  }});

  renderChips();
  renderChart();
  renderMiniTable(state.md);
}})();

(function () {{
  "use strict";

  // ---- Embedded data (build-time constant; no network calls) ----
  // PREV_STORIA: [giornata_riferimento, data_previsione, team, punti_medi, posizione_media,
  //               DR_medio, prob_titolo_%, prob_champions_top4_%, prob_europa_top6_%,
  //               prob_retrocessione_%, pos (piazzamento vero e proprio, 1-20)]
  // "posizione_media" (indice 4) e' la media grezza sulle simulazioni Monte Carlo; "pos"
  // (indice 10) e' il piazzamento vero e proprio nella classifica prevista (colonna "pos" di
  // classifica_prevista_2026_27.csv) - possono differire (una squadra puo' avere il miglior
  // piazzamento pur con posizione media > 1, se finisce spesso 2a/3a nelle singole simulazioni).
  // One full 20-team snapshot per predict_season.py run (re-running for an already-covered
  // giornata_riferimento replaces that snapshot, it does not duplicate it). Unlike HISTORY
  // above (progression within a single run), this tracks how the model's *final* predicted
  // table itself has shifted from run to run over the course of the season.
  var PREV_STORIA = {prev_storia_json};
  var SVGNS = "http://www.w3.org/2000/svg";

  var svg = document.getElementById("evo-svg");
  if (!svg) return;
  var axesG = document.getElementById("evo-axes");
  var linesG = document.getElementById("evo-lines");
  var labelsG = document.getElementById("evo-endlabels");
  var crosshair = document.getElementById("evo-crosshair");
  var chartWrap = document.getElementById("evo-chart-wrap");
  var tooltip = document.getElementById("evo-tooltip");
  var ttTitle = document.getElementById("evo-tt-title");
  var ttRows = document.getElementById("evo-tt-rows");
  var chipRow = document.getElementById("evo-chips");
  var evoInfo = document.getElementById("evo-info");
  var metricToggle = document.getElementById("evo-metric-toggle");

  var PLOT_X0 = 46, PLOT_X1 = 860, PLOT_Y0 = 16, PLOT_Y1 = 412, VB_W = 960;

  // ---- group PREV_STORIA rows by team (build-time data is already sorted by giornata,team) ----
  var byTeam = {{}};
  var teamOrderSeen = [];
  PREV_STORIA.forEach(function (r) {{
    var team = r[2];
    if (!byTeam[team]) {{ byTeam[team] = []; teamOrderSeen.push(team); }}
    byTeam[team].push({{
      gref: r[0], date: r[1], pts: r[3], pos: r[4], dr: r[5],
      pTitle: r[6], pTop4: r[7], pTop6: r[8], pReleg: r[9], rank: r[10]
    }});
  }});
  var TEAM_NAMES = teamOrderSeen.slice().sort();
  TEAM_NAMES.forEach(function (team) {{
    byTeam[team].sort(function (a, b) {{ return a.gref - b.gref; }});
  }});

  // distinct giornata_riferimento values present, ascending. Generic by construction: works the
  // same whether there is 1 snapshot (today, pre-season) or 38 of them (end of season).
  var GREFS = [];
  var seenGref = {{}};
  PREV_STORIA.forEach(function (r) {{
    if (!seenGref.hasOwnProperty(r[0])) {{ seenGref[r[0]] = true; GREFS.push(r[0]); }}
  }});
  GREFS.sort(function (a, b) {{ return a - b; }});
  var GREF_MIN = GREFS[0], GREF_MAX = GREFS[GREFS.length - 1];
  var SINGLE_X = GREF_MAX === GREF_MIN;

  var dateForGref = {{}};
  PREV_STORIA.forEach(function (r) {{ dateForGref[r[0]] = r[1]; }});

  var PTS_MAX = 0;
  PREV_STORIA.forEach(function (r) {{ if (r[3] > PTS_MAX) PTS_MAX = r[3]; }});
  PTS_MAX = Math.ceil(PTS_MAX / 10) * 10;
  if (PTS_MAX <= 0) PTS_MAX = 10;

  // most recent snapshot's standings drive the default highlight (predicted top4 + bottom3)
  var latestRows = PREV_STORIA.filter(function (r) {{ return r[0] === GREF_MAX; }})
    .slice()
    .sort(function (a, b) {{ return a[10] - b[10]; }});

  var state = {{ metric: "rank", hoverGref: null }};

  var SLOT_COUNT = 8;
  var teamSlot = {{}};
  var usedSlots = [false, false, false, false, false, false, false, false];
  var highlighted = new Set();

  function hashSlot(team) {{
    var h = 0;
    for (var i = 0; i < team.length; i++) h = (h * 31 + team.charCodeAt(i)) >>> 0;
    return h % SLOT_COUNT;
  }}
  function acquireSlot(team) {{
    for (var i = 0; i < SLOT_COUNT; i++) {{
      if (!usedSlots[i]) {{ usedSlots[i] = true; teamSlot[team] = i; return; }}
    }}
    teamSlot[team] = hashSlot(team);
  }}
  function releaseSlot(team) {{
    var s = teamSlot[team];
    if (s !== undefined && usedSlots[s] !== undefined) usedSlots[s] = false;
    delete teamSlot[team];
  }}
  function highlightTeam(team) {{
    if (highlighted.has(team)) return;
    highlighted.add(team);
    acquireSlot(team);
  }}
  function unhighlightTeam(team) {{
    if (!highlighted.has(team)) return;
    highlighted.delete(team);
    releaseSlot(team);
  }}
  function toggleTeam(team) {{
    if (highlighted.has(team)) unhighlightTeam(team);
    else highlightTeam(team);
    renderChart();
    renderChips();
  }}

  // default: predicted top 4 (Champions zone) + bottom 3 (relegation zone), same convention
  // used by the Cronologia chart above
  var defaultTop = latestRows.slice(0, 4).map(function (r) {{ return r[2]; }});
  var defaultBottom = latestRows.slice(-3).map(function (r) {{ return r[2]; }});
  defaultTop.concat(defaultBottom).forEach(highlightTeam);

  function svgEl(tag, attrs) {{
    var el = document.createElementNS(SVGNS, tag);
    if (attrs) {{
      for (var k in attrs) {{ if (attrs.hasOwnProperty(k)) el.setAttribute(k, attrs[k]); }}
    }}
    return el;
  }}

  function xForGref(g) {{
    if (SINGLE_X) return (PLOT_X0 + PLOT_X1) / 2;
    return PLOT_X0 + (g - GREF_MIN) / (GREF_MAX - GREF_MIN) * (PLOT_X1 - PLOT_X0);
  }}
  function yForValue(v, metric) {{
    if (metric === "pos" || metric === "rank") {{
      var vv = Math.min(20, Math.max(1, v));
      return PLOT_Y0 + (vv - 1) / (20 - 1) * (PLOT_Y1 - PLOT_Y0);
    }}
    var t = Math.min(1, Math.max(0, v / PTS_MAX));
    return PLOT_Y1 - t * (PLOT_Y1 - PLOT_Y0);
  }}
  function grefForX(px) {{
    if (SINGLE_X) return GREF_MIN;
    var t = (px - PLOT_X0) / (PLOT_X1 - PLOT_X0);
    var raw = GREF_MIN + t * (GREF_MAX - GREF_MIN);
    var best = GREFS[0], bestD = Infinity;
    GREFS.forEach(function (g) {{
      var d = Math.abs(g - raw);
      if (d < bestD) {{ bestD = d; best = g; }}
    }});
    return best;
  }}
  function fmtVal(v, metric) {{
    if (metric === "rank") return String(Math.round(v)) + "°";
    var s = (Math.round(v * 10) / 10).toFixed(1);
    return metric === "pos" ? s + "°" : s;
  }}
  function fmtDate(iso) {{
    if (!iso) return "";
    var parts = String(iso).split("-");
    if (parts.length !== 3) return iso;
    return parts[2] + "/" + parts[1] + "/" + parts[0];
  }}

  function renderAxes() {{
    axesG.textContent = "";
    var metric = state.metric;
    var ticks;
    if (metric === "pos" || metric === "rank") {{
      ticks = [1, 5, 10, 15, 20];
    }} else {{
      var step = PTS_MAX <= 60 ? 10 : 20;
      ticks = [];
      for (var v = 0; v <= PTS_MAX; v += step) ticks.push(v);
    }}
    ticks.forEach(function (t) {{
      var y = yForValue(t, metric);
      axesG.appendChild(svgEl("line", {{ x1: PLOT_X0, x2: PLOT_X1, y1: y, y2: y, "class": "hx-gridline" }}));
      var lbl = svgEl("text", {{ x: PLOT_X0 - 8, y: y + 3.5, "class": "hx-axis-text", "text-anchor": "end" }});
      lbl.textContent = String(t);
      axesG.appendChild(lbl);
    }});

    // x ticks: the distinct giornata_riferimento values actually present, thinned if many
    var xTicks = GREFS;
    if (GREFS.length > 12) {{
      xTicks = [];
      var n = GREFS.length;
      var everyK = Math.ceil(n / 10);
      for (var i = 0; i < n; i += everyK) xTicks.push(GREFS[i]);
      if (xTicks[xTicks.length - 1] !== GREFS[n - 1]) xTicks.push(GREFS[n - 1]);
    }}
    xTicks.forEach(function (g) {{
      var x = xForGref(g);
      var lbl = svgEl("text", {{ x: x, y: PLOT_Y1 + 20, "class": "hx-axis-text", "text-anchor": "middle" }});
      lbl.textContent = String(g);
      axesG.appendChild(lbl);
    }});
    var xTitle = svgEl("text", {{ x: PLOT_X0, y: PLOT_Y1 + 36, "class": "hx-axis-text" }});
    xTitle.textContent = "Giornata di riferimento (0 = pre-stagione)";
    axesG.appendChild(xTitle);
    var yTitle = svgEl("text", {{ x: PLOT_X0, y: 10, "class": "hx-axis-text" }});
    yTitle.textContent = metric === "rank" ? "Piazzamento finale previsto (1° in alto)" :
      (metric === "pos" ? "Posizione media finale prevista (1° in alto)" : "Punti finali medi previsti");
    axesG.appendChild(yTitle);
  }}

  function valueFor(row, metric) {{
    return metric === "rank" ? row.rank : (metric === "pos" ? row.pos : row.pts);
  }}

  function pathFor(team, metric) {{
    var rows = byTeam[team];
    var d = "";
    for (var i = 0; i < rows.length; i++) {{
      var x = xForGref(rows[i].gref);
      var y = yForValue(valueFor(rows[i], metric), metric);
      d += (d === "" ? "M" : "L") + x.toFixed(1) + "," + y.toFixed(1) + " ";
    }}
    return d;
  }}

  function renderLines() {{
    linesG.textContent = "";
    labelsG.textContent = "";
    var metric = state.metric;

    // muted teams first (background layer, all 20 teams). With a single giornata_riferimento
    // in the data a 1-point path is invisible (a lone "M", no segment) so we draw a dot instead;
    // as soon as a 2nd snapshot appears the same code naturally starts drawing a full line.
    TEAM_NAMES.forEach(function (team) {{
      if (highlighted.has(team)) return;
      var rows = byTeam[team];
      if (rows.length < 2) {{
        var last = rows[rows.length - 1];
        if (!last) return;
        var y = yForValue(valueFor(last, metric), metric);
        linesG.appendChild(svgEl("circle", {{ cx: xForGref(last.gref), cy: y, r: 3, "class": "hx-dot" }}));
      }} else {{
        linesG.appendChild(svgEl("path", {{ d: pathFor(team, metric), "class": "hx-line" }}));
      }}
    }});

    // highlighted teams on top, full color, thicker stroke, end marker + direct label
    var endPoints = [];
    TEAM_NAMES.forEach(function (team) {{
      if (!highlighted.has(team)) return;
      var slot = teamSlot[team];
      var rows = byTeam[team];
      var last = rows[rows.length - 1];
      if (!last) return;
      if (rows.length >= 2) {{
        linesG.appendChild(svgEl("path", {{ d: pathFor(team, metric), "class": "hx-line active slot-" + slot }}));
      }}
      var val = valueFor(last, metric);
      var y = yForValue(val, metric);
      linesG.appendChild(svgEl("circle", {{ cx: xForGref(last.gref), cy: y, r: 4, "class": "hx-dot active slot-" + slot }}));
      endPoints.push({{ team: team, slot: slot, trueY: y, y: y }});
    }});

    // direct end-of-line labels; nudge apart + leader line when they'd collide
    endPoints.sort(function (a, b) {{ return a.trueY - b.trueY; }});
    var minGap = 14;
    for (var i = 1; i < endPoints.length; i++) {{
      if (endPoints[i].y - endPoints[i - 1].y < minGap) endPoints[i].y = endPoints[i - 1].y + minGap;
    }}
    var lastX = xForGref(GREF_MAX);
    endPoints.forEach(function (ep) {{
      if (Math.abs(ep.y - ep.trueY) > 1) {{
        labelsG.appendChild(svgEl("line", {{ x1: lastX + 5, y1: ep.trueY, x2: lastX + 14, y2: ep.y, "class": "hx-leader" }}));
      }}
      var t = svgEl("text", {{ x: lastX + 16, y: ep.y + 4, "class": "hx-endlabel" }});
      t.textContent = ep.team;
      labelsG.appendChild(t);
    }});
  }}

  function renderChart() {{
    renderAxes();
    renderLines();
    updateCrosshair(state.hoverGref !== null ? state.hoverGref : GREF_MAX, state.hoverGref !== null);
  }}

  function renderChips() {{
    chipRow.textContent = "";
    TEAM_NAMES.forEach(function (team) {{
      var active = highlighted.has(team);
      var chip = document.createElement("button");
      chip.type = "button";
      chip.className = "team-chip" + (active ? " active" : "");
      chip.setAttribute("aria-pressed", active ? "true" : "false");
      var dot = document.createElement("span");
      dot.className = "dot" + (active ? " slot-" + teamSlot[team] : "");
      var label = document.createElement("span");
      label.textContent = team;
      chip.appendChild(dot);
      chip.appendChild(label);
      chip.addEventListener("click", (function (teamName) {{
        return function () {{ toggleTeam(teamName); }};
      }})(team));
      chipRow.appendChild(chip);
    }});
  }}

  function updateCrosshair(gref, isHover) {{
    var x = xForGref(gref);
    crosshair.setAttribute("x1", x);
    crosshair.setAttribute("x2", x);
    crosshair.style.display = "block";
    crosshair.setAttribute("class", "hx-crosshair" + (isHover ? " hovering" : ""));

    var rows = [];
    TEAM_NAMES.forEach(function (team) {{
      if (!highlighted.has(team)) return;
      var teamRows = byTeam[team];
      var r = null;
      for (var i = 0; i < teamRows.length; i++) {{
        if (teamRows[i].gref === gref) {{ r = teamRows[i]; break; }}
      }}
      if (!r) return;
      rows.push({{ team: team, slot: teamSlot[team], pos: r.pos, pts: r.pts, rank: r.rank }});
    }});
    rows.sort(function (a, b) {{ return a.rank - b.rank; }});

    ttTitle.textContent = "Giornata di riferimento " + gref + " · " + fmtDate(dateForGref[gref]);
    ttRows.textContent = "";
    rows.forEach(function (r) {{
      var row = document.createElement("div");
      row.className = "hx-tt-row";
      var key = document.createElement("span");
      key.className = "hx-tt-key slot-" + r.slot;
      var name = document.createElement("span");
      name.className = "hx-tt-name";
      name.textContent = r.team;
      var val = document.createElement("span");
      val.className = "hx-tt-val";
      val.textContent = state.metric === "rank" ? fmtVal(r.rank, "rank") :
        (state.metric === "pos" ? fmtVal(r.pos, "pos") : fmtVal(r.pts, "pts") + " pt");
      row.appendChild(key);
      row.appendChild(name);
      row.appendChild(val);
      ttRows.appendChild(row);
    }});

    tooltip.classList.toggle("visible", isHover && rows.length > 0);
    if (isHover) {{
      var wrapRect = chartWrap.getBoundingClientRect();
      var relX = x / VB_W * wrapRect.width;
      var left = Math.min(Math.max(relX + 10, 4), Math.max(4, wrapRect.width - 172));
      tooltip.style.left = left + "px";
    }}
  }}

  metricToggle.addEventListener("click", function (e) {{
    var btn = e.target.closest ? e.target.closest(".metric-btn") : null;
    if (!btn) return;
    var metric = btn.getAttribute("data-metric");
    if (metric === state.metric) return;
    state.metric = metric;
    Array.prototype.forEach.call(metricToggle.querySelectorAll(".metric-btn"), function (b) {{
      b.classList.toggle("active", b === btn);
    }});
    renderChart();
    if (state.hoverGref === null) updateCrosshair(GREF_MAX, false);
  }});

  svg.addEventListener("pointermove", function (evt) {{
    var rect = svg.getBoundingClientRect();
    if (!rect.width) return;
    var scaleX = VB_W / rect.width;
    var svgX = (evt.clientX - rect.left) * scaleX;
    state.hoverGref = grefForX(svgX);
    updateCrosshair(state.hoverGref, true);
  }});
  svg.addEventListener("pointerleave", function () {{
    state.hoverGref = null;
    updateCrosshair(GREF_MAX, false);
  }});

  evoInfo.textContent = GREFS.length + (GREFS.length === 1 ? " rilevazione disponibile" : " rilevazioni disponibili") +
    " — ultima: giornata di riferimento " + GREF_MAX + " (" + fmtDate(dateForGref[GREF_MAX]) + ")";

  renderChips();
  renderChart();
}})();
</script>

</body>
</html>
"""

if __name__ == "__main__":
    main()
