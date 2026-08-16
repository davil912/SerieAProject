"""
Fase 1 - Grafici esplorativi sul dataset Serie A pulito.
Genera due PNG in notebooks/: trend vantaggio-casa per stagione e
distribuzione degli esiti 1X2. Palette e stile seguono la skill "dataviz"
(palette categorica validata, linee sottili, griglia recessiva).
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

BASE = Path(__file__).resolve().parent.parent
df = pd.read_csv(BASE / "data" / "processed" / "serieA_matches.csv", parse_dates=["date"])

# Escludiamo le stagioni con dati grezzi incompleti (2000/01-2004/05, note dal
# report EDA) per non distorcere il trend con campioni troppo piccoli.
COMPLETE_FROM = "2005/2006"
df_complete = df[df["season"] >= COMPLETE_FROM]

# --- Stile base coerente con la skill dataviz ---
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

SLOT_BLUE = "#2a78d6"
SLOT_ORANGE = "#eb6834"
SLOT_AQUA = "#1baf7a"

plt.rcParams.update({
    "font.family": "sans-serif",
    "text.color": INK_PRIMARY,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
})

# ============ Grafico 1: trend vantaggio casa per stagione ============
home_rate = df_complete.groupby("season").apply(lambda g: (g["result"] == "H").mean() * 100)

fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
ax.plot(home_rate.index, home_rate.values, color=SLOT_BLUE, linewidth=2, marker="o", markersize=4)
ax.set_title("Percentuale vittorie in casa per stagione — Serie A", loc="left", fontsize=12, color=INK_PRIMARY, pad=14)
ax.set_ylabel("% vittorie casa")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color(BASELINE)
plt.xticks(rotation=60, ha="right", fontsize=8)
plt.tight_layout()
fig.savefig(BASE / "notebooks" / "home_advantage_trend.png", facecolor=SURFACE)
plt.close(fig)

# ============ Grafico 2: distribuzione esiti 1X2 ============
counts = df_complete["result"].value_counts(normalize=True).reindex(["H", "D", "A"]) * 100
labels = ["Vittoria Casa (H)", "Pareggio (D)", "Vittoria Trasferta (A)"]
colors = [SLOT_BLUE, SLOT_ORANGE, SLOT_AQUA]

fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
bars = ax.bar(labels, counts.values, color=colors, width=0.55, zorder=3)
ax.set_title("Distribuzione esiti 1X2 — Serie A (2005/06 - 2024/25)", loc="left", fontsize=12, color=INK_PRIMARY, pad=14)
ax.set_ylabel("% partite")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color(BASELINE)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 1, f"{val:.1f}%", ha="center", fontsize=10, color=INK_PRIMARY)
plt.tight_layout()
fig.savefig(BASE / "notebooks" / "result_distribution.png", facecolor=SURFACE)
plt.close(fig)

print("Grafici salvati in notebooks/: home_advantage_trend.png, result_distribution.png")
