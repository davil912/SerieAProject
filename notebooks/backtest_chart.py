"""Grafico di confronto log-loss per stagione: modello baseline vs mercato vs baseline banale."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
df = pd.read_csv(BASE / "notebooks" / "backtest_results.csv")

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE_COLOR = "#c3c2b7"
SURFACE = "#fcfcfb"
SLOT_BLUE = "#2a78d6"     # LogisticRegression
SLOT_ORANGE = "#eb6834"   # Quote di mercato
SLOT_AQUA = "#1baf7a"     # Baseline vince sempre casa

plt.rcParams.update({
    "font.family": "sans-serif", "text.color": INK_PRIMARY,
    "axes.edgecolor": BASELINE_COLOR, "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})

pivot = df.pivot(index="season", columns="modello", values="log_loss")
order = ["Baseline_vince_sempre_casa", "Quote_di_mercato", "LogisticRegression"]
colors = {"LogisticRegression": SLOT_BLUE, "Quote_di_mercato": SLOT_ORANGE, "Baseline_vince_sempre_casa": SLOT_AQUA}
labels = {"LogisticRegression": "Modello (Logistic Regression)", "Quote_di_mercato": "Quote di mercato",
          "Baseline_vince_sempre_casa": "Baseline: vince sempre casa"}

fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
for col in order:
    ax.plot(pivot.index, pivot[col], color=colors[col], linewidth=2, marker="o", markersize=4, label=labels[col])

ax.set_title("Log-loss per stagione — backtest walk-forward (piu' basso = meglio)", loc="left", fontsize=12, color=INK_PRIMARY, pad=14)
ax.set_ylabel("Log-loss")
ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color(BASELINE_COLOR)
ax.legend(frameon=False, loc="upper right", fontsize=9)
plt.xticks(rotation=60, ha="right", fontsize=8)
plt.tight_layout()
fig.savefig(BASE / "notebooks" / "backtest_logloss.png", facecolor=SURFACE)
plt.close(fig)
print("Salvato notebooks/backtest_logloss.png")
