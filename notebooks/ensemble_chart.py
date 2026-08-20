"""Grafico: log-loss sulle stagioni di selezione al variare del peso dell'ensemble."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
df_all = pd.read_csv(BASE / "notebooks" / "ensemble_weight_search.csv")
# Fase 4e: il file ora contiene una griglia congiunta (peso w x half-life stagioni recenti).
# Per il grafico (peso vs log-loss) teniamo solo le righe dell'half-life scelto.
# season_half_life=None (nessuna ponderazione) viene letto da pandas come NaN.
best_row = df_all.loc[df_all["log_loss"].idxmin()]
best_hl = best_row["season_half_life"]
if pd.isna(best_hl):
    df = df_all[df_all["season_half_life"].isna()].reset_index(drop=True)
    best_hl_label = "nessuno (uniforme)"
else:
    df = df_all[df_all["season_half_life"] == best_hl].reset_index(drop=True)
    best_hl_label = f"{best_hl:.0f} stagioni"

INK_PRIMARY, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE_COLOR, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SLOT_BLUE = "#2a78d6"

plt.rcParams.update({
    "font.family": "sans-serif", "text.color": INK_PRIMARY,
    "axes.edgecolor": BASELINE_COLOR, "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})

best_idx = df["log_loss"].idxmin()

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
ax.plot(df["w_logreg"], df["log_loss"], color=SLOT_BLUE, linewidth=2, marker="o", markersize=5, zorder=3)
ax.scatter([df.loc[best_idx, "w_logreg"]], [df.loc[best_idx, "log_loss"]], color="#eb6834", s=90, zorder=4,
           label=f"Peso scelto: w={df.loc[best_idx, 'w_logreg']:.1f}")
ax.set_title(f"Log-loss dell'ensemble al variare del peso — stagioni di selezione (half-life stagioni={best_hl_label})", loc="left", fontsize=11.5, color=INK_PRIMARY, pad=14)
ax.set_xlabel("Peso Logistic Regression (1 - peso XGBoost)")
ax.set_ylabel("Log-loss")
ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color(BASELINE_COLOR)
ax.legend(frameon=False, loc="upper right", fontsize=9)
plt.tight_layout()
fig.savefig(BASE / "notebooks" / "ensemble_weight_search.png", facecolor=SURFACE)
plt.close(fig)
print("Salvato notebooks/ensemble_weight_search.png")
