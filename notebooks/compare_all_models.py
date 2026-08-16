"""Confronto finale: Logistic Regression (Fase 2) vs XGBoost/XGBoost calibrato (Fase 3)
vs quote di mercato vs baseline banale, sulle stesse 15 stagioni di backtest."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
fase2 = pd.read_csv(BASE / "notebooks" / "backtest_results.csv")
fase3 = pd.read_csv(BASE / "notebooks" / "backtest_results_fase3.csv")

# Dalla Fase 2 teniamo solo il modello Logistic Regression (mercato/baseline li
# ricalcoliamo identici in Fase 3, li usiamo da li' per evitare duplicati con arrotondamenti diversi)
logreg = fase2[fase2["modello"] == "LogisticRegression"]
combined = pd.concat([logreg, fase3], ignore_index=True)

summary = combined.groupby("modello")[["accuracy", "log_loss", "brier"]].mean().round(4).sort_values("log_loss")
print("=== Confronto finale (media sulle 15 stagioni di backtest) ===")
print(summary.to_string())
summary.to_csv(BASE / "notebooks" / "final_comparison_summary.csv")

# --- Grafico ---
INK_PRIMARY, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE_COLOR, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
COLORS = {
    "LogisticRegression": "#2a78d6", "XGBoost": "#eb6834", "XGBoost_calibrato": "#1baf7a",
    "Quote_di_mercato": "#eda100", "Baseline_vince_sempre_casa": "#4a3aa7",
}
LABELS = {
    "LogisticRegression": "Logistic Regression", "XGBoost": "XGBoost",
    "XGBoost_calibrato": "XGBoost calibrato", "Quote_di_mercato": "Quote di mercato",
    "Baseline_vince_sempre_casa": "Baseline: vince sempre casa",
}

plt.rcParams.update({
    "font.family": "sans-serif", "text.color": INK_PRIMARY,
    "axes.edgecolor": BASELINE_COLOR, "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})

order = summary.index.tolist()
fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
bars = ax.barh([LABELS[m] for m in order], summary.loc[order, "log_loss"], color=[COLORS[m] for m in order], height=0.55, zorder=3)
ax.invert_yaxis()
ax.set_title("Log-loss media sul backtest (15 stagioni) — piu' basso = meglio", loc="left", fontsize=11.5, color=INK_PRIMARY, pad=14)
ax.set_xlabel("Log-loss")
ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color(BASELINE_COLOR)
for bar, val in zip(bars, summary.loc[order, "log_loss"]):
    ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=9.5, color=INK_PRIMARY)
plt.tight_layout()
fig.savefig(BASE / "notebooks" / "final_comparison.png", facecolor=SURFACE)
plt.close(fig)
print("\nSalvato notebooks/final_comparison.png")
