"""Step 4 — publication figures (dark ECHO theme) into results/plots/."""
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from echo.data import tokenize  # noqa: E402
from echo.models import DISPLAY, KINDS  # noqa: E402

M, P = ROOT / "results" / "metrics", ROOT / "results" / "plots"

# Theme tokens — categorical slots validated for CVD on the dark surface.
SURFACE, PANEL = "#0c0e14", "#12151d"
INK, INK2, MUTED, GRID = "#f0efec", "#c3c2b7", "#8a8a85", "#232733"
COLOR = {"rnn": "#3987e5", "birnn": "#d95926", "lstm": "#199e70", "gru": "#c98500", "tfm": "#d55181"}
MARKER = {"rnn": "o", "birnn": "s", "lstm": "D", "gru": "^", "tfm": "P"}  # secondary encoding
SEQ = LinearSegmentedColormap.from_list("seq", ["#0d366b", "#1c5cab", "#3987e5", "#86b6ef", "#cde2fb"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID, "grid.linewidth": 1,
    "axes.grid": True, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans", "font.size": 10.5, "axes.titlesize": 12.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False, "legend.labelcolor": INK2,
    "lines.linewidth": 2, "lines.solid_capstyle": "round",
})


def title(fig, text, sub):
    fig.text(0.012, 0.975, text, fontsize=16, fontweight="bold", color=INK, va="top")
    fig.text(0.012, 0.925, sub, fontsize=10.5, color=INK2, va="top")


def save(fig, name):
    P.mkdir(parents=True, exist_ok=True)
    fig.savefig(P / name, dpi=170)
    plt.close(fig)
    print("wrote", name)


def dataset_figure():
    stats = json.loads((M / "dataset_stats.json").read_text())
    stories = json.loads((ROOT / "data" / "raw" / "stories.json").read_text())["train"]
    toks = [tokenize(s) for s in stories]
    lengths = [len(t) for t in toks]
    counts = Counter(t for s in toks for t in s)
    freqs = np.array(sorted(counts.values(), reverse=True))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1.1, 1, 1.2]})
    fig.subplots_adjust(top=0.78, bottom=0.14, left=0.05, right=0.98, wspace=0.28)
    title(fig, "TinyStories — what the models read",
          f"{stats['stories']['train']:,} training stories · {stats['tokens']['train']:,} words · "
          f"{stats['vocab_size']:,}-word vocabulary covers {stats['train_coverage']:.2%} of text")

    ax = axes[0]
    ax.hist(lengths, bins=60, color=COLOR["rnn"], edgecolor=SURFACE, linewidth=1)
    ax.axvline(np.mean(lengths), color=INK2, lw=1)
    ax.text(np.mean(lengths) + 12, ax.get_ylim()[1] * 0.9, f"mean {np.mean(lengths):.0f}", color=INK2)
    ax.set(title="Story length", xlabel="words per story", ylabel="stories", xlim=(0, 600))

    ax = axes[1]
    ranks = np.arange(1, len(freqs) + 1)
    ax.loglog(ranks, freqs, color=COLOR["lstm"])
    ax.axvline(stats["vocab_size"], color=INK2, lw=1)
    ax.text(stats["vocab_size"] * 0.55, freqs[0] * 0.3, "vocab\ncut-off", color=INK2, ha="right")
    ax.set(title="Zipf's law: few words do most work", xlabel="word rank", ylabel="frequency")

    ax = axes[2]
    top = stats["top_words"][:15][::-1]
    ax.barh([w for w, _ in top], [c for _, c in top], color=COLOR["gru"], height=0.62)
    ax.set(title="Most frequent tokens", xlabel="count")
    ax.grid(axis="y", visible=False)
    save(fig, "01_dataset.png")


def learning_curves():
    runs = pd.read_csv(M / "runs.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.subplots_adjust(top=0.8, bottom=0.12, left=0.06, right=0.86, wspace=0.22)
    title(fig, "Learning curves", "Mean over 3 seeds, band = min–max · each epoch = 2M freshly sampled windows")
    for kind in KINDS:
        hs = [pd.DataFrame(json.loads((ROOT / "results" / "runs" / r / "history.json").read_text()))
              for r in runs[runs.kind == kind].run]
        for ax, col in zip(axes, ["train_perplexity", "val_perplexity"]):
            L = min(len(h) for h in hs)
            arr = np.stack([h[col].values[:L] for h in hs])
            x = np.arange(1, L + 1)
            ax.fill_between(x, arr.min(0), arr.max(0), color=COLOR[kind], alpha=0.12, lw=0)
            ax.plot(x, arr.mean(0), color=COLOR[kind], marker=MARKER[kind], ms=6,
                    mec=SURFACE, mew=2, label=DISPLAY[kind])
            if col == "val_perplexity":
                ax.annotate(f"{DISPLAY[kind]}  {arr.mean(0)[-1]:.1f}", (x[-1], arr.mean(0)[-1]),
                            xytext=(10, 0), textcoords="offset points", va="center", color=INK2,
                            annotation_clip=False)
    axes[0].set(title="Training perplexity", xlabel="epoch", ylabel="perplexity (lower is better)", yscale="log")
    axes[1].set(title="Validation perplexity", xlabel="epoch", yscale="log")
    axes[0].legend(loc="upper right")
    save(fig, "02_learning_curves.png")


def test_metrics():
    s = pd.read_csv(M / "summary_by_model.csv", index_col=0).loc[list(KINDS)]
    panels = [("test_ppl", "Test perplexity", "lower is better", "{:.1f}"),
              ("test_top1", "Top-1 accuracy", "higher is better", "{:.1%}"),
              ("test_top5", "Top-5 accuracy", "higher is better", "{:.1%}"),
              ("time_per_epoch_s", "Seconds per epoch", "training cost", "{:.0f}s"),
              ("ms_per_token", "ms per generated word", "inference latency (M4 GPU)", "{:.2f}")]
    fig, axes = plt.subplots(1, len(panels), figsize=(17, 4.4))
    fig.subplots_adjust(top=0.74, bottom=0.08, left=0.07, right=0.99, wspace=0.55)
    title(fig, "Head-to-head on the held-out test set", "Mean ± std over 3 seeds; best checkpoint by validation loss")
    y = np.arange(len(KINDS))[::-1]
    for ax, (col, t, sub, fmt) in zip(axes, panels):
        vals = s[f"{col}_mean"].values
        err = s.get(f"{col}_std", pd.Series(0, index=s.index)).fillna(0).values
        ax.barh(y, vals, xerr=err, color=[COLOR[k] for k in KINDS], height=0.55,
                error_kw=dict(ecolor=INK2, lw=1, capsize=3))
        for yi, v in zip(y, vals):
            ax.text(v + vals.max() * 0.04 + err.max(), yi, fmt.format(v), va="center", color=INK, fontsize=10)
        ax.set_yticks(y, [DISPLAY[k] for k in KINDS] if ax is axes[0] else [""] * len(KINDS))
        ax.set_xlim(0, vals.max() * 1.45 + err.max())
        ax.set_title(f"{t}\n", loc="left")
        ax.text(0, 1.04, sub, transform=ax.transAxes, color=MUTED, fontsize=9)
        ax.grid(axis="y", visible=False)
    save(fig, "03_test_metrics.png")


def temperature_heatmaps():
    tq = pd.read_csv(M / "text_quality.csv")
    human = tq[tq.kind == "human"].iloc[0]
    tq = tq[tq.kind != "human"]
    temps = sorted(tq.temperature.unique())
    panels = [("gpt2_ppl", "Coherence: GPT-2 perplexity", "lower = more natural", True, "{:.0f}"),
              ("consistency", "Contextual consistency", "seed↔text similarity, higher = on topic", False, "{:.2f}"),
              ("distinct_2", "Diversity: distinct-2", "higher = more varied", False, "{:.2f}"),
              ("repetition", "Repetition: repeated 3-grams", "lower = fewer loops", True, "{:.2f}")]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6))
    fig.subplots_adjust(top=0.72, bottom=0.12, left=0.07, right=0.99, wspace=0.3)
    title(fig, "How temperature reshapes the text",
          "5 seeds × 5 samples per cell · brighter = better on that metric · "
          "T=0 is greedy decoding")
    for ax, (col, t, sub, lower_better, fmt) in zip(axes, panels):
        grid = tq.pivot(index="kind", columns="temperature", values=col).loc[list(KINDS), temps]
        z = np.log(grid.values) if col == "gpt2_ppl" else grid.values
        z = -z if lower_better else z
        ax.imshow(z, cmap=SEQ, aspect="auto")
        for i in range(len(KINDS)):
            for j in range(len(temps)):
                v = grid.values[i, j]
                bright = (z[i, j] - z.min()) / (np.ptp(z) + 1e-9) > 0.6
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=9.5,
                        color=SURFACE if bright else INK)
        ax.set_xticks(range(len(temps)), ["greedy" if x == 0 else f"{x:g}" for x in temps])
        ax.set_yticks(range(len(KINDS)), [DISPLAY[k] for k in KINDS] if ax is axes[0] else [""] * 4)
        ax.set_title(f"{t}\n", loc="left")
        ax.text(0, 1.04, f"{sub} · human: {fmt.format(human[col])}", transform=ax.transAxes, color=MUTED, fontsize=9)
        ax.grid(False)
        ax.set_xlabel("temperature")
    save(fig, "04_temperature_heatmaps.png")


def temperature_frontier():
    tq = pd.read_csv(M / "text_quality.csv")
    human = tq[tq.kind == "human"].iloc[0]
    fig, ax = plt.subplots(figsize=(11, 6.6))
    fig.subplots_adjust(top=0.84, bottom=0.1, left=0.08, right=0.8)
    title(fig, "The creativity–coherence trade-off",
          "Each line walks one model from greedy (left) to T = 1.3 (right). Closer to the human ★ is better.")
    for kind in KINDS:
        g = tq[tq.kind == kind].sort_values("temperature")
        ax.plot(g.distinct_2, g.gpt2_ppl, color=COLOR[kind], marker=MARKER[kind], ms=8,
                mec=SURFACE, mew=2, label=DISPLAY[kind])
        for _, r in g.iterrows():
            if r.temperature in (0.0, 1.3):
                ax.annotate("greedy" if r.temperature == 0 else "T 1.3", (r.distinct_2, r.gpt2_ppl),
                            xytext=(6, -12), textcoords="offset points", color=MUTED, fontsize=8.5)
        last = g.iloc[-1]
        ax.annotate(DISPLAY[kind], (last.distinct_2, last.gpt2_ppl), xytext=(10, 4),
                    textcoords="offset points", color=INK2, annotation_clip=False)
    ax.scatter([human.distinct_2], [human.gpt2_ppl], marker="*", s=380, color=INK, edgecolor=SURFACE, lw=2, zorder=5)
    ax.annotate("human text", (human.distinct_2, human.gpt2_ppl), xytext=(12, -4), textcoords="offset points",
                color=INK)
    ax.set(yscale="log", xlabel="diversity  (distinct-2 → more varied)",
           ylabel="incoherence  (GPT-2 perplexity, log → less natural)")
    ax.legend(loc="upper left")
    save(fig, "05_temperature_frontier.png")


def matched_figure():
    mp = M / "matched.csv"
    if not mp.exists() or mp.stat().st_size < 10:
        return
    m = pd.read_csv(mp).set_index("kind").loc[list(KINDS)]
    runs = pd.read_csv(M / "runs.csv")
    base = runs[runs.seed == m.seed.iloc[0]].set_index("kind").loc[list(KINDS)]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    fig.subplots_adjust(top=0.78, bottom=0.12, left=0.08, right=0.97)
    title(fig, "Fair fight: equal parameter budget",
          f"Hidden size re-tuned so every model has ≈{m.params_total.iloc[0] / 1e6:.1f}M parameters (same as the LSTM)")
    x = np.arange(len(KINDS))
    w = 0.34
    ax.bar(x - w / 2 - 0.01, base.test_ppl, w, color=[COLOR[k] for k in KINDS], alpha=0.45, label="default size")
    ax.bar(x + w / 2 + 0.01, m.test_ppl, w, color=[COLOR[k] for k in KINDS], label="parameter-matched")
    for xi, (b, v, h) in enumerate(zip(base.test_ppl, m.test_ppl, m.hidden_dim)):
        ax.text(xi - w / 2, b + 0.4, f"{b:.1f}", ha="center", color=INK2, fontsize=9.5)
        ax.text(xi + w / 2, v + 0.4, f"{v:.1f}\nh={h}", ha="center", color=INK, fontsize=9.5)
    ax.set_xticks(x, [DISPLAY[k] for k in KINDS])
    ax.set(ylabel="test perplexity (lower is better)", ylim=(0, max(base.test_ppl.max(), m.test_ppl.max()) * 1.25))
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    save(fig, "06_parameter_matched.png")


if __name__ == "__main__":
    dataset_figure()
    for fn in (learning_curves, test_metrics, temperature_heatmaps, temperature_frontier, matched_figure):
        try:
            fn()
        except FileNotFoundError as e:
            print(f"skip {fn.__name__}: {e.filename} missing")
