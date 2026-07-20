"""
Generate benchmark graphs from LEMA notebook results.
Run: pip install matplotlib && python tools/generate_graphs.py
"""
import os
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not installed. Run: pip install matplotlib")
    raise

os.makedirs("docs/assets", exist_ok=True)
plt.style.use("ggplot")
COLORS = {"PEFT": "#e74c3c", "LEMA": "#2ecc71", "C++": "#3498db", "Python": "#f39c12", "SGD": "#9b59b6"}


def vram_comparison():
    models = ["TinyLlama\n1.1B", "Llama-2\n7B"]
    peft = [5.0, 0]
    lema = [1.4, 3.2]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(models))
    ax.bar(x - 0.15, peft, 0.3, label="PEFT", color=COLORS["PEFT"])
    ax.bar(x + 0.15, lema, 0.3, label="LEMA", color=COLORS["LEMA"])

    for i, v in enumerate(lema):
        ax.text(i + 0.15, v + 0.1, f"{v:.1f} GB", ha="center", va="bottom", fontweight="bold")
    ax.text(0 - 0.15, 5.2, "5.0 GB", ha="center", va="bottom", fontweight="bold")
    ax.text(1 - 0.15, 0.3, "OOM", ha="center", va="bottom", color="red", fontweight="bold")

    ax.set_ylabel("Peak VRAM (GB)")
    ax.set_title("VRAM Usage: PEFT vs LEMA (bs=1, seq=512)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    plt.tight_layout()
    plt.savefig("docs/assets/vram_benchmark.png", dpi=150)
    print("Generated docs/assets/vram_benchmark.png")
    plt.close()


def speed_comparison():
    models = ["TinyLlama\n1.1B", "Llama-2\n7B"]
    peft = [310, 0]
    lema = [2297, 3719]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(models))
    ax.bar(x - 0.15, peft, 0.3, label="PEFT", color=COLORS["PEFT"])
    ax.bar(x + 0.15, lema, 0.3, label="LEMA", color=COLORS["LEMA"])

    for i, (p, l) in enumerate(zip(peft, lema)):
        if p > 0:
            ax.text(i - 0.15, p + 30, f"{p}ms", ha="center", va="bottom")
        else:
            ax.text(i - 0.15, 100, "OOM", ha="center", va="bottom", color="red", fontweight="bold")
        ax.text(i + 0.15, l + 30, f"{l}ms", ha="center", va="bottom")

    ax.set_ylabel("Time per Step (ms)")
    ax.set_title("Training Speed: PEFT vs LEMA (bs=1, seq=512)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    plt.tight_layout()
    plt.savefig("docs/assets/speed_benchmark.png", dpi=150)
    print("Generated docs/assets/speed_benchmark.png")
    plt.close()


def longseq_vram():
    seqs = [512, 1024, 2048]
    lema = [3.2, 4.0, 6.3]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(seqs, lema, "o-", color=COLORS["LEMA"], linewidth=2, label="LEMA")
    ax.axhline(y=14.6, color="red", linestyle="--", alpha=0.5, label="GPU VRAM (14.6 GB)")

    for s, v in zip(seqs, lema):
        ax.text(s, v + 0.2, f"{v:.1f} GB", ha="center", fontweight="bold")

    ax.text(600, 4.8, "PEFT OOM at seq=512", color="red", fontsize=10, fontweight="bold")

    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Peak VRAM (GB)")
    ax.set_title("Long Sequence Headroom — Llama-2 7B")
    ax.set_xlim(400, 2200)
    ax.legend()
    plt.tight_layout()
    plt.savefig("docs/assets/longseq_vram.png", dpi=150)
    print("Generated docs/assets/longseq_vram.png")
    plt.close()


def cpp_benchmark():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: Pack (microseconds)
    labels = ["Pack\n(memcpy)"]
    python_t = [1.33]
    cpp_t = [0.64]
    x = np.arange(len(labels))
    ax1.bar(x - 0.15, python_t, 0.3, label="Python", color=COLORS["Python"])
    ax1.bar(x + 0.15, cpp_t, 0.3, label="C++", color=COLORS["C++"])
    for i, (p, c) in enumerate(zip(python_t, cpp_t)):
        ax1.text(i - 0.15, p + 0.05, f"{p:.2f}ms", ha="center", va="bottom", fontsize=9)
        ax1.text(i + 0.15, c + 0.05, f"{c:.2f}ms", ha="center", va="bottom", fontsize=9)
    pct = (python_t[0] - cpp_t[0]) / python_t[0] * 100
    ax1.text(0, max(python_t[0], cpp_t[0]) + 0.2, f"{pct:.0f}% faster", ha="center", fontweight="bold", fontsize=10)
    ax1.set_ylabel("Time (ms)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)

    # Right: Transfer + End-to-End (milliseconds)
    labels2 = ["Transfer\n(2 GB)", "End-to-End"]
    python_t2 = [180.0, 227.1]
    cpp_t2 = [179.5, 203.0]
    x2 = np.arange(len(labels2))
    ax2.bar(x2 - 0.15, python_t2, 0.3, label="Python", color=COLORS["Python"])
    ax2.bar(x2 + 0.15, cpp_t2, 0.3, label="C++", color=COLORS["C++"])
    for i, (p, c) in enumerate(zip(python_t2, cpp_t2)):
        ax2.text(i - 0.15, p + 2, f"{p:.0f}ms", ha="center", va="bottom", fontsize=9)
        ax2.text(i + 0.15, c + 2, f"{c:.0f}ms", ha="center", va="bottom", fontsize=9)
        pct2 = (p - c) / p * 100
        ax2.text(i, max(p, c) + 8, f"{pct2:.0f}%", ha="center", fontweight="bold", fontsize=10)
    ax2.set_ylabel("Time (ms)")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels2)

    ax1.legend()
    fig.suptitle("C++ vs Python Backend (Tesla T4)", fontsize=13)
    plt.tight_layout()
    plt.savefig("docs/assets/cpp_benchmark.png", dpi=150)
    print("Generated docs/assets/cpp_benchmark.png")
    plt.close()


def vram_stability():
    steps = [1, 10, 20, 30, 40, 50, 60]
    vram = [2.87, 3.12, 3.12, 3.12, 3.12, 3.12, 3.12]
    loss = [10.77, 9.52, 7.59, 5.68, 4.95, 4.71, 4.57]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    ax1.plot(steps, vram, "s-", color=COLORS["LEMA"], linewidth=2, label="VRAM")
    ax2.plot(steps, loss, "o-", color=COLORS["SGD"], linewidth=2, label="Loss")

    for s, v in zip(steps, vram):
        ax1.text(s, v - 0.01, f"{v:.2f}", ha="center", fontsize=8, color="black")

    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("VRAM (GB)", color=COLORS["LEMA"])
    ax2.set_ylabel("Loss", color=COLORS["SGD"])
    ax1.set_title("VRAM Stability — Llama-2 7B (60 steps, SGD)")
    ax1.tick_params(axis="y", labelcolor=COLORS["LEMA"])
    ax2.tick_params(axis="y", labelcolor=COLORS["SGD"])
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=2)

    plt.tight_layout()
    plt.savefig("docs/assets/vram_stability.png", dpi=150)
    print("Generated docs/assets/vram_stability.png")
    plt.close()


def scaling_heatmap():
    batch = [1, 2, 4, 8]
    seq = [128, 256, 512]
    # LEMA VRAM for TinyLlama 1.1B
    vram = [[1.2, 1.3, 1.4],
            [1.3, 1.4, 1.7],
            [1.3, 1.6, 2.3],
            [1.6, 2.1, 3.5]]

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(vram, cmap="YlGn", aspect="auto")

    ax.set_xticks(range(len(seq)))
    ax.set_yticks(range(len(batch)))
    ax.set_xticklabels(seq)
    ax.set_yticklabels(batch)
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Batch Size")
    ax.set_title("LEMA VRAM (GB) — TinyLlama 1.1B")

    for i in range(len(batch)):
        for j in range(len(seq)):
            ax.text(j, i, f"{vram[i][j]:.1f}", ha="center", va="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig("docs/assets/scaling_heatmap.png", dpi=150)
    print("Generated docs/assets/scaling_heatmap.png")
    plt.close()


if __name__ == "__main__":
    vram_comparison()
    speed_comparison()
    longseq_vram()
    cpp_benchmark()
    vram_stability()
    scaling_heatmap()
    print("\nAll graphs generated in docs/assets/")
