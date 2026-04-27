"""Generate an illustrative figure that explains the concepts of
confidence level and confidence interval.

Output: confidence_interval_illustration.png in the same directory.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager as fm
import numpy as np
from scipy import stats


def _register_chinese_font() -> str:
    """Register a CJK-capable TTF/TTC font and return its family name."""
    candidate_paths = [
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            family = fm.FontProperties(fname=path).get_name()
            plt.rcParams["font.family"] = family
            plt.rcParams["axes.unicode_minus"] = False
            return family
    plt.rcParams["axes.unicode_minus"] = False
    return plt.rcParams["font.family"]


def _draw_normal_panel(ax: plt.Axes) -> None:
    """Top panel: standard normal PDF with the central 95% region shaded."""
    z_crit = stats.norm.ppf(0.975)
    x = np.linspace(-4.0, 4.0, 1000)
    y = stats.norm.pdf(x)

    ax.plot(x, y, color="#1f3b73", linewidth=2.2)

    mask_center = (x >= -z_crit) & (x <= z_crit)
    ax.fill_between(
        x[mask_center],
        y[mask_center],
        color="#3b82f6",
        alpha=0.35,
        label="95% 置信水平 (覆盖区域)",
    )

    mask_left = x <= -z_crit
    mask_right = x >= z_crit
    ax.fill_between(x[mask_left], y[mask_left], color="#dc2626", alpha=0.55)
    ax.fill_between(
        x[mask_right],
        y[mask_right],
        color="#dc2626",
        alpha=0.55,
        label="尾部各 2.5% (拒绝域)",
    )

    ax.axvline(-z_crit, color="#7f1d1d", linestyle="--", linewidth=1.2)
    ax.axvline(z_crit, color="#7f1d1d", linestyle="--", linewidth=1.2)
    ax.axvline(0.0, color="#111827", linestyle=":", linewidth=1.0)

    ax.annotate(
        f"−1.96 σ",
        xy=(-z_crit, stats.norm.pdf(-z_crit)),
        xytext=(-3.4, 0.18),
        fontsize=11,
        color="#7f1d1d",
        arrowprops=dict(arrowstyle="->", color="#7f1d1d", lw=1.0),
    )
    ax.annotate(
        f"+1.96 σ",
        xy=(z_crit, stats.norm.pdf(z_crit)),
        xytext=(2.0, 0.18),
        fontsize=11,
        color="#7f1d1d",
        arrowprops=dict(arrowstyle="->", color="#7f1d1d", lw=1.0),
    )

    ax.text(
        0.0,
        0.16,
        "中央 95%\n(置信水平)",
        ha="center",
        va="center",
        fontsize=13,
        color="#0b3d91",
        fontweight="bold",
    )
    ax.text(
        -3.2,
        0.025,
        "2.5%",
        ha="center",
        fontsize=10,
        color="#7f1d1d",
    )
    ax.text(
        3.2,
        0.025,
        "2.5%",
        ha="center",
        fontsize=10,
        color="#7f1d1d",
    )

    ax.set_xlim(-4.0, 4.0)
    ax.set_ylim(0.0, 0.45)
    ax.set_xlabel("Z (标准化偏离均值的 σ 数)", fontsize=11)
    ax.set_ylabel("概率密度", fontsize=11)
    ax.set_title(
        "图 (a) 置信水平：抽样统计量落在中央 95% 区域的长期频率",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.4)


def _draw_coverage_panel(ax: plt.Axes) -> None:
    """Middle panel: simulate many CIs and show how often they cover μ."""
    rng = np.random.default_rng(seed=20260427)
    true_mu = 10.0
    true_sigma = 2.0
    n_per_sample = 30
    n_trials = 100
    z_crit = stats.norm.ppf(0.975)
    half_width = z_crit * true_sigma / np.sqrt(n_per_sample)

    samples = rng.normal(true_mu, true_sigma, size=(n_trials, n_per_sample))
    sample_means = samples.mean(axis=1)
    lowers = sample_means - half_width
    uppers = sample_means + half_width
    contains = (lowers <= true_mu) & (uppers >= true_mu)

    y_positions = np.arange(n_trials)
    for y, lo, hi, ok in zip(y_positions, lowers, uppers, contains):
        color = "#16a34a" if ok else "#dc2626"
        ax.hlines(y, lo, hi, color=color, linewidth=1.4, alpha=0.85)
        ax.plot(
            (lo + hi) / 2.0,
            y,
            marker="o",
            color=color,
            markersize=2.6,
        )

    ax.axvline(
        true_mu,
        color="#111827",
        linestyle="--",
        linewidth=1.4,
        label=f"真实均值 μ = {true_mu}",
    )

    coverage = contains.mean() * 100.0
    n_missed = int((~contains).sum())

    green_patch = mpatches.Patch(
        color="#16a34a", label=f"包含真值 μ ({n_trials - n_missed} / {n_trials})"
    )
    red_patch = mpatches.Patch(
        color="#dc2626", label=f"未包含真值 ({n_missed} / {n_trials})"
    )
    mu_line = plt.Line2D(
        [0], [0], color="#111827", linestyle="--", label=f"真实均值 μ = {true_mu}"
    )
    ax.legend(
        handles=[green_patch, red_patch, mu_line],
        loc="upper right",
        fontsize=10,
        framealpha=0.9,
    )

    ax.set_xlim(true_mu - 1.8, true_mu + 1.8)
    ax.set_ylim(-1, n_trials)
    ax.set_xlabel("数值轴 (样本均值 ± 误差)", fontsize=11)
    ax.set_ylabel("第 k 次抽样实验", fontsize=11)
    ax.set_title(
        f"图 (b) 置信区间：100 次重复抽样中实际覆盖率 = {coverage:.0f}%"
        f"   (理论值 95%)",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax.grid(True, axis="x", linestyle=":", alpha=0.4)


def _draw_explanation_panel(ax: plt.Axes) -> None:
    """Bottom panel: textual explanation of the two concepts."""
    ax.axis("off")

    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.01, 0.46),
            0.46,
            0.52,
            boxstyle="round,pad=0.02",
            linewidth=1.2,
            edgecolor="#1f3b73",
            facecolor="#eff6ff",
            transform=ax.transAxes,
        )
    )
    ax.text(
        0.24,
        0.94,
        "置信水平 (Confidence Level)",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=13,
        fontweight="bold",
        color="#0b3d91",
    )
    ax.text(
        0.03,
        0.86,
        (
            "• 通常记为 1 − α，例如 95%、99%。\n"
            "• 含义：在长期重复抽样下，由同一方法\n"
            "  构造出的置信区间能够覆盖真实参数\n"
            "  的频率。\n"
            "• 它描述的是“方法的可靠性”，而非\n"
            "  某一次具体区间是否包含真值的概率。"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        color="#1f2937",
        linespacing=1.6,
    )

    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.53, 0.46),
            0.46,
            0.52,
            boxstyle="round,pad=0.02",
            linewidth=1.2,
            edgecolor="#166534",
            facecolor="#ecfdf5",
            transform=ax.transAxes,
        )
    )
    ax.text(
        0.76,
        0.94,
        "置信区间 (Confidence Interval)",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=13,
        fontweight="bold",
        color="#14532d",
    )
    ax.text(
        0.55,
        0.86,
        (
            "• 由样本计算出的一个数值区间 [L, U]。\n"
            "• 形式：点估计 ± 误差边界，\n"
            "  例如 (样本均值) ± z · (σ / √n)。\n"
            "• 区间宽度受样本量 n、总体波动 σ\n"
            "  及置信水平共同决定。\n"
            "• 同一置信水平下，n 越大区间越窄。"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        color="#1f2937",
        linespacing=1.6,
    )

    ax.text(
        0.5,
        0.34,
        "关键区分",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=13,
        fontweight="bold",
        color="#7c2d12",
    )
    ax.text(
        0.5,
        0.25,
        (
            "正确表述：在 95% 置信水平下，长期重复抽样得到的区间约有 95% 会覆盖真实参数。\n"
            "错误表述：“真实参数有 95% 的概率落在本次得到的区间里。”\n"
            "原因：在频率学派框架下，真实参数是固定常数，区间端点才是随机变量。"
        ),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11,
        color="#1f2937",
        linespacing=1.7,
    )


def main() -> None:
    _register_chinese_font()

    fig = plt.figure(figsize=(11.0, 14.5))
    gs = fig.add_gridspec(
        nrows=3,
        ncols=1,
        height_ratios=[1.0, 1.25, 1.20],
        hspace=0.45,
    )
    ax_top = fig.add_subplot(gs[0, 0])
    ax_mid = fig.add_subplot(gs[1, 0])
    ax_bot = fig.add_subplot(gs[2, 0])

    fig.suptitle(
        "置信度 (Confidence Level) 与 置信区间 (Confidence Interval)",
        fontsize=17,
        fontweight="bold",
        y=0.985,
    )

    _draw_normal_panel(ax_top)
    _draw_coverage_panel(ax_mid)
    _draw_explanation_panel(ax_bot)

    fig.text(
        0.5,
        0.012,
        "示意图：以 95% 置信水平为例 · 数值仿真自 N(μ=10, σ=2)，n=30，重复 100 次",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#4b5563",
        style="italic",
    )

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "confidence_interval_illustration.png",
    )
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
