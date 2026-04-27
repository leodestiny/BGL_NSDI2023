"""Generate an illustrative figure that explains the concepts of
confidence level and confidence interval (English version).

Output: confidence_interval_illustration.png in the same directory.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats


def _configure_fonts() -> None:
    """Use a default Latin/Greek-capable font and fix the unicode minus."""
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False


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
        label="95% confidence level (central region)",
    )

    mask_left = x <= -z_crit
    mask_right = x >= z_crit
    ax.fill_between(x[mask_left], y[mask_left], color="#dc2626", alpha=0.55)
    ax.fill_between(
        x[mask_right],
        y[mask_right],
        color="#dc2626",
        alpha=0.55,
        label="2.5% in each tail (rejection region)",
    )

    ax.axvline(-z_crit, color="#7f1d1d", linestyle="--", linewidth=1.2)
    ax.axvline(z_crit, color="#7f1d1d", linestyle="--", linewidth=1.2)
    ax.axvline(0.0, color="#111827", linestyle=":", linewidth=1.0)

    ax.annotate(
        "\u22121.96 \u03c3",
        xy=(-z_crit, stats.norm.pdf(-z_crit)),
        xytext=(-3.4, 0.18),
        fontsize=11,
        color="#7f1d1d",
        arrowprops=dict(arrowstyle="->", color="#7f1d1d", lw=1.0),
    )
    ax.annotate(
        "+1.96 \u03c3",
        xy=(z_crit, stats.norm.pdf(z_crit)),
        xytext=(2.0, 0.18),
        fontsize=11,
        color="#7f1d1d",
        arrowprops=dict(arrowstyle="->", color="#7f1d1d", lw=1.0),
    )

    ax.text(
        0.0,
        0.16,
        "Central 95%\n(confidence level)",
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
    ax.set_xlabel("Z (standardised deviation from the mean, in \u03c3)", fontsize=11)
    ax.set_ylabel("Probability density", fontsize=11)
    ax.set_title(
        "(a) Confidence level: long-run frequency of statistics falling in the central 95% region",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.4)


def _draw_coverage_panel(ax: plt.Axes) -> None:
    """Middle panel: simulate many CIs and show how often they cover mu."""
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
    )

    coverage = contains.mean() * 100.0
    n_missed = int((~contains).sum())

    green_patch = mpatches.Patch(
        color="#16a34a",
        label=f"Contains \u03bc  ({n_trials - n_missed} / {n_trials})",
    )
    red_patch = mpatches.Patch(
        color="#dc2626",
        label=f"Misses \u03bc  ({n_missed} / {n_trials})",
    )
    mu_line = plt.Line2D(
        [0],
        [0],
        color="#111827",
        linestyle="--",
        label=f"True mean \u03bc = {true_mu}",
    )
    ax.legend(
        handles=[green_patch, red_patch, mu_line],
        loc="upper right",
        fontsize=10,
        framealpha=0.9,
    )

    ax.set_xlim(true_mu - 1.8, true_mu + 1.8)
    ax.set_ylim(-1, n_trials)
    ax.set_xlabel("Value (sample mean \u00b1 margin of error)", fontsize=11)
    ax.set_ylabel("k-th repeated sample", fontsize=11)
    ax.set_title(
        f"(b) Confidence interval: empirical coverage over {n_trials} repeated samples = "
        f"{coverage:.0f}%   (theoretical 95%)",
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
        "Confidence Level",
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
            "\u2022 Usually written as 1 \u2212 \u03b1, e.g. 95% or 99%.\n"
            "\u2022 Meaning: under indefinite repeated sampling,\n"
            "  the long-run frequency at which intervals built\n"
            "  by the same procedure cover the true parameter.\n"
            "\u2022 It describes the reliability of the procedure,\n"
            "  not the probability of any single interval."
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
        "Confidence Interval",
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
            "\u2022 A numerical range [L, U] computed from data.\n"
            "\u2022 Form: point estimate \u00b1 margin of error,\n"
            "  e.g. (sample mean) \u00b1 z \u00b7 (\u03c3 / \u221an).\n"
            "\u2022 Width depends on sample size n, variability \u03c3\n"
            "  and the chosen confidence level.\n"
            "\u2022 At a fixed confidence level, larger n \u2192 narrower."
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
        "Key distinction",
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
            "Correct: at the 95% confidence level, about 95% of intervals from repeated sampling cover the true parameter.\n"
            "Incorrect: \u201cthere is a 95% probability that the true parameter lies in this particular interval.\u201d\n"
            "Reason: in the frequentist view the parameter is a fixed constant; the interval endpoints are the random quantities."
        ),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11,
        color="#1f2937",
        linespacing=1.7,
    )


def main() -> None:
    _configure_fonts()

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
        "Confidence Level and Confidence Interval",
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
        "Illustration at 95% confidence level \u00b7 Simulated from N(\u03bc=10, \u03c3=2), n=30, repeated 100 times",
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
