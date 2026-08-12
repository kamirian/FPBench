"""
fp_cdf_density_plots.py
=========================
Reusable plotting functions for FP vs DFT force analysis.

Public API
----------
build_cdf_from_all_results(all_results, ...)       – build CDFs from raw results
build_cdf_by_regime(all_results, regimes, ...)     – build CDFs split by F_DFT regime

plot_cdf_with_inset_on_ax(fig, subspec, ...)       – CDF line plot + inset zoom
plot_rank_curves_from_cdfs_on_ax(fig, subspec, ...)– rank-vs-threshold curves

plot_density_with_marginals(fig, subspec, x, y, .) – 2-D density + marginals
panel_Fdft_vs_abs_dF(fig, subspec, all_results, .) – |Δ|F|| vs F_DFT panel
panel_Fdft_vs_dtheta(fig, subspec, all_results, .) – Δθ vs F_DFT panel
panel_abs_dF_vs_dtheta_cond_on_Fdft(...)           – |Δ|F|| vs Δθ panel

plot_cdf_regimes(all_results, regimes, ...)        – convenience: one figure per regime
add_panel_label / add_panel_label_from_top         – absolute-pixel panel labels
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.transforms import ScaledTranslation
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# ============================================================
# Style presets (pass as **style to most functions)
# ============================================================
PAPER_STYLE_DENSITY = dict(
    label_fontsize=9,
    title_fontsize=9,
    tick_major=8,
    tick_minor=8,
    cbar_fontsize=10,
    hist_label_fontsize=9,
    title_pad=6,
)

PAPER_STYLE_CDF = dict(
    label_fontsize=9,
    title_fontsize=9,
    tick_major=8,
    tick_minor=8,
    title_pad=6,
    legend_fontsize=9,
    inset_tick_fontsize=8,
)

# Default force regimes
DEFAULT_REGIMES = {
    "near-eq":      (0.0,  0.1),   # |F_DFT| < 0.1 eV/Å
    "intermediate": (0.1,  1.0),   # 0.1 ≤ |F_DFT| < 1 eV/Å
    "large":        (1.0, None),   # |F_DFT| ≥ 1 eV/Å
}

# ============================================================
# Internal helpers
# ============================================================

def _as_finite(*arrs):
    mask = None
    for a in arrs:
        a = np.asarray(a, dtype=float)
        m = np.isfinite(a)
        mask = m if mask is None else (mask & m)
    return mask


def _prep_xy(x_raw, y_raw, x_abs=False, y_abs=False,
             x_min_eps=1e-12, y_min_eps=1e-12, mask=None):
    x = np.asarray(x_raw, dtype=float)
    y = np.asarray(y_raw, dtype=float)
    if x_abs:
        x = np.abs(x)
    if y_abs:
        y = np.abs(y)
    mfin = _as_finite(x, y)
    if mask is not None:
        mfin = mfin & mask
    x, y = x[mfin], y[mfin]
    x = x + x_min_eps
    y = y + y_min_eps
    return x, y


def _make_inner_axes(fig, subspec,
                     width_ratios=(1, 1, 1, 1, 0.11),
                     height_ratios=(1, 1, 1, 1, 1),
                     wspace=0.05, hspace=0.05):
    gs = GridSpecFromSubplotSpec(
        len(height_ratios), len(width_ratios),
        subplot_spec=subspec,
        width_ratios=list(width_ratios),
        height_ratios=list(height_ratios),
        wspace=wspace, hspace=hspace,
    )
    ax_main  = fig.add_subplot(gs[1:, :3])
    ax_xhist = fig.add_subplot(gs[0,  :3], sharex=ax_main)
    ax_yhist = fig.add_subplot(gs[1:,  3], sharey=ax_main)
    ax_cbar  = fig.add_subplot(gs[1:,  4])
    return ax_main, ax_xhist, ax_yhist, ax_cbar


def _style_ticks(ax_list, major=12, minor=10):
    for ax in ax_list:
        ax.tick_params(axis="both", which="major", labelsize=major)
        ax.tick_params(axis="both", which="minor", labelsize=minor)


def _rotate_right_hist_xticks(ax_yhist, rotation=45):
    for label in ax_yhist.get_xticklabels():
        label.set_rotation(rotation)
        label.set_ha("right")


def _shift_xticklabels(ax, dx_pts=3):
    trans = ScaledTranslation(dx_pts / 72, 0, ax.figure.dpi_scale_trans)
    for lbl in ax.get_xticklabels():
        lbl.set_transform(lbl.get_transform() + trans)


def _shift_yticklabels(ax, dy_pts=0, dx_pts=0):
    trans = ScaledTranslation(dx_pts / 72, dy_pts / 72, ax.figure.dpi_scale_trans)
    for lbl in ax.get_yticklabels():
        lbl.set_transform(lbl.get_transform() + trans)

# ============================================================
# Panel labels
# ============================================================

def add_panel_label(fig, label, x_px, y_px, fontsize=11, weight="bold"):
    """Place text at absolute pixel coords (0,0 = bottom-left)."""
    fig.text(x_px, y_px, label,
             transform=fig.dpi_scale_trans,
             ha="left", va="bottom",
             fontsize=fontsize, fontweight=weight)


def add_panel_label_from_top(fig, label, x_px, y_from_top_px,
                              fontsize=11, weight="bold"):
    w_px, h_px = fig.canvas.get_width_height()
    add_panel_label(fig, label, x_px, h_px - y_from_top_px,
                    fontsize=fontsize, weight=weight)

# ============================================================
# CDF helpers
# ============================================================

def compute_cdf(data):
    """Return (sorted_x, cdf) for a 1-D array, ignoring non-finite values."""
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return np.array([]), np.array([])
    x = np.sort(data)
    cdf = np.arange(1, x.size + 1) / x.size
    return x, cdf


def _regime_mask(F_dft_abs, lo, hi):
    """Boolean mask for lo <= |F_DFT| < hi (hi=None means no upper bound)."""
    m = F_dft_abs >= lo
    if hi is not None:
        m &= F_dft_abs < hi
    return m


def build_cdf_from_all_results(all_results, f_dft_thr=0.1, use_abs_f_dft=True):
    """
    Build CDFs of |Δ|F|| and Δθ for every model, filtered by F_DFT threshold.

    Parameters
    ----------
    all_results : dict[str, dict]  keys: "F_dft", "deltaF", "deltaTheta"
    f_dft_thr   : lower bound on |F_DFT| (or F_DFT if use_abs_f_dft=False)
    use_abs_f_dft : apply threshold to |F_DFT| instead of F_DFT

    Returns
    -------
    dict[model] -> {"dF": (x, cdf), "theta": (x, cdf)}
    """
    out = {}
    for model, d in all_results.items():
        F_dft = np.asarray(d.get("F_dft", d.get("all_F_dft_mags", [])), dtype=float)
        dF    = np.asarray(d.get("deltaF", d.get("all_deltaF",    [])), dtype=float)
        dTh   = np.asarray(d.get("deltaTheta", d.get("all_deltaTheta", [])), dtype=float)

        n = min(F_dft.size, dF.size, dTh.size)
        F_dft, dF, dTh = F_dft[:n], dF[:n], dTh[:n]

        mask = np.isfinite(F_dft) & np.isfinite(dF) & np.isfinite(dTh)
        mask &= (np.abs(F_dft) > f_dft_thr) if use_abs_f_dft else (F_dft > f_dft_thr)

        out[model] = {
            "dF":    compute_cdf(np.abs(dF[mask])),
            "theta": compute_cdf(np.abs(dTh[mask])),
        }
    return out


def build_cdf_by_regime(all_results,
                        regimes=None,
                        use_abs_f_dft=True):
    """
    Build CDFs split by F_DFT regime.

    Parameters
    ----------
    all_results : dict[str, dict]
    regimes     : dict[name -> (lo, hi)]  hi=None means unbounded above
                  Defaults to DEFAULT_REGIMES.
    use_abs_f_dft : apply regime bounds to |F_DFT|

    Returns
    -------
    dict[regime_name] -> dict[model] -> {"dF": (x, cdf), "theta": (x, cdf)}
    """
    if regimes is None:
        regimes = DEFAULT_REGIMES

    result = {}
    for regime_name, (lo, hi) in regimes.items():
        regime_cdfs = {}
        for model, d in all_results.items():
            F_dft = np.asarray(d.get("F_dft", []), dtype=float)
            dF    = np.asarray(d.get("deltaF", []), dtype=float)
            dTh   = np.asarray(d.get("deltaTheta", []), dtype=float)

            n = min(F_dft.size, dF.size, dTh.size)
            F_dft, dF, dTh = F_dft[:n], dF[:n], dTh[:n]

            F_ref = np.abs(F_dft) if use_abs_f_dft else F_dft
            mask  = np.isfinite(F_dft) & np.isfinite(dF) & np.isfinite(dTh)
            mask &= _regime_mask(F_ref, lo, hi)

            regime_cdfs[model] = {
                "dF":    compute_cdf(np.abs(dF[mask])),
                "theta": compute_cdf(np.abs(dTh[mask])),
            }
        result[regime_name] = regime_cdfs
    return result

# ============================================================
# Core: 2-D density + marginal histograms
# ============================================================

def plot_density_with_marginals(
    fig, subspec,
    x, y,
    *,
    xscale="log",
    yscale="log",
    x_bins=200,
    y_bins=200,
    x_bin_mode="log",        # "log" | "linear" | array of edges
    y_bin_mode="log",        # "log" | "linear" | array of edges
    y_range=None,            # (lo, hi) for linear y
    cmap="viridis",
    norm=None,               # defaults to LogNorm()
    cbar_label="Density (log scale)",
    xlabel="",
    ylabel="",
    title="",
    xlim=None,
    ylim=None,
    hist_bins=100,
    hist_color="blue",
    hist_alpha=0.7,
    hist_edgecolor="black",
    hist_count_log=True,
    tick_major=12,
    tick_minor=10,
    right_hist_xtick_rotation=45,
    loglocator_base10=True,
    inner_wspace=0.18,
    inner_hspace=0.18,
    title_on="xhist",        # "xhist" | "main"
    title_pad=2,
    label_fontsize=12,
    title_fontsize=14,
    cbar_fontsize=12,
    hist_label_fontsize=11,
    # colorbar width / spacing can be tuned via width_ratios:
    cbar_width_ratio=0.11,   # fraction of each data column
):
    """
    Draw a 2-D density heatmap with top and right marginal histograms and a
    colour bar, all inside *subspec* of *fig*.

    Returns
    -------
    ax_main, ax_xhist, ax_yhist, ax_cbar
    """
    if norm is None:
        norm = LogNorm()

    ax_main, ax_xhist, ax_yhist, ax_cbar = _make_inner_axes(
        fig, subspec,
        width_ratios=(1, 1, 1, 1, cbar_width_ratio),
        wspace=inner_wspace, hspace=inner_hspace,
    )

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # ---------- bin edges ----------
    if isinstance(x_bin_mode, (list, tuple, np.ndarray)):
        xedges = np.asarray(x_bin_mode, dtype=float)
    elif x_bin_mode == "log":
        xedges = np.logspace(np.log10(x.min()), np.log10(x.max()), int(x_bins))
    elif x_bin_mode == "linear":
        xedges = np.linspace(x.min(), x.max(), int(x_bins))
    else:
        raise ValueError("x_bin_mode must be 'log', 'linear', or edge array")

    if isinstance(y_bin_mode, (list, tuple, np.ndarray)):
        yedges = np.asarray(y_bin_mode, dtype=float)
    else:
        ylo, yhi = (y_range if y_range is not None else (y.min(), y.max()))
        if y_bin_mode == "log":
            yedges = np.logspace(np.log10(max(ylo, 1e-12)), np.log10(yhi), int(y_bins))
        elif y_bin_mode == "linear":
            yedges = np.linspace(ylo, yhi, int(y_bins))
        else:
            raise ValueError("y_bin_mode must be 'log', 'linear', or edge array")

    # ---------- 2-D histogram ----------
    heatmap, xedges2, yedges2 = np.histogram2d(x, y, bins=[xedges, yedges])

    ax_main.set_xscale(xscale)
    ax_main.set_yscale(yscale)
    pcm = ax_main.pcolormesh(xedges2, yedges2, heatmap.T,
                              cmap=cmap, norm=norm, rasterized=True)

    cbar = fig.colorbar(pcm, cax=ax_cbar)
    cbar.set_label(cbar_label, fontsize=cbar_fontsize, rotation=90)
    cbar.ax.tick_params(which="major", labelsize=tick_major)
    cbar.ax.tick_params(which="minor", labelsize=tick_minor)

    ax_main.set_xlabel(xlabel, fontsize=label_fontsize)
    ax_main.set_ylabel(ylabel, fontsize=label_fontsize)

    if title:
        if title_on == "xhist":
            ax_xhist.set_title(title, fontsize=title_fontsize, pad=title_pad)
        else:
            ax_main.set_title(title, fontsize=title_fontsize, pad=title_pad)

    if xlim is not None:
        ax_main.set_xlim(*xlim)
    if ylim is not None:
        ax_main.set_ylim(*ylim)

    if loglocator_base10 and xscale == "log":
        ax_main.xaxis.set_major_locator(
            ticker.LogLocator(base=10.0, subs=(1.0,), numticks=10)
        )

    # ---------- marginal histograms ----------
    if xscale == "log":
        x_hist_edges = np.logspace(np.log10(x.min()), np.log10(x.max()), int(hist_bins))
    else:
        x_hist_edges = int(hist_bins)

    ax_xhist.hist(x, bins=x_hist_edges, color=hist_color,
                  alpha=hist_alpha, edgecolor=hist_edgecolor)
    ax_xhist.set_xscale(xscale)
    if hist_count_log:
        ax_xhist.set_yscale("log")
    ax_xhist.set_ylabel("Count", fontsize=hist_label_fontsize)

    if yscale == "log":
        y_hist_edges = np.logspace(np.log10(y.min()), np.log10(y.max()), int(hist_bins))
    else:
        y_hist_edges = int(hist_bins)

    ax_yhist.hist(y, bins=y_hist_edges, orientation="horizontal",
                  color=hist_color, alpha=hist_alpha, edgecolor=hist_edgecolor)
    if hist_count_log:
        ax_yhist.set_xscale("log")
    ax_yhist.set_xlabel("Count", fontsize=hist_label_fontsize)

    _rotate_right_hist_xticks(ax_yhist, rotation=right_hist_xtick_rotation)
    _shift_xticklabels(ax_yhist, dx_pts=10)
    _style_ticks([ax_main, ax_xhist, ax_yhist], major=tick_major, minor=tick_minor)

    plt.setp(ax_xhist.get_xticklabels(), visible=False)
    plt.setp(ax_yhist.get_yticklabels(), visible=False)

    ax_main.spines["top"].set_visible(True)
    ax_main.spines["right"].set_visible(True)
    ax_xhist.spines["bottom"].set_visible(False)
    ax_yhist.spines["left"].set_visible(False)
    ax_xhist.tick_params(axis="x", which="both", bottom=False)
    ax_yhist.tick_params(axis="y", which="both", left=False)

    return ax_main, ax_xhist, ax_yhist, ax_cbar

# ============================================================
# Pre-built panel functions
# ============================================================
_FD_THRESHOLD = 0.01   # default module-level threshold


def panel_Fdft_vs_abs_dF(
    fig, subspec, all_results, fp,
    *,
    force_threshold=_FD_THRESHOLD,
    bins2d=200,
    hist_bins=100,
    xlim=(None, 1000),
    title=None,
    cmap="viridis",
    tick_major=12, tick_minor=10,
    inner_wspace=0.18,
    inner_hspace=0.18,
    **kwargs,
):
    """|Δ|F|| vs F_DFT density panel."""
    if title is None:
        title = (
            rf"$|\Delta\left|F\right||$ ({fp}) vs $|F_{{\mathrm{{DFT}}}}|$"
            "\n"
            rf"$|F_{{\mathrm{{DFT}}}}| > {force_threshold}$ only"
        )
    x_raw = np.array(all_results[fp].get("F_dft", all_results[fp].get("all_F_dft_mags")))
    dF_raw = np.array(all_results[fp].get("deltaF", all_results[fp].get("all_deltaF")))

    mask = x_raw > force_threshold
    x, y = _prep_xy(x_raw, dF_raw, x_abs=False, y_abs=True, mask=mask)

    xlim_use = None
    if xlim is not None:
        xlo = x.min() if xlim[0] is None else xlim[0]
        xlim_use = (xlo, xlim[1])

    return plot_density_with_marginals(
        fig, subspec, x, y,
        xscale="log", yscale="log",
        x_bins=bins2d, y_bins=bins2d,
        x_bin_mode="log", y_bin_mode="log",
        cmap=cmap, norm=LogNorm(),
        cbar_label=r"Density",
        xlabel=r"$|F_{\mathrm{DFT}}|$ (eV/$\mathrm{\AA}$)",
        ylabel=r"$|\Delta\left|F\right||$ (eV/$\mathrm{\AA}$)",
        title=title,
        xlim=xlim_use,
        hist_bins=hist_bins,
        tick_major=tick_major, tick_minor=tick_minor,
        inner_wspace=inner_wspace, inner_hspace=inner_hspace,
        title_on="xhist",
        **kwargs,
    )


def panel_Fdft_vs_dtheta(
    fig, subspec, all_results, fp,
    *,
    force_threshold=_FD_THRESHOLD,
    bins2d=200,
    hist_bins=100,
    xlim=(None, 1000),
    title=None,
    cmap="viridis",
    tick_major=12, tick_minor=10,
    inner_wspace=0.18,
    inner_hspace=0.18,
    **kwargs,
):
    """Δθ vs F_DFT density panel."""
    if title is None:
        title = (
            rf"$\Delta\theta$ ({fp}) vs $|F_{{\mathrm{{DFT}}}}|$"
            "\n"
            rf"$|F_{{\mathrm{{DFT}}}}| > {force_threshold}$ only"
        )
    x_raw = np.array(all_results[fp].get("F_dft", all_results[fp].get("all_F_dft_mags")))
    dtheta_raw = np.array(all_results[fp].get("deltaTheta", all_results[fp].get("all_deltaTheta")))

    mask = x_raw > force_threshold
    x, y = _prep_xy(x_raw, dtheta_raw, x_abs=False, y_abs=False,
                    mask=mask, y_min_eps=0.0)

    xlim_use = None
    if xlim is not None:
        xlo = x.min() if xlim[0] is None else xlim[0]
        xlim_use = (xlo, xlim[1])

    return plot_density_with_marginals(
        fig, subspec, x, y,
        xscale="log", yscale="linear",
        x_bins=bins2d, y_bins=bins2d,
        x_bin_mode="log", y_bin_mode="linear",
        y_range=(0, 180),
        cmap=cmap, norm=LogNorm(),
        cbar_label=r"Density",
        xlabel=r"$|F_{\mathrm{DFT}}|$ (eV/$\mathrm{\AA}$)",
        ylabel=r"$\Delta\theta$ (°)",
        title=title,
        xlim=xlim_use,
        hist_bins=hist_bins,
        tick_major=tick_major, tick_minor=tick_minor,
        inner_wspace=inner_wspace, inner_hspace=inner_hspace,
        title_on="xhist",
        **kwargs,
    )


def panel_abs_dF_vs_dtheta_cond_on_Fdft(
    fig, subspec, all_results, fp,
    *,
    fdft_threshold=0.1,
    bins2d=200,
    hist_bins=100,
    title=None,
    xlim=(None, 1e3),
    cmap="plasma",
    tick_major=12, tick_minor=10,
    inner_wspace=0.18,
    inner_hspace=0.18,
    **kwargs,
):
    """|Δ|F|| vs Δθ, conditioned on |F_DFT| > fdft_threshold."""
    if title is None:
        title = (
            rf"DFT vs {fp}: $|\Delta\left|F\right||$ vs $\Delta\theta$"
            "\n"
            rf"$|F_{{\mathrm{{DFT}}}}| > {fdft_threshold}$ only"
        )
    Fdft = np.array(all_results[fp].get("F_dft", all_results[fp].get("all_F_dft_mags")))
    dF   = np.array(all_results[fp].get("deltaF", all_results[fp].get("all_deltaF")))
    dth  = np.array(all_results[fp].get("deltaTheta", all_results[fp].get("all_deltaTheta")))

    mask = np.abs(Fdft) > fdft_threshold
    x, y = _prep_xy(dF, dth, x_abs=True, y_abs=False,
                    mask=mask, y_min_eps=0.0)

    xlim_use = None
    if xlim is not None:
        xlo = x.min() if xlim[0] is None else xlim[0]
        xlim_use = (xlo, xlim[1])

    return plot_density_with_marginals(
        fig, subspec, x, y,
        xscale="log", yscale="linear",
        x_bins=bins2d, y_bins=bins2d,
        x_bin_mode="log", y_bin_mode="linear",
        y_range=(0, 180),
        cmap=cmap, norm=LogNorm(),
        cbar_label=r"Density",
        xlabel=r"$|\Delta\left|F\right||$ (eV/$\mathrm{\AA}$)",
        ylabel=r"$\Delta\theta$ (°)",
        title=title,
        xlim=xlim_use,
        hist_bins=hist_bins,
        tick_major=tick_major, tick_minor=tick_minor,
        inner_wspace=inner_wspace, inner_hspace=inner_hspace,
        title_on="xhist",
        **kwargs,
    )

# ============================================================
# CDF plots
# ============================================================

def plot_cdf_with_inset_on_ax(
    fig, subspec,
    cdf_results,
    *,
    kind="dF",                        # "dF" or "theta"
    title="",
    xlabel="",
    xlim_main=(-0.1, 2),
    xlim_inset=(0.5, 4),
    ylim_inset=(0.97, 1.0),
    inset_bbox=(0.25, 0.16, 1, 1),    # (x0, y0, w, h) in axes fraction — anchor point + size
    inset_width="35%",                # width  of the inset axes (% of parent or absolute inches)
    inset_height="35%",               # height of the inset axes
    inset_loc="center",               # loc anchor inside parent axes
    legend_bbox=(0.55, 0.02),
    legend_ncols=1,
    legend_loc="lower left",
    show_legend=True,
    show_inset=True,
    linewidth=1.0,
    do_mark_inset=False,
    textsize=10,
    grid_alpha_main=0.6,
    grid_alpha_inset=0.4,
    model_colors=None,                # dict {model: color} or None → default prop_cycle
    model_linestyles=None,            # dict {model: ls} or None → solid "-"
    model_linewidths=None,            # dict {model: lw} or None → use linewidth param
    **style,
):
    """
    CDF line plot with an inset zoom window.

    *style* keys understood: label_fontsize, title_fontsize, tick_major,
    tick_minor, legend_fontsize, inset_tick_fontsize, title_pad.

    Returns
    -------
    ax_main, ax_inset
    """
    ax = fig.add_subplot(subspec)

    label_fs  = style.get("label_fontsize",     textsize)
    title_fs  = style.get("title_fontsize",     textsize)
    tick_maj  = style.get("tick_major",         textsize)
    tick_min  = style.get("tick_minor",         textsize)
    legend_fs = style.get("legend_fontsize",    max(8, label_fs - 2))
    title_pad = style.get("title_pad",          2)
    inset_tick = style.get("inset_tick_fontsize", max(8, tick_maj - 2))

    # build color lookup (None entries → matplotlib auto-cycles)
    if model_colors is not None:
        _clookup = model_colors
    else:
        _cycle   = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        _clookup = {m: _cycle[i % len(_cycle)]
                    for i, m in enumerate(cdf_results)}

    # --- main lines ---
    for model, vals in cdf_results.items():
        x, cdf = vals[kind]
        if x.size == 0:
            continue
        _lw = (model_linewidths.get(model, linewidth) if model_linewidths else linewidth)
        _ls = (model_linestyles.get(model, "-") if model_linestyles else "-")
        ax.plot(x, cdf, label=model, linewidth=_lw, linestyle=_ls,
                color=_clookup.get(model))

    ax.set_xlim(*xlim_main)
    ax.set_xlabel(xlabel, fontsize=label_fs)
    ax.set_ylabel("CDF", fontsize=label_fs)
    ax.set_title(title, fontsize=title_fs, pad=title_pad)
    ax.tick_params(axis="both", which="major", labelsize=tick_maj)
    ax.tick_params(axis="both", which="minor", labelsize=tick_min)
    ax.grid(True, which="both", ls="--", alpha=grid_alpha_main)

    if show_legend:
        ax.legend(
            loc=legend_loc,
            bbox_to_anchor=legend_bbox,
            borderaxespad=0.0,
            fontsize=legend_fs,
            ncols=legend_ncols,
            frameon=False,
        )

    # --- inset ---
    if not show_inset:
        return ax, None

    ax_inset = inset_axes(
        ax,
        width=inset_width,
        height=inset_height,
        loc=inset_loc,
        bbox_to_anchor=inset_bbox,
        bbox_transform=ax.transAxes,
        borderpad=0,
    )

    for model, vals in cdf_results.items():
        x, cdf = vals[kind]
        if x.size == 0:
            continue
        _lw = (model_linewidths.get(model, linewidth) if model_linewidths else linewidth)
        _ls = (model_linestyles.get(model, "-") if model_linestyles else "-")
        ax_inset.plot(x, cdf, linewidth=_lw, linestyle=_ls, color=_clookup.get(model))

    ax_inset.set_xlim(*xlim_inset)
    ax_inset.set_ylim(*ylim_inset)
    ax_inset.tick_params(axis="both", which="major", labelsize=inset_tick)
    ax_inset.grid(True, which="both", ls="--", alpha=grid_alpha_inset)

    if do_mark_inset:
        mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="0.5")

    return ax, ax_inset


def plot_rank_curves_from_cdfs_on_ax(
    fig, subspec,
    cdf_results,
    *,
    kind="dF",
    model_names=None,
    model_colors=None,
    x_max=2.5,
    n_grid=300,
    x_min_valid=0.0,
    smooth=True,
    n_fit_points=25,
    sigma=1.0,
    lw=1.6,
    title="",
    xlabel=None,
    textsize=10,
    legend_bbox=(1.0, 0.5),
    legend_loc="center left",
    legend_ncols=1,
    use_discrete_final_rank=True,
    show_legend=True,
):
    """
    Rank-vs-threshold plot (1 = best model at each threshold).

    Returns
    -------
    ax, (x_grid, rank_matrix)
    """
    from scipy.interpolate import PchipInterpolator
    from scipy.ndimage import gaussian_filter1d

    ax = fig.add_subplot(subspec)

    if model_names is None:
        model_names = list(cdf_results.keys())

    x_common = np.linspace(0, x_max, n_grid)
    x_common = x_common[x_common >= x_min_valid]

    # interpolate each CDF onto common grid
    cdf_matrix = []
    for model in model_names:
        x, cdf = cdf_results[model][kind]
        if x.size == 0:
            cdf_matrix.append(np.full_like(x_common, np.nan))
        else:
            cdf_matrix.append(np.interp(x_common, x, cdf))
    cdf_matrix = np.array(cdf_matrix)

    cdf_fill = np.where(np.isfinite(cdf_matrix), cdf_matrix, -np.inf)
    ranks = np.argsort(np.argsort(-cdf_fill, axis=0), axis=0) + 1

    model_lines = []
    smooth_final = []

    for i, model in enumerate(model_names):
        color = (model_colors.get(model) if model_colors else None)
        y = ranks[i].astype(float)

        if smooth:
            x_fit = np.linspace(x_common.min(), x_common.max(), n_fit_points)
            x_smooth = np.linspace(x_common.min(), x_common.max(), 1000)
            y_fit = np.interp(x_fit, x_common, y)
            y_s = PchipInterpolator(x_fit, y_fit)(x_smooth)
            y_s = gaussian_filter1d(y_s, sigma=sigma)
            line, = ax.plot(x_smooth, y_s, lw=lw, color=color, label=model)
            model_lines.append(line)
            smooth_final.append(float(y_s[-1]))
        else:
            line, = ax.plot(x_common, y, lw=lw, color=color, label=model)
            model_lines.append(line)

    final_ranks = ranks[:, -1] if use_discrete_final_rank else (
        np.array(smooth_final) if smooth else ranks[:, -1]
    )
    order = np.argsort(final_ranks)
    handles = [model_lines[i] for i in order]
    labels  = [model_names[i] for i in order]

    ax.set_xlim(x_common.min(), x_common.max())
    ax.invert_yaxis()
    ax.set_ylabel("Rank", fontsize=textsize)

    if xlabel is None:
        xlabel = (r"$|\Delta\left|F\right||$ (eV/$\mathrm{\AA}$)"
                  if kind == "dF" else r"$\Delta\theta$ (deg)")
    ax.set_xlabel(xlabel, fontsize=textsize)

    if title:
        ax.set_title(title, fontsize=textsize)

    ax.tick_params(axis="both", which="major", labelsize=textsize)
    ax.grid(True, ls="--", alpha=0.4)

    if show_legend:
        ax.legend(handles, labels,
                  loc=legend_loc,
                  bbox_to_anchor=legend_bbox,
                  fontsize=textsize - 2,
                  frameon=False,
                  ncols=legend_ncols)

    return ax, (x_common, ranks)

# ============================================================
# Convenience: one CDF figure per regime
# ============================================================

def plot_cdf_regimes(
    all_results,
    regimes=None,
    *,
    kinds=("dF", "theta"),
    figsize=None,
    dpi=150,
    wspace=0.4,
    hspace=0.5,
    cdf_kwargs=None,
    shared_legend=True,
    legend_ncols=3,
    legend_fontsize=9,
    legend_loc="lower center",
    legend_bbox=(0.5, -0.04),
    show_suptitle=True,
    suptitle_fontsize=10,
    savefig=None,
):
    """
    For each regime produce a figure with one CDF subplot per *kind*.

    Parameters
    ----------
    all_results   : raw results dict
    regimes       : dict[name -> (lo, hi)]; defaults to DEFAULT_REGIMES
    kinds         : which quantities to plot ("dF", "theta")
    cdf_kwargs    : dict passed to plot_cdf_with_inset_on_ax (can be list
                    parallel to *kinds*, or a single dict applied to all)
    shared_legend : add a shared legend at bottom of each figure
    savefig       : path template, e.g. "cdf_{regime}.pdf" – {regime} replaced

    Returns
    -------
    dict[regime_name] -> fig
    """
    if regimes is None:
        regimes = DEFAULT_REGIMES

    cdf_by_regime = build_cdf_by_regime(all_results, regimes=regimes)

    n_kinds = len(kinds)
    if figsize is None:
        figsize = (4.5 * n_kinds, 4.0)

    figs = {}
    for regime_name, cdf_results in cdf_by_regime.items():
        lo, hi = regimes[regime_name]
        hi_str = f"{hi}" if hi is not None else "∞"
        regime_label = rf"$|F_{{\mathrm{{DFT}}}}| \in [{lo}, {hi_str})$ eV/Å"

        fig = plt.figure(figsize=figsize, dpi=dpi)
        outer = fig.add_gridspec(1, n_kinds, wspace=wspace, hspace=hspace)

        axes = []
        for col, kind in enumerate(kinds):
            kw_base = dict(
                kind=kind,
                title=(rf"CDF of $|\Delta\left|F\right||$" if kind == "dF"
                       else rf"CDF of $\Delta\theta$")
                      + "\n" + regime_label,
                xlabel=(r"$|\Delta\left|F\right||$ (eV/$\mathrm{\AA}$)"
                        if kind == "dF" else r"$\Delta\theta$ (deg)"),
                show_legend=(not shared_legend),
            )
            # per-kind kwargs override
            extra = {}
            if cdf_kwargs is not None:
                if isinstance(cdf_kwargs, list):
                    extra = cdf_kwargs[col] if col < len(cdf_kwargs) else {}
                else:
                    extra = cdf_kwargs
            kw_base.update(extra)

            ax, _ = plot_cdf_with_inset_on_ax(
                fig, outer[0, col], cdf_results, **kw_base
            )
            axes.append(ax)

        if shared_legend and axes:
            handles, labels = axes[0].get_legend_handles_labels()
            # collect all – in case some axes have extra models
            seen = set(labels)
            for ax in axes[1:]:
                h, l = ax.get_legend_handles_labels()
                for hh, ll in zip(h, l):
                    if ll not in seen:
                        handles.append(hh)
                        labels.append(ll)
                        seen.add(ll)

            fig.legend(handles, labels,
                       loc=legend_loc,
                       bbox_to_anchor=legend_bbox,
                       ncol=legend_ncols,
                       frameon=False,
                       fontsize=legend_fontsize)

        if show_suptitle:
            fig.suptitle(f"Regime: {regime_name}  —  {regime_label}",
                         fontsize=suptitle_fontsize, y=1.02)

        if savefig:
            path = savefig.format(regime=regime_name)
            fig.savefig(path, bbox_inches="tight")
            print(f"Saved: {path}")

        figs[regime_name] = fig

    return figs
