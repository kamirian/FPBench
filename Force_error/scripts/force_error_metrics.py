"""
force_error_metrics.py
=======================
Analysis and visualization helpers for MatPES foundation potential (FP) force/energy evaluation.

Public API
----------
Computation
  frac_percent(dF, dtheta, F_dft, dF_cut, angle_cut, fdf_min)
      → fraction (%) of atoms passing Δ|F| < dF_cut AND Δθ < angle_cut

  build_joint_dF_theta_accuracy_table(all_results, dF_cuts, angle_cuts, fdf_min)
      → {angle_cut: DataFrame(index=models, columns=dF_cuts)}

  build_highly_accurate_force_fraction_table(all_results, dF_thresholds, fdf_min)
      → DataFrame(index=models, columns=thresholds)  — fraction < threshold, no angle filter

  build_large_force_error_fraction_table(all_results, dF_thresholds, fdf_min)
      → DataFrame(index=models, columns=thresholds)  — fraction > threshold (large errors)

  build_far_from_equilibrium_regime_panels(all_results, abs_thresh, rel_thresh, threshold, fdf_min)
      → (df_panel_A, df_panel_B)  — near-eq / far-from-eq atom statistics

  build_large_error_fdft_distribution_table(all_results, df_thresh, fdft_thresh, fdf_min)
      → DataFrame  — F_DFT distribution for atoms with Δ|F| > df_thresh

  merge_mae_rmse_as_string(mae_df, rmse_df, fmt)
      → DataFrame  — cells formatted as "mae / rmse"

  build_fdft_distribution_fraction_table(all_results, fdft_thresholds)
      → DataFrame  — fraction (%) of atoms with |F_DFT| > threshold

  build_dF_mae_rmse_fdft_subset(all_results, fdft_thresholds)
      → (mae_df, rmse_df)  — MAE/RMSE of Δ|F| conditioned on |F_DFT| > threshold

  build_angle_accuracy_fraction_table(all_results, theta_thresholds, fdf_min)
      → DataFrame  — fraction (%) of atoms with Δθ < threshold

  build_theta_mae_rmse_angle_subset(all_results, theta_thresholds, fdf_min)
      → (mae_df, rmse_df)  — mean/RMS of Δθ conditioned on Δθ < threshold

  build_theta_far_from_equilibrium_regime_panels(all_results, theta_thresh, threshold, fdf_min)
      → (df_panel_A, df_panel_B)  — near-eq / far-from-eq Δθ fraction statistics

Plotting
  split_triangle_heatmap(data_lower, data_upper, row_labels, col_labels, ...)
      → fig, ax  — each cell split diagonally: lower ← data_lower, upper ← data_upper

  single_heatmap(data, row_labels, col_labels, ...)
      → fig, ax  — standard rectangular heatmap, one value per cell

  merged_heatmaps(data_left, ..., data_lower, data_upper, ...)
      → fig, (ax_left, ax_right)  — single_heatmap + split_triangle side-by-side

  heatmap_fraction_delta_theta(df_frac, ...)
      → fig, ax  — imshow-based heatmap for Δθ fraction data

  triangular_heatmap_with_fraction_row_word_style(mae_df, rmse_df, frac_row_str, ...)
      → fig, ax  — triangular heatmap with horizontal colorbars placed below

  plot_error_histograms(all_results_filtered, ...)
      — Δ|F| (log-x, log-y) and Δθ (linear-x, log-y) histograms
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Polygon, Rectangle
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D


# ── global matplotlib style (matches tested figure output) ────────────────────
plt.rcParams["svg.fonttype"]       = "none"
plt.rcParams["font.family"]        = "Arial"
plt.rcParams["mathtext.rm"]        = "Arial"
plt.rcParams["mathtext.it"]        = "Arial:italic"
plt.rcParams["mathtext.fontset"]   = "custom"
plt.rcParams["mathtext.default"]   = "it"


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper
# ─────────────────────────────────────────────────────────────────────────────

def _sigfig_formatter(n):
    """
    Return a callable that formats a float to *n* significant figures as plain
    decimal (never scientific notation).  E.g. n=2: 99.5→"100", 7.3→"7.3",
    0.12→"0.12".
    """
    import math
    def _f(val):
        if not np.isfinite(val):
            return ""
        if val == 0:
            return "0"
        s = f"{val:.{n}g}"
        if "e" in s or "E" in s:
            mag = math.floor(math.log10(abs(val)))
            dp  = max(0, n - 1 - mag)
            s   = f"{val:.{dp}f}"
        return s
    return _f


def _fmtval(fmt, val):
    """Apply *fmt* to *val*: callable → fmt(val), str → fmt.format(val)."""
    return fmt(val) if callable(fmt) else fmt.format(val)


def _text_color(rgba, threshold=0.5):
    """Return 'black' or 'white' based on WCAG background luminance."""
    r, g, b, _ = rgba
    return "black" if (0.2126 * r + 0.7152 * g + 0.0722 * b) > threshold else "white"


# ═════════════════════════════════════════════════════════════════════════════
# Computation
# ═════════════════════════════════════════════════════════════════════════════

def frac_percent(dF, dtheta, F_dft, dF_cut, angle_cut, fdf_min=0.01, fdf_max=None):
    """
    Fraction (%) of atoms satisfying both thresholds simultaneously.

    Only atoms with fdf_min < |F_dft| (< fdf_max, if given) are considered
    (near-zero DFT forces are excluded from both numerator and denominator;
    an upper bound restricts to a near-equilibrium regime).

    Parameters
    ----------
    dF        : array-like – force magnitude error |F_fp| − |F_dft| (eV/Å)
    dtheta    : array-like – angular error between force vectors (degrees)
    F_dft     : array-like – DFT force magnitude (eV/Å)
    dF_cut    : float – threshold on Δ|F| (eV/Å)
    angle_cut : float – threshold on Δθ (degrees); use 180 to disable angle filter
    fdf_min   : float – minimum |F_dft| for an atom to be included
    fdf_max   : float or None – maximum |F_dft| for an atom to be included
                (None = no upper bound)

    Returns
    -------
    float in [0, 100], or np.nan if no valid atoms exist
    """
    F_dft  = np.asarray(F_dft,  float)
    dF     = np.asarray(dF,     float)
    dtheta = np.asarray(dtheta, float)

    base = np.isfinite(F_dft) & np.isfinite(dF) & np.isfinite(dtheta)
    base &= np.abs(F_dft) > fdf_min
    if fdf_max is not None:
        base &= np.abs(F_dft) < fdf_max

    denom = base.sum()
    if denom == 0:
        return np.nan

    ok = base & (np.abs(dF) < dF_cut) & (np.abs(dtheta) < angle_cut)
    return 100.0 * ok.sum() / denom


def build_joint_dF_theta_accuracy_table(all_results, dF_cuts, angle_cuts, fdf_min=0.01, fdf_max=None):
    """
    Compute fraction tables for one or more angle-cut thresholds.

    Parameters
    ----------
    all_results : dict[model_name → dict]
        Each value must have keys "F_dft", "deltaF", "deltaTheta".
    dF_cuts     : list of float – force-error thresholds (eV/Å)
    angle_cuts  : list of float – angular thresholds (degrees), e.g. [1, 20]
    fdf_min     : float – minimum |F_dft| for inclusion
    fdf_max     : float or None – maximum |F_dft| for inclusion (None = no cap)

    Returns
    -------
    dict[angle_cut → pd.DataFrame]
        Rows = models, columns = dF_cuts, values = fraction (%) of atoms
        passing Δ|F| < dF_cut AND Δθ < angle_cut.
    """
    tables = {}
    for angle_cut in angle_cuts:
        rows = {}
        for model, data in all_results.items():
            F_dft  = data.get("F_dft", data.get("all_F_dft_mags"))
            dF     = data.get("deltaF", data.get("all_deltaF"))
            dtheta = data.get("deltaTheta", data.get("all_deltaTheta"))
            rows[model] = {
                thr: frac_percent(dF, dtheta, F_dft, thr, angle_cut, fdf_min, fdf_max)
                for thr in dF_cuts
            }
        df = pd.DataFrame(rows).T          # rows = models, columns = thresholds
        df.columns = dF_cuts
        tables[angle_cut] = df
    return tables


def build_highly_accurate_force_fraction_table(all_results, dF_thresholds, fdf_min=0.01, fdf_max=None):
    """
    Fraction (%) of atoms with Δ|F| < threshold — no angle filter.

    Parameters
    ----------
    all_results   : dict[model_name → dict]  must have keys "F_dft", "deltaF"
    dF_thresholds : list of float
    fdf_min       : float
    fdf_max       : float or None – maximum |F_dft| for inclusion (None = no cap)

    Returns
    -------
    pd.DataFrame – rows = models, columns = thresholds, values = fraction (%)
    """
    rows = {}
    for model, data in all_results.items():
        F_dft = np.asarray(data.get("F_dft", data.get("all_F_dft_mags")), float)
        dF    = np.asarray(data.get("deltaF", data.get("all_deltaF")),    float)

        mask = np.isfinite(F_dft) & np.isfinite(dF) & (np.abs(F_dft) > fdf_min)
        if fdf_max is not None:
            mask &= np.abs(F_dft) < fdf_max
        dF_f = dF[mask]

        rows[model] = {
            thr: (np.round(np.mean(np.abs(dF_f) < thr) * 100, 2) if dF_f.size > 0 else np.nan)
            for thr in dF_thresholds
        }
    df = pd.DataFrame(rows).T
    df.columns = dF_thresholds
    return df


def build_large_force_error_fraction_table(all_results, dF_thresholds, fdf_min=0.01):
    """
    Fraction (%) of atoms with Δ|F| **greater than** threshold — no angle filter.

    Complement of build_highly_accurate_force_fraction_table; useful for identifying the tail of large
    force errors (e.g. how many atoms have Δ|F| > 1, > 5, > 10 eV/Å).

    Parameters
    ----------
    all_results   : dict[model_name → dict]  must have keys "F_dft", "deltaF"
    dF_thresholds : list of float
    fdf_min       : float – minimum |F_dft| for inclusion

    Returns
    -------
    pd.DataFrame – rows = models, columns = thresholds, values = fraction (%)
    """
    rows = {}
    for model, data in all_results.items():
        F_dft = np.asarray(data.get("F_dft", data.get("all_F_dft_mags")),  float)
        dF    = np.asarray(data.get("deltaF", data.get("all_deltaF")), float)

        mask = np.isfinite(F_dft) & np.isfinite(dF) & (np.abs(F_dft) > fdf_min)
        dF_f = dF[mask]

        rows[model] = {
            thr: (np.round(np.mean(np.abs(dF_f) > thr) * 100, 4) if dF_f.size > 0 else np.nan)
            for thr in dF_thresholds
        }
    df = pd.DataFrame(rows).T
    df.columns = dF_thresholds
    return df


def build_far_from_equilibrium_regime_panels(
    all_results,
    abs_thresh,
    rel_thresh,
    threshold=1.0,
    fdf_min=0.0,
):
    r"""
    Split atoms into two regimes by |F_DFT| and compute absolute/relative
    force-error fraction tables.

    Panel A — non-FE (near-equilibrium)   : |F_DFT| <= threshold
    Panel B — FE (far-from-equilibrium)   : |F_DFT| > threshold

    Each row in the returned DataFrames contains:
      "N atoms"               — number of atoms in this regime for this model
      "Frac of all atoms (%)" — share of the total valid atom pool
      "$|\Delta\left|F\right|| < {thr}$ (%)" — for each absolute threshold
      "r < {thr} (%)"        — for each relative threshold ($r_F$ = |Δ|F|| / |F_DFT|)

    Parameters
    ----------
    all_results : dict[model_name → dict]  keys: "F_dft", "deltaF"
    abs_thresh  : list of float – absolute Δ|F| thresholds (eV/Å)
    rel_thresh  : list of float – relative error thresholds (dimensionless)
    threshold   : float – |F_DFT| boundary between panels (default 1.0 eV/Å)
    fdf_min     : float – minimum |F_dft| for an atom to be included at all

    Returns
    -------
    (df_panel_A, df_panel_B) : two pd.DataFrames with rows = models
    """
    panel_A, panel_B = {}, {}

    for model, data in all_results.items():
        F_dft = np.asarray(data.get("F_dft", data.get("all_F_dft_mags")),  float)
        dF    = np.asarray(data.get("deltaF", data.get("all_deltaF")), float)

        n = min(len(F_dft), len(dF))
        F_dft, dF = F_dft[:n], dF[:n]

        valid = np.isfinite(F_dft) & np.isfinite(dF)
        if fdf_min > 0:
            valid &= np.abs(F_dft) > fdf_min

        F_abs  = np.abs(F_dft[valid])
        dF_abs = np.abs(dF[valid])
        rel    = dF_abs / F_abs
        total  = F_abs.size

        for panel_dict, mask in [
            (panel_A, F_abs <= threshold),
            (panel_B, F_abs >  threshold),
        ]:
            n_regime = int(mask.sum())
            row = {
                "N atoms":               n_regime,
                "Frac of all atoms (%)": round(100 * n_regime / total, 4) if total > 0 else np.nan,
            }
            for thr in abs_thresh:
                row[rf"$|\Delta\left|F\right|| < {thr}$ eV/Å (%)"] = (
                    round(100 * np.mean(dF_abs[mask] < thr), 4) if n_regime > 0 else np.nan
                )
            for thr in rel_thresh:
                row[f"r < {thr} (%)"] = (
                    round(100 * np.mean(rel[mask] < thr), 4) if n_regime > 0 else np.nan
                )
            panel_dict[model] = row

    return pd.DataFrame(panel_A).T, pd.DataFrame(panel_B).T


def build_large_error_fdft_distribution_table(all_results, df_thresh, fdft_thresh, fdf_min=0.01):
    """
    For atoms with Δ|F| > df_thresh, show what fraction have |F_DFT| below
    each of the given thresholds.

    Useful for understanding whether large force errors occur predominantly
    in near-equilibrium (low F_DFT) or high-force atoms.

    Parameters
    ----------
    all_results  : dict[model_name → dict]  keys: "F_dft", "deltaF"
    df_thresh    : float – Δ|F| threshold defining "large-error" atoms (eV/Å)
    fdft_thresh  : list of float – |F_DFT| thresholds to report (eV/Å)
    fdf_min      : float – minimum |F_dft| for an atom to be included

    Returns
    -------
    pd.DataFrame
        Rows = models.
        First column  : "% of all atoms (|Δ|F||>{df_thresh:g})"
        Other columns : "|F_DFT| < {thr} (%)" for each fdft_thresh value.
    """
    first_col = f"% of all atoms\n(|Δ|F||>{df_thresh:g})"
    rows = {}

    for model, data in all_results.items():
        F_dft = np.abs(np.asarray(data.get("F_dft", data.get("all_F_dft_mags")),  float))
        dF    = np.asarray(data.get("deltaF", data.get("all_deltaF")), float)

        n = min(len(F_dft), len(dF))
        F_dft, dF = F_dft[:n], dF[:n]

        valid  = np.isfinite(F_dft) & np.isfinite(dF) & (F_dft > fdf_min)
        F_dft  = F_dft[valid]
        abs_dF = np.abs(dF[valid])
        total  = int(valid.sum())

        mask_large = abs_dF > df_thresh
        n_large    = int(mask_large.sum())

        row = {first_col: round(100 * n_large / total, 4) if total > 0 else np.nan}
        for thr in fdft_thresh:
            row[f"|F_DFT| < {thr} (%)"] = (
                round(100 * np.mean(F_dft[mask_large] < thr), 2) if n_large > 0 else np.nan
            )
        rows[model] = row

    return pd.DataFrame(rows).T


# ═════════════════════════════════════════════════════════════════════════════
# Plotting
# ═════════════════════════════════════════════════════════════════════════════

def split_triangle_heatmap(
    data_lower, data_upper,
    row_labels, col_labels,
    title=None,
    cmap_lower="viridis",
    cmap_upper="viridis",
    norm_lower=None,
    norm_upper=None,
    annotate=True,
    fmt="{:.1f}",
    sig_figs=None,             # significant figures: overrides fmt (e.g. sig_figs=2 → 2 sig figs)
    textsize=8,
    addsize=1,
    cbar=True,
    cbar_label_lower="(%) (<1°)",
    cbar_label_upper="(%) (<20°)",
    figsize=(3.5, 5.5),
    gap=0.2,
    cbar_width=0.015,
    gap_between_cbars=0.15,
    left=0.22, right=0.80, bottom=0.0225, top=1.0,
    savepath=None,
    show_row_labels=False,
):
    """
    Heatmap where each cell is split diagonally into two triangles.

    Lower triangle  ← data_lower  (e.g. fraction at Δθ < 1°)
    Upper triangle  ← data_upper  (e.g. fraction at Δθ < 20°)

    All layout parameters (figsize, margins, font sizes) match tested output.

    Parameters
    ----------
    data_lower, data_upper : 2D array-like, shape (n_models, n_cols)
    row_labels  : list of str – model names (top → bottom)
    col_labels  : list of str – column header labels
    cmap_lower, cmap_upper : str – Matplotlib colormap names
    norm_lower, norm_upper : Normalize or None – auto-computed if None
    fmt         : format string for annotations (applied to both triangles)
    textsize    : int – base font size for annotations
    addsize     : int – added to textsize for tick labels and title
    gap         : float – gap from heatmap right edge to first colorbar
    cbar_width  : float – width of each colorbar (figure-fraction units)
    gap_between_cbars : float
    left, right, bottom, top : float – subplots_adjust margins
    savepath    : str or None – save path (e.g. "out.svg"); None = don't save

    Returns
    -------
    fig, ax
    """
    if sig_figs is not None:
        fmt = _sigfig_formatter(sig_figs)
    data_lower = np.asarray(data_lower, float)
    data_upper = np.asarray(data_upper, float)
    assert data_lower.shape == data_upper.shape, "data_lower and data_upper must have the same shape"
    nrows, ncols = data_lower.shape

    if norm_lower is None:
        v = data_lower[np.isfinite(data_lower)]
        norm_lower = Normalize(vmin=v.min(), vmax=v.max()) if v.size else Normalize(0, 1)
    if norm_upper is None:
        v = data_upper[np.isfinite(data_upper)]
        norm_upper = Normalize(vmin=v.min(), vmax=v.max()) if v.size else Normalize(0, 1)

    cm_lower = mpl.cm.get_cmap(cmap_lower)
    cm_upper = mpl.cm.get_cmap(cmap_upper)

    fig, ax = plt.subplots(figsize=figsize, dpi=350, constrained_layout=False)
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

    for i in range(nrows):
        for j in range(ncols):
            x0, x1 = j, j + 1
            y0, y1 = i, i + 1

            tri_lower = np.array([[x1, y0], [x1, y1], [x0, y0]])
            tri_upper = np.array([[x0, y1], [x0, y0], [x1, y1]])

            v_lo = data_lower[i, j]
            v_up = data_upper[i, j]

            face_lo = cm_lower(norm_lower(v_lo)) if np.isfinite(v_lo) else (0.9, 0.9, 0.9, 1)
            face_up = cm_upper(norm_upper(v_up)) if np.isfinite(v_up) else (0.9, 0.9, 0.9, 1)

            ax.add_patch(Polygon(tri_lower, closed=True,
                                 facecolor=face_lo, edgecolor="black", linewidth=0.5))
            ax.add_patch(Polygon(tri_upper, closed=True,
                                 facecolor=face_up, edgecolor="black", linewidth=0.5))

            if annotate:
                if np.isfinite(v_lo):
                    ax.text(j + 0.28, i + 0.72, _fmtval(fmt, v_lo),
                            ha="center", va="center", fontsize=textsize,
                            color=_text_color(face_lo))
                if np.isfinite(v_up):
                    ax.text(j + 0.65, i + 0.28, _fmtval(fmt, v_up),
                            ha="center", va="center", fontsize=textsize,
                            color=_text_color(face_up))

    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)
    ax.invert_yaxis()
    ax.set_aspect("equal")

    ax.set_xticks(np.arange(ncols) + 0.5)
    ax.set_yticks(np.arange(nrows) + 0.5)
    ax.set_xticklabels(col_labels, rotation=45, ha="right",
                       rotation_mode="anchor", fontsize=textsize + addsize)
    if show_row_labels:
        ax.set_yticklabels(row_labels, fontsize=textsize + addsize)
    else:
        ax.set_yticklabels([])

    ax.tick_params(axis="y", length=4, width=1)
    ax.tick_params(axis="x", length=4, width=1)
    for spine in ax.spines.values():
        spine.set_visible(True)

    if title:
        ax.set_title(title, pad=4, fontsize=textsize + 2 + addsize)

    if cbar:
        fig.canvas.draw()
        bbox = ax.get_position()

        sm_lo = mpl.cm.ScalarMappable(norm=norm_lower, cmap=cm_lower)
        sm_up = mpl.cm.ScalarMappable(norm=norm_upper, cmap=cm_upper)
        sm_lo.set_array([])
        sm_up.set_array([])

        cax1 = fig.add_axes([bbox.x1 + gap,
                             bbox.y0, cbar_width, bbox.height])
        cax2 = fig.add_axes([bbox.x1 + gap + cbar_width + gap_between_cbars,
                             bbox.y0, cbar_width, bbox.height])

        cb1 = fig.colorbar(sm_lo, cax=cax1)
        cb1.set_label(cbar_label_lower, fontsize=textsize + 2, rotation=90, labelpad=0)
        cb1.ax.tick_params(labelsize=textsize + 2 + addsize)

        cb2 = fig.colorbar(sm_up, cax=cax2)
        cb2.set_label(cbar_label_upper, fontsize=textsize + 2 + addsize, rotation=90, labelpad=1)
        cb2.ax.tick_params(labelsize=textsize + 2 + addsize)

    else:
        plt.tight_layout()

    fig.canvas.draw()

    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight", dpi=350, pad_inches=0.05)

    plt.show()
    plt.close(fig)

    return fig, ax


# ═════════════════════════════════════════════════════════════════════════════
# Utility
# ═════════════════════════════════════════════════════════════════════════════

def merge_mae_rmse_as_string(mae_df, rmse_df, fmt="{:.3f}"):
    """
    Combine two numeric DataFrames into a string DataFrame with cells
    formatted as "mae / rmse".

    Parameters
    ----------
    mae_df, rmse_df : pd.DataFrame – must share index and columns
    fmt             : str – Python format string applied to each value

    Returns
    -------
    pd.DataFrame of strings, same shape as the inputs
    """
    result = pd.DataFrame("", index=mae_df.index, columns=mae_df.columns)
    for col in mae_df.columns:
        for idx in mae_df.index:
            try:
                m = float(mae_df.loc[idx, col])
                r = float(rmse_df.loc[idx, col])
                result.loc[idx, col] = (
                    f"{fmt.format(m)} / {fmt.format(r)}"
                    if np.isfinite(m) and np.isfinite(r) else "—"
                )
            except (TypeError, ValueError):
                result.loc[idx, col] = "—"
    return result


def single_heatmap(
    data,
    row_labels, col_labels,
    title=None,
    title_pad=4,
    cmap="viridis",
    norm=None,
    annotate=True,
    fmt="{:.1f}",
    sig_figs=None,             # significant figures: overrides fmt (e.g. sig_figs=2 → 2 sig figs)
    textsize=8,
    addsize=0,
    cbar=True,
    cbar_label="(%)",
    figsize=(3.5, 5.5),
    gap=0.025,
    cbar_width=0.015,
    left=0.22, right=0.80, bottom=0.02, top=1.0,
    savepath=None,
    show_row_labels=False,
):
    """
    Standard rectangular heatmap with one value per cell.

    Layout parameters match the style of split_triangle_heatmap.

    Parameters
    ----------
    data       : 2D array-like, shape (n_models, n_cols)
    row_labels : list of str – model names (top → bottom)
    col_labels : list of str – column header labels
    cmap       : str – Matplotlib colormap name
    norm       : Normalize or None – auto-computed from data if None
    fmt        : format string for cell annotations
    textsize   : int – base font size for annotations
    addsize    : int – added to textsize for tick labels and title
    gap        : float – gap from heatmap right edge to colorbar
    cbar_width : float – colorbar width (figure-fraction units)
    left, right, bottom, top : float – subplots_adjust margins
    savepath   : str or None – save path (e.g. "out.svg"); None = don't save

    Returns
    -------
    fig, ax
    """
    if sig_figs is not None:
        fmt = _sigfig_formatter(sig_figs)
    data = np.asarray(data, float)
    nrows, ncols = data.shape

    if norm is None:
        v = data[np.isfinite(data)]
        norm = Normalize(vmin=v.min(), vmax=v.max()) if v.size else Normalize(0, 1)

    cm = mpl.cm.get_cmap(cmap)

    fig, ax = plt.subplots(figsize=figsize, dpi=350, constrained_layout=False)
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

    for i in range(nrows):
        for j in range(ncols):
            val  = data[i, j]
            face = cm(norm(val)) if np.isfinite(val) else (0.9, 0.9, 0.9, 1)
            ax.add_patch(Rectangle((j, i), 1, 1,
                                   facecolor=face, edgecolor="black", linewidth=0.5))
            if annotate and np.isfinite(val):
                ax.text(j + 0.5, i + 0.5, _fmtval(fmt, val),
                        ha="center", va="center", fontsize=textsize,
                        color=_text_color(face))

    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)
    ax.invert_yaxis()
    ax.set_aspect("equal")

    ax.set_xticks(np.arange(ncols) + 0.5)
    ax.set_yticks(np.arange(nrows) + 0.5)
    ax.set_xticklabels(col_labels, rotation=45, ha="right",
                       rotation_mode="anchor", fontsize=textsize + addsize)
    if show_row_labels:
        ax.set_yticklabels(row_labels, fontsize=textsize + addsize)
    else:
        ax.set_yticklabels([])

    ax.tick_params(axis="y", length=4, width=1)
    ax.tick_params(axis="x", length=4, width=1)
    for spine in ax.spines.values():
        spine.set_visible(True)

    if title:
        ax.set_title(title, pad=title_pad, fontsize=textsize + 2 + addsize)

    if cbar:
        fig.canvas.draw()
        bbox = ax.get_position()
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cm)
        sm.set_array([])
        cax = fig.add_axes([bbox.x1 + gap, bbox.y0, cbar_width, bbox.height])
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label(cbar_label, fontsize=textsize + 2, rotation=90, labelpad=0)
        cb.ax.tick_params(labelsize=textsize + 2 + addsize)

    else:
        plt.tight_layout()

    fig.canvas.draw()

    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight", dpi=350, pad_inches=0.05)

    plt.show()
    plt.close(fig)

    return fig, ax


def single_heatmap_with_frac_row(
    data,
    row_labels, col_labels,
    frac_row_str,
    frac_row_fontweight="bold",
    title=None,
    title_pad=4,
    cmap="viridis",
    norm=None,
    annotate=True,
    fmt="{:.1f}",
    sig_figs=None,             # significant figures: overrides fmt (e.g. sig_figs=2 → 2 sig figs)
    textsize=8,
    addsize=0,
    cbar=True,
    cbar_label="(%)",
    figsize=(3.5, 6.0),
    gap=0.025,
    cbar_width=0.015,
    left=0.22, right=0.80, bottom=0.02, top=1.0,
    savepath=None,
    show_row_labels=False,
):
    """
    single_heatmap with one extra un-colored header row on top, showing
    pre-formatted strings (e.g. what % of atoms each column is computed over).

    Layout
    ------
    Top row   : plain white cells with pre-formatted text (frac_row_str)
    Data rows : standard colored heatmap cells, same as single_heatmap

    Parameters
    ----------
    data          : 2D array-like, shape (n_models, n_cols)
    row_labels    : list of str – model names (top → bottom)
    col_labels    : list of str – column header labels
    frac_row_str  : sequence of str, length n_cols – pre-formatted header-row text
    (all other parameters match single_heatmap)

    Returns
    -------
    fig, ax
    """
    if sig_figs is not None:
        fmt = _sigfig_formatter(sig_figs)
    data = np.asarray(data, float)
    nrows, ncols = data.shape
    nrows_total = nrows + 1

    if norm is None:
        v = data[np.isfinite(data)]
        norm = Normalize(vmin=v.min(), vmax=v.max()) if v.size else Normalize(0, 1)

    cm = mpl.cm.get_cmap(cmap)

    fig, ax = plt.subplots(figsize=figsize, dpi=350, constrained_layout=False)
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

    # ── header row (row 0): plain white cells with pre-formatted text ────────
    for j in range(ncols):
        ax.add_patch(Rectangle((j, 0), 1, 1,
                               facecolor="white", edgecolor="black", linewidth=0.8))
        ax.text(j + 0.5, 0.5, frac_row_str[j],
                ha="center", va="center", fontsize=textsize,
                fontweight=frac_row_fontweight)
    ax.plot([0, ncols], [1, 1], color="black", linewidth=1.2)

    # ── data cells (offset down by one row for the header) ───────────────────
    for i in range(nrows):
        for j in range(ncols):
            val  = data[i, j]
            face = cm(norm(val)) if np.isfinite(val) else (0.9, 0.9, 0.9, 1)
            ax.add_patch(Rectangle((j, i + 1), 1, 1,
                                   facecolor=face, edgecolor="black", linewidth=0.5))
            if annotate and np.isfinite(val):
                ax.text(j + 0.5, i + 1.5, _fmtval(fmt, val),
                        ha="center", va="center", fontsize=textsize,
                        color=_text_color(face))

    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows_total)
    ax.invert_yaxis()
    ax.set_aspect("equal")

    ax.set_xticks(np.arange(ncols) + 0.5)
    ax.set_yticks(np.arange(nrows) + 1.5)
    ax.set_xticklabels(col_labels, rotation=45, ha="right",
                       rotation_mode="anchor", fontsize=textsize + addsize)
    if show_row_labels:
        ax.set_yticklabels(row_labels, fontsize=textsize + addsize)
    else:
        ax.set_yticklabels([])

    ax.tick_params(axis="y", length=4, width=1)
    ax.tick_params(axis="x", length=4, width=1)
    for spine in ax.spines.values():
        spine.set_visible(True)

    if title:
        ax.set_title(title, pad=title_pad, fontsize=textsize + 2 + addsize)

    if cbar:
        fig.canvas.draw()
        bbox = ax.get_position()
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cm)
        sm.set_array([])
        cax = fig.add_axes([bbox.x1 + gap, bbox.y0, cbar_width, bbox.height])
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label(cbar_label, fontsize=textsize + 2, rotation=90, labelpad=0)
        cb.ax.tick_params(labelsize=textsize + 2 + addsize)
    else:
        plt.tight_layout()

    fig.canvas.draw()

    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight", dpi=350, pad_inches=0.05)

    plt.show()
    plt.close(fig)

    return fig, ax


# ═════════════════════════════════════════════════════════════════════════════
# Merged two-panel figure: single_heatmap (left) + split_triangle (right)
# ═════════════════════════════════════════════════════════════════════════════

def merged_heatmaps(
    # ── Left panel data (single_heatmap) ─────────────────────────────────
    data_left,
    col_labels_left,
    title_left=None,
    cmap_left="viridis",
    norm_left=None,
    fmt_left="{:.1f}",
    sig_figs_left=None,        # significant figures for left panel (e.g. sig_figs_left=2)
    textsize_left=8,           # cell-number size in the left panel only
    cbar_label_left="(%)",
    labelpad_left = 0,
    annotate_left=True,

    # ── Right panel data (split_triangle_heatmap) ─────────────────────────
    data_lower=None,
    data_upper=None,
    col_labels_right=None,
    title_right=None,
    cmap_lower="viridis",
    cmap_upper="viridis",
    norm_lower=None,
    norm_upper=None,
    fmt_right="{:.1f}",
    sig_figs_right=None,       # significant figures for right panel (e.g. sig_figs_right=2)
    textsize_right=7.5,        # cell-number size in the right panel only
    cbar_label_lower="(%) (<1°)",
    cbar_label_upper="(%) (<20°)",
    cbar_upper_labelpad=1,
    annotate_right=True,
    text_lower_x=0.28,
    text_lower_y=0.72,

    # ── Shared font size (titles, tick labels, cbar labels, suptitle) ────
    fontsize=9,

    # ── Shared ────────────────────────────────────────────────────────────
    row_labels=None,
    suptitle=None,
    suptitle_y=1.01,
    suptitle_x=0.5,            # 0.5 = figure centre; adjust if panels are off-centre
    show_row_labels_left=True,   # right panel always hides row labels

    # ── Figure layout ─────────────────────────────────────────────────────
    figsize=(7.5, 5.5),
    dpi=350,
    bottom=0.025,
    top=0.92,
    left_margin=0.12,          # left edge of left axes (figure fraction)

    # ── Panel spacing ─────────────────────────────────────────────────────
    gap_between_panels=0.06,   # gap between left colorbar and right axes

    # ── Colorbar tuning ───────────────────────────────────────────────────
    cbar_width=0.015,          # width of every colorbar (figure fraction)
    gap_cbar_left=0.025,       # left panel: axes → colorbar gap
    gap_cbar_right=0.02,       # right panel: axes → first colorbar gap
    gap_between_cbars=0.12,    # right panel: cbar1 → cbar2 gap

    savepath=None,
):
    """
    Combine single_heatmap (left) and split_triangle_heatmap (right) into one
    figure, with full control over spacing, colorbar placement, and font sizes.

    Font sizes
    ----------
    fontsize       : controls everything except cell numbers — titles, tick
                     labels, colorbar labels, suptitle all use this value
    textsize_left  : size of the numbers printed inside left-panel cells
    textsize_right : size of the numbers printed inside right-panel cells

    Other parameters
    ----------------
    data_left              : 2-D array for the left (single) heatmap
    col_labels_left        : column labels for the left panel
    data_lower, data_upper : 2-D arrays for the right (split-triangle) panel
    col_labels_right       : column labels for the right panel
    row_labels             : shared row labels (same order for both panels)
    suptitle               : overall figure title (None = no title)
    show_row_labels_left   : show model names on the left panel y-axis
    left_margin            : figure-fraction x offset to the first axes
    gap_between_panels     : figure-fraction gap between left cbar and right axes
    gap_cbar_left/right    : axes → first colorbar gap for each panel
    gap_between_cbars      : gap between the two colorbars of the right panel
    cbar_width             : width of every colorbar in figure fraction

    Returns
    -------
    fig, (ax_left, ax_right)
    """
    if sig_figs_left  is not None:
        fmt_left  = _sigfig_formatter(sig_figs_left)
    if sig_figs_right is not None:
        fmt_right = _sigfig_formatter(sig_figs_right)
    data_left  = np.asarray(data_left,  float)
    data_lower = np.asarray(data_lower, float)
    data_upper = np.asarray(data_upper, float)
    assert data_lower.shape == data_upper.shape
    assert data_left.shape[0] == data_lower.shape[0], "Both panels must have the same number of rows"

    nrows        = data_left.shape[0]
    ncols_left   = data_left.shape[1]
    ncols_right  = data_lower.shape[1]

    # ── norms ────────────────────────────────────────────────────────────
    def _autonorm(arr, n):
        v = arr[np.isfinite(arr)]
        return n if n is not None else (Normalize(v.min(), v.max()) if v.size else Normalize(0, 1))

    norm_left   = _autonorm(data_left,   norm_left)
    norm_lower  = _autonorm(data_lower,  norm_lower)
    norm_upper  = _autonorm(data_upper,  norm_upper)

    cm_left   = mpl.cm.get_cmap(cmap_left)
    cm_lower  = mpl.cm.get_cmap(cmap_lower)
    cm_upper  = mpl.cm.get_cmap(cmap_upper)

    # ── axes geometry ─────────────────────────────────────────────────────
    # With set_aspect("equal") the rectangle must have width = h * ncols/nrows
    # (in figure-fraction space: scale by figsize[1]/figsize[0]).
    h       = top - bottom
    scale   = figsize[1] / figsize[0]
    w_left  = h * scale * ncols_left  / nrows
    w_right = h * scale * ncols_right / nrows

    x0_left     = left_margin
    x0_cbar_l   = x0_left  + w_left  + gap_cbar_left
    x0_right    = x0_cbar_l + cbar_width + gap_between_panels
    x0_cbar_r1  = x0_right + w_right + gap_cbar_right
    x0_cbar_r2  = x0_cbar_r1 + cbar_width + gap_between_cbars

    # ── create figure and axes ────────────────────────────────────────────
    fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=False)

    ax_left  = fig.add_axes([x0_left,    bottom, w_left,     h])
    ax_right = fig.add_axes([x0_right,   bottom, w_right,    h])
    cax_l    = fig.add_axes([x0_cbar_l,  bottom, cbar_width, h])
    cax_r1   = fig.add_axes([x0_cbar_r1, bottom, cbar_width, h])
    cax_r2   = fig.add_axes([x0_cbar_r2, bottom, cbar_width, h])

    # ── draw left panel ───────────────────────────────────────────────────
    for i in range(nrows):
        for j in range(ncols_left):
            val  = data_left[i, j]
            face = cm_left(norm_left(val)) if np.isfinite(val) else (0.9, 0.9, 0.9, 1)
            ax_left.add_patch(Rectangle((j, i), 1, 1,
                                        facecolor=face, edgecolor="black", linewidth=0.5))
            if annotate_left and np.isfinite(val):
                ax_left.text(j + 0.5, i + 0.5, _fmtval(fmt_left, val),
                             ha="center", va="center", fontsize=textsize_left,
                             color=_text_color(face))

    ax_left.set_xlim(0, ncols_left)
    ax_left.set_ylim(0, nrows)
    ax_left.invert_yaxis()
    ax_left.set_aspect("equal")
    ax_left.set_xticks(np.arange(ncols_left) + 0.5)
    ax_left.set_yticks(np.arange(nrows) + 0.5)
    ax_left.set_xticklabels(col_labels_left, rotation=45, ha="right",
                            rotation_mode="anchor", fontsize=fontsize)
    if show_row_labels_left and row_labels is not None:
        ax_left.set_yticklabels(row_labels, fontsize=fontsize)
    else:
        ax_left.set_yticklabels([])
    ax_left.tick_params(axis="both", length=4, width=1)
    for sp in ax_left.spines.values():
        sp.set_visible(True)
    if title_left:
        ax_left.set_title(title_left, pad=4, fontsize=fontsize)

    sm_l = mpl.cm.ScalarMappable(norm=norm_left, cmap=cm_left)
    sm_l.set_array([])
    cb_l = fig.colorbar(sm_l, cax=cax_l)
    cb_l.set_label(cbar_label_left, fontsize=fontsize, rotation=90, labelpad=labelpad_left)
    cb_l.ax.tick_params(labelsize=fontsize)

    # ── draw right panel ──────────────────────────────────────────────────
    for i in range(nrows):
        for j in range(ncols_right):
            x0c, x1c = j, j + 1
            y0c, y1c = i, i + 1
            tri_lo = np.array([[x1c, y0c], [x1c, y1c], [x0c, y0c]])
            tri_up = np.array([[x0c, y1c], [x0c, y0c], [x1c, y1c]])
            v_lo = data_lower[i, j]
            v_up = data_upper[i, j]
            face_lo = cm_lower(norm_lower(v_lo)) if np.isfinite(v_lo) else (0.9, 0.9, 0.9, 1)
            face_up = cm_upper(norm_upper(v_up)) if np.isfinite(v_up) else (0.9, 0.9, 0.9, 1)
            ax_right.add_patch(Polygon(tri_lo, closed=True,
                                       facecolor=face_lo, edgecolor="black", linewidth=0.5))
            ax_right.add_patch(Polygon(tri_up, closed=True,
                                       facecolor=face_up, edgecolor="black", linewidth=0.5))
            if annotate_right:
                if np.isfinite(v_lo):
                    ax_right.text(j + text_lower_x, i + text_lower_y, _fmtval(fmt_right, v_lo),
                                  ha="center", va="center", fontsize=textsize_right,
                                  color=_text_color(face_lo))
                if np.isfinite(v_up):
                    ax_right.text(j + 0.65, i + 0.28, _fmtval(fmt_right, v_up),
                                  ha="center", va="center", fontsize=textsize_right,
                                  color=_text_color(face_up))

    ax_right.set_xlim(0, ncols_right)
    ax_right.set_ylim(0, nrows)
    ax_right.invert_yaxis()
    ax_right.set_aspect("equal")
    ax_right.set_xticks(np.arange(ncols_right) + 0.5)
    ax_right.set_yticks(np.arange(nrows) + 0.5)
    ax_right.set_xticklabels(col_labels_right, rotation=45, ha="right",
                             rotation_mode="anchor", fontsize=fontsize)
    ax_right.set_yticklabels([])   # row labels shown on left panel only
    ax_right.tick_params(axis="both", length=4, width=1)
    for sp in ax_right.spines.values():
        sp.set_visible(True)
    if title_right:
        ax_right.set_title(title_right, pad=4, fontsize=fontsize)

    sm_lo = mpl.cm.ScalarMappable(norm=norm_lower, cmap=cm_lower)
    sm_up = mpl.cm.ScalarMappable(norm=norm_upper, cmap=cm_upper)
    sm_lo.set_array([])
    sm_up.set_array([])
    cb_r1 = fig.colorbar(sm_lo, cax=cax_r1)
    cb_r1.set_label(cbar_label_lower, fontsize=fontsize, rotation=90, labelpad=0)
    cb_r1.ax.tick_params(labelsize=fontsize)
    cb_r2 = fig.colorbar(sm_up, cax=cax_r2)
    cb_r2.set_label(cbar_label_upper, fontsize=fontsize, rotation=90, labelpad=cbar_upper_labelpad)
    cb_r2.ax.tick_params(labelsize=fontsize)

    # ── overall title ─────────────────────────────────────────────────────
    if suptitle:
        fig.suptitle(suptitle, fontsize=fontsize, x=suptitle_x, y=suptitle_y,
                     ha="center")

    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight", dpi=dpi, pad_inches=0.05)

    return fig, (ax_left, ax_right)


def merged_heatmaps_with_frac_row(
    # ── Left panel data (single_heatmap) ─────────────────────────────────
    data_left,
    col_labels_left,
    title_left=None,
    cmap_left="viridis",
    norm_left=None,
    fmt_left="{:.1f}",
    sig_figs_left=None,        # significant figures for left panel (e.g. sig_figs_left=2)
    textsize_left=8,           # cell-number size in the left panel only
    cbar_label_left="(%)",
    labelpad_left=0,
    annotate_left=True,

    # ── Right panel data (split_triangle_heatmap) ─────────────────────────
    data_lower=None,
    data_upper=None,
    col_labels_right=None,
    title_right=None,
    cmap_lower="viridis",
    cmap_upper="viridis",
    norm_lower=None,
    norm_upper=None,
    fmt_right="{:.1f}",
    sig_figs_right=None,       # significant figures for right panel (e.g. sig_figs_right=2)
    textsize_right=7.5,        # cell-number size in the right panel only
    cbar_label_lower="(%) (<1°)",
    cbar_label_upper="(%) (<20°)",
    cbar_upper_labelpad=1,
    annotate_right=True,
    text_lower_x=0.28,
    text_lower_y=0.72,

    # ── Header (fraction) row — shared across both panels ─────────────────
    frac_row_str=None,            # sequence aligned to columns; same values drawn above both panels
    frac_row_fontsize=None,        # None → falls back to `fontsize`
    frac_row_fontweight="bold",

    # ── Shared font size (titles, tick labels, cbar labels, suptitle) ────
    fontsize=9,

    # ── Shared ────────────────────────────────────────────────────────────
    row_labels=None,
    suptitle=None,
    suptitle_y=1.01,
    suptitle_x=0.5,            # 0.5 = figure centre; adjust if panels are off-centre
    show_row_labels_left=True,   # right panel always hides row labels

    # ── Figure layout ─────────────────────────────────────────────────────
    figsize=(7.5, 5.5),
    dpi=350,
    bottom=0.025,
    top=0.92,
    left_margin=0.12,          # left edge of left axes (figure fraction)

    # ── Panel spacing ─────────────────────────────────────────────────────
    gap_between_panels=0.06,   # gap between left colorbar and right axes

    # ── Colorbar tuning ───────────────────────────────────────────────────
    cbar_width=0.015,          # width of every colorbar (figure fraction)
    gap_cbar_left=0.025,       # left panel: axes → colorbar gap
    gap_cbar_right=0.02,       # right panel: axes → first colorbar gap
    gap_between_cbars=0.12,    # right panel: cbar1 → cbar2 gap

    savepath=None,
):
    """
    merged_heatmaps, plus one shared un-colored header row on top of BOTH
    panels showing pre-formatted text (e.g. what % of atoms each column is
    computed over) — the two-panel analogue of single_heatmap_with_frac_row.

    frac_row_str must have the same length as col_labels_left and
    col_labels_right (both panels share the same column meaning here, e.g.
    the same sweeping |F_DFT| windows).

    All other parameters match merged_heatmaps exactly.

    Returns
    -------
    fig, (ax_left, ax_right)
    """
    if sig_figs_left  is not None:
        fmt_left  = _sigfig_formatter(sig_figs_left)
    if sig_figs_right is not None:
        fmt_right = _sigfig_formatter(sig_figs_right)
    if frac_row_fontsize is None:
        frac_row_fontsize = fontsize

    data_left  = np.asarray(data_left,  float)
    data_lower = np.asarray(data_lower, float)
    data_upper = np.asarray(data_upper, float)
    assert data_lower.shape == data_upper.shape
    assert data_left.shape[0] == data_lower.shape[0], "Both panels must have the same number of rows"
    assert frac_row_str is not None, "frac_row_str is required"
    assert len(frac_row_str) == data_left.shape[1] == data_lower.shape[1], \
        "frac_row_str must match the (shared) column count of both panels"

    nrows        = data_left.shape[0]
    nrows_total  = nrows + 1
    ncols_left   = data_left.shape[1]
    ncols_right  = data_lower.shape[1]

    # ── norms ────────────────────────────────────────────────────────────
    def _autonorm(arr, n):
        v = arr[np.isfinite(arr)]
        return n if n is not None else (Normalize(v.min(), v.max()) if v.size else Normalize(0, 1))

    norm_left   = _autonorm(data_left,   norm_left)
    norm_lower  = _autonorm(data_lower,  norm_lower)
    norm_upper  = _autonorm(data_upper,  norm_upper)

    cm_left   = mpl.cm.get_cmap(cmap_left)
    cm_lower  = mpl.cm.get_cmap(cmap_lower)
    cm_upper  = mpl.cm.get_cmap(cmap_upper)

    # ── axes geometry ─────────────────────────────────────────────────────
    h       = top - bottom
    scale   = figsize[1] / figsize[0]
    w_left  = h * scale * ncols_left  / nrows_total
    w_right = h * scale * ncols_right / nrows_total

    x0_left     = left_margin
    x0_cbar_l   = x0_left  + w_left  + gap_cbar_left
    x0_right    = x0_cbar_l + cbar_width + gap_between_panels
    x0_cbar_r1  = x0_right + w_right + gap_cbar_right
    x0_cbar_r2  = x0_cbar_r1 + cbar_width + gap_between_cbars

    # ── create figure and axes ────────────────────────────────────────────
    fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=False)

    ax_left  = fig.add_axes([x0_left,    bottom, w_left,     h])
    ax_right = fig.add_axes([x0_right,   bottom, w_right,    h])
    cax_l    = fig.add_axes([x0_cbar_l,  bottom, cbar_width, h])
    cax_r1   = fig.add_axes([x0_cbar_r1, bottom, cbar_width, h])
    cax_r2   = fig.add_axes([x0_cbar_r2, bottom, cbar_width, h])

    # ── shared header row (row 0 of both panels) ──────────────────────────
    for ax, ncols in [(ax_left, ncols_left), (ax_right, ncols_right)]:
        for j in range(ncols):
            ax.add_patch(Rectangle((j, 0), 1, 1,
                                   facecolor="white", edgecolor="black", linewidth=0.8))
            ax.text(j + 0.5, 0.5, frac_row_str[j],
                    ha="center", va="center", fontsize=frac_row_fontsize,
                    fontweight=frac_row_fontweight)
        ax.plot([0, ncols], [1, 1], color="black", linewidth=1.2)

    # ── draw left panel (data rows offset down by one for the header) ────
    for i in range(nrows):
        for j in range(ncols_left):
            val  = data_left[i, j]
            face = cm_left(norm_left(val)) if np.isfinite(val) else (0.9, 0.9, 0.9, 1)
            ax_left.add_patch(Rectangle((j, i + 1), 1, 1,
                                        facecolor=face, edgecolor="black", linewidth=0.5))
            if annotate_left and np.isfinite(val):
                ax_left.text(j + 0.5, i + 1.5, _fmtval(fmt_left, val),
                             ha="center", va="center", fontsize=textsize_left,
                             color=_text_color(face))

    ax_left.set_xlim(0, ncols_left)
    ax_left.set_ylim(0, nrows_total)
    ax_left.invert_yaxis()
    ax_left.set_aspect("equal")
    ax_left.set_xticks(np.arange(ncols_left) + 0.5)
    ax_left.set_yticks(np.arange(nrows) + 1.5)
    ax_left.set_xticklabels(col_labels_left, rotation=45, ha="right",
                            rotation_mode="anchor", fontsize=fontsize)
    if show_row_labels_left and row_labels is not None:
        ax_left.set_yticklabels(row_labels, fontsize=fontsize)
    else:
        ax_left.set_yticklabels([])
    ax_left.tick_params(axis="both", length=4, width=1)
    for sp in ax_left.spines.values():
        sp.set_visible(True)
    if title_left:
        ax_left.set_title(title_left, pad=4, fontsize=fontsize)

    sm_l = mpl.cm.ScalarMappable(norm=norm_left, cmap=cm_left)
    sm_l.set_array([])
    cb_l = fig.colorbar(sm_l, cax=cax_l)
    cb_l.set_label(cbar_label_left, fontsize=fontsize, rotation=90, labelpad=labelpad_left)
    cb_l.ax.tick_params(labelsize=fontsize)

    # ── draw right panel (data rows offset down by one for the header) ───
    for i in range(nrows):
        for j in range(ncols_right):
            x0c, x1c = j, j + 1
            y0c, y1c = i + 1, i + 2
            tri_lo = np.array([[x1c, y0c], [x1c, y1c], [x0c, y0c]])
            tri_up = np.array([[x0c, y1c], [x0c, y0c], [x1c, y1c]])
            v_lo = data_lower[i, j]
            v_up = data_upper[i, j]
            face_lo = cm_lower(norm_lower(v_lo)) if np.isfinite(v_lo) else (0.9, 0.9, 0.9, 1)
            face_up = cm_upper(norm_upper(v_up)) if np.isfinite(v_up) else (0.9, 0.9, 0.9, 1)
            ax_right.add_patch(Polygon(tri_lo, closed=True,
                                       facecolor=face_lo, edgecolor="black", linewidth=0.5))
            ax_right.add_patch(Polygon(tri_up, closed=True,
                                       facecolor=face_up, edgecolor="black", linewidth=0.5))
            if annotate_right:
                if np.isfinite(v_lo):
                    ax_right.text(j + text_lower_x, i + 1 + text_lower_y, _fmtval(fmt_right, v_lo),
                                  ha="center", va="center", fontsize=textsize_right,
                                  color=_text_color(face_lo))
                if np.isfinite(v_up):
                    ax_right.text(j + 0.65, i + 1 + 0.28, _fmtval(fmt_right, v_up),
                                  ha="center", va="center", fontsize=textsize_right,
                                  color=_text_color(face_up))

    ax_right.set_xlim(0, ncols_right)
    ax_right.set_ylim(0, nrows_total)
    ax_right.invert_yaxis()
    ax_right.set_aspect("equal")
    ax_right.set_xticks(np.arange(ncols_right) + 0.5)
    ax_right.set_yticks(np.arange(nrows) + 1.5)
    ax_right.set_xticklabels(col_labels_right, rotation=45, ha="right",
                             rotation_mode="anchor", fontsize=fontsize)
    ax_right.set_yticklabels([])   # row labels shown on left panel only
    ax_right.tick_params(axis="both", length=4, width=1)
    for sp in ax_right.spines.values():
        sp.set_visible(True)
    if title_right:
        ax_right.set_title(title_right, pad=4, fontsize=fontsize)

    sm_lo = mpl.cm.ScalarMappable(norm=norm_lower, cmap=cm_lower)
    sm_up = mpl.cm.ScalarMappable(norm=norm_upper, cmap=cm_upper)
    sm_lo.set_array([])
    sm_up.set_array([])
    cb_r1 = fig.colorbar(sm_lo, cax=cax_r1)
    cb_r1.set_label(cbar_label_lower, fontsize=fontsize, rotation=90, labelpad=0)
    cb_r1.ax.tick_params(labelsize=fontsize)
    cb_r2 = fig.colorbar(sm_up, cax=cax_r2)
    cb_r2.set_label(cbar_label_upper, fontsize=fontsize, rotation=90, labelpad=cbar_upper_labelpad)
    cb_r2.ax.tick_params(labelsize=fontsize)

    # ── overall title ─────────────────────────────────────────────────────
    if suptitle:
        fig.suptitle(suptitle, fontsize=fontsize, x=suptitle_x, y=suptitle_y,
                     ha="center")

    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight", dpi=dpi, pad_inches=0.05)

    return fig, (ax_left, ax_right)


# ═════════════════════════════════════════════════════════════════════════════
# F_DFT fraction and conditioned MAE/RMSE
# ═════════════════════════════════════════════════════════════════════════════

def build_fdft_distribution_fraction_table(all_results, fdft_thresholds):
    """
    Fraction (%) of atoms with |F_DFT| > threshold.

    No fdf_min filtering — the whole atom pool is used as denominator
    so the numbers reflect the actual force-magnitude distribution.

    Parameters
    ----------
    all_results      : dict[model → dict]  key: "F_dft"
    fdft_thresholds  : list of float – |F_DFT| cut-offs (eV/Å)

    Returns
    -------
    pd.DataFrame – rows = models, columns = thresholds, values = fraction (%)
    """
    rows = {}
    for model, data in all_results.items():
        F_dft = np.abs(np.asarray(data.get("F_dft", data.get("all_F_dft_mags")), float))
        valid = np.isfinite(F_dft)
        F_f   = F_dft[valid]
        rows[model] = {
            thr: (round(np.mean(F_f > thr) * 100, 2) if F_f.size > 0 else np.nan)
            for thr in fdft_thresholds
        }
    df = pd.DataFrame(rows).T
    df.columns = fdft_thresholds
    return df


def build_dF_mae_rmse_fdft_subset(all_results, fdft_thresholds):
    """
    MAE and RMSE of Δ|F| for atoms with |F_DFT| > threshold.

    Conditioning on larger DFT forces reveals how prediction error
    scales with force magnitude.

    Parameters
    ----------
    all_results     : dict[model → dict]  keys: "F_dft", "deltaF"
    fdft_thresholds : list of float – |F_DFT| cut-offs (eV/Å)

    Returns
    -------
    (mae_df, rmse_df) : two pd.DataFrames
        rows = models, columns = thresholds, values in eV/Å
    """
    mae_rows, rmse_rows = {}, {}

    for model, data in all_results.items():
        F_dft = np.abs(np.asarray(data.get("F_dft", data.get("all_F_dft_mags")),  float))
        dF    = np.asarray(data.get("deltaF", data.get("all_deltaF")), float)
        n     = min(len(F_dft), len(dF))
        F_dft, dF = F_dft[:n], dF[:n]

        valid = np.isfinite(F_dft) & np.isfinite(dF)
        F_f   = F_dft[valid]
        dF_f  = np.abs(dF[valid])

        mae_rows[model], rmse_rows[model] = {}, {}
        for thr in fdft_thresholds:
            sub = dF_f if thr == 0 else dF_f[F_f > thr]
            if sub.size > 0:
                mae_rows[model][thr]  = round(float(np.mean(sub)), 4)
                rmse_rows[model][thr] = round(float(np.sqrt(np.mean(sub ** 2))), 4)
            else:
                mae_rows[model][thr]  = np.nan
                rmse_rows[model][thr] = np.nan

    mae_df  = pd.DataFrame(mae_rows).T;  mae_df.columns  = fdft_thresholds
    rmse_df = pd.DataFrame(rmse_rows).T; rmse_df.columns = fdft_thresholds
    return mae_df, rmse_df


def build_highly_accurate_force_fraction_table_fdft_windows(all_results, dF_cut, fdft_thresholds, fdf_min=0.01):
    """
    Fraction (%) of atoms with Δ|F| < dF_cut, computed WITHIN each near-
    equilibrium window fdf_min < |F_DFT| < threshold (threshold == 0 disables
    the upper cap, i.e. "all atoms" with |F_DFT| > fdf_min).

    Each column is conditioned independently — i.e. the denominator is the
    atom count inside that specific window, not the full dataset.

    Parameters
    ----------
    all_results     : dict[model → dict]  keys: "F_dft", "deltaF"
    dF_cut          : float – Δ|F| threshold defining the fraction (eV/Å)
    fdft_thresholds : list of float – |F_DFT| upper cut-offs (eV/Å); 0 = no cap
    fdf_min         : float – lower |F_DFT| floor applied to every column

    Returns
    -------
    pd.DataFrame – rows = models, columns = thresholds, values = fraction (%)
    """
    rows = {}
    for model, data in all_results.items():
        F_dft = np.abs(np.asarray(data.get("F_dft", data.get("all_F_dft_mags")), float))
        dF    = np.asarray(data.get("deltaF", data.get("all_deltaF")), float)
        n     = min(len(F_dft), len(dF))
        F_dft, dF = F_dft[:n], dF[:n]

        valid = np.isfinite(F_dft) & np.isfinite(dF) & (F_dft > fdf_min)
        F_f   = F_dft[valid]
        dF_f  = np.abs(dF[valid])

        rows[model] = {}
        for thr in fdft_thresholds:
            sub = dF_f if thr == 0 else dF_f[F_f < thr]
            rows[model][thr] = (
                round(float(np.mean(sub < dF_cut) * 100), 2) if sub.size > 0 else np.nan
            )
    df = pd.DataFrame(rows).T
    df.columns = fdft_thresholds
    return df


def build_joint_dF_theta_accuracy_table_fdft_windows(all_results, dF_cut, angle_cuts, fdft_thresholds, fdf_min=0.01):
    """
    Fraction (%) of atoms with Δ|F| < dF_cut AND Δθ < angle_cut, computed
    WITHIN each near-equilibrium window fdf_min < |F_DFT| < threshold
    (threshold == 0 disables the upper cap).

    Joint-condition companion to build_highly_accurate_force_fraction_table_fdft_windows —
    same sweeping |F_DFT| windows, but the fraction requires both the
    force-error AND angle-error cuts simultaneously. Reuses frac_percent's
    fdf_max support.

    Parameters
    ----------
    all_results     : dict[model → dict]  keys: "F_dft", "deltaF", "deltaTheta"
    dF_cut          : float – Δ|F| threshold (eV/Å)
    angle_cuts      : list of float – Δθ thresholds (degrees), e.g. [1, 20]
    fdft_thresholds : list of float – |F_DFT| upper cut-offs (eV/Å); 0 = no cap
    fdf_min         : float – lower |F_DFT| floor applied to every column

    Returns
    -------
    dict[angle_cut → pd.DataFrame]
        Rows = models, columns = fdft_thresholds, values = fraction (%).
    """
    tables = {}
    for angle_cut in angle_cuts:
        rows = {}
        for model, data in all_results.items():
            F_dft  = data.get("F_dft", data.get("all_F_dft_mags"))
            dF     = data.get("deltaF", data.get("all_deltaF"))
            dtheta = data.get("deltaTheta", data.get("all_deltaTheta"))
            rows[model] = {
                thr: frac_percent(
                    dF, dtheta, F_dft, dF_cut, angle_cut,
                    fdf_min=fdf_min, fdf_max=(None if thr == 0 else thr),
                )
                for thr in fdft_thresholds
            }
        df = pd.DataFrame(rows).T
        df.columns = fdft_thresholds
        tables[angle_cut] = df
    return tables


def build_fdft_distribution_fraction_table_windows(all_results, fdft_thresholds, fdf_min=0.01):
    """
    Fraction (%) of atoms — out of all atoms with |F_DFT| > fdf_min — that
    fall inside each near-equilibrium window fdf_min < |F_DFT| < threshold
    (threshold == 0 disables the cap, i.e. 100% of the pool by definition).

    This is the population weight of each window; use it as the header row
    above build_highly_accurate_force_fraction_table_fdft_windows to show what fraction of the
    dataset each column's percentage is computed over.

    Mirror of build_fdft_distribution_fraction_table, sweeping an upper bound instead of a
    lower one.

    Parameters
    ----------
    all_results     : dict[model → dict]  key: "F_dft"
    fdft_thresholds : list of float – |F_DFT| upper cut-offs (eV/Å); 0 = no cap
    fdf_min         : float – lower |F_DFT| floor defining the atom pool

    Returns
    -------
    pd.DataFrame – rows = models, columns = thresholds, values = fraction (%)
    """
    rows = {}
    for model, data in all_results.items():
        F_dft = np.abs(np.asarray(data.get("F_dft", data.get("all_F_dft_mags")), float))
        valid = np.isfinite(F_dft) & (F_dft > fdf_min)
        F_f   = F_dft[valid]
        rows[model] = {
            thr: (100.0 if thr == 0 else round(float(np.mean(F_f < thr) * 100), 2))
            if F_f.size > 0 else np.nan
            for thr in fdft_thresholds
        }
    df = pd.DataFrame(rows).T
    df.columns = fdft_thresholds
    return df


def build_theta_mae_rmse_fdft_subset(all_results, fdft_thresholds):
    """
    mean and RMS of Δθ for atoms with |F_DFT| > threshold.

    Mirrors build_dF_mae_rmse_fdft_subset but for angle errors.

    Parameters
    ----------
    all_results     : dict[model → dict]  keys: "F_dft", "deltaTheta"
    fdft_thresholds : list of float – |F_DFT| cut-offs (eV/Å)

    Returns
    -------
    (mae_df, rmse_df) : two pd.DataFrames
        rows = models, columns = thresholds, values in degrees
    """
    mae_rows, rmse_rows = {}, {}

    for model, data in all_results.items():
        F_dft  = np.abs(np.asarray(data.get("F_dft", data.get("all_F_dft_mags")), float))
        dtheta = np.abs(np.asarray(data.get("deltaTheta", data.get("all_deltaTheta")), float))
        n      = min(len(F_dft), len(dtheta))
        F_dft, dtheta = F_dft[:n], dtheta[:n]

        valid  = np.isfinite(F_dft) & np.isfinite(dtheta)
        F_f    = F_dft[valid]
        dth_f  = dtheta[valid]

        mae_rows[model], rmse_rows[model] = {}, {}
        for thr in fdft_thresholds:
            sub = dth_f[F_f > thr]
            if sub.size > 0:
                mae_rows[model][thr]  = round(float(np.mean(sub)), 4)
                rmse_rows[model][thr] = round(float(np.sqrt(np.mean(sub ** 2))), 4)
            else:
                mae_rows[model][thr]  = np.nan
                rmse_rows[model][thr] = np.nan

    mae_df  = pd.DataFrame(mae_rows).T;  mae_df.columns  = fdft_thresholds
    rmse_df = pd.DataFrame(rmse_rows).T; rmse_df.columns = fdft_thresholds
    return mae_df, rmse_df


def build_dF_mae_rmse_smalldF_subset(all_results, dF_thresholds, fdft_min=0.01):
    """
    MAE and RMSE of Δ|F| for atoms with Δ|F| < threshold AND |F_DFT| > fdft_min.

    Shows how well the FP captures the "easy" (small-error) regime:
    as the threshold rises, more atoms are included and MAE/RMSE grows.

    Parameters
    ----------
    all_results    : dict[model → dict]  keys: "F_dft"/"all_F_dft_mags", "deltaF"/"all_deltaF"
    dF_thresholds  : list of float – upper Δ|F| bounds (eV/Å)
    fdft_min       : float – minimum |F_DFT| for inclusion (eV/Å)

    Returns
    -------
    (mae_df, rmse_df) : two pd.DataFrames
        rows = models, columns = thresholds, values in eV/Å
    """
    mae_rows, rmse_rows = {}, {}

    for model, data in all_results.items():
        F_dft = np.abs(np.asarray(data.get("F_dft", data.get("all_F_dft_mags")), float))
        dF    = np.abs(np.asarray(data.get("deltaF", data.get("all_deltaF")),     float))
        n     = min(len(F_dft), len(dF))
        F_dft, dF = F_dft[:n], dF[:n]

        valid = np.isfinite(F_dft) & np.isfinite(dF) & (F_dft > fdft_min)
        dF_f  = dF[valid]

        mae_rows[model], rmse_rows[model] = {}, {}
        for thr in dF_thresholds:
            sub = dF_f[dF_f < thr]
            if sub.size > 0:
                mae_rows[model][thr]  = round(float(np.mean(sub)), 4)
                rmse_rows[model][thr] = round(float(np.sqrt(np.mean(sub ** 2))), 4)
            else:
                mae_rows[model][thr]  = np.nan
                rmse_rows[model][thr] = np.nan

    mae_df  = pd.DataFrame(mae_rows).T;  mae_df.columns  = dF_thresholds
    rmse_df = pd.DataFrame(rmse_rows).T; rmse_df.columns = dF_thresholds
    return mae_df, rmse_df


# ═════════════════════════════════════════════════════════════════════════════
# Δθ fraction and conditioned MAE/RMSE
# ═════════════════════════════════════════════════════════════════════════════

def build_angle_accuracy_fraction_table(all_results, theta_thresholds, fdf_min=0.01):
    """
    Fraction (%) of atoms with Δθ < threshold.

    Only atoms with |F_DFT| > fdf_min are included (angular errors are
    ill-conditioned for near-zero DFT forces).

    Parameters
    ----------
    all_results       : dict[model → dict]  keys: "F_dft", "deltaTheta"
    theta_thresholds  : list of float – angular cut-offs (degrees)
    fdf_min           : float – minimum |F_DFT| for inclusion (eV/Å)

    Returns
    -------
    pd.DataFrame – rows = models, columns = thresholds, values = fraction (%)
    """
    rows = {}
    for model, data in all_results.items():
        F_dft  = np.abs(np.asarray(data.get("F_dft", data.get("all_F_dft_mags")),      float))
        dtheta = np.abs(np.asarray(data.get("deltaTheta", data.get("all_deltaTheta")), float))
        n      = min(len(F_dft), len(dtheta))
        F_dft, dtheta = F_dft[:n], dtheta[:n]

        valid  = np.isfinite(F_dft) & np.isfinite(dtheta) & (F_dft > fdf_min)
        dth_f  = dtheta[valid]

        rows[model] = {
            thr: (round(np.mean(dth_f < thr) * 100, 2) if dth_f.size > 0 else np.nan)
            for thr in theta_thresholds
        }
    df = pd.DataFrame(rows).T
    df.columns = theta_thresholds
    return df


def build_theta_mae_rmse_angle_subset(all_results, theta_thresholds, fdf_min=0.01):
    """
    mean and RMS of Δθ for atoms with Δθ < threshold.

    As the threshold increases, more atoms (those with larger angular errors)
    are included, so MAE/RMSE grows monotonically.

    Parameters
    ----------
    all_results      : dict[model → dict]  keys: "F_dft", "deltaTheta"
    theta_thresholds : list of float – upper angular bounds (degrees)
    fdf_min          : float – minimum |F_DFT| for inclusion (eV/Å)

    Returns
    -------
    (mae_df, rmse_df) : two pd.DataFrames
        rows = models, columns = thresholds, values in degrees
    """
    mae_rows, rmse_rows = {}, {}

    for model, data in all_results.items():
        F_dft  = np.abs(np.asarray(data.get("F_dft", data.get("all_F_dft_mags")),      float))
        dtheta = np.abs(np.asarray(data.get("deltaTheta", data.get("all_deltaTheta")), float))
        n      = min(len(F_dft), len(dtheta))
        F_dft, dtheta = F_dft[:n], dtheta[:n]

        valid  = np.isfinite(F_dft) & np.isfinite(dtheta) & (F_dft > fdf_min)
        dth_f  = dtheta[valid]

        mae_rows[model], rmse_rows[model] = {}, {}
        for thr in theta_thresholds:
            sub = dth_f[dth_f < thr]
            if sub.size > 0:
                mae_rows[model][thr]  = round(float(np.mean(sub)), 4)
                rmse_rows[model][thr] = round(float(np.sqrt(np.mean(sub ** 2))), 4)
            else:
                mae_rows[model][thr]  = np.nan
                rmse_rows[model][thr] = np.nan

    mae_df  = pd.DataFrame(mae_rows).T;  mae_df.columns  = theta_thresholds
    rmse_df = pd.DataFrame(rmse_rows).T; rmse_df.columns = theta_thresholds
    return mae_df, rmse_df


# ═════════════════════════════════════════════════════════════════════════════
# Force-vector error (e_vec) — counterparts of the force-magnitude-error
# analyses above, keyed on "e_vec" instead of "deltaF". e_vec = ||F_FP - F_DFT||
# is already non-negative, so no np.abs() is needed anywhere below.
# ═════════════════════════════════════════════════════════════════════════════

def build_evec_high_accuracy_fraction_table(all_results, evec_thresholds, fdf_min=0.01, fdf_max=None):
    """
    Fraction (%) of atoms with the force-vector error e_vec < threshold.

    Force-vector-error counterpart of build_highly_accurate_force_fraction_table.

    Parameters
    ----------
    all_results     : dict[model → dict]  keys: "F_dft", "e_vec"
    evec_thresholds : list of float – e_vec cut-offs (eV/Å)
    fdf_min         : float
    fdf_max         : float or None – maximum |F_dft| for inclusion (None = no cap)

    Returns
    -------
    pd.DataFrame – rows = models, columns = thresholds, values = fraction (%)
    """
    rows = {}
    for model, data in all_results.items():
        F_dft = np.abs(np.asarray(data["F_dft"], float))
        evec  = np.asarray(data["e_vec"], float)

        mask = np.isfinite(F_dft) & np.isfinite(evec) & (F_dft > fdf_min)
        if fdf_max is not None:
            mask &= F_dft < fdf_max
        evec_f = evec[mask]

        rows[model] = {
            thr: (np.round(np.mean(evec_f < thr) * 100, 2) if evec_f.size > 0 else np.nan)
            for thr in evec_thresholds
        }
    df = pd.DataFrame(rows).T
    df.columns = evec_thresholds
    return df


def build_large_evec_fraction_table(all_results, evec_thresholds, fdf_min=0.01):
    """
    Fraction (%) of atoms with the force-vector error e_vec **greater than** threshold.

    Force-vector-error counterpart of build_large_force_error_fraction_table;
    identifies the tail of large-force-vector-error atoms.

    Parameters
    ----------
    all_results     : dict[model → dict]  keys: "F_dft", "e_vec"
    evec_thresholds : list of float – e_vec cut-offs (eV/Å)
    fdf_min         : float – minimum |F_dft| for inclusion

    Returns
    -------
    pd.DataFrame – rows = models, columns = thresholds, values = fraction (%)
    """
    rows = {}
    for model, data in all_results.items():
        F_dft = np.abs(np.asarray(data["F_dft"], float))
        evec  = np.asarray(data["e_vec"], float)

        mask = np.isfinite(F_dft) & np.isfinite(evec) & (F_dft > fdf_min)
        evec_f = evec[mask]

        rows[model] = {
            thr: (np.round(np.mean(evec_f > thr) * 100, 4) if evec_f.size > 0 else np.nan)
            for thr in evec_thresholds
        }
    df = pd.DataFrame(rows).T
    df.columns = evec_thresholds
    return df


def build_evec_mae_rmse_fdft_subset(all_results, fdft_thresholds):
    """
    MAE and RMSE of the force-vector error e_vec for atoms with |F_DFT| > threshold.

    Force-vector-error counterpart of build_dF_mae_rmse_fdft_subset.

    Parameters
    ----------
    all_results     : dict[model → dict]  keys: "F_dft", "e_vec"
    fdft_thresholds : list of float – |F_DFT| cut-offs (eV/Å); 0 = all atoms

    Returns
    -------
    (mae_df, rmse_df) : two pd.DataFrames
        rows = models, columns = thresholds, values in eV/Å
    """
    mae_rows, rmse_rows = {}, {}

    for model, data in all_results.items():
        F_dft = np.abs(np.asarray(data["F_dft"], float))
        evec  = np.asarray(data["e_vec"], float)
        n     = min(len(F_dft), len(evec))
        F_dft, evec = F_dft[:n], evec[:n]

        valid  = np.isfinite(F_dft) & np.isfinite(evec)
        F_f    = F_dft[valid]
        evec_f = evec[valid]

        mae_rows[model], rmse_rows[model] = {}, {}
        for thr in fdft_thresholds:
            sub = evec_f if thr == 0 else evec_f[F_f > thr]
            if sub.size > 0:
                mae_rows[model][thr]  = round(float(np.mean(sub)), 4)
                rmse_rows[model][thr] = round(float(np.sqrt(np.mean(sub ** 2))), 4)
            else:
                mae_rows[model][thr]  = np.nan
                rmse_rows[model][thr] = np.nan

    mae_df  = pd.DataFrame(mae_rows).T;  mae_df.columns  = fdft_thresholds
    rmse_df = pd.DataFrame(rmse_rows).T; rmse_df.columns = fdft_thresholds
    return mae_df, rmse_df


def build_theta_far_from_equilibrium_regime_panels(all_results, theta_thresh, threshold=1.0, fdf_min=0.01):
    """
    Split atoms into two |F_DFT| regimes and compute Δθ fraction tables.

    Panel A — non-FE (near-equilibrium)    : |F_DFT| <= threshold
    Panel B — FE (far-from-equilibrium)    : |F_DFT| > threshold

    Each row contains:
      "N atoms"               — atom count in this regime
      "Frac of all atoms (%)" — share of the total valid pool
      "Δθ < {thr} (%)"     — fraction passing each angular threshold

    Parameters
    ----------
    all_results  : dict[model → dict]  keys: "F_dft", "deltaTheta"
    theta_thresh : list of float – angular thresholds (degrees)
    threshold    : float – |F_DFT| boundary between panels (eV/Å)
    fdf_min      : float – minimum |F_DFT| for inclusion (eV/Å)

    Returns
    -------
    (df_panel_A, df_panel_B) : two pd.DataFrames with rows = models
    """
    panel_A, panel_B = {}, {}

    for model, data in all_results.items():
        F_dft  = np.abs(np.asarray(data.get("F_dft", data.get("all_F_dft_mags")),      float))
        dtheta = np.abs(np.asarray(data.get("deltaTheta", data.get("all_deltaTheta")), float))
        n      = min(len(F_dft), len(dtheta))
        F_dft, dtheta = F_dft[:n], dtheta[:n]

        valid = np.isfinite(F_dft) & np.isfinite(dtheta)
        if fdf_min > 0:
            valid &= F_dft > fdf_min

        F_abs  = F_dft[valid]
        dth    = dtheta[valid]
        total  = F_abs.size

        for panel_dict, mask in [
            (panel_A, F_abs <= threshold),
            (panel_B, F_abs >  threshold),
        ]:
            n_regime = int(mask.sum())
            row = {
                "N atoms":               n_regime,
                "Frac of all atoms (%)": round(100 * n_regime / total, 4) if total > 0 else np.nan,
            }
            for thr in theta_thresh:
                row[f"Δθ < {thr}° (%)"] = (
                    round(100 * np.mean(dth[mask] < thr), 4) if n_regime > 0 else np.nan
                )
            panel_dict[model] = row

    return pd.DataFrame(panel_A).T, pd.DataFrame(panel_B).T


# ═════════════════════════════════════════════════════════════════════════════
# Additional plot functions
# ═════════════════════════════════════════════════════════════════════════════

def heatmap_fraction_delta_theta(
    df_frac,
    title=None,
    title_pad=4,
    cmap="viridis",
    fmt="{:.1f}",
    sig_figs=None,             # significant figures: overrides fmt (e.g. sig_figs=2 → 2 sig figs)
    textsize=8,
    figsize=(4.5, 5.5),
    cbar_width=0.025,   # width of the vertical colorbar (figure-fraction units)
    cbar_pad=0.01,      # gap between the table right edge and the colorbar
    col_labels=None,    # override auto-generated column labels; None → use "< X°" format
    savepath=None,
):
    """
    imshow-based heatmap for Δθ fraction data.

    Rows = models, columns = angular thresholds (degrees).
    Cell colour encodes the fraction (%) of atoms with Δθ < threshold.

    Parameters
    ----------
    df_frac   : pd.DataFrame – output of build_angle_accuracy_fraction_table
    title     : str or None
    cmap      : str – Matplotlib colormap name
    fmt       : str – annotation format string
    textsize  : int – base font size
    figsize   : (w, h) in inches
    savepath  : str or None

    Returns
    -------
    fig, ax
    """
    if sig_figs is not None:
        fmt = _sigfig_formatter(sig_figs)
    data       = df_frac.values.astype(float)
    row_labels = df_frac.index.tolist()
    nrows, ncols = data.shape

    v    = data[np.isfinite(data)]
    norm = Normalize(vmin=v.min(), vmax=v.max()) if v.size else Normalize(0, 100)
    cm   = mpl.cm.get_cmap(cmap)

    fig, ax = plt.subplots(figsize=figsize, dpi=350)
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="equal")

    for i in range(nrows):
        for j in range(ncols):
            val = data[i, j]
            if np.isfinite(val):
                ax.text(j, i, _fmtval(fmt, val),
                        ha="center", va="center", fontsize=textsize,
                        color=_text_color(cm(norm(val))))

    ax.set_xticks(np.arange(ncols))
    _col_labels = col_labels if col_labels is not None else [f"< {c}°" for c in df_frac.columns]
    ax.set_xticklabels(
        _col_labels,
        rotation=45, ha="right", rotation_mode="anchor", fontsize=textsize + 1,
    )
    ax.set_yticks(np.arange(nrows))
    ax.set_yticklabels(row_labels, fontsize=textsize + 1)

    # cell borders (imshow has no per-cell patches, so draw them via minor gridlines)
    ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    if title:
        ax.set_title(title, fontsize=textsize + 2, pad=title_pad)

    # Let tight_layout settle the axes position first (aspect="equal" shrinks it),
    # then place the colorbar manually at the exact axes height.
    plt.tight_layout()
    fig.canvas.draw()
    ax_pos = ax.get_position()

    cax = fig.add_axes([ax_pos.x1 + cbar_pad, ax_pos.y0, cbar_width, ax_pos.height])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("(%)", fontsize=textsize + 1, rotation=90)
    cb.ax.tick_params(labelsize=textsize)

    fig.canvas.draw()

    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight", dpi=350, pad_inches=0.05)

    plt.show()
    plt.close(fig)
    return fig, ax


def triangular_heatmap_with_fraction_row_word_style(
    mae_df, rmse_df, frac_row_str,
    col_labels=None,
    title=r"(MAE / RMSE) $\Delta F$ (eV/Å)",
    cmap_mae="Blues",
    cmap_rmse="Reds",
    figsize=(6.3, 8),
    fmt_mae="{:.2f}",
    fmt_rmse="{:.2f}",
    sig_figs=None,             # significant figures: sets both fmt_mae and fmt_rmse (e.g. sig_figs=2)
    text_size=12,
    # ── horizontal colorbars below (default) ─────────────────────────────
    cbar_height=0.025,       # thickness of each colorbar (figure-fraction units)
    cbar_gap=0.045,          # gap between table bottom and first (MAE) colorbar
    cbar_between_gap=0.045,  # gap between MAE colorbar and RMSE colorbar
    bottom=0.32,             # figure bottom margin (must fit both bars + gaps)
    # ── vertical colorbars on the right (active when cbar_side=True) ─────
    cbar_side=False,
    cbar_width_right=0.015,       # width of each vertical colorbar (figure-fraction)
    cbar_gap_right=0.02,          # horizontal gap: axes right edge → first colorbar
    cbar_between_gap_right=0.08,  # horizontal gap: first colorbar → second colorbar
    right=0.82,                   # subplots_adjust right margin (leave room for cbars)
    bottom_side=0.05,             # subplots_adjust bottom when cbar_side=True
    xlabel="thresholds",        # x-axis label; set to "" to hide it
    xlabel_pad=14,
    rmse_text_x_nudge=0.0,     # shift RMSE text left(-) / right(+) within cell
    rmse_text_y_nudge=0.0,     # shift RMSE text down(-) / up(+) within cell
    cbar_labelpad_mae=4,        # padding between MAE colorbar and its label
    cbar_labelpad_rmse=4,       # padding between RMSE colorbar and its label
    savepath=None,
):
    """
    Triangular heatmap with a fraction header row and colorbars.

    By default colorbars are placed horizontally below the axes.
    Set ``cbar_side=True`` to place them vertically on the right instead.

    Layout
    ------
    Top row   : pre-formatted fraction strings (white cells, word-style labels)
    Data rows : diagonal-split MAE / RMSE cells

    Spacing controls — horizontal mode (cbar_side=False)
    ----------------
    cbar_height     : thickness of each colorbar bar (figure-fraction units)
    cbar_gap        : vertical gap between the table bottom and the MAE bar
    cbar_between_gap: vertical gap between the MAE bar and the RMSE bar
    bottom          : figure bottom margin reserved for the colorbars
                      (rule of thumb: ≥ cbar_gap + cbar_height + cbar_between_gap + cbar_height + 0.05)

    Spacing controls — vertical mode (cbar_side=True)
    ----------------
    cbar_width_right      : width of each vertical colorbar (figure-fraction)
    cbar_gap_right        : horizontal gap from axes right edge to first colorbar
    cbar_between_gap_right: horizontal gap between the two colorbars
    right                 : subplots_adjust right (e.g. 0.82 leaves 18% for cbars)
    bottom_side           : subplots_adjust bottom (can be small, e.g. 0.05)

    Parameters
    ----------
    mae_df, rmse_df : pd.DataFrame – rows = models, columns = thresholds
    frac_row_str    : pd.Series    – pre-formatted strings indexed by threshold
    col_labels      : list or None – override column header labels
    title           : str
    cmap_mae, cmap_rmse : str – Matplotlib colormap names
    figsize         : (w, h) in inches
    fmt_mae, fmt_rmse : str – format strings for cell annotations
    text_size       : int  – base font size
    cbar_side       : bool – False = colorbars below (horizontal), True = right (vertical)
    savepath        : str or None

    Returns
    -------
    fig, ax
    """
    if sig_figs is not None:
        fmt_mae = fmt_rmse = _sigfig_formatter(sig_figs)
    from heatmap_table import (
        draw_rectangular_row, draw_triangular_column,
        setup_frame, setup_ticks_and_labels,
    )

    mae_df, rmse_df = mae_df.align(rmse_df, join="inner", axis=1)
    frac_row_str    = frac_row_str.reindex(mae_df.columns)

    nrows, ncols  = mae_df.shape
    nrows_total   = nrows + 1

    mae_vals  = mae_df.to_numpy(float)
    rmse_vals = rmse_df.to_numpy(float)

    if col_labels is None:
        col_labels = list(mae_df.columns)

    mae_vmin,  mae_vmax  = np.nanpercentile(mae_vals,  [ 5, 95])
    rmse_vmin, rmse_vmax = np.nanpercentile(rmse_vals, [15, 95])

    mae_norm  = plt.Normalize(mae_vmin,  mae_vmax)
    rmse_norm = plt.Normalize(rmse_vmin, rmse_vmax)
    mae_cmap  = mpl.cm.get_cmap(cmap_mae)
    rmse_cmap = mpl.cm.get_cmap(cmap_rmse)

    fig, ax = plt.subplots(figsize=figsize, dpi=350)
    if cbar_side:
        plt.subplots_adjust(bottom=bottom_side, right=right)
    else:
        plt.subplots_adjust(bottom=bottom)

    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows_total)
    ax.set_aspect("equal")

    # ── fraction header row ────────────────────────────────────────────────
    draw_rectangular_row(ax, row_y=nrows, ncols=ncols,
                         vals=frac_row_str.values,
                         cmap=None, norm=None, text_size=text_size)
    ax.plot([0, ncols], [nrows, nrows], color="black", linewidth=1.0)

    # ── triangular data cells ──────────────────────────────────────────────
    for col_idx in range(ncols):
        draw_triangular_column(
            ax,
            col_idx=col_idx, nrows=nrows,
            vals_lower=mae_vals[:, col_idx],
            vals_upper=rmse_vals[:, col_idx],
            cmap_lower=mae_cmap,  norm_lower=mae_norm,
            cmap_upper=rmse_cmap, norm_upper=rmse_norm,
            row_offset=0,
            fmt_lower=fmt_mae, fmt_upper=fmt_rmse,
            text_size=text_size,
            upper_text_pos=(0.65 + rmse_text_x_nudge, 0.78 + rmse_text_y_nudge),
        )

    setup_frame(ax, ncols, nrows_total)
    setup_ticks_and_labels(
        ax,
        ncols=ncols, nrows_data=nrows, nrows_total=nrows_total,
        row_labels=mae_df.index, col_labels=col_labels,
        xlabel=xlabel, xlabel_pad=xlabel_pad,
        title=title, text_size=text_size, extra_row_label="Fraction of atoms with\n" + r"$|F_{\mathrm{DFT}}| >$ threshold (%)",
    )

    # ── colorbars ─────────────────────────────────────────────────────────
    fig.canvas.draw()
    ax_pos = ax.get_position()

    if cbar_side:
        # Vertical colorbars stacked side-by-side to the right of the table
        x_mae  = ax_pos.x1 + cbar_gap_right
        x_rmse = x_mae + cbar_width_right + cbar_between_gap_right
        for cbar_x, cmap_k, norm_k, label_k, lpad in [
            (x_mae,  mae_cmap,  mae_norm,  "MAE (eV/Å)",  cbar_labelpad_mae),
            (x_rmse, rmse_cmap, rmse_norm, "RMSE (eV/Å)", cbar_labelpad_rmse),
        ]:
            cax = fig.add_axes([cbar_x, ax_pos.y0, cbar_width_right, ax_pos.height])
            sm  = mpl.cm.ScalarMappable(norm=norm_k, cmap=cmap_k)
            sm.set_array([])
            cb  = fig.colorbar(sm, cax=cax, orientation="vertical")
            cb.set_label(label_k, fontsize=text_size, labelpad=lpad, rotation=90)
            cb.ax.tick_params(labelsize=text_size - 1)
    else:
        # Horizontal colorbars spanning the full table width below the axes
        cbar_w  = ax_pos.width
        cbar_x0 = ax_pos.x0
        cbar_y_mae  = ax_pos.y0 - cbar_gap - cbar_height
        cbar_y_rmse = cbar_y_mae - cbar_between_gap - cbar_height
        for cbar_y, cmap_k, norm_k, label_k in [
            (cbar_y_mae,  mae_cmap,  mae_norm,  "MAE (eV/Å)"),
            (cbar_y_rmse, rmse_cmap, rmse_norm, "RMSE (eV/Å)"),
        ]:
            cax = fig.add_axes([cbar_x0, cbar_y, cbar_w, cbar_height])
            sm  = mpl.cm.ScalarMappable(norm=norm_k, cmap=cmap_k)
            sm.set_array([])
            cb  = fig.colorbar(sm, cax=cax, orientation="horizontal")
            cb.set_label(label_k, fontsize=text_size, labelpad=2)
            cb.ax.tick_params(labelsize=text_size - 1)

    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight", dpi=350, pad_inches=0.05)

    plt.show()
    plt.close(fig)
    return fig, ax


def plot_error_histograms(
    all_results_filtered,
    fdf_min=0.01,
    fdft_max=None,
    bins_dF=100,
    bins_theta=90,
    bins_fdft=100,
    drawstyle="line",
    figsize=(5, 4),
    textsize=10,
    model_colors=None,      # dict {model: color} or None → use default prop_cycle
    model_linestyles=None,  # dict {model: ls} or None → solid "-"
    model_linewidths=None,  # dict {model: lw} or None → use hardcoded default
    legend_style="auto",    # "auto" (default, matches drawstyle) | "line" | "patch"
    title_dF=None,          # override title of the Δ|F| histogram (None → default)
    title_theta=None,       # override title of the Δθ histogram (None → default)
    title_fdft=None,        # override title of the |F_DFT| histogram (None → default)
    savepath_dF=None,
    savepath_theta=None,
    savepath_fdft=None,
):
    """
    Three histograms: Δ|F| (log-x, log-y), Δθ (linear-x, log-y),
    and |F_DFT| (log-x, log-y) for all models.

    Parameters
    ----------
    all_results_filtered : dict[model → dict]  keys: "F_dft", "deltaF", "deltaTheta"
    fdf_min        : float – minimum |F_DFT| for inclusion (eV/Å)
    fdft_max       : float or None – maximum |F_DFT| for inclusion; set e.g. 1.0 to
                     restrict all three plots to near-equilibrium atoms only
    bins_dF        : int   – bins for Δ|F| (log-spaced)
    bins_theta     : int   – bins for Δθ (linear-spaced 0–180°)
    bins_fdft      : int   – bins for |F_DFT| (log-spaced)
    drawstyle      : str   – "line" (smooth curve) or "step" (step histogram)
    figsize        : (w, h) in inches
    textsize       : int   – base font size
    legend_style   : str   – legend icon style, independent of drawstyle:
                     "auto"  → matches drawstyle (line for "line", rectangle for "step" —
                               this is Matplotlib's default histtype="step" legend icon)
                     "line"  → force a line icon in the legend even for step histograms
                     "patch" → force a rectangle icon in the legend even for line plots
    title_dF, title_theta, title_fdft
                   : str or None – custom title for each of the three plots;
                     None → default ("Histogram of $\\Delta|F|$", etc., with the
                     |F_DFT| cutoff suffix appended when fdft_max is set)
    savepath_dF    : str or None
    savepath_theta : str or None
    savepath_fdft  : str or None
    """
    if model_colors is not None:
        # explicit dict {model: color} — look up per model below
        _color_lookup = model_colors
        colors = None
    else:
        # use the same default prop_cycle that CDF plots use
        _cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        _color_lookup = {m: _cycle[i % len(_cycle)]
                         for i, m in enumerate(all_results_filtered)}
        colors = None

    def _draw(ax, vals, edges, color, label, xlog=False, ls="-", lw=None, legend_entries=None):
        """Plot one model's histogram in the requested drawstyle."""
        default_lw = 1.6 if drawstyle == "step" else 1.5
        if drawstyle == "step":
            ax.hist(vals, bins=edges, histtype="step",
                    color=color, label=label, lw=(lw or default_lw), linestyle=ls, alpha=0.9)
        else:
            counts, _ = np.histogram(vals, bins=edges)
            centers   = (
                np.sqrt(edges[:-1] * edges[1:]) if xlog
                else 0.5 * (edges[:-1] + edges[1:])
            )
            m = counts > 0
            ax.plot(centers[m], counts[m], color=color, label=label,
                    lw=(lw or default_lw), linestyle=ls)
        if legend_entries is not None:
            legend_entries.append((label, color, ls, lw or default_lw))

    def _draw_legend(ax, legend_entries, fontsize):
        """Attach a legend, honoring legend_style independent of drawstyle."""
        style = legend_style
        if style == "auto":
            style = "patch" if drawstyle == "step" else "line"

        if style == "line":
            handles = [Line2D([0], [0], color=c, linestyle=ls, linewidth=lw)
                       for _, c, ls, lw in legend_entries]
        elif style in ("patch", "rectangle"):
            handles = [Rectangle((0, 0), 1, 1, facecolor="none",
                                  edgecolor=c, linestyle=ls, linewidth=lw)
                       for _, c, ls, lw in legend_entries]
        else:
            raise ValueError(f"legend_style must be 'auto', 'line', or 'patch' (got {style!r})")

        labels = [lbl for lbl, *_ in legend_entries]
        ax.legend(handles, labels, fontsize=fontsize, loc="best")

    fdft_label = (
        rf"  ($|F_{{\mathrm{{DFT}}}}| < {fdft_max}$ eV/Å)" if fdft_max else ""
    )

    # ── Δ|F| histogram (log-x, log-y) ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize, dpi=350)
    edges_dF = np.logspace(np.log10(1e-4), np.log10(50), bins_dF + 1)
    legend_entries = []

    for model, data in all_results_filtered.items():
        color = _color_lookup.get(model)
        _ls = (model_linestyles.get(model, "-") if model_linestyles else "-")
        _lw = (model_linewidths.get(model) if model_linewidths else None)
        F_dft = np.abs(np.asarray(data.get("F_dft", data.get("all_F_dft_mags")),  float))
        dF    = np.abs(np.asarray(data.get("deltaF", data.get("all_deltaF")), float))
        n     = min(len(F_dft), len(dF))
        F_dft, dF = F_dft[:n], dF[:n]
        valid = np.isfinite(F_dft) & np.isfinite(dF) & (F_dft > fdf_min) & (dF > 0)
        if fdft_max is not None:
            valid &= F_dft < fdft_max
        _draw(ax, dF[valid], edges_dF, color, model, xlog=True, ls=_ls, lw=_lw,
              legend_entries=legend_entries)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$|\Delta\left|F\right||$ (eV/Å)", fontsize=textsize)
    ax.set_ylabel("Number of Atoms", fontsize=textsize)
    ax.set_title(title_dF if title_dF is not None else r"Histogram of $|\Delta\left|F\right||$" + fdft_label,
                 fontsize=textsize + 1)
    ax.tick_params(labelsize=textsize - 1)
    _draw_legend(ax, legend_entries, fontsize=textsize - 2)
    ax.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    if savepath_dF is not None:
        fig.savefig(savepath_dF, bbox_inches="tight", dpi=350, pad_inches=0.05)
    plt.show()
    plt.close(fig)

    # ── Δθ histogram (linear-x, log-y) ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize, dpi=350)
    edges_th = np.linspace(0, 180, bins_theta + 1)
    legend_entries = []

    for model, data in all_results_filtered.items():
        color = _color_lookup.get(model)
        _ls = (model_linestyles.get(model, "-") if model_linestyles else "-")
        _lw = (model_linewidths.get(model) if model_linewidths else None)
        F_dft  = np.abs(np.asarray(data.get("F_dft", data.get("all_F_dft_mags")),      float))
        dtheta = np.abs(np.asarray(data.get("deltaTheta", data.get("all_deltaTheta")), float))
        n      = min(len(F_dft), len(dtheta))
        F_dft, dtheta = F_dft[:n], dtheta[:n]
        valid  = np.isfinite(F_dft) & np.isfinite(dtheta) & (F_dft > fdf_min)
        if fdft_max is not None:
            valid &= F_dft < fdft_max
        _draw(ax, dtheta[valid], edges_th, color, model, xlog=False, ls=_ls, lw=_lw,
              legend_entries=legend_entries)

    ax.set_yscale("log")
    ax.set_xlabel(r"$\Delta\theta$ (°)", fontsize=textsize)
    ax.set_ylabel("Number of Atoms", fontsize=textsize)
    ax.set_title(title_theta if title_theta is not None else r"Histogram of $\Delta\theta$" + fdft_label,
                 fontsize=textsize + 1)
    ax.tick_params(labelsize=textsize - 1)
    _draw_legend(ax, legend_entries, fontsize=textsize - 2)
    ax.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    if savepath_theta is not None:
        fig.savefig(savepath_theta, bbox_inches="tight", dpi=350, pad_inches=0.05)
    plt.show()
    plt.close(fig)

    # ── |F_DFT| histogram (log-x, log-y) ─────────────────────────────────────
    # F_DFT is DFT reference data — identical across models, so plot only once.
    fig, ax = plt.subplots(figsize=figsize, dpi=350)

    first_data = next(iter(all_results_filtered.values()))
    F_dft = np.abs(np.asarray(first_data.get("F_dft", first_data.get("all_F_dft_mags")), float))
    valid = np.isfinite(F_dft) & (F_dft > fdf_min)
    if fdft_max is not None:
        valid &= F_dft < fdft_max

    if fdft_max is not None:
        fdft_upper = fdft_max
    else:
        fdft_upper = float(F_dft[valid].max()) if valid.any() else 100.0
    edges_fd   = np.logspace(np.log10(max(fdf_min, 1e-4)), np.log10(fdft_upper), bins_fdft + 1)
    legend_entries = []
    _draw(ax, F_dft[valid], edges_fd, "steelblue", r"$|F_{\mathrm{DFT}}|$", xlog=True,
          legend_entries=legend_entries)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$|F_{\mathrm{DFT}}|$ (eV/Å)", fontsize=textsize)
    ax.set_ylabel("Number of Atoms", fontsize=textsize)
    ax.set_title(title_fdft if title_fdft is not None else r"Histogram of $|F_{\mathrm{DFT}}|$" + fdft_label,
                 fontsize=textsize + 1)
    ax.tick_params(labelsize=textsize - 1)
    _draw_legend(ax, legend_entries, fontsize=textsize - 2)
    ax.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    if savepath_fdft is not None:
        fig.savefig(savepath_fdft, bbox_inches="tight", dpi=350, pad_inches=0.05)
    plt.show()
    plt.close(fig)


def get_bad_atom_indices(all_results, model,
                         dF_gt=None, dF_lt=None,
                         dtheta_gt=None, dtheta_lt=None,
                         fdft_gt=None, fdft_lt=None):
    """
    Return flat atom-level indices and the original structure indices of
    structures that contain at least one matching atom.

    Structure-level mapping prefers direct per-atom provenance, checked in
    this order:
      1. d["structure_id"] (standardized-schema field name, one entry per
         metric-valid atom, same length/order as the dF/dtheta/fdft arrays).
      2. d["all_structure_ids"] (merge_one_potential() per-potential merged
         output field name) — same shape/meaning as (1), different name.
      3. Falls back to the legacy "forces_mlip" per-structure reconstruction
         (counts atoms per structure from forces_mlip, maps back via
         original_indices) for data that predates per-atom provenance.

    Parameters
    ----------
    all_results : dict[model → dict]
    model       : str – key into all_results
    dF_gt/lt    : float or None – Δ|F| > / < threshold (eV/Å)
    dtheta_gt/lt: float or None – Δθ > / < threshold (degrees)
    fdft_gt/lt  : float or None – |F_DFT| > / < threshold (eV/Å)

    Returns
    -------
    atom_indices      : np.ndarray of int – positions in the flat per-atom arrays
    structure_indices : list of int – original_index of structures with ≥1 matching atom
    """
    d    = all_results[model]
    dF   = np.abs(np.array(d.get("all_deltaF",     d.get("deltaF",     [])), dtype=float))
    dth  = np.abs(np.array(d.get("all_deltaTheta", d.get("deltaTheta", [])), dtype=float))
    fdft = np.abs(np.array(d.get("all_F_dft_mags", d.get("F_dft",      [])), dtype=float))

    mask = np.ones(len(dF), dtype=bool)
    if dF_gt     is not None: mask &= dF   > dF_gt
    if dF_lt     is not None: mask &= dF   < dF_lt
    if dtheta_gt is not None: mask &= dth  > dtheta_gt
    if dtheta_lt is not None: mask &= dth  < dtheta_lt
    if fdft_gt   is not None: mask &= fdft > fdft_gt
    if fdft_lt   is not None: mask &= fdft < fdft_lt

    atom_indices = np.where(mask)[0]

    # Preferred path: direct per-atom structure_id, same flat indexing as
    # dF/dth/fdft above — no atom-counting/boundary reconstruction needed.
    per_atom_structure_id = d.get("structure_id", d.get("all_structure_ids", []))
    structure_indices = []
    if len(per_atom_structure_id) == len(dF):
        sid_arr = np.asarray(per_atom_structure_id)
        structure_indices = sorted(set(sid_arr[atom_indices].tolist()))
    else:
        # Fallback: map flat atom indices → structure original_indices via
        # forces_mlip atom counts (data that predates per-atom provenance).
        orig_idxs   = d.get("original_indices", [])
        forces_mlip = d.get("forces_mlip", [])
        if orig_idxs and forces_mlip:
            atom_counts  = np.array([len(f) for f in forces_mlip])
            struct_start = np.concatenate([[0], np.cumsum(atom_counts[:-1])])
            struct_end   = np.cumsum(atom_counts)
            bad_set      = set(atom_indices.tolist())
            for i, (s, e) in enumerate(zip(struct_start, struct_end)):
                if any(a in bad_set for a in range(s, e)):
                    structure_indices.append(orig_idxs[i])

    return atom_indices, structure_indices


def get_bad_structure_indices(results_list,
                               dF_gt=None, dF_lt=None,
                               dtheta_gt=None, dtheta_lt=None,
                               fdft_gt=None, fdft_lt=None,
                               require_any=True):
    """
    From a results_*.json list (per-structure records), return the
    original_index of structures where at least one atom (require_any=True)
    or ALL atoms (require_any=False) satisfy the conditions.

    Parameters
    ----------
    results_list : list of dict – loaded from results_*.json
    require_any  : bool – True  → flag structure if ANY atom matches
                          False → flag structure only if ALL atoms match

    Returns
    -------
    list of int – original_index values of matching structures
    """
    bad = []
    for record in results_list:
        dF   = np.abs(np.array(record.get("deltaF",     []), dtype=float))
        dth  = np.abs(np.array(record.get("deltaTheta", []), dtype=float))
        fdft_raw = record.get("forces_dft", None)
        fdft = (np.linalg.norm(np.array(fdft_raw, dtype=float), axis=1)
                if fdft_raw is not None else np.full(len(dF), np.nan))

        n = min(len(dF), len(dth), len(fdft))
        dF, dth, fdft = dF[:n], dth[:n], fdft[:n]

        mask = np.ones(n, dtype=bool)
        if dF_gt     is not None: mask &= dF   > dF_gt
        if dF_lt     is not None: mask &= dF   < dF_lt
        if dtheta_gt is not None: mask &= dth  > dtheta_gt
        if dtheta_lt is not None: mask &= dth  < dtheta_lt
        if fdft_gt   is not None: mask &= fdft > fdft_gt
        if fdft_lt   is not None: mask &= fdft < fdft_lt

        hit = mask.any() if require_any else mask.all()
        if hit:
            bad.append(record["original_index"])
    return bad
