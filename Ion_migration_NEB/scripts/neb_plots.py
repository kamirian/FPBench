"""
neb_plots.py
============
Reusable NEB plotting and analysis functions.

Usage in notebook
-----------------
    from neb_plots import (
        plot_neb_fit_exact,
        plot_neb_dft_mlip,
        plot_neb_dft_multi_mlips,
        area_between_curves,
        classify_neb_path,
    )
"""

import re
import math
import textwrap
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec
from pathlib import Path

import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error

from pymatgen.core import Composition
from ase.utils.forcecurve import fit_images
from heatmap_table import (
    create_figure, draw_triangular_column, draw_rectangular_column,
    setup_frame, setup_ticks_and_labels, fmt_half_away_from_zero,
)

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


# ─────────────────────────────────────────────────────────────────────────────
# Formula helpers
# ─────────────────────────────────────────────────────────────────────────────

def to_subscript(formula):
    """Wrap numbers in HTML subscript tags (for HTML display)."""
    return re.sub(r'(\d+\.?\d*)', r'<sub>\1</sub>', formula)


def to_latex_subscript(formula):
    """Wrap numbers in LaTeX subscript notation (for matplotlib math text)."""
    return re.sub(r'(\d+\.?\d*)', r'$_{\1}$', formula)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

_COLOR_LIST = ["orange", "green", "red", "purple", "cyan", "blue", "brown", "pink"]

# Display names for FPs (internal key → label shown in plots and tables)
MODEL_NAMES = {
    "MACE-MP0_medium":      "MACE",
    "CHGNET":               "CHGNet",
    "M3GNET_pes":           "M3GNet",
    "UMA_s1_p1":            "UMA",
    "M3GNET_matpes_PBE":    "M3GNet-MatPES",
    "TensorNET_matpes_PBE": "TensorNet-MatPES",
    "MACE_matpes_pbe":      "MACE-MatPES",
}


def _get_barrier_text(energies, show_delta=False, show_range=False):
    """Return a formatted string with forward/backward barriers."""
    max_e    = max(energies)
    forward  = max_e - energies[0]
    backward = max_e - energies[-1]
    text = f"Forward:  {forward:.3f} eV\nBackward: {backward:.3f} eV"
    if show_delta:
        text += f"\nΔE: {energies[-1] - energies[0]:.3f} eV"
    if show_range:
        text += f"\nRange: {max_e - min(energies):.3f} eV"
    return text


def _fit_and_normalize(images, normalized, zero_start):
    """Run fit_images and optionally normalize path length and shift energy."""
    fit   = fit_images(images)
    E_raw = np.array(fit.energies)
    E_fit = np.array(fit.fit_energies)
    s_raw = np.array(fit.path)
    s_fit = np.array(fit.fit_path)

    if normalized:
        s_raw /= s_raw[-1]
        s_fit /= s_fit[-1]

    if zero_start:
        offset = E_raw[0]
        E_raw -= offset
        E_fit -= offset

    return E_raw, E_fit, s_raw, s_fit


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def _get_neb_data(images, normalized=True, zero_start=False, use_fit=True):
    """
    Extract NEB path and energy data, with optional spline fit.

    When use_fit=True (default), uses ASE's spline fit_images.
    When use_fit=False, computes raw image-to-image distances via MIC.
    """
    from ase.geometry import find_mic

    fit         = fit_images(images)
    E_raw       = np.array(fit.energies)
    s_fit       = np.array(fit.path)
    E_fit       = np.array(fit.fit_energies)
    s_fit_dense = np.array(fit.fit_path)

    if use_fit:
        s_raw  = s_fit
        s_plot = s_fit_dense
        E_plot = E_fit
    else:
        cell  = images[0].get_cell()
        pbc   = images[0].get_pbc()
        dists = []
        for i in range(1, len(images)):
            dr = images[i].get_positions() - images[i - 1].get_positions()
            dr_mic, _ = find_mic(dr, cell, pbc)
            dists.append(np.linalg.norm(dr_mic))
        s_raw  = np.concatenate([[0], np.cumsum(dists)])
        s_plot = s_raw.copy()
        E_plot = E_raw.copy()

    if normalized and s_raw[-1] > 0:
        s_raw  /= s_raw[-1]
        s_plot /= s_plot[-1]
    if zero_start:
        E_raw  -= E_raw[0]
        E_plot -= E_plot[0]

    return s_raw, E_raw, s_plot, E_plot


def area_between_curves(dft_images, mlip_images, zero_start=False, normalized=True, use_fit=True):
    """
    Compute the area between the fitted DFT and FP NEB energy curves.

    Returns
    -------
    float : area (eV · coordinate unit)
    """
    _, _, sfit_dft,  Efit_dft  = _get_neb_data(dft_images,  normalized, zero_start, use_fit)
    _, _, sfit_mlip, Efit_mlip = _get_neb_data(mlip_images, normalized, zero_start, use_fit)

    s_common    = np.linspace(0, 1, 300)
    dft_interp  = np.interp(s_common, sfit_dft,  Efit_dft)
    mlip_interp = np.interp(s_common, sfit_mlip, Efit_mlip)
    return float(np.trapz(np.abs(dft_interp - mlip_interp), s_common))


def plot_neb_fit_exact(
    images,
    label="NEB",
    color="tab:blue",
    linestyle="--",
    show=True,
    ax=None,
    zero_start=False,
    normalized=True,
    add_barrier_info=True,
    figsize=(6, 4),
    fontsize=12,
    barrier_text_loc=(1.02, 0.99),
    right_margin=0.75,
):
    """Plot a single NEB energy band with its spline fit."""
    E_raw, E_fit, s_raw, s_fit = _fit_and_normalize(images, normalized, zero_start)
    xlabel = "Reaction Coordinate (Normalized)" if normalized else "Path Length (Å)"

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.plot(s_raw, E_raw, "o", label=label, color=color)
    ax.plot(s_fit, E_fit, linestyle, color=color)
    ax.set_xlabel(xlabel,        fontsize=fontsize)
    ax.set_ylabel("Energy (eV)", fontsize=fontsize)
    ax.set_title(label,          fontsize=fontsize)
    ax.legend(fontsize=fontsize - 2)
    ax.tick_params(labelsize=fontsize - 2)
    fig.tight_layout()

    if add_barrier_info:
        fig.subplots_adjust(right=right_margin)
        fig.text(
            *barrier_text_loc,
            _get_barrier_text(E_raw, show_delta=True, show_range=True),
            ha="left", va="top", fontsize=fontsize - 2,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
        )

    if show:
        plt.show()
    return ax


def plot_neb_dft_mlip(
    dft_images,
    mlip_images,
    zero_start=False,
    normalized=True,
    show=True,
    ax=None,
    show_area=False,
    add_barrier_info=True,
    # ── classification labels ─────────────────────────────────────────────────
    mlip_path_label=None,   # e.g. topology string for the FP
    DFT_path_label=None,    # e.g. topology string for DFT
    # ── label & text ──────────────────────────────────────────────────────────
    dft_label="DFT",
    mlip_label="MLIP",
    title_label=None,
    show_legend=True,
    show_title=True,
    show_axis_labels=True,
    show_ylabel=True,
    # ── layout knobs ──────────────────────────────────────────────────────────
    figsize=(8, 4),
    fontsize=16,
    legend_loc="upper center",
    legend_bbox=(-0.35, 0.325),
    barrier_text_loc_dft=(-0.08, 0.9),
    barrier_text_loc_mlip=(1.15, 0.9),
    subplots_adjust=dict(left=0.15, right=0.85, top=0.82, bottom=0.25),
    save_path=None,
):
    """
    Plot one DFT and one FP NEB band with optional barrier/classification info.

    Parameters
    ----------
    mlip_path_label : str or None
        Text box shown near the FP barrier region, e.g. topology classification.
    DFT_path_label  : str or None
        Text box shown below mlip_path_label for DFT classification.
    barrier_text_loc_dft/mlip : (x, y) in axes-fraction coords for text boxes.
    """
    E_dft,  Efit_dft,  s_dft,  sfit_dft  = _fit_and_normalize(dft_images,  normalized, zero_start)
    E_mlip, Efit_mlip, s_mlip, sfit_mlip = _fit_and_normalize(mlip_images, normalized, zero_start)

    xlabel = "Reaction Coordinate\n(Normalized)" if normalized else "Path Length (Å)"

    s_common    = np.linspace(0, 1, 300)
    dft_interp  = np.interp(s_common, sfit_dft,  Efit_dft)
    mlip_interp = np.interp(s_common, sfit_mlip, Efit_mlip)
    area_diff   = float(np.trapz(np.abs(dft_interp - mlip_interp), s_common))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=350)
    else:
        fig = ax.figure

    ax.plot(s_dft,  E_dft,  "o", color="black",     label=dft_label)
    ax.plot(s_mlip, E_mlip, "o", color="tab:orange", label=mlip_label)
    ax.plot(sfit_dft,  Efit_dft,  "-",  color="black")
    ax.plot(sfit_mlip, Efit_mlip, "--", color="tab:orange")

    if show_area:
        ax.fill_between(s_common, dft_interp, mlip_interp,
                        color="purple", alpha=0.3,
                        label=f"|Area Δ| = {area_diff:.4f} eV·unit")

    if show_axis_labels:
        ax.set_xlabel(xlabel, fontsize=fontsize)
        if show_ylabel:
            ax.set_ylabel("Energy (eV)", fontsize=fontsize)
    if show_title:
        title = f"{dft_label} vs. {mlip_label}"
        if title_label:
            title += f"  {title_label}"
        ax.set_title(title, fontsize=fontsize)
    if show_legend:
        ax.legend(loc=legend_loc, bbox_to_anchor=legend_bbox,
                  frameon=True, fontsize=fontsize - 6)

    ax.tick_params(labelsize=fontsize - 2)

    if add_barrier_info:
        ax.text(*barrier_text_loc_dft,
                f"{dft_label}\n" + _get_barrier_text(E_dft),
                transform=ax.transAxes,
                ha="left", va="top", fontsize=fontsize - 4,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
        ax.text(*barrier_text_loc_mlip,
                f"{mlip_label}\n" + _get_barrier_text(E_mlip),
                transform=ax.transAxes,
                ha="left", va="top", fontsize=fontsize - 4,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="tab:orange", alpha=0.8))

    if mlip_path_label:
        ax.text(barrier_text_loc_mlip[0], barrier_text_loc_mlip[1] - 0.25,
                mlip_path_label, transform=ax.transAxes,
                ha="left", va="top", fontsize=fontsize - 4,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="orange", alpha=0.8))
    if DFT_path_label:
        ax.text(barrier_text_loc_mlip[0], barrier_text_loc_mlip[1] - 0.42,
                DFT_path_label, transform=ax.transAxes,
                ha="left", va="top", fontsize=fontsize - 4,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()

    return (ax, area_diff) if show_area else ax


def plot_neb_dft_multi_mlips(
    dft_images,
    mlip_dict,
    zero_start=False,
    normalized=True,
    show=True,
    ax=None,
    show_area=False,
    add_barrier_info=True,
    # ── classification labels ─────────────────────────────────────────────────
    df_classification=None,     # dict {mlip_name: {mode: DataFrame}}
    df_classification_dft=None, # DataFrame indexed by path_idx
    path_idx=None,              # int — row index into classification DataFrames
    mode=None,                  # "static" or "full" — used for df_classification lookup
    # ── label & text ──────────────────────────────────────────────────────────
    dft_label="DFT",
    mlip_label_prefix="",       # set to "" to use mlip_name as-is in legend
    material_label=None,
    show_legend=True,
    show_title=True,
    show_axis_labels=True,
    show_ylabel=True,
    # ── layout knobs ──────────────────────────────────────────────────────────
    figsize=(8, 4),
    fontsize=16,
    legend_loc="upper center",
    legend_bbox=(0.5, -0.3),
    legend_ncol=2,
    barrier_text_locs=None,         # dict {mlip_name: (x, y)} in axes coords
    dft_barrier_loc=(-0.11, 0.92),  # (x, y) in axes coords for DFT text box
    colors=None,
    save_path=None,
):
    """
    Plot DFT vs. multiple FP NEB bands with individual barrier info boxes.

    Classification labels
    ---------------------
    Pass df_classification, df_classification_dft, and path_idx to
    automatically append topology/asymmetry info to each barrier text box.

    df_classification      : dict {mlip_name: {mode: DataFrame}}
                             DataFrame must have columns "Pathway Topology"
                             and "Endpoint Energy Asymmetry", indexed by path_idx.
    df_classification_dft  : DataFrame with same columns, indexed by path_idx.
    path_idx               : integer index used to look up the row.
    mode                   : "static" or "full" — selects the inner DataFrame.

    barrier_text_locs      : dict {mlip_name: (x, y)} — positions in axes
                             fraction coordinates (0–1 relative to the axes).
    dft_barrier_loc        : (x, y) for the DFT text box, also in axes coords.
    """
    if colors is None:
        colors = _COLOR_LIST

    E_dft, Efit_dft, s_dft, sfit_dft = _fit_and_normalize(dft_images, normalized, zero_start)
    xlabel = "Reaction Coordinate\n(Normalized)" if normalized else "Path Length (Å)"

    s_common   = np.linspace(0, 1, 300)
    dft_interp = np.interp(s_common, sfit_dft, Efit_dft)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=350)
    else:
        fig = ax.figure

    # ── DFT curve ─────────────────────────────────────────────────────────────
    ax.plot(s_dft, E_dft, "o", color="black", label=dft_label)
    ax.plot(sfit_dft, Efit_dft, "-", color="black")

    if add_barrier_info:
        dft_text = f"{dft_label}\n" + _get_barrier_text(E_dft)
        if df_classification_dft is not None and path_idx is not None:
            row = df_classification_dft.loc[path_idx]
            dft_text += (f"\nTopology: {row['Pathway Topology']}"
                         f"\nAsymmetry: {row['Endpoint Energy Asymmetry']}")
        ax.text(*dft_barrier_loc, dft_text,
                transform=ax.transAxes,
                ha="left", va="top", fontsize=fontsize - 4,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

    # ── FP curves ───────────────────────────────────────────────────────────
    for i, (mlip_name, mlip_images) in enumerate(mlip_dict.items()):
        color = f"tab:{colors[i % len(colors)]}"
        E_mlip, Efit_mlip, s_mlip, sfit_mlip = _fit_and_normalize(
            mlip_images, normalized, zero_start
        )

        mlip_interp = np.interp(s_common, sfit_mlip, Efit_mlip)
        area_diff   = float(np.trapz(np.abs(dft_interp - mlip_interp), s_common))

        legend_label = f"{mlip_label_prefix}{mlip_name}" if mlip_label_prefix else mlip_name
        ax.plot(s_mlip,    E_mlip,    "o",  color=color, label=legend_label)
        ax.plot(sfit_mlip, Efit_mlip, "--", color=color)

        if show_area:
            ax.fill_between(s_common, dft_interp, mlip_interp,
                            alpha=0.2, color=color,
                            label=f"|ΔArea| {mlip_name}: {area_diff:.4f} eV·unit")

        if add_barrier_info:
            if barrier_text_locs and mlip_name in barrier_text_locs:
                loc = barrier_text_locs[mlip_name]
            else:
                loc = (1.05, 0.92 - 0.22 * i)

            mlip_text = f"{mlip_name}\n" + _get_barrier_text(E_mlip)
            mlip_text += f"\n|ΔArea|: {area_diff:.4f} eV·unit"

            # append classification if provided
            if (df_classification is not None and path_idx is not None
                    and mlip_name in df_classification
                    and mode in df_classification[mlip_name]):
                row = df_classification[mlip_name][mode].loc[path_idx]
                mlip_text += (f"\nTopology: {row['Pathway Topology']}"
                              f"\nAsymmetry: {row['Endpoint Energy Asymmetry']}")

            fc = "lightyellow" if mode == "static" else "white"
            ax.text(*loc, mlip_text,
                    transform=ax.transAxes,
                    ha="left", va="top", fontsize=fontsize - 4,
                    bbox=dict(boxstyle="round,pad=0.3", fc=fc, ec=color, alpha=0.8))

    # ── labels / legend ───────────────────────────────────────────────────────
    if show_axis_labels:
        ax.set_xlabel(xlabel, fontsize=fontsize)
        if show_ylabel:
            ax.set_ylabel("Energy (eV)", fontsize=fontsize - 2)
    if show_title:
        title = f"DFT vs. MLIP ({mode})" if mode else "DFT vs. MLIP"
        if material_label:
            title += f", {material_label}"
        ax.set_title(title, fontsize=fontsize)
    if show_legend:
        ax.legend(loc=legend_loc, bbox_to_anchor=legend_bbox,
                  ncol=legend_ncol, frameon=True, fontsize=fontsize - 4)

    ax.tick_params(labelsize=fontsize - 4)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()

    return ax


# ─────────────────────────────────────────────────────────────────────────────
# NEB path classification (no plotting)
# ─────────────────────────────────────────────────────────────────────────────

def classify_neb_path(images, tolerance=0.01):
    """
    Classify the topology of a NEB energy path.

    Returns
    -------
    dict with keys:
        Pathway Topology, Endpoint Energy Asymmetry,
        energy_forward_barrier, energy_backward_barrier,
        delta_E, energy_range, n_local_max, n_local_min,
        end_type, reason
    """
    energies = [atoms.get_potential_energy() for atoms in images]

    max_e   = max(energies)
    fwd     = max_e - energies[0]
    bwd     = max_e - energies[-1]
    delta_E = energies[-1] - energies[0]
    span    = max_e - min(energies)

    local_maxima = [i for i in range(1, len(energies) - 1)
                    if energies[i] > energies[i - 1] and energies[i] > energies[i + 1]]
    local_minima = [i for i in range(1, len(energies) - 1)
                    if energies[i] < energies[i - 1] and energies[i] < energies[i + 1]]

    def _end_type():
        if abs(energies[0] - energies[-1]) <= tolerance:
            return "Symmetric Endpoints"
        return "Higher Initial State" if energies[0] > energies[-1] else "Higher Final State"

    base = dict(
        energy_forward_barrier  = fwd,
        energy_backward_barrier = bwd,
        delta_E                 = delta_E,
        energy_range            = span,
        n_local_max             = len(local_maxima),
        n_local_min             = len(local_minima),
        end_type                = _end_type(),
    )

    if energies[1] < energies[0] or energies[-2] < energies[-1]:
        base.update({
            "Pathway Topology":          "Not Valid (Endpoint-Relaxation Error)",
            "Endpoint Energy Asymmetry": _end_type(),
            "reason":                   "Bad endpoint relaxation",
        })
        return base

    valley = any(
        energies[i] < energies[0] and energies[i] < energies[-1]
        for i in local_minima
    )
    base.update({
        "Pathway Topology":          "Intermediate Minimum Between Endpoints" if valley else "Normal-Hill Barrier",
        "Endpoint Energy Asymmetry": _end_type(),
        "reason":                    None,
    })
    return base


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — Barrier metric table helpers
# ─────────────────────────────────────────────────────────────────────────────

def combine_mae_rmse(df, mae_col, rmse_col, fmt="{:.2f} / {:.2f}"):
    """Combine MAE and RMSE columns into a single 'MAE / RMSE' string column."""
    out = []
    for a, b in zip(df[mae_col], df[rmse_col]):
        out.append(fmt.format(a, b) if pd.notna(a) and pd.notna(b) else np.nan)
    return pd.Series(out, index=df.index)


def make_3col_table(df_full, df_static, kind, T, fmt="{:.2f} / {:.2f}"):
    """
    Build a 3-column MAE/RMSE summary for one barrier kind (Forward / Backward / Range).

    Columns: full all-paths | threshold-filtered full | static all-paths
    """
    s_full    = combine_mae_rmse(df_full,   f"MAE {kind} (eV)",       f"RMSE {kind} (eV)",       fmt=fmt)
    s_full_lt = combine_mae_rmse(df_full,   f"MAE {kind} (< {T} eV)", f"RMSE {kind} (< {T} eV)", fmt=fmt)
    s_static  = combine_mae_rmse(df_static, f"MAE {kind} (eV)",       f"RMSE {kind} (eV)",       fmt=fmt)
    return pd.concat([
        s_full.rename(f"{kind} (MAE/RMSE)"),
        s_full_lt.rename(f"{kind} <{T} eV (MAE/RMSE)"),
        s_static.rename(f"{kind} (static) (MAE/RMSE)"),
    ], axis=1)


def combine_two(df, a_col, b_col, fmt="{:.2f} / {:.2f}"):
    """Combine two numeric columns into a single 'A / B' string column."""
    out = []
    for a, b in zip(df[a_col], df[b_col]):
        out.append(fmt.format(a, b) if pd.notna(a) and pd.notna(b) else np.nan)
    return pd.Series(out, index=df.index)


# ─────────────────────────────────────────────────────────────────────────────
# Section 8 — Parity plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_parity_grid(
    mode, metric,
    df_classification_common, df_dft_valid,
    mlip_order,
    outlier_threshold=1.0,
    figsize=(9, 4),
    model_names=None,
):
    """Parity plot grid: one subplot per FP model, shared axes."""
    _names = model_names if model_names is not None else MODEL_NAMES
    _dn = lambda n: _names.get(n, n)
    textsize = 9
    metric_labels = {
        "energy_forward_barrier":  "Forward Barrier (eV)",
        "energy_backward_barrier": "Backward Barrier (eV)",
        "energy_range":            "Energy Range (eV)",
    }
    label  = metric_labels[metric]
    mlips  = [m for m in mlip_order if mode in df_classification_common.get(m, {})]
    n_cols = math.ceil(len(mlips) / 2)
    n_rows = 2

    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey=True, dpi=350)
    axs = axs.flatten()

    df_v = df_dft_valid.copy()
    df_v["ICSD"] = df_v["ICSD"].astype(str)
    df_v["Path"] = df_v["Path"].astype(str)

    for i, mlip in enumerate(mlips):
        ax   = axs[i]
        df_m = df_classification_common[mlip][mode].copy()
        df_m["ICSD"] = df_m["ICSD"].astype(str)
        df_m["Path"] = df_m["Path"].astype(str)

        merged = df_m.merge(df_v, on=["ICSD", "Path"], suffixes=("_FP", "_DFT"))
        merged = merged[merged[f"{metric}_FP"].abs() < outlier_threshold]

        ax.scatter(merged[f"{metric}_DFT"], merged[f"{metric}_FP"], alpha=0.5, s=5)
        max_val = max(merged[f"{metric}_DFT"].max(), merged[f"{metric}_FP"].max())
        ax.plot([0, max_val], [0, max_val], "--", color="gray", lw=1.2)
        ax.set_title(_dn(mlip), fontsize=textsize + 2)
        ax.set_xlim(-0.1, max_val * 1.05)
        ax.set_ylim(-0.1, max_val * 1.05)
        ax.tick_params(labelsize=textsize)
        if i >= n_cols:
            ax.set_xlabel(f"DFT {label}", fontsize=textsize + 2)

    for i in range(0, len(axs), n_cols):
        axs[i].set_ylabel(f"FP\n{label}", fontsize=textsize)
    for j in range(len(mlips), len(axs)):
        fig.delaxes(axs[j])

    fig.suptitle(f"Parity: {label} — {mode}", fontsize=textsize + 2, y=0.97)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_parity_2x2(
    df_classification_common,
    df_dft_valid,
    mlip_order,
    conv_data=None,
    outlier_threshold=None,
    figsize=(12, 9),
    dpi=350,
    textsize=9,
    alpha=0.5,
    marker_size=8,
    xlim=None,
    ylim=None,
    model_names=None,
    panel_titles=None,          # list of 4 strings to override the default panel titles (a-d order); None entries keep the default
    panel_label_fontsize=16,    # fontsize for the (a)/(b)/(c)/(d) annotation
    panel_label_offset=(-25, -1),  # (x, y) offset in points for the panel-label annotation
    ylabel="FP  (eV)",          # y-axis label (left-column panels only)
    xlabel="DFT  (eV)",         # x-axis label (bottom-row panels only)
):
    """
    2×2 parity plot: rows = forward/backward barrier, cols = full/static.
    Each panel overlays all FPs with distinct colors.
    Convergence filtering applied when conv_data is provided:
      - all modes: neb_converged must be True
      - full mode: endpoint_status must be 'both_converged'
    outlier_threshold: if set, drops rows where |FP value| > threshold.
    xlim / ylim: (min, max) tuples to fix axis ranges; None = auto.
    panel_titles: optional list of 4 custom titles (order: a, b, c, d);
        pass None for any entry to keep that panel's default title.
    """
    panels = [
        ("energy_forward_barrier",  "full",   "Forward barrier — full",    "(a)"),
        ("energy_forward_barrier",  "static", "Forward barrier — static",  "(b)"),
        ("energy_backward_barrier", "full",   "Backward barrier — full",   "(c)"),
        ("energy_backward_barrier", "static", "Backward barrier — static", "(d)"),
    ]
    if panel_titles is not None:
        panels = [
            (metric, mode, (panel_titles[i] if panel_titles[i] is not None else title), label)
            for i, (metric, mode, title, label) in enumerate(panels)
        ]

    _names = model_names if model_names is not None else MODEL_NAMES
    _dn = lambda n: _names.get(n, n)
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    mlip_colors = {m: palette[i % len(palette)] for i, m in enumerate(mlip_order)}

    df_v = df_dft_valid.copy()
    df_v["ICSD"] = df_v["ICSD"].astype(str)
    df_v["Path"] = df_v["Path"].astype(str)

    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)

    for idx, (metric, mode, title, panel_label) in enumerate(panels):
        row, col = divmod(idx, 2)
        ax = axes[row][col]

        all_dft, all_mlip, all_mlip_names = [], [], []

        for mlip in mlip_order:
            if mode not in df_classification_common.get(mlip, {}):
                continue
            df_m = df_classification_common[mlip][mode].copy()
            df_m["ICSD"] = df_m["ICSD"].astype(str)
            df_m["Path"] = df_m["Path"].astype(str)

            merged = df_m.merge(df_v, on=["ICSD", "Path"], suffixes=("_MLIP", "_DFT"))

            # convergence filter
            if conv_data is not None:
                clookup = conv_data.get(mlip, {}).get(mode, {})
                if clookup:
                    neb_ok = merged.apply(
                        lambda r: clookup.get(
                            (str(int(float(r["ICSD"]))), str(int(float(r["Path"])))), {}
                        ).get("neb_converged", True), axis=1)
                    merged = merged[neb_ok]
                    if mode == "full":
                        ep_ok = merged.apply(
                            lambda r: clookup.get(
                                (str(int(float(r["ICSD"]))), str(int(float(r["Path"])))), {}
                            ).get("endpoint_status", "both_converged") == "both_converged", axis=1)
                        merged = merged[ep_ok]

            if outlier_threshold is not None:
                merged = merged[merged[f"{metric}_MLIP"].abs() <= outlier_threshold]

            if merged.empty:
                continue

            ax.scatter(
                merged[f"{metric}_DFT"], merged[f"{metric}_MLIP"],
                alpha=alpha, s=marker_size,
                color=mlip_colors[mlip], label=mlip, zorder=3,
            )
            all_dft.extend(merged[f"{metric}_DFT"].tolist())
            all_mlip.extend(merged[f"{metric}_MLIP"].tolist())

        if not all_dft:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=textsize, color="gray")
            ax.set_title(title, fontsize=textsize, fontweight="bold")
            ax.annotate(panel_label, xy=(0, 1), xycoords="axes fraction",
                        xytext=panel_label_offset, textcoords="offset points",
                        fontsize=panel_label_fontsize, fontweight="bold", va="bottom",
                        annotation_clip=False)
            continue

        lo = min(min(all_dft), min(all_mlip))
        hi = max(max(all_dft), max(all_mlip))
        pad = (hi - lo) * 0.05
        _xl = xlim if xlim is not None else (lo - pad, hi + pad)
        _yl = ylim if ylim is not None else (lo - pad, hi + pad)
        ax.plot([_xl[0], _xl[1]], [_xl[0], _xl[1]],
                "--", color="gray", lw=1.0, zorder=2)
        ax.set_xlim(_xl)
        ax.set_ylim(_yl)

        ax.set_title(title, fontsize=textsize, fontweight="bold")
        ax.annotate(panel_label, xy=(0, 1), xycoords="axes fraction",
                    xytext=panel_label_offset, textcoords="offset points",
                    fontsize=panel_label_fontsize, fontweight="bold", va="bottom",
                    annotation_clip=False)
        ax.tick_params(labelsize=textsize - 1)

        # y-label: left column only
        if col == 0:
            ax.set_ylabel(ylabel, fontsize=textsize)
        else:
            ax.set_ylabel("")

        # x-label and tick labels: bottom row only
        if row == 1:
            ax.set_xlabel(xlabel, fontsize=textsize)
        else:
            ax.set_xlabel("")
            ax.set_xticklabels([])

    # shared legend outside the grid
    handles = [
        mlines.Line2D([], [], color=mlip_colors[m], marker="o", ls="",
                      markersize=5, label=_dn(m))
        for m in mlip_order
    ]
    fig.legend(handles=handles, fontsize=textsize - 1,
               loc="lower center", ncol=math.ceil(len(mlip_order) / 2),
               bbox_to_anchor=(0.5, -0.04), frameon=True, edgecolor="#ccc")
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Section 9 — Violin error distribution plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_violin_errors(
    metric, mode,
    df_classification_common, df_classification_DFT_common,
    mlip_order,
    threshold=None,
    figsize=(6.5, 4),
    model_names=None,
):
    """
    Violin plot of (MLIP − DFT) error for one metric and mode.

    Parameters
    ----------
    threshold : float or None
        If given, only include paths where |error| ≤ threshold.
    """
    _names = model_names if model_names is not None else MODEL_NAMES
    _dn = lambda n: _names.get(n, n)
    textsize = 9
    metric_labels = {
        "energy_forward_barrier":  "Forward Barrier (eV)",
        "energy_backward_barrier": "Backward Barrier (eV)",
        "energy_range":            "Energy Range (eV)",
    }
    label   = metric_labels[metric]
    comment = f"  |error| ≤ {threshold} eV" if threshold is not None else ""

    df_v = df_classification_DFT_common.copy()
    df_v["ICSD"] = df_v["ICSD"].astype(str)
    df_v["Path"] = df_v["Path"].astype(str)

    all_errors = []
    for mlip in mlip_order:
        if mode not in df_classification_common.get(mlip, {}):
            continue
        df_m = df_classification_common[mlip][mode].copy()
        df_m["ICSD"] = df_m["ICSD"].astype(str)
        df_m["Path"] = df_m["Path"].astype(str)

        merged = df_m.merge(df_v, on=["ICSD", "Path"], suffixes=("_MLIP", "_DFT"))
        merged = merged[~merged["Pathway Topology_DFT"].isin(_INVALID_TOPOLOGY_LABELS)]
        error  = merged[f"{metric}_MLIP"] - merged[f"{metric}_DFT"]

        if threshold is not None:
            error = error[error.abs() <= threshold]

        for e in error:
            all_errors.append({"MLIP": _dn(mlip), "Error (eV)": e})

    if not all_errors:
        print(f"No data for {label} [{mode}]")
        return

    df_error        = pd.DataFrame(all_errors)
    mlip_order_plot = [_dn(m) for m in mlip_order if _dn(m) in df_error["MLIP"].values]

    fig, ax = plt.subplots(figsize=figsize, dpi=350)
    sns.violinplot(x="MLIP", y="Error (eV)", data=df_error,
                   order=mlip_order_plot, inner="box", cut=0, ax=ax)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1.0)

    y_min, y_max = ax.get_ylim()
    offset = (y_max - y_min) * 0.03
    for i, dn_mlip in enumerate(mlip_order_plot):
        vals    = df_error.loc[df_error["MLIP"] == dn_mlip, "Error (eV)"]
        pct_pos = 100 * (vals > 0).sum() / len(vals)
        pct_neg = 100 * (vals < 0).sum() / len(vals)
        ax.text(i, y_max - offset, f"{pct_pos:.0f}%",
                ha="center", va="top", fontsize=textsize - 1,
                color="firebrick", fontweight="bold")
        ax.text(i, y_min + offset, f"{pct_neg:.0f}%",
                ha="center", va="bottom", fontsize=textsize - 1,
                color="steelblue", fontweight="bold")

    ax.set_title(f"Error Distribution: {label} — {mode}{comment}", fontsize=textsize + 2)
    ax.set_ylabel("Barrier Error (eV)", fontsize=textsize)
    ax.set_xlabel("FP Model",           fontsize=textsize)
    ax.tick_params(labelsize=textsize)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()


def plot_violin_2x2(
    df_classification_common,
    df_classification_DFT_common,
    mlip_order,
    conv_data=None,
    figsize=(13, 7),
    dpi=350,
    textsize=9,
    wspace=0.25,
    hspace=0.35,
    x_rotation=30,
    x_ha="right",
    x_label_pad=2,
    x_label_shift=0,
    violin_inner="box",
    box_alpha=0.55,
    model_names=None,
    panel_titles=None,          # list of 4 strings to override the default panel titles (a-d order); None entries keep the default
    panel_label_fontsize=16,    # fontsize for the (a)/(b)/(c)/(d) annotation
    panel_label_offset=(-25, -1),  # (x, y) offset in points for the panel-label annotation
    ylabel="Barrier error (eV)",   # y-axis label (left-column panels only)
    xlabel="FP",                   # x-axis label (bottom-row panels only)
):
    """
    2×2 violin plot grid: rows = forward/backward, cols = full/static.
    Only converged paths are included (neb_converged=True; for full mode
    also endpoint_status='both_converged'). Pass conv_data to enable
    convergence filtering; if None, all paths are used.

    wspace / hspace  — column and row spacing.
    x_rotation       — rotation angle of x-tick labels (bottom row).
    x_ha             — horizontal alignment of rotated labels ('right'/'center').
    x_label_pad      — padding between tick and label in points.
    x_label_shift    — horizontal shift of tick labels in points (positive = right).
    violin_inner      — inner mark style: 'box', 'quartile', 'point', 'stick', or None.
    box_alpha         — alpha (transparency) of the inner box/whisker lines (0–1).
    panel_titles      — optional list of 4 custom titles (order: a, b, c, d);
        pass None for any entry to keep that panel's default title.
    """
    panels = [
        ("energy_forward_barrier",  "full",   "Forward barrier — full",    "(a)"),
        ("energy_forward_barrier",  "static", "Forward barrier — static",  "(b)"),
        ("energy_backward_barrier", "full",   "Backward barrier — full",   "(c)"),
        ("energy_backward_barrier", "static", "Backward barrier — static", "(d)"),
    ]
    if panel_titles is not None:
        panels = [
            (metric, mode, (panel_titles[i] if panel_titles[i] is not None else title), label)
            for i, (metric, mode, title, label) in enumerate(panels)
        ]

    _names = model_names if model_names is not None else MODEL_NAMES
    _dn = lambda n: _names.get(n, n)
    df_v = df_classification_DFT_common.copy()
    df_v["ICSD"] = df_v["ICSD"].astype(str)
    df_v["Path"] = df_v["Path"].astype(str)

    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)

    for idx, (metric, mode, title, panel_label) in enumerate(panels):
        row, col = divmod(idx, 2)
        ax = axes[row][col]

        all_errors = []
        for mlip in mlip_order:
            if mode not in df_classification_common.get(mlip, {}):
                continue
            df_m = df_classification_common[mlip][mode].copy()
            df_m["ICSD"] = df_m["ICSD"].astype(str)
            df_m["Path"] = df_m["Path"].astype(str)
            merged = df_m.merge(df_v, on=["ICSD", "Path"], suffixes=("_MLIP", "_DFT"))
            merged = merged[~merged["Pathway Topology_DFT"].isin(_INVALID_TOPOLOGY_LABELS)]

            # ── convergence filter ────────────────────────────────────────
            if conv_data is not None:
                clookup = conv_data.get(mlip, {}).get(mode, {})
                if clookup:
                    neb_ok = merged.apply(
                        lambda r: clookup.get(
                            (str(int(float(r["ICSD"]))), str(int(float(r["Path"])))), {}
                        ).get("neb_converged", True), axis=1)
                    merged = merged[neb_ok]
                    if mode == "full":
                        ep_ok = merged.apply(
                            lambda r: clookup.get(
                                (str(int(float(r["ICSD"]))), str(int(float(r["Path"])))), {}
                            ).get("endpoint_status", "both_converged") == "both_converged", axis=1)
                        merged = merged[ep_ok]

            error = merged[f"{metric}_MLIP"] - merged[f"{metric}_DFT"]
            for e in error:
                all_errors.append({"MLIP": _dn(mlip), "Error (eV)": e})

        if not all_errors:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=textsize, color="gray")
            ax.set_title(title, fontsize=textsize, fontweight="bold")
            ax.annotate(panel_label, xy=(0, 1), xycoords="axes fraction",
                        xytext=panel_label_offset, textcoords="offset points",
                        fontsize=panel_label_fontsize, fontweight="bold", va="bottom",
                        annotation_clip=False)
            continue

        df_error        = pd.DataFrame(all_errors)
        mlip_order_plot = [_dn(m) for m in mlip_order if _dn(m) in df_error["MLIP"].values]

        _n_lines_before = len(ax.lines)
        sns.violinplot(x="MLIP", y="Error (eV)", data=df_error,
                       order=mlip_order_plot, inner=violin_inner, cut=0, ax=ax)
        for _ln in ax.lines[_n_lines_before:]:
            _ln.set_alpha(box_alpha)
        ax.axhline(0, color="gray", linestyle="--", linewidth=1.0)

        y_min, y_max = ax.get_ylim()
        offset = (y_max - y_min) * 0.03
        for i, dn_mlip in enumerate(mlip_order_plot):
            vals    = df_error.loc[df_error["MLIP"] == dn_mlip, "Error (eV)"]
            pct_pos = 100 * (vals > 0).sum() / len(vals)
            pct_neg = 100 * (vals < 0).sum() / len(vals)
            ax.text(i, y_max - offset, f"{pct_pos:.0f}%",
                    ha="center", va="top", fontsize=textsize - 2,
                    color="firebrick", fontweight="bold")
            ax.text(i, y_min + offset, f"{pct_neg:.0f}%",
                    ha="center", va="bottom", fontsize=textsize - 2,
                    color="steelblue", fontweight="bold")

        ax.set_title(title, fontsize=textsize, fontweight="bold")
        ax.annotate(panel_label, xy=(0, 1), xycoords="axes fraction",
                    xytext=panel_label_offset, textcoords="offset points",
                    fontsize=panel_label_fontsize, fontweight="bold", va="bottom",
                    annotation_clip=False)
        ax.tick_params(axis="y", labelsize=textsize - 1)

        # y-label: left column only
        if col == 0:
            ax.set_ylabel(ylabel, fontsize=textsize)
        else:
            ax.set_ylabel("")

        # x-tick labels and x-label: bottom row only
        if row == 1:
            ax.set_xlabel(xlabel, fontsize=textsize)
            ax.tick_params(axis="x", labelsize=textsize - 1, pad=x_label_pad)
            plt.setp(ax.get_xticklabels(), rotation=x_rotation, ha=x_ha)
            if x_label_shift != 0:
                import matplotlib.transforms as _mtrans
                _shift = _mtrans.ScaledTranslation(
                    x_label_shift / 72, 0, ax.get_figure().dpi_scale_trans)
                for _lbl in ax.get_xticklabels():
                    _lbl.set_transform(_lbl.get_transform() + _shift)
        else:
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelbottom=False)  # hide labels, keep ticks aligned

    plt.tight_layout()
    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Section 10 — Area / classification analysis helpers
# ─────────────────────────────────────────────────────────────────────────────

_INVALID_TOPOLOGY_LABELS = {"Not Valid (Endpoint-Relaxation Error)", "Invalid"}


def simplify_class(cls):
    """Map detailed topology string to 3-category label. Recognizes both the
    legacy label ("Not Valid (Endpoint-Relaxation Error)", still produced by
    the live notebook's own classification cell) and the manuscript-aligned
    label ("Invalid", produced by neb_analysis.classify_energy_profile) as
    the same category, so this stays correct for either caller."""
    if cls in _INVALID_TOPOLOGY_LABELS: return "not valid"
    if cls == "Abnormal":               return "abnormal"
    return "normal-hill"


def safe_mae(x):
    """MAE against zero, ignoring non-finite values."""
    a = np.array(x, dtype=float); a = a[np.isfinite(a)]
    return float(mean_absolute_error(np.zeros(len(a)), a)) if a.size else np.nan


def safe_rmse(x):
    """RMSE against zero, ignoring non-finite values."""
    a = np.array(x, dtype=float); a = a[np.isfinite(a)]
    return float(np.sqrt(mean_squared_error(np.zeros(len(a)), a))) if a.size else np.nan


# ─────────────────────────────────────────────────────────────────────────────
# Section 11 — Heatmap summary tables
# ─────────────────────────────────────────────────────────────────────────────

def parse_mae_rmse(df):
    """Split 'MAE / RMSE' string DataFrame into two numeric DataFrames."""
    mae  = df.apply(lambda col: col.str.split(" / ").str[0].astype(float))
    rmse = df.apply(lambda col: col.str.split(" / ").str[1].astype(float))
    return mae, rmse


def plot_3col_triangular(
    df,
    title      = "",
    text_size  = 6.5,
    figsize    = (3.5, 5.5),
    dpi        = 350,
    cbar_w     = 0.007,
    cbar_gap   = 0.075,
    cbar_offset = 0.025,   # gap between the right edge of the table and the first colorbar
    save_path  = None,
    cmap_mae   = "Blues",
    cmap_rmse  = "Reds",
    col_labels    = None,
    rect_cols     = None,
    rect_cols_pre = None,   # rectangular columns drawn BEFORE the triangular columns
    model_names         = None,
    tick_rotation       = 35,   # rotation angle for column (x-axis) tick labels
    cbar_mae_labelpad   = 10,   # labelpad for the MAE colorbar label
    cbar_rmse_labelpad  = 10,   # labelpad for the RMSE colorbar label
    title_pad           = None, # gap (points) between title and axes (None → Matplotlib default)
    xtick_pad           = 2,    # gap (points) between x-axis and its tick labels
):
    """
    Triangular split heatmap (lower triangle = MAE, upper triangle = RMSE)
    with optional rectangular columns appended on the right (rect_cols) or
    prepended on the left (rect_cols_pre).

    Parameters
    ----------
    df            : DataFrame — each cell is 'MAE / RMSE' as a string
    rect_cols     : list of 1-D arrays — rectangular columns after the triangular block
    rect_cols_pre : list of 1-D arrays — rectangular columns before the triangular block
    col_labels    : list of strings — all columns left-to-right (pre-rect, tri, post-rect)
    """
    _mn = model_names if model_names is not None else MODEL_NAMES
    mae_df, rmse_df = parse_mae_rmse(df)
    models        = [_mn.get(m, m) for m in df.index.tolist()]
    ncols_tri     = len(df.columns)
    n_rect_pre    = len(rect_cols_pre) if rect_cols_pre else 0
    n_rect        = len(rect_cols)     if rect_cols     else 0
    ncols_total   = n_rect_pre + ncols_tri + n_rect

    if col_labels is None:
        col_labels = df.columns.tolist()

    nrows     = len(models)
    mae_vals  = mae_df.to_numpy(float)
    rmse_vals = rmse_df.to_numpy(float)

    mae_norm  = plt.Normalize(*np.nanpercentile(mae_vals,  [5, 95]))
    rmse_norm = plt.Normalize(*np.nanpercentile(rmse_vals, [5, 95]))
    mae_cmap  = plt.cm.get_cmap(cmap_mae)
    rmse_cmap = plt.cm.get_cmap(cmap_rmse)

    fig, ax, _ = create_figure(ncols_total, n_colorbars=0, figsize=figsize, dpi=dpi)
    ax.set_xlim(0, ncols_total)
    ax.set_ylim(0, nrows)
    ax.set_aspect("equal")

    if rect_cols_pre:
        for k, col_vals in enumerate(rect_cols_pre):
            draw_rectangular_column(
                ax, col_idx=k, nrows=nrows,
                vals=col_vals, cmap=None, norm=None, fmt="{:.1f}", text_size=text_size,
            )
        ax.plot([n_rect_pre, n_rect_pre], [0, nrows], color="black", lw=1.2, ls="--")

    for j in range(ncols_tri):
        draw_triangular_column(
            ax, col_idx=n_rect_pre + j, nrows=nrows,
            vals_lower=mae_vals[:, j], vals_upper=rmse_vals[:, j],
            cmap_lower=mae_cmap,  norm_lower=mae_norm,
            cmap_upper=rmse_cmap, norm_upper=rmse_norm,
            fmt_lower=fmt_half_away_from_zero(2), fmt_upper=fmt_half_away_from_zero(2), text_size=text_size,
        )

    if rect_cols:
        ax.plot([n_rect_pre + ncols_tri, n_rect_pre + ncols_tri], [0, nrows],
                color="black", lw=1.2, ls="--")
        for k, col_vals in enumerate(rect_cols):
            draw_rectangular_column(
                ax, col_idx=n_rect_pre + ncols_tri + k, nrows=nrows,
                vals=col_vals, cmap=None, norm=None, fmt="{:.1f}", text_size=text_size,
            )

    setup_frame(ax, ncols_total, nrows)
    setup_ticks_and_labels(
        ax, ncols=ncols_total, nrows_data=nrows, nrows_total=nrows,
        row_labels=models, col_labels=col_labels,
        title=title, title_pad=title_pad, xlabel="", extra_row_label="", extra_row_index=nrows,
        text_size=text_size, tick_rotation=tick_rotation,
    )
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right"); lbl.set_rotation(tick_rotation); lbl.set_rotation_mode("anchor")
    ax.tick_params(axis="x", pad=xtick_pad)

    fig.canvas.draw()
    ax_pos = ax.get_position()
    x0     = ax_pos.x0 + ax_pos.width + cbar_offset
    _cbar_labelpads = [cbar_mae_labelpad, cbar_rmse_labelpad]
    for k, (cmap_k, norm_k, label_k) in enumerate([
        (mae_cmap,  mae_norm,  "MAE (eV)"),
        (rmse_cmap, rmse_norm, "RMSE (eV)"),
    ]):
        cax = fig.add_axes([x0 + k * (cbar_w + cbar_gap), ax_pos.y0, cbar_w, ax_pos.height])
        cb  = fig.colorbar(plt.cm.ScalarMappable(norm=norm_k, cmap=cmap_k), cax=cax)
        cb.ax.tick_params(labelsize=text_size)
        cb.set_label(label_k, fontsize=text_size, rotation=90, labelpad=_cbar_labelpads[k])

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Section 12 — NEB path visualisation (multi-panel summary)
# ─────────────────────────────────────────────────────────────────────────────

# Layout constants for plot_neb_summary (tweak here)
_NEB_SUMMARY_LAYOUT = dict(
    FIG_W              = 6.5,
    FIG_H              = 8.5,
    HSPACE             = 0.35,
    WSPACE             = 0.3,
    LEGEND_COL_W_IN    = 1.1,    # inches — width of the vertical legend column (right of the panels)
    LEGEND_X_SHIFT_IN  = 0.0,    # inches — shifts the whole legend box left(-)/right(+)
    PLOT_ROW_RATIO     = 4,      # height_ratio of the top row (curve panels + legend)
    TABLE_ROW_RATIO    = 1.8,    # height_ratio of the table row — set PLOT_ROW_RATIO/TABLE_ROW_RATIO
                                 # together to control the plots-vs-tables size split
    TABLE_GAP_IN       = -0.65,  # inches — space between the two tables (increase for more gap)
    PLOT_GAP_IN        = 0.0,    # inches — extra space between the full/static curve panels (0 = gridspec
                                 # default; positive = more gap; negative = pulls them closer/overlapping)
    LEFT_PLOT_SHIFT_IN = 0.0,    # inches — moves only the left (full) panel left(-)/right(+); everything
                                 # else (right panel, legend, tables) stays exactly where it is
    RIGHT_PLOT_SHIFT_IN = 0.0,   # inches — moves only the right (static) panel left(-)/right(+); everything
                                 # else (left panel, legend, tables) stays exactly where it is
    PLOT_TABLE_GAP_IN  = 0.17,   # inches — pulls the tables up toward the plots (bigger = closer; negative = further away)
    WRAP_MAP           = [12, 8, 8, 8, 15, 15],
    FONTSIZE_TBL       = 9.3,
    HEADER_LINES       = 1.8,
    LINE_HEIGHT_MULT   = 1.35,   # line height, in units of font size
    ROW_PAD_IN         = 0.035,  # vertical padding above/below each row's text, in inches
    CELL_PAD           = 0.15,   # horizontal text padding inside each table cell (mpl default is 0.1).
                                 # NOTE: the table is always rescaled to exactly fill its axes width, so past a
                                 # certain point a bigger CELL_PAD paradoxically shrinks everything instead of
                                 # adding room (it asks for more room than is available). Keep changes small
                                 # (e.g. 0.1-0.2); if columns still feel cramped, widen TABLE_W_EXTRA_IN instead
                                 # (and bump TABLE_GAP_IN to match, so the two tables don't collide).
    LEGEND_X           = 0.5,
    LEGEND_Y           = 0.5,
    LEGEND_HANDLELENGTH = 3.0,   # legend line-sample length — longer makes the dash pattern visible past the marker
    LEGEND_HANDLETEXTPAD = 0.8,  # space between the dashed line sample and the model name (mpl default 0.8; smaller = tighter)
    TABLE_W_EXTRA_IN   = 0.975,  # inches — how far each table widens outward from its plot column
)


def _barrier_text_from_energies(energies):
    """Return (text, fwd, bwd) from a list of NEB image energies."""
    intermediates = list(energies[1:-1])
    E_low  = min(energies[0], energies[-1])
    E_high = max(energies[0], energies[-1])
    max_i  = max(intermediates)
    char_e = (max_i if max_i > E_high
               else max(intermediates,
                        key=lambda e: max(abs(e - energies[0]), abs(e - energies[-1]))))
    fwd = char_e - E_low
    bwd = char_e - E_high
    return f"Forward: {fwd:.3f} eV\nBackward: {bwd:.3f} eV", fwd, bwd


def _plot_neb_bands_multi(
    dft_images, mlip_dict,
    zero_start=False, normalized=True, show_ylabel=True, show=True,
    ax=None, show_area=False, add_barrier_info=True,
    figsize=(6.5, 5.5), barrier_text_locs=None,
    DFT_text_loc=(-0.11, 0.92), legend_bbox=(0.5, -0.3),
    dft_label="DFT", mlip_label_prefix="MLIP", material_label=None,
    mode=None, columnnum=3, show_legend=True, show_title=True,
    show_axis_labels=True, fontsize=16, save_path=None,
    return_barrier_data=False, use_fit=True,
    color_map=None, fp_label="FP", mode_label=None, xlabel_override=None,
    title_fontsize=None,   # overrides fontsize for the panel title only (xlabel/ylabel keep using fontsize)
    tick_fontsize_offset=-4,   # tick labelsize = fontsize + this offset (default keeps old -4pt-smaller behavior)
    title_prefix="DFT vs. ",   # prepended to mode_disp for the panel title; set to "" to drop it entirely
    xlabel_pad=None,   # padding (points) between the xlabel and its tick labels; None = matplotlib default
    n_xticks=5,   # number of evenly-spaced x-ticks, fixed regardless of panel width (set to None to fall
                  # back to matplotlib's auto locator, which silently drops ticks on narrow panels)
):
    """
    Internal helper: plot DFT + multiple FP NEB bands with barrier annotations.
    Returns (ax, barrier_data) when return_barrier_data=True.
    """
    if xlabel_override is not None:
        xlabel = xlabel_override
    else:
        xlabel = "Reaction Coordinate \n(Normalized)" if normalized else "Path Length (Å)"
    barrier_data = {}

    s_dft, E_dft, sfit_dft, Efit_dft = _get_neb_data(
        dft_images, normalized=normalized, zero_start=zero_start, use_fit=use_fit)
    s_common   = np.linspace(0, 1, 300)
    dft_interp = np.interp(s_common, sfit_dft, Efit_dft)

    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(dpi=350)
        fig.set_size_inches(figsize)
        fig.subplots_adjust(left=0.15, right=0.85, top=0.82, bottom=0.25)
    else:
        fig = ax.figure
        fig.set_dpi(350)

    ax.plot(s_dft,    E_dft,    "o", color="black", label=dft_label)
    ax.plot(sfit_dft, Efit_dft, "-", color="black")
    if add_barrier_info:
        barrier_str, _, _ = _barrier_text_from_energies(E_dft)
        ax.text(*DFT_text_loc, f"{dft_label}\n{barrier_str}",
                ha="left", va="top", fontsize=fontsize - 4,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7))

    barrier_str, _, _ = _barrier_text_from_energies(E_dft)
    barrier_data["DFT"] = (f"DFT\n{barrier_str}", "black")

    COLOR_LIST = ["orange", "green", "red", "purple", "cyan", "blue", "brown", "pink"]
    for i, (mlip_name, mlip_images) in enumerate(mlip_dict.items()):
        s_m, E_m, sf_m, Ef_m = _get_neb_data(
            mlip_images, normalized=normalized, zero_start=zero_start, use_fit=use_fit)
        mlip_interp = np.interp(s_common, sf_m, Ef_m)
        area_diff   = float(np.trapz(np.abs(dft_interp - mlip_interp), s_common))
        color       = (color_map[mlip_name] if color_map and mlip_name in color_map
                       else f"tab:{COLOR_LIST[i % len(COLOR_LIST)]}")

        barrier_str, _, _ = _barrier_text_from_energies(E_m)
        barrier_data[mlip_name] = (
            f"{mlip_name}\n{barrier_str}\n|ΔArea|: {area_diff:.4f} eV·unit", color
        )

        ax.plot(s_m,  E_m,  "o",  color=color, label=f"{mlip_label_prefix}{mlip_name}")
        ax.plot(sf_m, Ef_m, "--", color=color)

        if show_area:
            ax.fill_between(s_common, dft_interp, mlip_interp,
                            alpha=0.2, color=color,
                            label=f"|ΔArea| {mlip_name}: {area_diff:.4f} eV·unit")
        if add_barrier_info:
            loc    = (barrier_text_locs[mlip_name]
                      if barrier_text_locs and mlip_name in barrier_text_locs
                      else (1.05, 0.9 - 0.15 * i))
            box_fc = "lightyellow" if mode == "static" else "white"
            ax.text(*loc, f"{mlip_name}\n{barrier_str}\n|ΔArea|: {area_diff:.4f} eV·unit",
                    ha="left", va="top", fontsize=fontsize - 4,
                    bbox=dict(boxstyle="round,pad=0.3", fc=box_fc, ec=color, alpha=0.7))

    if n_xticks is not None:
        # matplotlib's auto locator silently drops ticks on narrow panels; fixed
        # ticks over the actual plotted range keep the count stable regardless.
        ax.set_xticks(np.linspace(float(np.min(s_dft)), float(np.max(s_dft)), n_xticks))
    if show_axis_labels:
        ax.set_xlabel(xlabel, fontsize=fontsize, labelpad=xlabel_pad)
        if show_ylabel:
            ax.set_ylabel("Energy (eV)", fontsize=fontsize)
    if show_title:
        mode_disp = mode_label if mode_label is not None else f"{fp_label} ({mode})"
        title_txt = f"{title_prefix}{mode_disp}"
        if material_label:
            title_txt += f"  {material_label}"
        ax.set_title(title_txt, fontsize=title_fontsize if title_fontsize is not None else fontsize)
    if show_legend:
        ax.legend(loc="upper center", bbox_to_anchor=legend_bbox,
                  ncol=columnnum, frameon=True, fontsize=fontsize - 2)
    ax.tick_params(axis="both", labelsize=fontsize + tick_fontsize_offset)

    if save_path:
        fig.savefig(save_path, format="svg", bbox_inches="tight")
    if show:
        plt.show()
    if return_barrier_data:
        return ax, barrier_data
    return ax


def plot_neb_summary(
    target_icsd, target_path, mlip_names,
    # ── data produced by the notebook ────────────────────────────────────────
    dft_index_lookup,
    df_ICSD,
    neb_paths_dft,
    neb_structures_all_mlip,
    mlip_index_lookup,
    df_classification,
    df_classification_DFT,
    # ── optional ─────────────────────────────────────────────────────────────
    use_fit=False,
    save_dir=None,
    figures_dir=None,
    mlip_names_full_plot=None,   # if set, only these are plotted in the full panel
                                 # (mlip_names still controls both tables)
    legend_ncol=None,
    model_names=None,   # dict {internal_key: display_name}; defaults to module-level MODEL_NAMES
    show_mode=None,     # None=both panels, "full"=full only, "static"=static only
    topology_col_header="Energy Profile\nShape Agr.",
    endpoint_col_header="End-site\nEnergy Rank.",
    area_col_header="|ΔArea|",
    fp_label="FP",                    # fallback curve-title label, used only if full/static_curve_label is None
    full_curve_label=None,       # title text for the full-mode curve panel: "<curve_title_prefix><full_curve_label>".
                                  # Defaults to full_table_title, so the curve panel and the table below it
                                  # show matching text unless you override this separately.
    static_curve_label=None,     # same as full_curve_label, for the static-mode panel. Defaults to static_table_title.
    curve_title_prefix="DFT vs. ",  # text prepended to the curve panel titles. Set to "" to make the panel
                                     # title exactly match full_curve_label/static_curve_label with no prefix.
    full_table_title="Full FP-NEB",   # title above the full-mode summary table
    static_table_title="FP Static",   # title above the static-mode summary table
    table_fontsize=None,   # overrides _NEB_SUMMARY_LAYOUT["FONTSIZE_TBL"] (default 9.3)
    col_widths=None,       # list of 6 floats or None per column — [Model, Fwd, Bwd, |ΔArea|, topology, endpoint]
                           # None → auto; float → axes-fraction width (e.g. 0.12 for a narrow column)
    table_gap_in=None,          # inches — space between the two tables (bigger = more gap). Default -0.65
    plot_gap_in=None,           # inches — space between the two curve panels (full/static). Default 0.0;
                                 # positive = more gap, negative = pulls them closer together. No effect
                                 # when only one panel is shown (show_mode set).
    left_plot_shift_in=None,    # inches — moves ONLY the left (full) curve panel left(-)/right(+); the right
                                 # panel, legend, and tables stay exactly where they are. Default 0.0.
    right_plot_shift_in=None,   # inches — moves ONLY the right (static) curve panel left(-)/right(+); the left
                                 # panel, legend, and tables stay exactly where they are. Default 0.0.
    show_ylabel=True,           # "Energy (eV)" on the curve panels: True = both, False = neither,
                                 # "left" = only the left (full) panel, "right" = only the right (static) panel
    legend_x_shift_in=None,     # inches — shifts the whole legend box left(-)/right(+). Default 0.0
    legend_handlelength=None,   # legend line-sample length — longer makes dashes visible past the marker. Default 3.0
    cell_pad=None,              # horizontal text padding inside each table cell. Default 0.15 (mpl default is 0.1).
                                 # NOTE: the table always rescales to exactly fill its axes width, so pushing this
                                 # too high can paradoxically shrink everything instead of adding room. Keep it
                                 # small (0.1-0.2); for more room, widen table_w_extra_in instead (and bump
                                 # table_gap_in to match, so the two tables don't collide).
    table_w_extra_in=None,      # inches — how far each table widens outward from its plot column. Default 0.975
    plot_table_gap_in=None,     # inches — pulls the tables up toward the plots (bigger = closer). Default 0.17
    panel_title_fontsize=None,  # fontsize of each curve panel's title ("DFT vs. Full/Static FP"). Default 12
    panel_label_fontsize=None,  # fontsize of each curve panel's x/y axis labels. Default 12
    panel_tick_fontsize_offset=None,  # tick label fontsize relative to panel_label_fontsize: 0 = same size,
                                       # +1 = one pt larger than the labels, -1 = one pt smaller. Default 0.
    panel_xlabel_pad=None,   # padding (points) between "Reaction Coordinate" and its tick labels. mpl default ~4.
    panel_n_xticks=None,     # number of evenly-spaced x-ticks on each curve panel. Default 5 (0, 0.25, 0.5,
                              # 0.75, 1), fixed regardless of panel width. Pass an int for a different count,
                              # or "auto" to fall back to matplotlib's own locator (which can silently drop
                              # ticks on narrow panels — that's the behavior this default replaces).
    table_title_fontsize=None,   # fontsize of the table titles only (e.g. "Full FP-NEB"/"Static FP") — independent
                                  # of table_fontsize, which controls the table body/cell text. Default 9.
                                  # To hide a table's title entirely, pass full_table_title=None and/or
                                  # static_table_title=None (that table is drawn with no title text).
    col_width_adjust_in=None,   # list of 6 floats or None per column — [Model, Fwd, Bwd, |ΔArea|, topology, endpoint]
                                 # inches to add to (or subtract from, if negative) that column's auto-computed
                                 # width. Applied on top of the normal auto-sizing, so e.g. [None, None, None,
                                 # None, None, 0.15] only widens the last column, leaving the rest auto-sized.
    legend_fontsize=None,          # legend text fontsize. Defaults to panel_label_fontsize (so it matches
                                    # the panel labels unless overridden here).
    legend_handletextpad=None,     # space between each dashed line sample and its model name. mpl default is
                                    # 0.8; make it smaller (e.g. 0.3) to pull the text closer to the line.
    fig_w_in=None,          # inches — width of the curve-panel area (legend column is added on top via
                             # legend_col_w_in, so total figure width = fig_w_in + legend_col_w_in). Default 6.5
    fig_h_in=None,          # inches — overall figure height. Default 8.5
    legend_col_w_in=None,   # inches — width of the legend column (bigger = more room for long labels). Default 1.1
    plot_row_ratio=None,    # relative height of the curve-panel(+legend) row vs. the table row. Default 4
    table_row_ratio=None,   # relative height of the table row vs. the plot row. Default 1.8
                             # (plot_row_ratio / table_row_ratio sets the plots-vs-tables size split;
                             # the absolute figure height still comes from fig_h_in)
):
    """
    Full NEB summary for one (ICSD, Path) entry.

    Panels: [full curve | static curve] / legend / [full table | static table]
    Use show_mode="full" or show_mode="static" to render only one side.

    Parameters
    ----------
    dft_index_lookup       : dict {(icsd_str, path_str): int}
    df_ICSD                : DataFrame with CollectionCode and SumFormula columns
    neb_paths_dft          : list of ASE image lists
    neb_structures_all_mlip: dict {mlip: {mode: [image_list]}}
    mlip_index_lookup      : dict {mlip: {mode: {key: int}}}
    df_classification      : dict {mlip: {mode: DataFrame}}
    df_classification_DFT  : DataFrame
    model_names            : optional dict overriding the module-level MODEL_NAMES
    show_mode              : None (both), "full", or "static"
    """
    L = dict(_NEB_SUMMARY_LAYOUT)
    if table_gap_in         is not None: L["TABLE_GAP_IN"]        = table_gap_in
    if plot_gap_in          is not None: L["PLOT_GAP_IN"]         = plot_gap_in
    if left_plot_shift_in   is not None: L["LEFT_PLOT_SHIFT_IN"]  = left_plot_shift_in
    if right_plot_shift_in  is not None: L["RIGHT_PLOT_SHIFT_IN"] = right_plot_shift_in
    if legend_x_shift_in    is not None: L["LEGEND_X_SHIFT_IN"]   = legend_x_shift_in
    if legend_handlelength  is not None: L["LEGEND_HANDLELENGTH"] = legend_handlelength
    if cell_pad             is not None: L["CELL_PAD"]            = cell_pad
    if table_w_extra_in     is not None: L["TABLE_W_EXTRA_IN"]    = table_w_extra_in
    if plot_table_gap_in    is not None: L["PLOT_TABLE_GAP_IN"]   = plot_table_gap_in
    if fig_w_in             is not None: L["FIG_W"]               = fig_w_in
    if fig_h_in             is not None: L["FIG_H"]               = fig_h_in
    if legend_col_w_in      is not None: L["LEGEND_COL_W_IN"]     = legend_col_w_in
    if plot_row_ratio       is not None: L["PLOT_ROW_RATIO"]      = plot_row_ratio
    if table_row_ratio      is not None: L["TABLE_ROW_RATIO"]     = table_row_ratio
    if legend_handletextpad is not None: L["LEGEND_HANDLETEXTPAD"] = legend_handletextpad
    _fontsize_tbl    = table_fontsize        if table_fontsize        is not None else L["FONTSIZE_TBL"]
    _fontsize_label  = panel_label_fontsize  if panel_label_fontsize  is not None else 12
    _fontsize_title  = panel_title_fontsize  if panel_title_fontsize  is not None else 12
    _fontsize_legend = legend_fontsize       if legend_fontsize       is not None else _fontsize_label
    _tick_offset     = panel_tick_fontsize_offset if panel_tick_fontsize_offset is not None else 0
    _full_curve_label   = full_curve_label   if full_curve_label   is not None else full_table_title
    _static_curve_label = static_curve_label if static_curve_label is not None else static_table_title
    _fontsize_tbl_title = table_title_fontsize if table_title_fontsize is not None else 9
    _n_xticks = 5 if panel_n_xticks is None else (None if panel_n_xticks == "auto" else panel_n_xticks)
    if figures_dir is None:
        figures_dir = Path("figures")
    save_dir = save_dir or figures_dir

    # display-name helper — falls back to internal name if not in dict
    _names = model_names if model_names is not None else MODEL_NAMES
    _dn = lambda n: _names.get(n, n)

    # determine which modes/columns to render
    if show_mode is None:
        modes_to_show = ["full", "static"]   # full on left, static on right
    elif show_mode in ("full", "static"):
        modes_to_show = [show_mode]
    else:
        raise ValueError(f"show_mode must be None, 'full', or 'static', got {show_mode!r}")
    ncols = len(modes_to_show)

    key = (str(target_icsd), str(target_path))
    if key not in dft_index_lookup:
        print(f"Skipping {key} — not in dft_index_lookup")
        return

    idx_dft  = dft_index_lookup[key]
    row_icsd = df_ICSD.loc[df_ICSD["CollectionCode"].astype(str) == str(target_icsd)]
    if row_icsd.empty:
        print(f"  Note: ICSD {target_icsd} not in df_ICSD (may be theoretical/computed) — using ICSD ID as label")
        compact_formula = f"ICSD-{target_icsd}"
    else:
        sum_formula     = row_icsd["SumFormula"].values[0]
        compact_formula = str(Composition(sum_formula).reduced_formula)
    compact_formula = to_latex_subscript(compact_formula)
    dft_images      = neb_paths_dft[idx_dft]
    names_order     = ["DFT"] + mlip_names

    base_fig_w   = L["FIG_W"] if ncols == 2 else L["FIG_W"] * 0.65
    legend_col_w = L["LEGEND_COL_W_IN"]
    fig_w        = base_fig_w + legend_col_w
    fig = plt.figure(figsize=(fig_w, L["FIG_H"]), dpi=350)
    width_ratios = [base_fig_w / ncols] * ncols + [legend_col_w]
    gs  = GridSpec(2, ncols + 1, figure=fig,
                   height_ratios=[L["PLOT_ROW_RATIO"], L["TABLE_ROW_RATIO"]],
                   width_ratios=width_ratios,
                   hspace=L["HSPACE"], wspace=L["WSPACE"],
                   top=0.95, bottom=0.02, left=0.06, right=0.97)

    # create NEB axes in column order (full=col 0, static=col 1 when both shown)
    neb_axes = {mode: fig.add_subplot(gs[0, col_idx])
                for col_idx, mode in enumerate(modes_to_show)}

    if ncols == 2 and L["PLOT_GAP_IN"]:
        # nudge the two curve panels apart (or together, if negative) without resizing them
        _plot_gap_shift = (L["PLOT_GAP_IN"] / fig.get_figwidth()) / 2
        _ax_full, _ax_static = neb_axes[modes_to_show[0]], neb_axes[modes_to_show[1]]
        for _ax, _sign in [(_ax_full, -1), (_ax_static, 1)]:
            _pos = _ax.get_position()
            _ax.set_position([_pos.x0 + _sign * _plot_gap_shift, _pos.y0, _pos.width, _pos.height])

    if ncols == 2 and L["LEFT_PLOT_SHIFT_IN"]:
        # moves only the left (full) panel — right panel, legend, and tables stay put
        _ax_full = neb_axes[modes_to_show[0]]
        _pos = _ax_full.get_position()
        _ax_full.set_position([_pos.x0 + L["LEFT_PLOT_SHIFT_IN"] / fig.get_figwidth(),
                                _pos.y0, _pos.width, _pos.height])

    if ncols == 2 and L["RIGHT_PLOT_SHIFT_IN"]:
        # moves only the right (static) panel — left panel, legend, and tables stay put
        _ax_static = neb_axes[modes_to_show[1]]
        _pos = _ax_static.get_position()
        _ax_static.set_position([_pos.x0 + L["RIGHT_PLOT_SHIFT_IN"] / fig.get_figwidth(),
                                  _pos.y0, _pos.width, _pos.height])

    all_barrier_data = {}
    model_colors     = {}

    # Pre-assign one fixed color per FP based on its position in mlip_names,
    # so colors are identical in both panels regardless of which paths have data.
    pre_colors = {
        name: f"tab:{_COLOR_LIST[i % len(_COLOR_LIST)]}"
        for i, name in enumerate(mlip_names)
    }

    for col_idx, mode in enumerate(modes_to_show):
        ax = neb_axes[mode]
        names_plot = (mlip_names_full_plot
                      if (mode == "full" and mlip_names_full_plot is not None)
                      else mlip_names)
        mlip_dict = {
            n: neb_structures_all_mlip[n][mode][mlip_index_lookup[n][mode][key]]
            for n in names_plot
            if key in mlip_index_lookup.get(n, {}).get(mode, {})
        }
        if show_ylabel == "left":
            _show_ylabel_this = (col_idx == 0)
        elif show_ylabel == "right":
            _show_ylabel_this = (col_idx == 1)
        else:
            _show_ylabel_this = bool(show_ylabel)
        _, barrier_data = _plot_neb_bands_multi(
            dft_images=dft_images, mlip_dict=mlip_dict,
            ax=ax, mode=mode, show_ylabel=_show_ylabel_this,
            use_fit=use_fit, show_area=True, columnnum=2, fontsize=_fontsize_label,
            title_fontsize=_fontsize_title, tick_fontsize_offset=_tick_offset,
            show_legend=False, add_barrier_info=False,
            return_barrier_data=True, DFT_text_loc=(0, 0),
            material_label="", zero_start=True, show=False, normalized=True,
            color_map=pre_colors, fp_label=fp_label,
            mode_label=(_full_curve_label if mode == "full" else _static_curve_label),
            title_prefix=curve_title_prefix,
            xlabel_override="Reaction Coordinate",
            xlabel_pad=panel_xlabel_pad,
            n_xticks=_n_xticks,
        )

        enriched = {}
        for name, (text, color) in barrier_data.items():
            model_colors[name] = color
            fwd_m  = re.search(r"Forward:\s*(-?[\d.]+)",  text)
            bwd_m  = re.search(r"Backward:\s*(-?[\d.]+)", text)
            area_m = re.search(r"\|ΔArea\|:\s*([\d.]+)",  text)
            fwd  = float(fwd_m.group(1))  if fwd_m  else float("nan")
            bwd  = float(bwd_m.group(1))  if bwd_m  else float("nan")
            area = float(area_m.group(1)) if area_m else None

            if name == "DFT":
                dft_row  = df_classification_DFT[
                    (df_classification_DFT["ICSD"].astype(str) == str(target_icsd)) &
                    (df_classification_DFT["Path"].astype(str) == str(target_path))
                ]
                topology = dft_row["Pathway Topology"].values[0]        if not dft_row.empty else "—"
                endpoint = dft_row["Endpoint Energy Ranking"].values[0]  if not dft_row.empty else "—"
            else:
                if key in mlip_index_lookup.get(name, {}).get(mode, {}):
                    mlip_row = df_classification[name][mode][
                        (df_classification[name][mode]["ICSD"].astype(str) == str(target_icsd)) &
                        (df_classification[name][mode]["Path"].astype(str) == str(target_path))
                    ]
                    topology = mlip_row["Pathway Topology"].values[0]        if not mlip_row.empty else "—"
                    endpoint = mlip_row["Endpoint Energy Ranking"].values[0]  if not mlip_row.empty else "—"
                else:
                    topology = endpoint = "—"

            enriched[name] = dict(color=color, forward=fwd, backward=bwd,
                                   area=area, topology=str(topology), endpoint=str(endpoint))
        # Add table-only rows: FPs in mlip_names but excluded from the full plot
        if mode == "full" and mlip_names_full_plot is not None:
            for name in mlip_names:
                if name in enriched:
                    continue
                try:
                    df_m = df_classification[name][mode]
                    mlip_row = df_m[
                        (df_m["ICSD"].astype(str) == str(target_icsd)) &
                        (df_m["Path"].astype(str) == str(target_path))
                    ]
                    if mlip_row.empty:
                        continue
                    r        = mlip_row.iloc[0]
                    fwd      = float(r["energy_forward_barrier"])
                    bwd      = float(r["energy_backward_barrier"])
                    topology = str(r.get("Pathway Topology", "—"))
                    endpoint = str(r.get("Endpoint Energy Ranking", "—"))
                    color    = model_colors.get(name, "lightgray")
                    enriched[name] = dict(color=color, forward=fwd, backward=bwd,
                                          area=None, topology=topology, endpoint=endpoint)
                except (KeyError, IndexError, TypeError):
                    pass

        all_barrier_data[mode] = (ax, enriched)

    # Legend column — vertical (top-to-bottom) list, right of the static panel
    leg_ax = fig.add_subplot(gs[0, ncols])
    leg_ax.axis("off")
    handles = [
        mlines.Line2D([], [], color=model_colors[n], marker="o", linestyle="--",
                      markersize=7, linewidth=2, label=_dn(n))
        for n in names_order if n in model_colors
    ]
    _ncol = legend_ncol if legend_ncol is not None else 1
    leg_ax.legend(handles=handles, loc="center", ncol=_ncol,
                  fontsize=_fontsize_legend, frameon=True, handlelength=L["LEGEND_HANDLELENGTH"],
                  handletextpad=L["LEGEND_HANDLETEXTPAD"],
                  bbox_to_anchor=(L["LEGEND_X"], L["LEGEND_Y"]))

    if L["LEGEND_X_SHIFT_IN"]:
        leg_pos = leg_ax.get_position()
        leg_ax.set_position([leg_pos.x0 + L["LEGEND_X_SHIFT_IN"] / fig.get_figwidth(),
                              leg_pos.y0, leg_pos.width, leg_pos.height])

    # Tables
    col_headers   = ["Model", "Fwd\n(eV)", "Bwd\n(eV)", area_col_header, topology_col_header, endpoint_col_header]
    tbl_axes_list = []
    tbl_list      = []
    line_in     = (_fontsize_tbl * L["LINE_HEIGHT_MULT"]) / 72.0
    pad_in      = L["ROW_PAD_IN"]
    header_lines = max(L["HEADER_LINES"], max(str(h).count("\n") + 1 for h in col_headers))

    for col_idx, (mode, (ax, enriched)) in enumerate(all_barrier_data.items()):
        rows_present = [n for n in names_order if n in enriched]
        cell_text    = []
        for name in rows_present:
            d      = enriched[name]
            area_s = f"{d['area']:.4f}" if d["area"] is not None else "—"
            cell_text.append([
                textwrap.fill(_dn(name),              width=L["WRAP_MAP"][0]),
                textwrap.fill(f"{d['forward']:.3f}",  width=L["WRAP_MAP"][1]),
                textwrap.fill(f"{d['backward']:.3f}", width=L["WRAP_MAP"][2]),
                textwrap.fill(area_s,                 width=L["WRAP_MAP"][3]),
                textwrap.fill(d["topology"],           width=L["WRAP_MAP"][4]),
                textwrap.fill(d["endpoint"],           width=L["WRAP_MAP"][5]),
            ])

        row_lines = [max(len(cell_text[i][j].split("\n")) for j in range(len(col_headers)))
                     for i in range(len(rows_present))]

        tbl_ax = fig.add_subplot(gs[1, col_idx])
        tbl_ax.axis("off")
        _this_table_title = static_table_title if mode == "static" else full_table_title
        if _this_table_title:
            tbl_ax.set_title(_this_table_title, fontsize=_fontsize_tbl_title, fontweight="bold", pad=4)
        tbl_axes_list.append(tbl_ax)

        # Tight row heights: size each row to its text (font size + a small pad)
        # instead of stretching to fill the whole axes, then anchor to the top.
        header_h_in = header_lines * line_in + pad_in
        data_h_in   = [lc * line_in + pad_in for lc in row_lines]
        total_h_in  = header_h_in + sum(data_h_in)
        axes_h_in   = tbl_ax.get_position().height * fig.get_figheight()
        frac        = min(1.0, total_h_in / axes_h_in) if axes_h_in > 0 else 1.0
        header_h    = header_h_in / total_h_in
        data_hs     = [d / total_h_in for d in data_h_in]

        tbl = tbl_ax.table(cellText=cell_text, colLabels=col_headers,
                           cellLoc="center", loc="upper center",
                           bbox=[0, 1 - frac, 1, frac])
        tbl_list.append(tbl)
        tbl.auto_set_font_size(False)
        tbl.auto_set_column_width(col=list(range(len(col_headers))))
        for cell in tbl.get_celld().values():
            cell.PAD = L["CELL_PAD"]

        for j in range(len(col_headers)):
            tbl[0, j].set_facecolor("#b0b0b0")
            tbl[0, j].set_height(header_h)
            tbl[0, j].set_text_props(fontweight="bold", fontsize=_fontsize_tbl)

        for i, name in enumerate(rows_present):
            color = enriched[name]["color"]
            rgba  = mc.to_rgba(color)
            lum   = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            txt_c = "white" if lum < 0.5 else "black"
            for j in range(len(col_headers)):
                cell = tbl[i + 1, j]
                cell.set_height(data_hs[i])
                if j == 0:
                    cell.set_facecolor(color)
                    cell.set_text_props(color=txt_c, fontsize=_fontsize_tbl, fontweight="bold")
                else:
                    cell.set_facecolor("#f4f4f4" if i % 2 == 0 else "white")
                    cell.set_text_props(fontsize=_fontsize_tbl)

    fig.canvas.draw()

    if col_widths is not None or col_width_adjust_in is not None:
        # auto_set_column_width re-derives each column's "natural" width from rendered
        # text on every render pass (including the one fig.savefig() triggers), which
        # would silently undo any width we set here. So instead of nudging widths
        # in-place, we take over sizing for the whole table in one shot: measure each
        # column's natural width ourselves (same method mpl's auto-sizing uses),
        # apply overrides/deltas on top, freeze the result, and stop auto-sizing from
        # ever touching these columns again.
        _renderer = fig.canvas.get_renderer()
        for tbl in tbl_list:
            rows_in_tbl = sorted({r for (r, _c) in tbl.get_celld().keys()})
            for j in range(len(col_headers)):
                cells_in_col = [tbl[r, j] for r in rows_in_tbl]
                new_w = max(c.get_required_width(_renderer) for c in cells_in_col)
                if col_widths is not None and j < len(col_widths) and col_widths[j] is not None:
                    new_w = col_widths[j]
                if col_width_adjust_in is not None and j < len(col_width_adjust_in) and col_width_adjust_in[j]:
                    new_w += col_width_adjust_in[j] / fig.get_figwidth()
                for c in cells_in_col:
                    c.set_width(new_w)
            tbl._autoColumns = []

    # inch-based offsets converted to this figure's fraction units, so the
    # widening stays visually the same regardless of the legend column's width
    table_w_extra   = L["TABLE_W_EXTRA_IN"]   / fig.get_figwidth()
    table_wspace    = L["TABLE_GAP_IN"]       / fig.get_figwidth()
    plot_table_gap  = L["PLOT_TABLE_GAP_IN"]  / fig.get_figheight()
    for col_idx, tbl_ax in enumerate(tbl_axes_list):
        pos = tbl_ax.get_position()
        x0, y0, w, h = pos.x0, pos.y0, pos.width, pos.height
        if ncols == 2:
            if col_idx == 0:
                x0 -= table_w_extra
                w  += table_w_extra - table_wspace / 2
            else:
                x0 += table_wspace / 2
                w  += table_w_extra - table_wspace / 2
        y0 += plot_table_gap
        tbl_ax.set_position([x0, y0, w, h])

    save_path = Path(save_dir) / f"{compact_formula}_{target_icsd}_path{target_path}_NEB.svg"
    fig.savefig(str(save_path), format="svg", bbox_inches="tight")
    # plt.show()'s inline capture does not use bbox_inches="tight" the way the
    # explicit savefig() above does, so a rotated ylabel sitting near the
    # right panel's edge gets silently clipped in the notebook's displayed
    # PNG even though it renders correctly in the saved SVG. Match the
    # savefig behavior for the inline display too, restoring the prior
    # setting immediately after so it doesn't leak into other cells' plots.
    _prev_savefig_bbox = plt.rcParams["savefig.bbox"]
    plt.rcParams["savefig.bbox"] = "tight"
    try:
        plt.show()
    finally:
        plt.rcParams["savefig.bbox"] = _prev_savefig_bbox
    plt.close(fig)
    print(f"Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Section 13 — NEB test comparison plot (original vs test cases, single table)
# ─────────────────────────────────────────────────────────────────────────────

_NEB_TEST_LAYOUT = dict(
    FIG_W            = 6.5,
    FIG_H            = 10,
    HSPACE           = 0.5,
    WSPACE           = 0.3,
    LEGEND_ROW_SIZE  = 1.0,
    LEGEND_TABLE_GAP = -0.13,
    WRAP_MAP         = [14, 8, 8, 8, 16, 16],
    FONTSIZE_TBL     = 8.5,
    HEADER_LINES     = 1.8,
    LEGEND_X         = 0.5,
    LEGEND_Y         = 1.5,
)


def plot_neb_test_comparison(
    target_icsd, target_path,
    mlip_key, case_keys,
    # ── data produced by the notebook ────────────────────────────────────────
    dft_index_lookup,
    df_ICSD,
    neb_paths_dft,
    neb_structures_all_mlip,
    mlip_index_lookup,
    df_classification,
    df_classification_DFT,
    # ── optional ─────────────────────────────────────────────────────────────
    mace_key         = "MACE-MP0_medium",
    use_fit          = False,
    save_dir         = None,
    figures_dir      = None,
    legend_row_size  = None,
    legend_y         = None,
    legend_ncol      = None,
    exclude_from_plot = None,   # set/list of FP names to hide from NEB barrier panels
):
    """
    Two-panel comparison for one high-error (ICSD, Path):

    Left  panel  — full NEB: DFT + original FP + MACE (shows the "wrong" path)
    Right panel  — full NEB: DFT + test cases
    Bottom table — DFT, original FP, MACE, all test cases (single wide table)

    Parameters
    ----------
    mlip_key  : str   e.g. "CHGNET"
    case_keys : list  e.g. ["CHGNET_case1_relax", "CHGNET_case2_interp", ...]
    """
    L = {**_NEB_TEST_LAYOUT}
    if legend_row_size is not None:
        L["LEGEND_ROW_SIZE"] = legend_row_size
    if legend_y is not None:
        L["LEGEND_Y"] = legend_y
    if figures_dir is None:
        figures_dir = Path("figures")
    save_dir = save_dir or figures_dir

    key = (str(target_icsd), str(target_path))
    if key not in dft_index_lookup:
        print(f"Skipping {key} — not in dft_index_lookup")
        return

    idx_dft    = dft_index_lookup[key]
    row_icsd   = df_ICSD.loc[df_ICSD["CollectionCode"].astype(str) == str(target_icsd)]
    if row_icsd.empty:
        print(f"  Note: ICSD {target_icsd} not in df_ICSD (may be theoretical/computed) — using ICSD ID as label")
        compact_formula = f"ICSD-{target_icsd}"
    else:
        sum_formula     = row_icsd["SumFormula"].values[0]
        compact_formula = to_latex_subscript(str(Composition(sum_formula).reduced_formula))
    dft_images      = neb_paths_dft[idx_dft]
    mode            = "full"

    # Display order and colors
    all_models   = ["DFT", mlip_key, mace_key] + list(case_keys)
    model_colors = {"DFT": "black"}
    for i, name in enumerate([mlip_key, mace_key] + list(case_keys)):
        model_colors[name] = f"tab:{_COLOR_LIST[i % len(_COLOR_LIST)]}"

    # mlip_dicts
    def _get_images(name):
        idx = mlip_index_lookup.get(name, {}).get(mode, {}).get(key)
        return None if idx is None else neb_structures_all_mlip[name][mode][idx]

    _exclude = set(exclude_from_plot or [])
    left_mlip_dict  = {n: img for n in [mlip_key, mace_key]
                       if n not in _exclude and (img := _get_images(n)) is not None}
    right_mlip_dict = {n: img for n in case_keys
                       if n not in _exclude and (img := _get_images(n)) is not None}

    # Figure
    fig = plt.figure(figsize=(L["FIG_W"], L["FIG_H"]), dpi=350)
    gs  = GridSpec(3, 2, figure=fig,
                   height_ratios=[4, L["LEGEND_ROW_SIZE"], 4],
                   hspace=L["HSPACE"], wspace=L["WSPACE"],
                   top=0.92, bottom=0.02, left=0.06, right=0.97)

    ax_left  = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    shared_kw = dict(
        mode=mode, use_fit=use_fit, show_area=True, fontsize=12,
        show_legend=False, add_barrier_info=False,
        return_barrier_data=True, DFT_text_loc=(0, 0),
        material_label="", zero_start=True, show=False, normalized=True,
        color_map=model_colors,
    )
    _, left_bd  = _plot_neb_bands_multi(dft_images=dft_images, mlip_dict=left_mlip_dict,
                                         ax=ax_left,  show_ylabel=True,  **shared_kw)
    _, right_bd = _plot_neb_bands_multi(dft_images=dft_images, mlip_dict=right_mlip_dict,
                                         ax=ax_right, show_ylabel=False, **shared_kw)

    ax_left.set_title(f"Original ({mlip_key} + {mace_key})", fontsize=9)
    ax_right.set_title("Test Cases", fontsize=9)

    # Legend
    leg_ax = fig.add_subplot(gs[1, :])
    leg_ax.axis("off")
    handles = [mlines.Line2D([], [], color=model_colors[n], marker="o",
                              markersize=7, linewidth=2, label=n)
               for n in all_models if n in model_colors]
    _ncol = legend_ncol if legend_ncol is not None else len(all_models)
    leg_ax.legend(handles=handles, loc="center", ncol=_ncol,
                  fontsize=8, frameon=True, title="Potential", title_fontsize=8,
                  bbox_to_anchor=(L["LEGEND_X"], L["LEGEND_Y"]))

    # Enrich barrier data from both panels
    def _enrich(bd):
        out = {}
        for name, (text, color) in bd.items():
            fwd_m  = re.search(r"Forward:\s*(-?[\d.]+)",  text)
            bwd_m  = re.search(r"Backward:\s*(-?[\d.]+)", text)
            area_m = re.search(r"\|ΔArea\|:\s*([\d.]+)",  text)
            fwd  = float(fwd_m.group(1))  if fwd_m  else float("nan")
            bwd  = float(bwd_m.group(1))  if bwd_m  else float("nan")
            area = float(area_m.group(1)) if area_m else None
            if name == "DFT":
                dft_row  = df_classification_DFT[
                    (df_classification_DFT["ICSD"].astype(str) == str(target_icsd)) &
                    (df_classification_DFT["Path"].astype(str) == str(target_path))]
                topology = dft_row["Pathway Topology"].values[0]        if not dft_row.empty else "—"
                endpoint = dft_row["Endpoint Energy Ranking"].values[0] if not dft_row.empty else "—"
            elif name in df_classification and mode in df_classification[name]:
                df_m     = df_classification[name][mode]
                mlip_row = df_m[(df_m["ICSD"].astype(str) == str(target_icsd)) &
                                (df_m["Path"].astype(str) == str(target_path))]
                topology = mlip_row["Pathway Topology"].values[0]        if not mlip_row.empty else "—"
                endpoint = mlip_row["Endpoint Energy Ranking"].values[0] if not mlip_row.empty else "—"
            else:
                topology = endpoint = "—"
            out[name] = dict(color=color, forward=fwd, backward=bwd,
                             area=area, topology=str(topology), endpoint=str(endpoint))
        return out

    enriched_left  = _enrich(left_bd)
    enriched_right = _enrich(right_bd)

    # Merge in display order
    enriched_all = {}
    for name in all_models:
        if name in enriched_left:
            enriched_all[name] = enriched_left[name]
        elif name in enriched_right:
            enriched_all[name] = enriched_right[name]

    # Single wide table
    col_headers  = ["Model", "Fwd\n(eV)", "Bwd\n(eV)", "|ΔArea|", "Topology", "Endpoint"]
    rows_present = [n for n in all_models if n in enriched_all]
    cell_text    = []
    for name in rows_present:
        d      = enriched_all[name]
        area_s = f"{d['area']:.4f}" if d["area"] is not None else "—"
        cell_text.append([
            textwrap.fill(name,                   width=L["WRAP_MAP"][0]),
            textwrap.fill(f"{d['forward']:.3f}",  width=L["WRAP_MAP"][1]),
            textwrap.fill(f"{d['backward']:.3f}", width=L["WRAP_MAP"][2]),
            textwrap.fill(area_s,                 width=L["WRAP_MAP"][3]),
            textwrap.fill(d["topology"],           width=L["WRAP_MAP"][4]),
            textwrap.fill(d["endpoint"],           width=L["WRAP_MAP"][5]),
        ])

    row_lines   = [max(len(cell_text[i][j].split("\n")) for j in range(len(col_headers)))
                   for i in range(len(rows_present))]
    total_lines = L["HEADER_LINES"] + sum(row_lines)
    header_h    = L["HEADER_LINES"] / total_lines
    data_hs     = [lc / total_lines for lc in row_lines]

    tbl_ax = fig.add_subplot(gs[2, :])
    tbl_ax.axis("off")
    tbl_ax.set_title("Full Relaxation — Barrier Summary", fontsize=9, fontweight="bold", pad=4)

    tbl = tbl_ax.table(cellText=cell_text, colLabels=col_headers,
                       cellLoc="center", loc="upper center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.auto_set_column_width(col=list(range(len(col_headers))))

    for j in range(len(col_headers)):
        tbl[0, j].set_facecolor("#b0b0b0")
        tbl[0, j].set_height(header_h)
        tbl[0, j].set_text_props(fontweight="bold", fontsize=L["FONTSIZE_TBL"])

    for i, name in enumerate(rows_present):
        color = enriched_all[name]["color"]
        rgba  = mc.to_rgba(color)
        lum   = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
        txt_c = "white" if lum < 0.5 else "black"
        for j in range(len(col_headers)):
            cell = tbl[i + 1, j]
            cell.set_height(data_hs[i])
            if j == 0:
                cell.set_facecolor(color)
                cell.set_text_props(color=txt_c, fontsize=L["FONTSIZE_TBL"], fontweight="bold")
            else:
                cell.set_facecolor("#f4f4f4" if i % 2 == 0 else "white")
                cell.set_text_props(fontsize=L["FONTSIZE_TBL"])

    fig.suptitle(f"Material: {compact_formula}  (ICSD {target_icsd})", fontsize=13, y=0.97)
    save_path = (Path(save_dir) /
                 f"{compact_formula}_{target_icsd}_path{target_path}_NEB_test.svg")
    fig.savefig(str(save_path), format="svg", bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Saved: {save_path}")


# ── extra import needed by the wrapper functions below ──────────────────────
from matplotlib.ticker import FuncFormatter


# ═══════════════════════════════════════════════════════════════════════════
# Notebook-cell wrappers: case-study path, two-path force-error figures,
# and the single-path DFT/Full/Static comparison. Added to let the
# corresponding notebook cells shrink to a single function call while every
# tunable knob remains overridable via keyword arguments. Defaults below
# match exactly what each original cell had hardcoded.
# ═══════════════════════════════════════════════════════════════════════════

def _angle_between(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1., 1.))))


def plot_case_study_neb_path(
    target_icsd, target_path,
    df_dft_valid, df_classification_common,
    dft_index_lookup, df_ICSD, neb_paths_dft, neb_structures_all_mlip,
    mlip_index_lookup, df_classification, df_classification_DFT,
    figures_dir, outlier_threshold,
    plot_mlips=("MACE-MP0_medium", "UMA_s1_p1", "CHGNET"),
    endpoint_col_header="Endpoint\nenergy\nrank.",
    area_col_header="Integrated\nenergy-profile \ndiff. (eV)",
    topology_col_header="Energy-profile\nshape",
    fp_label="FP",
    full_table_title="Full FP-NEB",
    static_table_title="Static FP",
    table_fontsize=10,
    plot_gap_in=0.3,
    curve_title_prefix="",
    table_gap_in=-0.5,
    legend_x_shift_in=-0.25,
    legend_handlelength=2.5,
    cell_pad=0.18,
    table_w_extra_in=1.5,
    plot_table_gap_in=0.1,
    panel_title_fontsize=14,
    panel_label_fontsize=14,
    fig_h_in=9.0,
    fig_w_in=None,
    legend_col_w_in=1.4,
    plot_row_ratio=4.5,
    table_row_ratio=2.5,
    legend_fontsize=12,
    legend_handletextpad=0.5,
    panel_tick_fontsize_offset=0,
    panel_xlabel_pad=2.5,
    table_title_fontsize=14,
    col_width_adjust_in=(0.1, None, None, None, None, None),
    panel_n_xticks=5,
    use_fit=False,
    apply_font_rcparams=True,
    full_curve_label=None,
    static_curve_label=None,
    show_ylabel=True,   # "Energy (eV)" on the curve panels: True = both, False = neither,
                        # "left" = only the full-mode panel, "right" = only the static-mode panel
):
    """Case-study single-path summary (was notebook cell "12a").

    Excludes FPs whose forward/backward barrier error exceeds
    outlier_threshold (checked across both full and static modes) from the
    barrier panel, then calls plot_neb_summary(...) with that filtered set.
    """
    if apply_font_rcparams:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["Arial"]
        plt.rcParams["mathtext.fontset"] = "custom"
        plt.rcParams["svg.fonttype"] = "none"
        plt.rcParams["mathtext.rm"] = "Arial"
        plt.rcParams["mathtext.it"] = "Arial:italic"
        plt.rcParams["mathtext.bf"] = "Arial:bold"

    plot_mlips = list(plot_mlips)

    # ── Determine which FPs have barrier error > outlier_threshold ────────
    # (checked across both full and static modes; fwd or bwd error above
    # threshold → excluded)
    _high_err_mlips = set()
    _df_v = df_dft_valid.copy()
    _df_v["ICSD"] = _df_v["ICSD"].astype(str)
    _df_v["Path"] = _df_v["Path"].astype(str)
    _row_d = _df_v[(_df_v["ICSD"] == str(target_icsd)) & (_df_v["Path"] == str(target_path))]

    for _m in plot_mlips:
        for _mode in ["full", "static"]:
            if _m not in df_classification_common or _mode not in df_classification_common[_m]:
                continue
            _df_m = df_classification_common[_m][_mode].copy()
            _df_m["ICSD"] = _df_m["ICSD"].astype(str)
            _df_m["Path"] = _df_m["Path"].astype(str)
            _row_m = _df_m[(_df_m["ICSD"] == str(target_icsd)) & (_df_m["Path"] == str(target_path))]
            if _row_m.empty or _row_d.empty:
                continue
            _fwd_err = abs(_row_m.iloc[0]["energy_forward_barrier"]  - _row_d.iloc[0]["energy_forward_barrier"])
            _bwd_err = abs(_row_m.iloc[0]["energy_backward_barrier"] - _row_d.iloc[0]["energy_backward_barrier"])
            if _fwd_err > outlier_threshold or _bwd_err > outlier_threshold:
                _high_err_mlips.add(_m)

    _mlip_names_for_plot = [m for m in plot_mlips if m not in _high_err_mlips]

    if _high_err_mlips:
        print(f"Excluded from barrier panel (err > {outlier_threshold} eV): {sorted(_high_err_mlips)}")
    print(f"Plotting: {_mlip_names_for_plot}")

    return plot_neb_summary(
        target_icsd             = target_icsd,
        target_path              = target_path,
        mlip_names               = plot_mlips,
        table_fontsize           = table_fontsize,
        plot_gap_in              = plot_gap_in,
        dft_index_lookup         = dft_index_lookup,
        df_ICSD                  = df_ICSD,
        neb_paths_dft            = neb_paths_dft,
        neb_structures_all_mlip  = neb_structures_all_mlip,
        mlip_index_lookup        = mlip_index_lookup,
        df_classification        = df_classification,
        df_classification_DFT    = df_classification_DFT,
        use_fit                  = use_fit,
        figures_dir              = figures_dir,
        mlip_names_full_plot     = _mlip_names_for_plot,
        endpoint_col_header      = endpoint_col_header,
        area_col_header          = area_col_header,
        topology_col_header      = topology_col_header,
        fp_label                 = fp_label,
        full_table_title         = full_table_title,
        static_table_title       = static_table_title,
        full_curve_label         = full_curve_label if full_curve_label is not None else full_table_title,
        static_curve_label       = static_curve_label if static_curve_label is not None else static_table_title,
        fig_w_in                 = fig_w_in,
        curve_title_prefix       = curve_title_prefix,
        table_gap_in             = table_gap_in,
        legend_x_shift_in        = legend_x_shift_in,
        legend_handlelength      = legend_handlelength,
        cell_pad                 = cell_pad,
        table_w_extra_in         = table_w_extra_in,
        plot_table_gap_in        = plot_table_gap_in,
        panel_title_fontsize     = panel_title_fontsize,
        panel_label_fontsize     = panel_label_fontsize,
        fig_h_in                 = fig_h_in,
        legend_col_w_in          = legend_col_w_in,
        plot_row_ratio           = plot_row_ratio,
        table_row_ratio          = table_row_ratio,
        legend_fontsize          = legend_fontsize,
        legend_handletextpad     = legend_handletextpad,
        panel_tick_fontsize_offset = panel_tick_fontsize_offset,
        panel_xlabel_pad         = panel_xlabel_pad,
        table_title_fontsize     = table_title_fontsize,
        col_width_adjust_in      = list(col_width_adjust_in),
        panel_n_xticks           = panel_n_xticks,
        show_ylabel              = show_ylabel,
    )


# mode -> the one protocol each force-error table is allowed to hold here.
# "full"   panels read force errors on the FP-generated (full-mode) NEB path,
#          i.e. DFT evaluated statically on the FP's own final full-mode images.
# "static" panels read force errors on the DFT-NEB path, i.e. the FP evaluated
#          statically on the DFT-NEB's own images.
_FORCE_ERROR_MODE_PROTOCOL = {
    "full": "dft_static_on_fp_neb",
    "static": "fp_static_on_dft_neb",
}


def _validate_force_errors(force_errors, expected_protocol, fn_name):
    required_cols = {"fp_key", "icsd_id", "source_path_id", "protocol",
                      "calculation_stage", "structure_source", "image_index",
                      "mean_abs_delta_force_magnitude", "mean_force_angle_error_deg"}
    missing = required_cols - set(force_errors.columns)
    if missing:
        raise ValueError(
            f"{fn_name}: force_errors is missing required column(s) {sorted(missing)} "
            f"— pass a table built by the force-error construction cells "
            f"(full_fp_neb_path_force_errors / dft_neb_path_force_errors), not an "
            f"ad-hoc or legacy df_all-style frame."
        )
    seen_protocols = set(force_errors["protocol"].unique())
    if seen_protocols - {expected_protocol}:
        raise ValueError(
            f"{fn_name}: force_errors contains protocol(s) {sorted(seen_protocols)} "
            f"but this function only accepts protocol={expected_protocol!r}. "
            f"Passing the wrong protocol-specific table (or a mixed one) would "
            f"silently plot force errors from the wrong scientific workflow."
        )


def _neb_two_path_force_error_figure(
    mode,   # "full" or "static" — which FP-NEB mode is compared against DFT
    sel_mlip, selected_paths,
    model_names, df_ICSD, df_dft_valid, df_classification_common,
    dft_index_lookup, mlip_index_lookup, neb_paths_dft, neb_structures_all_mlip,
    endpoint_rmsd_data, force_errors,
    base_font, x_axis_mode, n_xticks_rc, label_pad,
    rmsd_fontsize, show_endpoint_rmsd, rmsd_left_x_shift, rmsd_right_x_shift, title_pad,
    show_top_legend, top_legend_x, top_legend_y,
    degree_symbol,
    panel_label_x, panel_label_y, show_panel_label,
    wspace_top_in, wspace_bottom_in, hspace_rows_in,
    show_axis_labels, xlabel_image, xlabel_rc, ylabel_energy, ylabel_force_left, ylabel_force_right,
    show_axis_arrows, arrow_left_label, arrow_right_label,
    arrow_left_y_frac, arrow_right_y_frac, arrow_left_x_frac, arrow_right_x_frac,
    arrow_left_len_frac, arrow_right_len_frac,
    mean_colors, twin_colors,
    yticks_energy, yticks_force_left, yticks_force_right,
    legend_handletextpad,
    apply_font_rcparams,
):
    _validate_force_errors(force_errors, _FORCE_ERROR_MODE_PROTOCOL[mode],
                            f"_neb_two_path_force_error_figure(mode={mode!r})")
    if apply_font_rcparams:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["Arial"]
        plt.rcParams["mathtext.fontset"] = "custom"
        plt.rcParams["svg.fonttype"] = "none"
        plt.rcParams["mathtext.rm"] = "Arial"
        plt.rcParams["mathtext.it"] = "Arial:italic"
        plt.rcParams["mathtext.bf"] = "Arial:bold"

    marker      = "D" if mode == "full" else "P"
    series_color = mean_colors[0] if mode == "full" else twin_colors[0]

    def _style_neb_ax_img(ax, ylabel=True, fontsize=base_font):
        ax.tick_params(labelsize=fontsize - 1, direction="in", top=True, right=True)
        if x_axis_mode == "reaction_coordinate":
            ax.set_xticks(np.linspace(0, 1, n_xticks_rc))
        if show_axis_labels:
            if ylabel:
                ax.set_ylabel(ylabel_energy, fontsize=fontsize, labelpad=label_pad)

    def _annotate_rmsd_img(ax, x_img, rmsd, label, fontsize=base_font, x_text_shift=0.0):
        ylo, yhi = ax.get_ylim()
        dy = 0.1 * (yhi - ylo)
        ax.annotate(
            f"{label}\nRMSD={rmsd:.3f} Å",
            xy=(x_img, ylo), xytext=(x_img + x_text_shift, ylo - dy),
            fontsize=fontsize - 2, fontweight="bold", ha="center", va="top", color="black",
            annotation_clip=False,
        )

    def get_path_plot_data(mlip, icsd, path_id):
        ep_key = (str(icsd), str(path_id))
        _mn    = model_names.get(mlip, mlip)

        row_icsd = df_ICSD.loc[df_ICSD["CollectionCode"].astype(str) == str(icsd)]
        if row_icsd.empty:
            compact_formula = f"ICSD {icsd}"
        else:
            compact_formula = to_latex_subscript(
                str(Composition(row_icsd["SumFormula"].iloc[0]).reduced_formula)
            )

        _df_dv = df_dft_valid.copy()
        _df_dv["ICSD"] = _df_dv["ICSD"].astype(str)
        _df_dv["Path"] = _df_dv["Path"].astype(str)
        row_dv  = _df_dv[(_df_dv["ICSD"] == str(icsd)) & (_df_dv["Path"] == str(path_id))]
        dft_fwd = row_dv.iloc[0]["energy_forward_barrier"]  if not row_dv.empty else float("nan")
        dft_bwd = row_dv.iloc[0]["energy_backward_barrier"] if not row_dv.empty else float("nan")

        mode_barriers = {}
        for bm in ["full", "static"]:
            if bm not in df_classification_common.get(mlip, {}):
                continue
            _df_mc = df_classification_common[mlip][bm].copy()
            _df_mc["ICSD"] = _df_mc["ICSD"].astype(str)
            _df_mc["Path"] = _df_mc["Path"].astype(str)
            row_mc = _df_mc[(_df_mc["ICSD"] == str(icsd)) & (_df_mc["Path"] == str(path_id))]
            if not row_mc.empty:
                mf = row_mc.iloc[0]["energy_forward_barrier"]
                mb = row_mc.iloc[0]["energy_backward_barrier"]
                mode_barriers[bm] = {"fwd": mf, "bwd": mb,
                                    "dfe": abs(mf - dft_fwd),
                                    "dbe": abs(mb - dft_bwd)}

        dft_neb_idx     = dft_index_lookup.get(ep_key)
        mlip_full_idx   = mlip_index_lookup.get(mlip, {}).get("full",   {}).get(ep_key)
        mlip_static_idx = mlip_index_lookup.get(mlip, {}).get("static", {}).get(ep_key)

        print(f"  [{mlip} | ICSD={icsd} path={path_id}]  "
            f"dft_idx={dft_neb_idx}  full_idx={mlip_full_idx}  "
            f"static_idx={mlip_static_idx}  barriers={list(mode_barriers)}")

        def _imgs_to_xy(imgs):
            """x/e pair in the currently active x_axis_mode."""
            if imgs is None or len(imgs) == 0:
                return None, None
            try:
                if x_axis_mode == "reaction_coordinate":
                    s, e, _, _ = _get_neb_data(
                        imgs, normalized=True, zero_start=True, use_fit=False)
                    return s, e
                e = np.array([a.get_potential_energy() for a in imgs])
                return np.arange(len(imgs)), e - e[0]
            except Exception as ex:
                print(f"    energy extraction failed: {ex}")
                return None, None

        dft_imgs  = neb_paths_dft[dft_neb_idx] if dft_neb_idx is not None else None
        mode_idx  = mlip_full_idx if mode == "full" else mlip_static_idx
        mode_imgs = neb_structures_all_mlip[mlip][mode][mode_idx] if mode_idx is not None else None

        x_dft,  e_dft  = _imgs_to_xy(dft_imgs)
        x_mode, e_mode = _imgs_to_xy(mode_imgs)

        # reaction-coordinate mapping for panels (c)/(d) (force/angle data is
        # always indexed by image_idx on the plotted-mode path) — computed
        # regardless of the active x_axis_mode so switching modes doesn't
        # require recomputing data.
        rc_mode = None
        if mode_imgs is not None and len(mode_imgs) > 0:
            try:
                rc_mode, _, _, _ = _get_neb_data(
                    mode_imgs, normalized=True, zero_start=True, use_fit=False)
            except Exception as ex:
                print(f"    reaction-coordinate mapping failed: {ex}")

        ep_rd  = endpoint_rmsd_data.get(mlip, {}).get("full", {}).get(ep_key, {})
        rmsd0  = ep_rd.get("rmsd_img0",         float("nan"))
        rmsdN  = ep_rd.get("rmsd_img_last",     float("nan"))
        mdist0 = ep_rd.get("max_dist_img0",     float("nan"))
        mdistN = ep_rd.get("max_dist_img_last", float("nan"))

        mask   = ((force_errors["fp_key"]         == mlip) &
                (force_errors["icsd_id"]         == str(icsd)) &
                (force_errors["source_path_id"]  == str(path_id)))
        df_sel = force_errors[mask].sort_values("image_index").copy()

        return dict(
            ep_key=ep_key, compact_formula=compact_formula, mn=_mn,
            dft_fwd=dft_fwd, dft_bwd=dft_bwd, mode_barriers=mode_barriers,
            x_dft=x_dft, e_dft=e_dft,
            x_mode=x_mode, e_mode=e_mode,
            rmsd0=rmsd0, rmsdN=rmsdN, mdist0=mdist0, mdistN=mdistN,
            df_sel=df_sel, rc_mode=rc_mode,
            has_neb=(x_dft is not None or x_mode is not None),
        )

    # ── Collect data ─────────────────────────────────────────────────────────
    print("Collecting data …")
    path_data = [get_path_plot_data(sel_mlip, icsd, pid) for icsd, pid in selected_paths]
    _mn = model_names.get(sel_mlip, sel_mlip)

    # ── Build figure ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 6), dpi=350)
    neb_axes = [axes[0, 0], axes[0, 1]]
    fa_axes  = [axes[1, 0], axes[1, 1]]
    fa_twin  = [ax.twinx() for ax in fa_axes]

    panel_labels = [("(a)", "(c)"), ("(b)", "(d)")]

    for col, (d, (icsd, pid)) in enumerate(zip(path_data, selected_paths)):
        ax_neb  = neb_axes[col]
        ax_fa   = fa_axes[col]
        ax_fa_r = fa_twin[col]

        if not d["has_neb"]:
            ax_neb.text(0.5, 0.5, "NEB data not available",
                        transform=ax_neb.transAxes,
                        ha="center", va="center", fontsize=base_font, color="gray")
        else:
            if d["x_dft"] is not None:
                ax_neb.plot(d["x_dft"], d["e_dft"], "D-", color="#333333",
                            lw=1.8, ms=5, zorder=4, label="DFT")

            if d["x_mode"] is not None:
                ax_neb.plot(d["x_mode"], d["e_mode"], f"{marker}-", color=series_color,
                            lw=1.8, ms=5, zorder=3, label=f"{_mn} [{mode}]")
                if d["x_dft"] is not None:
                    _x_com = np.linspace(0, max(d["x_dft"][-1], d["x_mode"][-1]), 300)
                    ax_neb.fill_between(
                        _x_com,
                        np.interp(_x_com, d["x_dft"],  d["e_dft"]),
                        np.interp(_x_com, d["x_mode"], d["e_mode"]),
                        alpha=0.12, color=series_color, zorder=0,
                    )

            if x_axis_mode == "image" and d["x_dft"] is not None:
                ax_neb.set_xticks(d["x_dft"])

            _style_neb_ax_img(ax_neb, ylabel=True, fontsize=base_font)

            if yticks_energy[col] is not None:
                ax_neb.set_yticks(yticks_energy[col])

            fig.canvas.draw()
            if show_endpoint_rmsd and d["x_mode"] is not None:
                _annotate_rmsd_img(ax_neb, d["x_mode"][0],  d["rmsd0"],
                                    "Endpoint", fontsize=rmsd_fontsize,
                                    x_text_shift=-rmsd_left_x_shift)
                _annotate_rmsd_img(ax_neb, d["x_mode"][-1], d["rmsdN"],
                                    "Endpoint",   fontsize=rmsd_fontsize,
                                    x_text_shift=rmsd_right_x_shift)

        if show_panel_label[panel_labels[col][0].strip("()")]:
            ax_neb.text(panel_label_x, panel_label_y, panel_labels[col][0],
                        transform=ax_neb.transAxes,
                        fontsize=base_font - 1, va="top", fontweight="bold")
        ax_neb.set_title(f"{d['compact_formula']} (ICSD {icsd})",
                    fontsize=base_font, pad=title_pad)

        if not d["df_sel"].empty:
            idxs = d["df_sel"]["image_index"].values
            if x_axis_mode == "reaction_coordinate" and d.get("rc_mode") is not None:
                _rc    = d["rc_mode"]
                x_vals = np.array([_rc[i] if 0 <= i < len(_rc) else np.nan for i in idxs])
                ax_fa.set_xticks(np.linspace(0, 1, n_xticks_rc))
            else:
                x_vals = idxs
            ax_fa.plot(x_vals,   d["df_sel"]["mean_abs_delta_force_magnitude"], "o-",  color=series_color,
                        markersize=5, linewidth=1.8, zorder=3)
            ax_fa_r.plot(x_vals, d["df_sel"]["mean_force_angle_error_deg"], "s--", color=series_color,
                        markersize=4, linewidth=1.4, alpha=0.5)
            ax_fa.axhline(0, color="k", lw=0.5, ls="--", alpha=0.35)
        else:
            ax_fa.text(0.5, 0.5, "Force data not available",
                        transform=ax_fa.transAxes,
                        ha="center", va="center", fontsize=base_font, color="gray")

        if show_axis_labels:
            _xlabel_fa = xlabel_rc if x_axis_mode == "reaction_coordinate" else xlabel_image
            ax_fa.set_xlabel(_xlabel_fa, fontsize=base_font, labelpad=label_pad)
            ax_fa.set_ylabel(ylabel_force_left,  fontsize=base_font, labelpad=label_pad)
            ax_fa_r.set_ylabel(ylabel_force_right, fontsize=base_font - 1, labelpad=label_pad)

        ax_fa.tick_params(labelsize=base_font - 1, direction="in")
        ax_fa_r.tick_params(labelsize=base_font - 2, direction="in")
        if degree_symbol:
            ax_fa_r.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:.0f}°"))
        ax_fa.spines["top"].set_visible(False)

        if yticks_force_left[col] is not None:
            ax_fa.set_yticks(yticks_force_left[col])
        if yticks_force_right[col] is not None:
            ax_fa_r.set_yticks(yticks_force_right[col])

        if show_axis_arrows:
            _aly  = arrow_left_y_frac[col]
            _ary  = arrow_right_y_frac[col]
            _alx  = arrow_left_x_frac[col]
            _arx  = arrow_right_x_frac[col]
            _allen = arrow_left_len_frac[col]
            _arlen = arrow_right_len_frac[col]
            _text_dy = 0.03   # gap between arrow and the text above it (axes fraction)

            ax_fa.annotate(
                "", xy=(_alx, _aly), xycoords="axes fraction",
                xytext=(_alx + _allen, _aly), textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color=series_color, lw=1.8),
            )
            ax_fa.text(_alx + _allen / 2, _aly + _text_dy, arrow_left_label,
                    transform=ax_fa.transAxes, ha="center", va="bottom",
                    fontsize=base_font - 2, color=series_color)

            ax_fa.annotate(
                "", xy=(_arx, _ary), xycoords="axes fraction",
                xytext=(_arx - _arlen, _ary), textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color=series_color, lw=1.4),
            )
            ax_fa.text(_arx - _arlen / 2, _ary + _text_dy, arrow_right_label,
                    transform=ax_fa.transAxes, ha="center", va="bottom",
                    fontsize=base_font - 2, color=series_color)

        if show_panel_label[panel_labels[col][1].strip("()")]:
            ax_fa.text(panel_label_x, panel_label_y, panel_labels[col][1],
                        transform=ax_fa.transAxes,
                        fontsize=base_font - 1, va="top", fontweight="bold")

    # ── Top legend: DFT / <FP>, between panels (a) and (b) ─────────────────
    if show_top_legend:
        top_leg_entries = [
            mlines.Line2D([], [], color="#333333", marker="D", ls="-", ms=4, lw=1.6, label="DFT"),
            mlines.Line2D([], [], color=series_color, marker=marker, ls="-", ms=4, lw=1.6, label=_mn),
        ]
        fig.legend(
            handles=top_leg_entries, ncol=1, loc="center",
            bbox_to_anchor=(top_legend_x - 0.005, top_legend_y - 0.08),
            frameon=True, edgecolor="#ccc",
            handlelength=1.8, handletextpad=legend_handletextpad, columnspacing=1.0,
            prop={"size": base_font - 2},
        )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)

    # ── independent left/right spacing for the top row and bottom row ────────
    fig.canvas.draw()
    def _spread_pair(ax_left, ax_right, gap_in):
        if not gap_in:
            return
        shift = (gap_in / fig.get_figwidth()) / 2
        pL = ax_left.get_position();  ax_left.set_position([pL.x0 - shift, pL.y0, pL.width, pL.height])
        pR = ax_right.get_position(); ax_right.set_position([pR.x0 + shift, pR.y0, pR.width, pR.height])

    _spread_pair(neb_axes[0], neb_axes[1], wspace_top_in)
    _spread_pair(fa_axes[0],  fa_axes[1],  wspace_bottom_in)

    def _shift_row(ax_list, dy):
        if not dy:
            return
        for ax in ax_list:
            p = ax.get_position()
            ax.set_position([p.x0, p.y0 + dy, p.width, p.height])

    _hspace_shift = (hspace_rows_in / fig.get_figheight()) / 2
    _shift_row(neb_axes, _hspace_shift)           # push (a)/(b) up
    _shift_row(fa_axes + fa_twin, -_hspace_shift) # push (c)/(d) — and their twin axes — down

    plt.show()
    return fig, axes


def plot_two_path_force_error_full(
    sel_mlip, selected_paths,
    model_names, df_ICSD, df_dft_valid, df_classification_common,
    dft_index_lookup, mlip_index_lookup, neb_paths_dft, neb_structures_all_mlip,
    endpoint_rmsd_data, full_fp_neb_path_force_errors,
    base_font=12, x_axis_mode="reaction_coordinate", n_xticks_rc=5, label_pad=2,
    rmsd_fontsize=None, show_endpoint_rmsd=True, rmsd_left_x_shift=0.0, rmsd_right_x_shift=0.0,
    title_pad=6,
    show_top_legend=True, top_legend_x=0.505, top_legend_y=0.975,
    degree_symbol=False,
    panel_label_x=-0.15, panel_label_y=1.06,
    show_panel_label=None,
    wspace_top_in=0.04, wspace_bottom_in=0.04, hspace_rows_in=0.0,
    show_axis_labels=True, xlabel_image="Image index", xlabel_rc="Reaction Coordinate",
    ylabel_energy="Energy (eV)",
    ylabel_force_left=r"Mean $\left|\Delta\left|F\right|\right|$ (eV/Å)",
    ylabel_force_right=r"Mean $\Delta\theta$ (°)",
    show_axis_arrows=True,
    arrow_left_label=r"Mean $\left|\Delta\left|F\right|\right|$",
    arrow_right_label=r"Mean $\Delta\theta$",
    arrow_left_y_frac=(0.81, 0.72), arrow_right_y_frac=(0.25, 0.20),
    arrow_left_x_frac=(0.115, 0.08), arrow_right_x_frac=(0.95, 0.95),
    arrow_left_len_frac=(0.23, 0.23), arrow_right_len_frac=(0.23, 0.23),
    mean_colors=("#2166ac", "#4dac26", "#7b3294", "#1a9641"),
    twin_colors=("#d6604d", "#f4a582", "#c2523c", "#e08070"),
    yticks_energy=([0.00, 0.05, 0.10, 0.15, 0.20, 0.25], [-0.4, -0.2, 0.0, 0.2, 0.4, 0.6]),
    yticks_force_left=([0.000, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006], [0.000, 0.020, 0.040, 0.060, 0.080, 0.100]),
    yticks_force_right=([30, 40, 50, 60, 70, 80], [75, 80, 85, 90, 95, 100]),
    legend_handletextpad=0.1,
    apply_font_rcparams=True,
):
    """DFT vs Full-mode FP-NEB, 2 selected paths, 4-panel (a/b energy, c/d
    force+angle) figure. Was notebook cell "main force error figure".

    full_fp_neb_path_force_errors: force errors on the FP-generated (full-mode)
    NEB path (protocol dft_static_on_fp_neb, "Force errors on FP-NEB path").
    Passing any other protocol's table raises ValueError."""
    if rmsd_fontsize is None:
        rmsd_fontsize = base_font - 2
    if show_panel_label is None:
        show_panel_label = {"a": True, "b": True, "c": True, "d": True}
    return _neb_two_path_force_error_figure(
        mode="full",
        sel_mlip=sel_mlip, selected_paths=selected_paths,
        model_names=model_names, df_ICSD=df_ICSD, df_dft_valid=df_dft_valid,
        df_classification_common=df_classification_common,
        dft_index_lookup=dft_index_lookup, mlip_index_lookup=mlip_index_lookup,
        neb_paths_dft=neb_paths_dft, neb_structures_all_mlip=neb_structures_all_mlip,
        endpoint_rmsd_data=endpoint_rmsd_data, force_errors=full_fp_neb_path_force_errors,
        base_font=base_font, x_axis_mode=x_axis_mode, n_xticks_rc=n_xticks_rc, label_pad=label_pad,
        rmsd_fontsize=rmsd_fontsize, show_endpoint_rmsd=show_endpoint_rmsd,
        rmsd_left_x_shift=rmsd_left_x_shift, rmsd_right_x_shift=rmsd_right_x_shift, title_pad=title_pad,
        show_top_legend=show_top_legend, top_legend_x=top_legend_x, top_legend_y=top_legend_y,
        degree_symbol=degree_symbol,
        panel_label_x=panel_label_x, panel_label_y=panel_label_y, show_panel_label=show_panel_label,
        wspace_top_in=wspace_top_in, wspace_bottom_in=wspace_bottom_in, hspace_rows_in=hspace_rows_in,
        show_axis_labels=show_axis_labels, xlabel_image=xlabel_image, xlabel_rc=xlabel_rc,
        ylabel_energy=ylabel_energy, ylabel_force_left=ylabel_force_left, ylabel_force_right=ylabel_force_right,
        show_axis_arrows=show_axis_arrows, arrow_left_label=arrow_left_label, arrow_right_label=arrow_right_label,
        arrow_left_y_frac=arrow_left_y_frac, arrow_right_y_frac=arrow_right_y_frac,
        arrow_left_x_frac=arrow_left_x_frac, arrow_right_x_frac=arrow_right_x_frac,
        arrow_left_len_frac=arrow_left_len_frac, arrow_right_len_frac=arrow_right_len_frac,
        mean_colors=mean_colors, twin_colors=twin_colors,
        yticks_energy=yticks_energy, yticks_force_left=yticks_force_left, yticks_force_right=yticks_force_right,
        legend_handletextpad=legend_handletextpad,
        apply_font_rcparams=apply_font_rcparams,
    )


def plot_two_path_force_error_static(
    sel_mlip, selected_paths,
    model_names, df_ICSD, df_dft_valid, df_classification_common,
    dft_index_lookup, mlip_index_lookup, neb_paths_dft, neb_structures_all_mlip,
    endpoint_rmsd_data, dft_neb_path_force_errors,
    base_font=12, x_axis_mode="reaction_coordinate", n_xticks_rc=5, label_pad=2,
    rmsd_fontsize=None, show_endpoint_rmsd=False, rmsd_left_x_shift=0.0, rmsd_right_x_shift=0.0,
    title_pad=6,
    show_top_legend=True, top_legend_x=0.5, top_legend_y=0.982,
    degree_symbol=False,
    panel_label_x=-0.14, panel_label_y=1.12,
    show_panel_label=None,
    wspace_top_in=0.0, wspace_bottom_in=0.0, hspace_rows_in=0.0,
    show_axis_labels=True, xlabel_image="Image index", xlabel_rc="Reaction Coordinate",
    ylabel_energy="Energy (eV)",
    ylabel_force_left=r"Mean $\left|\Delta\left|F\right|\right|$ (eV/Å)",
    ylabel_force_right=r"Mean $\Delta\theta$ (°)",
    show_axis_arrows=True,
    arrow_left_label=r"Mean $\left|\Delta\left|F\right|\right|$",
    arrow_right_label=r"Mean $\Delta\theta$",
    arrow_left_y_frac=(0.76, 0.885), arrow_right_y_frac=(0.25, 0.28),
    arrow_left_x_frac=(0.283, 0.115), arrow_right_x_frac=(0.858, 0.845),
    arrow_left_len_frac=(0.23, 0.23), arrow_right_len_frac=(0.23, 0.23),
    mean_colors=("#2166ac", "#4dac26", "#7b3294", "#1a9641"),
    twin_colors=("#d6604d", "#f4a582", "#c2523c", "#e08070"),
    yticks_energy=(None, None),
    yticks_force_left=(None, None),
    yticks_force_right=(None, None),
    legend_handletextpad=0.5,
    apply_font_rcparams=True,
):
    """DFT vs Static-mode FP evaluation, 2 selected paths, 4-panel (a/b
    energy, c/d force+angle) figure. Was notebook cell "main static force
    figure".

    dft_neb_path_force_errors: force errors on the DFT-NEB path (protocol
    fp_static_on_dft_neb, "Force errors on DFT-NEB path"). Passing any other
    protocol's table raises ValueError."""
    if rmsd_fontsize is None:
        rmsd_fontsize = base_font - 1.5
    if show_panel_label is None:
        show_panel_label = {"a": True, "b": True, "c": True, "d": True}
    return _neb_two_path_force_error_figure(
        mode="static",
        sel_mlip=sel_mlip, selected_paths=selected_paths,
        model_names=model_names, df_ICSD=df_ICSD, df_dft_valid=df_dft_valid,
        df_classification_common=df_classification_common,
        dft_index_lookup=dft_index_lookup, mlip_index_lookup=mlip_index_lookup,
        neb_paths_dft=neb_paths_dft, neb_structures_all_mlip=neb_structures_all_mlip,
        endpoint_rmsd_data=endpoint_rmsd_data, force_errors=dft_neb_path_force_errors,
        base_font=base_font, x_axis_mode=x_axis_mode, n_xticks_rc=n_xticks_rc, label_pad=label_pad,
        rmsd_fontsize=rmsd_fontsize, show_endpoint_rmsd=show_endpoint_rmsd,
        rmsd_left_x_shift=rmsd_left_x_shift, rmsd_right_x_shift=rmsd_right_x_shift, title_pad=title_pad,
        show_top_legend=show_top_legend, top_legend_x=top_legend_x, top_legend_y=top_legend_y,
        degree_symbol=degree_symbol,
        panel_label_x=panel_label_x, panel_label_y=panel_label_y, show_panel_label=show_panel_label,
        wspace_top_in=wspace_top_in, wspace_bottom_in=wspace_bottom_in, hspace_rows_in=hspace_rows_in,
        show_axis_labels=show_axis_labels, xlabel_image=xlabel_image, xlabel_rc=xlabel_rc,
        ylabel_energy=ylabel_energy, ylabel_force_left=ylabel_force_left, ylabel_force_right=ylabel_force_right,
        show_axis_arrows=show_axis_arrows, arrow_left_label=arrow_left_label, arrow_right_label=arrow_right_label,
        arrow_left_y_frac=arrow_left_y_frac, arrow_right_y_frac=arrow_right_y_frac,
        arrow_left_x_frac=arrow_left_x_frac, arrow_right_x_frac=arrow_right_x_frac,
        arrow_left_len_frac=arrow_left_len_frac, arrow_right_len_frac=arrow_right_len_frac,
        mean_colors=mean_colors, twin_colors=twin_colors,
        yticks_energy=yticks_energy, yticks_force_left=yticks_force_left, yticks_force_right=yticks_force_right,
        legend_handletextpad=legend_handletextpad,
        apply_font_rcparams=apply_font_rcparams,
    )


def plot_single_path_dft_full_static(
    sel_mlip, sel_icsd, sel_path_id,
    df_ICSD, df_dft_valid, df_classification_common,
    mlip_index_lookup, dft_index_lookup, endpoint_rmsd_data,
    neb_paths_dft, neb_structures_all_mlip,
    full_fp_neb_path_force_errors, dft_neb_path_force_errors,
    base_font=12, x_axis_mode="reaction_coordinate", n_xticks_rc=5, label_pad=2,
    rmsd_fontsize=None, show_endpoint_rmsd=True, rmsd_left_x_shift=0.0, rmsd_right_x_shift=0.0,
    title_pad=4,
    degree_symbol=False,
    panel_label_x=-0.15, panel_label_y=1.15,
    show_panel_label=None,
    wspace_top_in=0.05, wspace_bottom_in=0.05, hspace_rows_in=0.0,
    show_axis_labels=True, xlabel_image="Image index", xlabel_rc="Reaction Coordinate",
    ylabel_energy="Energy (eV)",
    ylabel_force_left=r"Mean $\left|\Delta\left|F\right|\right|$ (eV/Å)",
    ylabel_force_right=r"Mean $\Delta\theta$ (°)",
    show_axis_arrows=True,
    arrow_left_label=r"Mean $\left|\Delta\left|F\right|\right|$",
    arrow_right_label=r"Mean $\Delta\theta$",
    arrow_left_y_frac=(0.65, 0.8), arrow_right_y_frac=(0.3, 0.45),
    arrow_left_x_frac=(0.155, 0.45), arrow_right_x_frac=(0.95, 0.95),
    arrow_left_len_frac=(0.23, 0.23), arrow_right_len_frac=(0.23, 0.23),
    mean_colors=("#2166ac", "#4dac26", "#7b3294", "#1a9641"),
    atom0_colors=("#d6604d", "#f4a582", "#c2523c", "#e08070"),
    show_top_legend=True, top_legend_x=0.485, top_legend_y=0.91,
    apply_font_rcparams=True,
):
    """Single path: (a) DFT vs Full-mode FP-NEB energy, (b) DFT vs Static-mode
    FP energy, (c) force errors on the FP-generated (full-mode) NEB path
    (protocol dft_static_on_fp_neb, "Force errors on FP-NEB path" — case-study
    coverage only, shows "Force data not available" outside it), (d) force
    errors on the DFT-NEB path (protocol fp_static_on_dft_neb, "Force errors
    on DFT-NEB path", full common-path coverage). Both panels read from
    pre-built, explicitly protocol-tagged tables — no hidden on-the-fly force
    computation happens inside this function. Was notebook cell "Full FP-NEB
    vs Static FP: energy profile and force error (single path)"."""
    _validate_force_errors(full_fp_neb_path_force_errors, "dft_static_on_fp_neb",
                            "plot_single_path_dft_full_static (panel c)")
    _validate_force_errors(dft_neb_path_force_errors, "fp_static_on_dft_neb",
                            "plot_single_path_dft_full_static (panel d)")
    if apply_font_rcparams:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["Arial"]
        plt.rcParams["mathtext.fontset"] = "custom"
        plt.rcParams["svg.fonttype"] = "none"
        plt.rcParams["mathtext.rm"] = "Arial"
        plt.rcParams["mathtext.it"] = "Arial:italic"
        plt.rcParams["mathtext.bf"] = "Arial:bold"

    if rmsd_fontsize is None:
        rmsd_fontsize = base_font - 2
    if show_panel_label is None:
        show_panel_label = {"a": True, "b": True, "c": True, "d": True}

    def _style_neb_ax_img(ax, ylabel=True, fontsize=base_font):
        ax.tick_params(labelsize=fontsize - 1, direction="in", top=True, right=True)
        if x_axis_mode == "reaction_coordinate":
            ax.set_xticks(np.linspace(0, 1, n_xticks_rc))
        if show_axis_labels:
            if ylabel:
                ax.set_ylabel(ylabel_energy, fontsize=fontsize, labelpad=label_pad)

    def _annotate_rmsd_img(ax, x_img, rmsd, mdist, label, fontsize=base_font, x_text_shift=0.0):
        ylo, yhi = ax.get_ylim()
        dy = 0.1 * (yhi - ylo)
        ax.annotate(
            f"{label}\nRMSD={rmsd:.3f} Å",
            xy=(x_img, ylo), xytext=(x_img + x_text_shift, ylo - dy),
            fontsize=fontsize - 2, fontweight="bold", ha="center", va="top", color="black",
            annotation_clip=False,
        )

    # plain-text material name for logging
    _row_icsd = df_ICSD.loc[df_ICSD["CollectionCode"].astype(str) == str(sel_icsd)]
    if not _row_icsd.empty:
        _material = str(Composition(_row_icsd["SumFormula"].iloc[0]).reduced_formula)
    else:
        _material = f"ICSD {sel_icsd}"

    print(f"\n{'='*60}")
    print(f"{_material} | {sel_mlip} | ICSD={sel_icsd} | Path={sel_path_id}")

    # ── force data: panel (c) = protocol dft_static_on_fp_neb (FP-NEB path,
    # case-study coverage only), panel (d) = protocol fp_static_on_dft_neb
    # (DFT-NEB path, full common-path coverage) ──────────────────────────────
    def _select_force_errors(table):
        _mask = (
            (table["fp_key"]        == sel_mlip) &
            (table["icsd_id"]       == str(sel_icsd)) &
            (table["source_path_id"] == str(sel_path_id))
        )
        return table[_mask].sort_values("image_index").copy()

    df_sel      = _select_force_errors(full_fp_neb_path_force_errors)
    df_sel_stat = _select_force_errors(dft_neb_path_force_errors)

    _ep_key = (str(sel_icsd), str(sel_path_id))

    _df_dv = df_dft_valid.copy()
    _df_dv["ICSD"] = _df_dv["ICSD"].astype(str)
    _df_dv["Path"] = _df_dv["Path"].astype(str)
    _row_dv = _df_dv[(_df_dv["ICSD"] == str(sel_icsd)) & (_df_dv["Path"] == str(sel_path_id))]

    _mode_barriers = {}
    _dft_fwd = _dft_bwd = float("nan")

    if not _row_dv.empty:
        _dft_fwd = _row_dv.iloc[0]["energy_forward_barrier"]
        _dft_bwd = _row_dv.iloc[0]["energy_backward_barrier"]
        for _bm in ["full", "static"]:
            if _bm not in df_classification_common.get(sel_mlip, {}):
                continue
            _df_mc = df_classification_common[sel_mlip][_bm].copy()
            _df_mc["ICSD"] = _df_mc["ICSD"].astype(str)
            _df_mc["Path"] = _df_mc["Path"].astype(str)
            _row_mc = _df_mc[(_df_mc["ICSD"] == str(sel_icsd)) & (_df_mc["Path"] == str(sel_path_id))]
            if not _row_mc.empty:
                _mf = _row_mc.iloc[0]["energy_forward_barrier"]
                _mb = _row_mc.iloc[0]["energy_backward_barrier"]
                _mode_barriers[_bm] = {"fwd": _mf, "bwd": _mb,
                                        "dfe": abs(_mf - _dft_fwd),
                                        "dbe": abs(_mb - _dft_bwd)}

    # ── color maps — always include sel_mlip even if df_sel is empty ────────
    _mlips_for_color = sorted(set(df_sel["fp_key"].unique()) | {sel_mlip})
    _mc_ = {m: mean_colors[i  % len(mean_colors)]  for i, m in enumerate(_mlips_for_color)}
    _ac_ = {m: atom0_colors[i % len(atom0_colors)] for i, m in enumerate(_mlips_for_color)}
    _full_color = _mc_[sel_mlip]
    _stat_color = _ac_[sel_mlip]

    print(f"ICSD {sel_icsd}  —  Path {sel_path_id}")

    # ── data lookups ──────────────────────────────────────────────────────
    _mlip_full_idx   = mlip_index_lookup.get(sel_mlip, {}).get("full",   {}).get(_ep_key)
    _mlip_static_idx = mlip_index_lookup.get(sel_mlip, {}).get("static", {}).get(_ep_key)
    _dft_neb_idx     = dft_index_lookup.get(_ep_key)
    # single available endpoint-RMSD dataset — represents the full-NEB path (see (c) below)
    _ep_rd_full      = endpoint_rmsd_data.get(sel_mlip, {}).get("full", {}).get(_ep_key, {})

    def _imgs_to_xy(imgs):
        """x/e pair in the currently active x_axis_mode."""
        if imgs is None or len(imgs) == 0:
            return None, None
        e = np.array([a.get_potential_energy() for a in imgs])
        e = e - e[0]
        if x_axis_mode == "reaction_coordinate":
            try:
                s, _, _, _ = _get_neb_data(
                    imgs, normalized=True, zero_start=True, use_fit=False)
                return s, e
            except Exception as ex:
                print(f"    reaction-coordinate mapping failed: {ex}")
                return np.arange(len(imgs)), e
        return np.arange(len(imgs)), e

    _dft_imgs  = neb_paths_dft[_dft_neb_idx]                                    if _dft_neb_idx     is not None else None
    _full_imgs = neb_structures_all_mlip[sel_mlip]["full"][_mlip_full_idx]      if _mlip_full_idx   is not None else None
    _stat_imgs = neb_structures_all_mlip[sel_mlip]["static"][_mlip_static_idx]  if _mlip_static_idx is not None else None

    _x_dft,  _e_dft  = _imgs_to_xy(_dft_imgs)
    _x_full, _e_full = _imgs_to_xy(_full_imgs)
    _x_stat, _e_stat = _imgs_to_xy(_stat_imgs)

    # reaction-coordinate mapping for the force/angle panel (c) — (c) always
    # represents the FULL-NEB path (the only force/angle dataset available;
    # see the note on (d) below) — computed regardless of the active
    # x_axis_mode so switching modes needs no recompute. rc_stat is also
    # computed for symmetry/future use, even though (d) currently has no data.
    rc_full = rc_stat = None
    if _full_imgs is not None and len(_full_imgs) > 0:
        try:
            rc_full, _, _, _ = _get_neb_data(
                _full_imgs, normalized=True, zero_start=True, use_fit=False)
        except Exception as ex:
            print(f"    reaction-coordinate mapping (full) failed: {ex}")
    if _stat_imgs is not None and len(_stat_imgs) > 0:
        try:
            rc_stat, _, _, _ = _get_neb_data(
                _stat_imgs, normalized=True, zero_start=True, use_fit=False)
        except Exception as ex:
            print(f"    reaction-coordinate mapping (static) failed: {ex}")

    _rmsd0  = _ep_rd_full.get("rmsd_img0",         float("nan"))
    _rmsdN  = _ep_rd_full.get("rmsd_img_last",     float("nan"))
    _mdist0 = _ep_rd_full.get("max_dist_img0",     float("nan"))
    _mdistN = _ep_rd_full.get("max_dist_img_last", float("nan"))

    # ── Build figure: (a) full energy | (b) static energy
    #                  (c) force/angle (full) | (d) force/angle (static, no data available) ──
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 6), dpi=350)
    neb_axes = [axes[0, 0], axes[0, 1]]
    fa_axes  = [axes[1, 0], axes[1, 1]]
    fa_twin  = [ax.twinx() for ax in fa_axes]

    ax_full = neb_axes[0]
    ax_stat = neb_axes[1]
    ax_fa_full,  ax_fa_stat  = fa_axes
    ax_fa_full_r, ax_fa_stat_r = fa_twin

    panel_labels = ("(a)", "(b)", "(c)", "(d)")

    # ── panel (a): DFT vs full ────────────────────────────────────────────
    if _dft_neb_idx is None or _mlip_full_idx is None:
        ax_full.text(0.5, 0.5, "NEB data unavailable",
                    transform=ax_full.transAxes, ha="center", va="center",
                    fontsize=base_font, color="gray")
    else:
        ax_full.plot(_x_dft, _e_dft, "o-", color="#333333",
                    lw=1.8, ms=5, zorder=4, label="DFT")
        ax_full.plot(_x_full, _e_full, "o-", color=_full_color,
                    lw=1.8, ms=5, zorder=3, label=f"{sel_mlip} [full]")
        _x_com = np.linspace(0, max(_x_dft[-1], _x_full[-1]), 300)
        ax_full.fill_between(_x_com,
                            np.interp(_x_com, _x_dft,  _e_dft),
                            np.interp(_x_com, _x_full, _e_full),
                            alpha=0.12, color=_full_color, zorder=0)

        if x_axis_mode == "image":
            ax_full.set_xticks(_x_dft)
        _style_neb_ax_img(ax_full, ylabel=True, fontsize=base_font)

        fig.canvas.draw()
        if show_endpoint_rmsd:
            _annotate_rmsd_img(ax_full, _x_full[0],  _rmsd0, _mdist0,
                                "Endpoint", fontsize=rmsd_fontsize,
                                x_text_shift=-rmsd_left_x_shift)
            _annotate_rmsd_img(ax_full, _x_full[-1], _rmsdN, _mdistN,
                                "Endpoint", fontsize=rmsd_fontsize,
                                x_text_shift=rmsd_right_x_shift)

    if show_panel_label["a"]:
        ax_full.text(panel_label_x, panel_label_y, panel_labels[0],
                    transform=ax_full.transAxes,
                    fontsize=base_font - 1, va="top", fontweight="bold")
    ax_full.set_title("Full FP-NEB", fontsize=base_font, pad=title_pad, color=_full_color)

    # ── panel (b): DFT vs static ──────────────────────────────────────────
    if _dft_neb_idx is None or _mlip_static_idx is None:
        ax_stat.text(0.5, 0.5, "NEB data unavailable",
                    transform=ax_stat.transAxes, ha="center", va="center",
                    fontsize=base_font, color="gray")
    else:
        ax_stat.plot(_x_dft, _e_dft, "o-", color="#333333",
                    lw=1.8, ms=5, zorder=4, label="DFT")
        ax_stat.plot(_x_stat, _e_stat, "P-", color=_stat_color,
                    lw=1.8, ms=5, zorder=3, label=f"{sel_mlip} [static]")
        _x_com = np.linspace(0, max(_x_dft[-1], _x_stat[-1]), 300)
        ax_stat.fill_between(_x_com,
                            np.interp(_x_com, _x_dft,  _e_dft),
                            np.interp(_x_com, _x_stat, _e_stat),
                            alpha=0.12, color=_stat_color, zorder=0)
        # NOTE: the original script never filled DFT-vs-static (only DFT-vs-full ever got a
        # fill_between) — preserved here; add one if you actually want it shaded too.

        if x_axis_mode == "image":
            ax_stat.set_xticks(_x_dft)
        _style_neb_ax_img(ax_stat, ylabel=True, fontsize=base_font)

    if show_panel_label["b"]:
        ax_stat.text(panel_label_x, panel_label_y, panel_labels[1],
                    transform=ax_stat.transAxes,
                    fontsize=base_font - 1, va="top", fontweight="bold")
    ax_stat.set_title("Static FP", fontsize=base_font, pad=title_pad, color=_stat_color)

    # ── panel (c): force errors on the FP-generated (full-mode) NEB path ────
    if not df_sel.empty:
        idxs = df_sel["image_index"].values
        if x_axis_mode == "reaction_coordinate" and rc_full is not None:
            x_vals = np.array([rc_full[i] if 0 <= i < len(rc_full) else np.nan for i in idxs])
            ax_fa_full.set_xticks(np.linspace(0, 1, n_xticks_rc))
        else:
            x_vals = idxs
        ax_fa_full.plot(x_vals,  df_sel["mean_abs_delta_force_magnitude"], "o-",  color=_full_color,
                        markersize=5, linewidth=1.8, zorder=3)
        ax_fa_full_r.plot(x_vals, df_sel["mean_force_angle_error_deg"], "s--", color=_full_color,
                        markersize=4, linewidth=1.4, alpha=0.5)
        ax_fa_full.axhline(0, color="k", lw=0.5, ls="--", alpha=0.35)

        if show_axis_arrows:
            _aly, _ary = arrow_left_y_frac[0], arrow_right_y_frac[0]
            _alx, _arx = arrow_left_x_frac[0], arrow_right_x_frac[0]
            _allen, _arlen = arrow_left_len_frac[0], arrow_right_len_frac[0]
            _text_dy = 0.03
            ax_fa_full.annotate(
                "", xy=(_alx, _aly), xycoords="axes fraction",
                xytext=(_alx + _allen, _aly), textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color=_full_color, lw=1.8),
            )
            ax_fa_full.text(_alx + _allen / 2, _aly + _text_dy, arrow_left_label,
                            transform=ax_fa_full.transAxes, ha="center", va="bottom",
                            fontsize=base_font - 2, color=_full_color)
            ax_fa_full.annotate(
                "", xy=(_arx, _ary), xycoords="axes fraction",
                xytext=(_arx - _arlen, _ary), textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color=_full_color, lw=1.4),
            )
            ax_fa_full.text(_arx - _arlen / 2, _ary + _text_dy, arrow_right_label,
                            transform=ax_fa_full.transAxes, ha="center", va="bottom",
                            fontsize=base_font - 2, color=_full_color)
    else:
        ax_fa_full.text(0.5, 0.5, "Force data not available",
                        transform=ax_fa_full.transAxes,
                        ha="center", va="center", fontsize=base_font, color="gray")

    if show_axis_labels:
        _xlabel_fa = xlabel_rc if x_axis_mode == "reaction_coordinate" else xlabel_image
        ax_fa_full.set_xlabel(_xlabel_fa, fontsize=base_font, labelpad=label_pad)
        ax_fa_full.set_ylabel(ylabel_force_left,  fontsize=base_font, labelpad=label_pad)
        ax_fa_full_r.set_ylabel(ylabel_force_right, fontsize=base_font - 1, labelpad=label_pad)
    ax_fa_full.tick_params(labelsize=base_font - 1, direction="in")
    ax_fa_full_r.tick_params(labelsize=base_font - 2, direction="in")
    if degree_symbol:
        ax_fa_full_r.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:.0f}°"))
    ax_fa_full.spines["top"].set_visible(False)
    if show_panel_label["c"]:
        ax_fa_full.text(panel_label_x, panel_label_y, panel_labels[2],
                        transform=ax_fa_full.transAxes,
                        fontsize=base_font - 1, va="top", fontweight="bold")

    # ── panel (d): force errors on the DFT-NEB path — read from the pre-built
    # dft_neb_path_force_errors table (fp_static_on_dft_neb protocol) instead
    # of recomputing per-atom |Δ|F|| / Δθ on the fly; this table is built from
    # the identical _stat_imgs vs. _dft_imgs pairing/aggregation, so values are
    # unchanged, just no longer a hidden second computation inside this
    # plotting function ──
    _d_idxs = _d_dF = _d_dTheta = None
    if not df_sel_stat.empty:
        _d_idxs   = df_sel_stat["image_index"].values
        _d_dF     = df_sel_stat["mean_abs_delta_force_magnitude"].values
        _d_dTheta = df_sel_stat["mean_force_angle_error_deg"].values

    if _d_idxs is not None:
        if x_axis_mode == "reaction_coordinate" and rc_stat is not None:
            x_vals_d = np.array([rc_stat[i] if 0 <= i < len(rc_stat) else np.nan for i in _d_idxs])
            ax_fa_stat.set_xticks(np.linspace(0, 1, n_xticks_rc))
        else:
            x_vals_d = _d_idxs
        ax_fa_stat.plot(x_vals_d, _d_dF, "o-", color=_stat_color,
                        markersize=5, linewidth=1.8, zorder=3)
        ax_fa_stat_r.plot(x_vals_d, _d_dTheta, "s--", color=_stat_color,
                        markersize=4, linewidth=1.4, alpha=0.5)
        ax_fa_stat.axhline(0, color="k", lw=0.5, ls="--", alpha=0.35)

        if show_axis_arrows:
            _aly, _ary = arrow_left_y_frac[1], arrow_right_y_frac[1]
            _alx, _arx = arrow_left_x_frac[1], arrow_right_x_frac[1]
            _allen, _arlen = arrow_left_len_frac[1], arrow_right_len_frac[1]
            _text_dy = 0.03
            ax_fa_stat.annotate(
                "", xy=(_alx, _aly), xycoords="axes fraction",
                xytext=(_alx + _allen, _aly), textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color=_stat_color, lw=1.8),
            )
            ax_fa_stat.text(_alx + _allen / 2, _aly + _text_dy, arrow_left_label,
                            transform=ax_fa_stat.transAxes, ha="center", va="bottom",
                            fontsize=base_font - 2, color=_stat_color)
            ax_fa_stat.annotate(
                "", xy=(_arx, _ary), xycoords="axes fraction",
                xytext=(_arx - _arlen, _ary), textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color=_stat_color, lw=1.4),
            )
            ax_fa_stat.text(_arx - _arlen / 2, _ary + _text_dy, arrow_right_label,
                            transform=ax_fa_stat.transAxes, ha="center", va="bottom",
                            fontsize=base_font - 2, color=_stat_color)
    else:
        ax_fa_stat.text(0.5, 0.5, "Force data not available\n(static)",
                        transform=ax_fa_stat.transAxes,
                        ha="center", va="center", fontsize=base_font, color="gray")
    if show_axis_labels:
        _xlabel_fa = xlabel_rc if x_axis_mode == "reaction_coordinate" else xlabel_image
        ax_fa_stat.set_xlabel(_xlabel_fa, fontsize=base_font, labelpad=label_pad)
        ax_fa_stat.set_ylabel(ylabel_force_left,  fontsize=base_font, labelpad=label_pad)
        ax_fa_stat_r.set_ylabel(ylabel_force_right, fontsize=base_font - 1, labelpad=label_pad)
    ax_fa_stat.tick_params(labelsize=base_font - 1, direction="in")
    ax_fa_stat_r.tick_params(labelsize=base_font - 2, direction="in")
    if degree_symbol:
        ax_fa_stat_r.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:.0f}°"))
    ax_fa_stat.spines["top"].set_visible(False)
    if show_panel_label["d"]:
        ax_fa_stat.text(panel_label_x, panel_label_y, panel_labels[3],
                        transform=ax_fa_stat.transAxes,
                        fontsize=base_font - 1, va="top", fontweight="bold")

    # ── top legend: Full / Static ──────────────────────────────────────────
    if show_top_legend:
        top_leg_entries = [
            mlines.Line2D([], [], color="#333333", marker="o", ls="-", ms=4, lw=1.6, label="DFT"),
            mlines.Line2D([], [], color=_full_color, marker="o", ls="-", ms=4, lw=1.6, label="Full"),
            mlines.Line2D([], [], color=_stat_color, marker="P", ls="-", ms=4, lw=1.6, label="Static"),
        ]
        fig.legend(
            handles=top_leg_entries, ncol=1, loc="center",
            bbox_to_anchor=(top_legend_x, top_legend_y),
            frameon=True, edgecolor="#ccc",
            handlelength=1.6, handletextpad=0.5, columnspacing=1.0,
            prop={"size": base_font - 2},
        )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.16)

    # ── independent spacing knobs ───────────────────────────────────────
    fig.canvas.draw()
    def _spread_pair(ax_left, ax_right, gap_in):
        if not gap_in:
            return
        shift = (gap_in / fig.get_figwidth()) / 2
        pL = ax_left.get_position();  ax_left.set_position([pL.x0 - shift, pL.y0, pL.width, pL.height])
        pR = ax_right.get_position(); ax_right.set_position([pR.x0 + shift, pR.y0, pR.width, pR.height])

    _spread_pair(neb_axes[0], neb_axes[1], wspace_top_in)
    _spread_pair(fa_axes[0],  fa_axes[1],  wspace_bottom_in)

    def _shift_row(ax_list, dy):
        if not dy:
            return
        for ax in ax_list:
            p = ax.get_position()
            ax.set_position([p.x0, p.y0 + dy, p.width, p.height])

    _hspace_shift = (hspace_rows_in / fig.get_figheight()) / 2
    _shift_row(neb_axes, _hspace_shift)
    _shift_row(fa_axes + fa_twin, -_hspace_shift)

    plt.show()
    return fig, axes
