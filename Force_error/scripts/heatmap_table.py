"""
heatmap_table.py
================
Modular building blocks for constructing mixed triangular/rectangular
heatmap tables.  Each helper operates on a single Matplotlib Axes and
can be composed freely to create custom table layouts.

Hierarchy
---------
Cell level      draw_triangular_cell        – one cell, two metrics (lower-left / upper-right)
                draw_rectangular_cell       – one cell, one metric (full square)

Block level     draw_triangular_column      – full column of triangular cells
                draw_rectangular_row        – full row of rectangular cells
                draw_rectangular_column     – full column of rectangular cells

Figure setup    create_figure               – figure + gridspec (main axes + colorbar axes)
                setup_frame                 – outer border lines
                setup_ticks_and_labels      – ticks, axis labels, title, row/col labels
                add_colorbar                – attach one colorbar, height-matched to main ax

High-level      triangular_heatmap_with_fraction_row  – original convenience wrapper
                plot_fraction_panel                  – rectangular + optional triangular MAE/RMSE panel
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helper
# ═══════════════════════════════════════════════════════════════════════════════

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


def _text_color_for_bg(rgba, lum_threshold=0.45):
    """
    Return "white" or "black" depending on the luminance of the background.

    Uses the WCAG relative luminance formula so text always has sufficient
    contrast against the cell background colour.

    Parameters
    ----------
    rgba          : (r, g, b, a) tuple in [0, 1]  – background colour
    lum_threshold : float – luminance below this value → white text
    """
    r, g, b, _ = rgba
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "black" if luminance > lum_threshold else "white"


# ═══════════════════════════════════════════════════════════════════════════════
# Cell-level helpers
# ═══════════════════════════════════════════════════════════════════════════════

def draw_triangular_cell(
    ax, x, y,
    val_lower, val_upper,
    cmap_lower, norm_lower,
    cmap_upper, norm_upper,
    fmt_lower="{:.2f}", fmt_upper="{:.2f}",
    text_size=12,
    edge_lw=0.4,
    lower_text_pos=(0.35, 0.22),
    upper_text_pos=(0.65, 0.78),
    auto_text_color=True,
    lum_threshold=0.45,
):
    """
    Draw one unit cell split diagonally into two triangles.

    Lower-left  triangle  ← val_lower  (e.g. MAE, coloured with cmap_lower)
    Upper-right triangle  ← val_upper  (e.g. RMSE, coloured with cmap_upper)

    The cell occupies the square [x, x+1] × [y, y+1] in data coordinates.
    NaN values are silently skipped (the triangle is simply not drawn).

    Parameters
    ----------
    ax                         : Matplotlib Axes
    x, y                       : float – bottom-left corner of the unit cell
    val_lower, val_upper       : float – data values; NaN → skip that triangle
    cmap_lower, cmap_upper     : Colormap objects
    norm_lower, norm_upper     : Normalize objects (maps data value → [0,1])
    fmt_lower, fmt_upper       : Python format strings, e.g. "{:.2f}"
    text_size                  : int – font size for both annotations
    edge_lw                    : float – linewidth of triangle edges
    lower_text_pos, upper_text_pos
                               : (dx, dy) offsets from (x, y) for text placement
    auto_text_color            : bool – if True, text colour (black/white) is
                                 chosen automatically based on background luminance
                                 so annotations remain readable on dark cells
    lum_threshold              : float – luminance cut-off for auto_text_color
    """
    # ── lower-left triangle ────────────────────────────────────────────────
    if np.isfinite(val_lower):
        rgba = cmap_lower(norm_lower(val_lower))
        ax.add_patch(patches.Polygon(
            [[x, y], [x + 1, y], [x, y + 1]],
            facecolor=rgba,
            edgecolor="black", linewidth=edge_lw,
        ))
        tcolor = _text_color_for_bg(rgba, lum_threshold) if auto_text_color else "black"
        ax.text(
            x + lower_text_pos[0], y + lower_text_pos[1],
            _fmtval(fmt_lower, val_lower),
            ha="center", va="center", fontsize=text_size, color=tcolor,
        )

    # ── upper-right triangle ───────────────────────────────────────────────
    if np.isfinite(val_upper):
        rgba = cmap_upper(norm_upper(val_upper))
        ax.add_patch(patches.Polygon(
            [[x + 1, y], [x + 1, y + 1], [x, y + 1]],
            facecolor=rgba,
            edgecolor="black", linewidth=edge_lw,
        ))
        tcolor = _text_color_for_bg(rgba, lum_threshold) if auto_text_color else "black"
        ax.text(
            x + upper_text_pos[0], y + upper_text_pos[1],
            _fmtval(fmt_upper, val_upper),
            ha="center", va="center", fontsize=text_size, color=tcolor,
        )


def draw_rectangular_cell(
    ax, x, y, val,
    cmap=None, norm=None,
    fmt="{:.2f}",
    text_size=12,
    edge_lw=0.6,
    facecolor=None,
    text_color=None,
    auto_text_color=True,
    lum_threshold=0.45,
):
    """
    Draw one full rectangular (non-split) heatmap cell.

    Background colour priority:
      1. `facecolor` if explicitly provided
      2. cmap(norm(val)) if val is finite and cmap/norm are given
      3. "white" otherwise (also used for string values)

    Parameters
    ----------
    ax               : Matplotlib Axes
    x, y             : float – bottom-left corner
    val              : float or str – numeric → coloured + formatted;
                       str → white background, displayed as-is;
                       NaN / non-finite → white background, empty label
    cmap             : Colormap or None
    norm             : Normalize or None
    fmt              : format string applied when val is numeric
    text_size        : int
    edge_lw          : float – rectangle edge linewidth
    facecolor        : str / colour or None – explicit background override
    text_color       : str or None – explicit text colour; overrides auto detection
    auto_text_color  : bool – if True and text_color is None, pick black/white
                       automatically based on background luminance
    lum_threshold    : float – luminance cut-off for auto_text_color
    """
    # ── resolve background colour ──────────────────────────────────────────
    if facecolor is not None:
        bg = facecolor
    else:
        try:
            fval = float(val)
            is_numeric = np.isfinite(fval)
        except (TypeError, ValueError):
            is_numeric = False

        if is_numeric and cmap is not None and norm is not None:
            bg = cmap(norm(float(val)))
        else:
            bg = "white"

    ax.add_patch(patches.Rectangle(
        (x, y), 1, 1,
        facecolor=bg, edgecolor="black", linewidth=edge_lw,
    ))

    # ── resolve display text ───────────────────────────────────────────────
    if isinstance(val, str):
        label = val if val == val else ""
    else:
        try:
            fval = float(val)
            label = _fmtval(fmt, fval) if np.isfinite(fval) else ""
        except (TypeError, ValueError):
            label = str(val) if val is not None else ""

    # ── resolve text colour ────────────────────────────────────────────────
    if text_color is not None:
        tcolor = text_color
    elif auto_text_color:
        # convert bg to RGBA so luminance can be computed
        import matplotlib.colors as _mc
        try:
            rgba = _mc.to_rgba(bg)
        except (ValueError, TypeError):
            rgba = (1, 1, 1, 1)   # fallback to white bg → black text
        tcolor = _text_color_for_bg(rgba, lum_threshold)
    else:
        tcolor = "black"

    ax.text(
        x + 0.5, y + 0.5, label,
        ha="center", va="center",
        fontsize=text_size, color=tcolor,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Block-level helpers
# ═══════════════════════════════════════════════════════════════════════════════

def draw_triangular_column(
    ax, col_idx, nrows,
    vals_lower, vals_upper,
    cmap_lower, norm_lower,
    cmap_upper, norm_upper,
    row_offset=0,
    fmt_lower="{:.2f}", fmt_upper="{:.2f}",
    text_size=12,
    edge_lw=0.4,
    **cell_kwargs,
):
    """
    Draw a full column of triangular (diagonally-split) cells.

    Data ordering: vals_lower[0] / vals_upper[0] → topmost displayed row.
    The column occupies x ∈ [col_idx, col_idx+1].

    Parameters
    ----------
    col_idx    : int – 0-based column index (sets x position)
    nrows      : int – number of data rows in this column
    vals_lower : array-like, length nrows – lower-triangle values (e.g. MAE)
    vals_upper : array-like, length nrows – upper-triangle values (e.g. RMSE)
    row_offset : int – shift all cells upward by this many rows
                 (use when extra rows sit below the data block)
    **cell_kwargs : forwarded to draw_triangular_cell
    """
    for i in range(nrows):
        x = col_idx
        # index 0 of the arrays maps to the highest y (topmost row)
        y = (nrows - i - 1) + row_offset
        draw_triangular_cell(
            ax, x, y,
            vals_lower[i], vals_upper[i],
            cmap_lower, norm_lower,
            cmap_upper, norm_upper,
            fmt_lower=fmt_lower, fmt_upper=fmt_upper,
            text_size=text_size, edge_lw=edge_lw,
            **cell_kwargs,
        )


def draw_rectangular_row(
    ax, row_y, ncols, vals,
    cmap=None, norm=None,
    fmt="{:.2f}",
    text_size=12,
    edge_lw=0.6,
    **cell_kwargs,
):
    """
    Draw a full row of rectangular cells.

    The row occupies y ∈ [row_y, row_y+1].  Values may be numeric or strings
    (e.g. pre-formatted percentage text like "12.3 %").

    Parameters
    ----------
    row_y  : int – y coordinate of the row's bottom edge
    ncols  : int
    vals   : array-like, length ncols – one value per column
    cmap, norm : optional colouring (pass None for plain white cells)
    **cell_kwargs : forwarded to draw_rectangular_cell
    """
    for j in range(ncols):
        draw_rectangular_cell(
            ax, j, row_y, vals[j],
            cmap=cmap, norm=norm,
            fmt=fmt, text_size=text_size, edge_lw=edge_lw,
            **cell_kwargs,
        )


def draw_rectangular_column(
    ax, col_idx, nrows, vals,
    cmap=None, norm=None,
    row_offset=0,
    fmt="{:.2f}",
    sig_figs=None,             # significant figures: overrides fmt (e.g. sig_figs=2)
    text_size=12,
    edge_lw=0.6,
    **cell_kwargs,
):
    """
    Draw a full column of rectangular cells.

    Data ordering: vals[0] → topmost displayed row.

    Parameters
    ----------
    col_idx    : int – 0-based column index
    nrows      : int
    vals       : array-like, length nrows
    row_offset : int – shift cells upward by this many rows
    sig_figs   : int or None – significant figures; overrides fmt when set
    **cell_kwargs : forwarded to draw_rectangular_cell
    """
    if sig_figs is not None:
        fmt = _sigfig_formatter(sig_figs)
    for i in range(nrows):
        y = (nrows - i - 1) + row_offset
        draw_rectangular_cell(
            ax, col_idx, y, vals[i],
            cmap=cmap, norm=norm,
            fmt=fmt, text_size=text_size, edge_lw=edge_lw,
            **cell_kwargs,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Figure / layout helpers
# ═══════════════════════════════════════════════════════════════════════════════

def create_figure(
    ncols,
    n_colorbars=2,
    figsize=(6.3, 8),
    dpi=350,
    wspace=0.15,
    colorbar_width_ratio=0.15,
    colorbar_nudge=0.12,
    cbar_width=None,
    cbar_gap=None,
    nrows=None,
):
    """
    Create a Figure with a GridSpec layout:  [main axes | cbar_0 | cbar_1 | …]

    The main axes width is proportional to `ncols`; each colorbar column has
    width `colorbar_width_ratio` × ncols (relative units).

    After drawing, colorbars are nudged leftward by `colorbar_nudge` to close
    the gap introduced by wspace.

    Parameters
    ----------
    ncols                : int   – number of data columns
    n_colorbars          : int   – how many colorbar axes to create (0 is valid)
    figsize              : (w, h) in inches
    dpi                  : int
    wspace               : float – horizontal spacing between subplots
    colorbar_width_ratio : float – width of each cbar column relative to ncols
    colorbar_nudge       : float – leftward shift for each colorbar axes

    Returns
    -------
    fig   : Figure
    ax    : Axes  – main heatmap axes
    caxes : list of Axes  – one Axes per requested colorbar (may be empty)
    """
    if cbar_width is not None:
        colorbar_width_ratio = cbar_width
    if cbar_gap is not None:
        colorbar_nudge = cbar_gap
    width_ratios = [ncols] + [colorbar_width_ratio] * n_colorbars

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(
        nrows=1, ncols=1 + n_colorbars,
        width_ratios=width_ratios,
        wspace=wspace,
    )
    ax = fig.add_subplot(gs[0, 0])

    caxes = []
    for k in range(n_colorbars):
        cax = fig.add_subplot(gs[0, k + 1])
        # nudge each colorbar axis leftward so it sits closer to the main axes
        pos = cax.get_position()
        cax.set_position([
            pos.x0 - colorbar_nudge,
            pos.y0,
            pos.width,
            pos.height,
        ])
        caxes.append(cax)

    return fig, ax, caxes


def setup_frame(ax, ncols, nrows_total, lw_outer=2.0, lw_inner=1.0):
    """
    Draw the four border lines of the table and hide Matplotlib's default spines.

    Parameters
    ----------
    ncols, nrows_total : int – grid extents in data coordinates
    lw_outer           : float – linewidth for bottom and right borders
    lw_inner           : float – linewidth for top and left borders
    """
    ax.plot([0, ncols],     [0, 0],               color="black", lw=lw_outer)  # bottom
    ax.plot([0, ncols],     [nrows_total]*2,       color="black", lw=lw_inner)  # top
    ax.plot([0, 0],         [0, nrows_total],      color="black", lw=lw_inner)  # left
    ax.plot([ncols]*2,      [0, nrows_total],      color="black", lw=lw_outer)  # right

    for spine in ax.spines.values():
        spine.set_visible(False)


def setup_ticks_and_labels(
    ax, ncols, nrows_data, nrows_total,
    row_labels, col_labels,
    title="",
    xlabel="thresholds",
    xlabel_pad=14,
    extra_row_label="Fraction\n(F_DFT > threshold) (%)",
    extra_row_index=None,         # y-index of the extra row's bottom edge (default: nrows_data)
    text_size=12,
    title_size=None,
    title_pad=None,
    xlabel_size=None,
    tick_rotation=45,
):
    """
    Configure tick positions, tick labels, axis labels, title, and the
    optional side-label for any extra (non-data) rows.

    Parameters
    ----------
    ncols, nrows_data, nrows_total
                   : ints describing the grid geometry
    row_labels     : sequence of str – model names, listed top-to-bottom
    col_labels     : sequence of str – column header labels
    title          : str – axes title
    xlabel         : str – x-axis label (set to "" to suppress)
    extra_row_label: str – label placed to the left of the extra row
                    (set to "" or None to suppress)
    extra_row_index: int – bottom-edge y of the extra row in data coordinates;
                    defaults to nrows_data (i.e. the row immediately above data)
    text_size      : int – base font size
    title_size, xlabel_size
                   : int or None – override sizes (default: text_size + 2)
    title_pad      : float or None – gap (points) between title and axes
                    (None → Matplotlib default)
    tick_rotation  : int – rotation angle of x-tick labels
    """
    if title_size is None:
        title_size = text_size + 2
    if xlabel_size is None:
        xlabel_size = text_size + 2
    if extra_row_index is None:
        extra_row_index = nrows_data

    # ── y ticks: one per data row, no ticks for extra rows ────────────────
    ax.set_yticks(np.arange(nrows_data) + 0.5)
    ax.set_yticklabels(list(row_labels)[::-1], fontsize=text_size)

    # ── x ticks at the bottom ─────────────────────────────────────────────
    ax.set_xticks(np.arange(ncols) + 0.5)
    ax.set_xticklabels(col_labels, rotation=tick_rotation, ha="right",
                       fontsize=text_size)
    ax.tick_params(axis="x", bottom=True, labelbottom=True,
                   top=False,  labeltop=False)

    if xlabel:
        ax.set_xlabel(xlabel, labelpad=xlabel_pad, fontsize=xlabel_size)
    if title:
        ax.set_title(title, fontsize=title_size, pad=title_pad)

    # ── side-label for the extra (non-data) row ───────────────────────────
    if extra_row_label:
        # centre of the extra row in axes-fraction coordinates
        frac_y_centre = (extra_row_index + 0.5) / nrows_total
        ax.text(
            -0.02, frac_y_centre, extra_row_label,
            transform=ax.transAxes,
            ha="right", va="center", fontsize=text_size,
        )


def add_colorbar(
    fig, ax, cax,
    cmap, norm,
    label="",
    text_size=12,
    label_size=None,
    label_pad=None,
    match_ax_height=True,
):
    """
    Attach a single colorbar to `cax`, optionally resizing its height to
    match the main heatmap axes so the colorbar spans exactly the table.

    Call this *after* all drawing is done so that `ax` has its final position.

    Parameters
    ----------
    fig             : Figure
    ax              : Axes – main heatmap axes (provides reference height)
    cax             : Axes – colorbar axes created by create_figure()
    cmap            : Colormap
    norm            : Normalize
    label           : str – colorbar label
    text_size       : int – tick label size
    label_size      : int or None – colorbar label size (default: text_size + 2)
    label_pad       : float or None – gap (points) between colorbar and its
                      label (None → Matplotlib default)
    match_ax_height : bool – resize cax so its y-extent equals ax's y-extent;
                      triggers fig.canvas.draw() to force layout computation

    Returns
    -------
    cb : Colorbar
    """
    if label_size is None:
        label_size = text_size + 2

    if match_ax_height:
        # force Matplotlib to compute actual axes positions
        fig.canvas.draw()
        ax_pos  = ax.get_position()
        cax_pos = cax.get_position()
        cax.set_position([
            cax_pos.x0,
            ax_pos.y0,
            cax_pos.width,
            ax_pos.height,
        ])

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax)
    cb.ax.tick_params(labelsize=text_size)
    cb.set_label(label, fontsize=label_size, rotation=90, labelpad=label_pad)
    return cb


# ═══════════════════════════════════════════════════════════════════════════════
# High-level convenience wrapper  (mirrors the original function)
# ═══════════════════════════════════════════════════════════════════════════════

def triangular_heatmap_with_fraction_row(
    mae_df, rmse_df, frac_row_str=None,
    col_labels=None,
    title=(
        r"(MAE / RMSE) $\Delta F$ (eV/Å)"
        r" $(F_{\mathrm{DFT}} < \mathrm{thresholds}$)"
    ),
    cmap_mae="Blues",
    cmap_rmse="Reds",
    figsize=(6.3, 8),
    fmt_mae="{:.2f}",
    fmt_rmse="{:.2f}",
    sig_figs=None,             # significant figures: overrides fmt_mae/fmt_rmse (e.g. sig_figs=2)
    text_size=12,
    wspace=0.15,               # gap between the two colorbars (GridSpec spacing)
    colorbar_nudge=0.12,       # shift colorbars leftward — controls gap to table
    colorbar_width_ratio=0.15, # width of each colorbar relative to ncols
    show_fraction_row=True,    # set False to omit the fraction header row
    mae_cbar_label="MAE (eV/Å)",   # colorbar label for MAE (override units here)
    rmse_cbar_label="RMSE (eV/Å)", # colorbar label for RMSE
    mae_cbar_labelpad=None,   # gap (points) between MAE colorbar and its label
    rmse_cbar_labelpad=None,  # gap (points) between RMSE colorbar and its label
    title_pad=None,           # gap (points) between title and axes
    xlabel="thresholds",      # x-axis label (set to "" to hide it)
):
    """
    Convenience wrapper that builds the full triangular heatmap table with:

      • A fraction header row at the top  (plain text, white background)
      • Diagonal-split cells for each (model, threshold) entry
      • Two colorbars (MAE in Blues, RMSE in Reds) height-matched to the table

    All internal details are delegated to the modular helpers in this module.

    Parameters
    ----------
    mae_df, rmse_df : pd.DataFrame – rows = models, columns = thresholds
    frac_row_str    : pd.Series    – pre-formatted strings, indexed like df columns
    col_labels      : list or None – override column labels
    title           : str          – axes title
    cmap_mae, cmap_rmse : str      – Matplotlib colormap names
    figsize         : (w, h) in inches
    fmt_mae, fmt_rmse : str        – format strings for cell annotations
    text_size       : int          – base font size
    wspace          : float        – GridSpec horizontal spacing
    colorbar_nudge  : float        – leftward shift of colorbar axes
    mae_cbar_label, rmse_cbar_label : str – colorbar labels (set units here,
                    e.g. "MAE (°)" / "RMSE (°)" for angular data)
    mae_cbar_labelpad, rmse_cbar_labelpad
                    : float or None – per-colorbar label padding (points);
                      None → Matplotlib default
    title_pad       : float or None – gap (points) between title and axes;
                      None → Matplotlib default
    """
    # ── 1. Align DataFrames and reindex the fraction Series ───────────────
    if sig_figs is not None:
        fmt_mae = fmt_rmse = _sigfig_formatter(sig_figs)
    mae_df, rmse_df = mae_df.align(rmse_df, join="inner", axis=1)
    if frac_row_str is not None:
        frac_row_str = frac_row_str.reindex(mae_df.columns)

    nrows, ncols  = mae_df.shape
    nrows_total   = nrows + 1 if show_fraction_row else nrows

    mae_vals  = mae_df.to_numpy(float)
    rmse_vals = rmse_df.to_numpy(float)

    if col_labels is None:
        col_labels = list(mae_df.columns)

    # ── 2. Robust percentile-based normalisation ───────────────────────────
    mae_vmin,  mae_vmax  = np.nanpercentile(mae_vals,  [5,  95])
    rmse_vmin, rmse_vmax = np.nanpercentile(rmse_vals, [15, 95])

    mae_norm  = plt.Normalize(mae_vmin,  mae_vmax)
    rmse_norm = plt.Normalize(rmse_vmin, rmse_vmax)

    mae_cmap  = plt.cm.get_cmap(cmap_mae)
    rmse_cmap = plt.cm.get_cmap(cmap_rmse)

    # ── 3. Figure and axes layout ─────────────────────────────────────────
    fig, ax, (cax_mae, cax_rmse) = create_figure(
        ncols, n_colorbars=2,
        figsize=figsize, wspace=wspace,
        colorbar_nudge=colorbar_nudge,
        colorbar_width_ratio=colorbar_width_ratio,
    )

    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows_total)
    ax.set_aspect("equal")

    # ── 4. Fraction header row (top of table) ─────────────────────────────
    if show_fraction_row and frac_row_str is not None:
        # row_y = nrows means the row sits in [nrows, nrows+1] → above all data
        draw_rectangular_row(
            ax, row_y=nrows, ncols=ncols,
            vals=frac_row_str.values,
            cmap=None, norm=None,           # plain white cells
            text_size=text_size,
        )
        # visual separator between header and data rows
        ax.plot([0, ncols], [nrows, nrows], color="black", linewidth=1.0)

    # ── 5. Triangular data cells (one column at a time) ───────────────────
    for col_idx in range(ncols):
        draw_triangular_column(
            ax,
            col_idx=col_idx,
            nrows=nrows,
            vals_lower=mae_vals[:, col_idx],    # lower-left  = MAE
            vals_upper=rmse_vals[:, col_idx],   # upper-right = RMSE
            cmap_lower=mae_cmap,  norm_lower=mae_norm,
            cmap_upper=rmse_cmap, norm_upper=rmse_norm,
            row_offset=0,                       # data rows start at y=0
            fmt_lower=fmt_mae, fmt_upper=fmt_rmse,
            text_size=text_size,
        )

    # ── 6. Outer frame and axis decorations ───────────────────────────────
    setup_frame(ax, ncols, nrows_total)

    setup_ticks_and_labels(
        ax,
        ncols=ncols,
        nrows_data=nrows,
    nrows_total=nrows_total,
        row_labels=mae_df.index,
        col_labels=col_labels,
        title=title,
        title_pad=title_pad,
        xlabel=xlabel,
        text_size=text_size,
        extra_row_label="" if not show_fraction_row else "Fraction\n(F_DFT > threshold) (%)",
    )

    # ── 7. Colorbars (height-matched to the table after layout) ───────────
    # add_colorbar calls fig.canvas.draw() internally to resolve positions
    add_colorbar(
        fig, ax, cax_mae,
        cmap=mae_cmap, norm=mae_norm,
        label=mae_cbar_label,
        label_pad=mae_cbar_labelpad,
        text_size=text_size,
    )
    add_colorbar(
        fig, ax, cax_rmse,
        cmap=rmse_cmap, norm=rmse_norm,
        label=rmse_cbar_label,
        label_pad=rmse_cbar_labelpad,
        text_size=text_size,
    )

    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# Fraction-panel convenience wrapper
# ═══════════════════════════════════════════════════════════════════════════════

def plot_fraction_panel(
    df_panel,
    panel_title="",
    title_pad=None,
    cmap_name="RdYlGn",       # low % = red (bad), high % = green (good)
    pct=(5, 95),              # percentile clipping for fraction norm
    cell_size=0.65,           # inches per cell
    left_margin=1.6,          # inches for row (model) labels on the left
    text_size=7,
    fmt="{:.1f}",             # format for fraction cell values
    sig_figs=None,            # significant figures: overrides fmt (e.g. sig_figs=2)
    # ── optional triangular MAE/RMSE columns appended on the right ────────────
    tri_cols=None,            # list of (key_mae, key_rmse, col_label, fmt_mae, fmt_rmse)
    df_tri=None,              # DataFrame with MAE/RMSE values (e.g. df_regime_summary)
    mae_cmap=None,            # colormap for MAE triangles  (default: Blues)
    rmse_cmap=None,           # colormap for RMSE triangles (default: Reds)
    tri_pct=(5, 95),          # percentile clipping for triangular norms
    mae_cbar_label="MAE",     # colorbar label for MAE (include units if needed)
    rmse_cbar_label="RMSE",   # colorbar label for RMSE
    cbar_w=0.018,             # colorbar width in figure-fraction units
    cbar_gap=0.055,           # gap between adjacent colorbars
    save_path=None,           # e.g. "panel_A.svg" or "panel_A.pdf" — None → don't save
    save_dpi=350,             # dpi used when saving raster formats (ignored for svg/pdf)
    fig_w=None,
    fig_h=None,
    cbar_label_pad=10,
    show_regime_row=True,     # set False to hide the grey "Regime avg" header row
):
    """
    Rectangular heatmap for one fraction panel, with an optional block of
    triangular MAE/RMSE columns appended on the right.

    Layout
    ------
    Top row   : single grey spanning cell showing avg regime fraction
    Data rows : rectangular fraction columns | (optional) triangular MAE/RMSE columns
    Colorbars : fraction (RdYlGn) + optional MAE + optional RMSE

    Parameters
    ----------
    df_panel         : DataFrame – rows=models; must contain "N atoms" and
                       "Frac of all atoms (%)" plus threshold fraction columns
    tri_cols         : list of (key_mae, key_rmse, col_label, fmt_mae, fmt_rmse)
                       One tuple per triangular column to append on the right.
    df_tri           : DataFrame – source of MAE/RMSE data (df_regime_summary)
    mae_cmap/rmse_cmap : Colormap – override default Blues/Reds for triangles
    mae/rmse_cbar_label : str   – colorbar labels (add units here, e.g. "MAE (eV/Å)")
    cbar_w, cbar_gap : float    – colorbar width and gap in figure-fraction units
    """
    if mae_cmap  is None: mae_cmap  = plt.cm.Blues
    if rmse_cmap is None: rmse_cmap = plt.cm.Reds

    has_tri = tri_cols is not None and df_tri is not None

    # ── separate regime-fraction header from threshold data ──────────────────
    frac_avg = df_panel["Frac of all atoms (%)"].mean()
    data_df  = df_panel.drop(columns=["N atoms", "Frac of all atoms (%)"])

    models     = data_df.index.tolist()
    col_keys   = data_df.columns.tolist()
    nrows      = len(models)
    nrows_total = nrows + (1 if show_regime_row else 0)

    ncols_rect  = len(col_keys)
    ncols_tri   = len(tri_cols) if has_tri else 0
    ncols_total = ncols_rect + ncols_tri

    col_labels_rect = [c.replace(" (%)", "") for c in col_keys]
    col_labels_tri  = [c[2] for c in tri_cols] if has_tri else []
    col_labels_all  = col_labels_rect + col_labels_tri

    # ── norms ─────────────────────────────────────────────────────────────────
    vals_rect  = data_df.values.ravel()
    vmin, vmax = np.nanpercentile(vals_rect[np.isfinite(vals_rect)], pct)
    norm_rect  = plt.Normalize(vmin, vmax)
    cmap_rect  = plt.cm.get_cmap(cmap_name)

    if has_tri:
        mae_keys  = [c[0] for c in tri_cols]
        rmse_keys = [c[1] for c in tri_cols]
        mae_vals  = df_tri[mae_keys].values.ravel()
        rmse_vals = df_tri[rmse_keys].values.ravel()
        mae_norm  = plt.Normalize(*np.nanpercentile(mae_vals[np.isfinite(mae_vals)],   tri_pct))
        rmse_norm = plt.Normalize(*np.nanpercentile(rmse_vals[np.isfinite(rmse_vals)], tri_pct))

    # ── figure sized from cell_size ───────────────────────────────────────────
    n_cbars = 1 + (2 if has_tri else 0)
    auto_fig_h = cell_size * nrows_total
    auto_fig_w = cell_size * ncols_total + left_margin + 0.9 * n_cbars
    # override if user provides values
    fig_h = auto_fig_h if fig_h is None else fig_h
    fig_w = auto_fig_w if fig_w is None else fig_w

    fig, ax, _ = create_figure(ncols_total, n_colorbars=0, figsize=(fig_w, fig_h))
    ax.set_xlim(0, ncols_total)
    ax.set_ylim(0, nrows_total)
    ax.set_aspect("equal")

    # ── header row: single grey cell spanning all columns ─────────────────────
    if show_regime_row:
        ax.add_patch(patches.Rectangle(
            (0, nrows), ncols_total, 1,
            facecolor="lightgrey", edgecolor="black", linewidth=0.8,
        ))
        ax.text(
            ncols_total / 2, nrows + 0.5,
            f"Regime avg: {frac_avg:.1f}% of all atoms",
            ha="center", va="center", fontsize=text_size, fontweight="bold",
        )
        ax.plot([0, ncols_total], [nrows, nrows], color="black", lw=1.2)

    # ── rectangular fraction data cells ───────────────────────────────────────
    if sig_figs is not None:
        fmt = _sigfig_formatter(sig_figs)
    for j, col_key in enumerate(col_keys):
        draw_rectangular_column(
            ax, col_idx=j, nrows=nrows,
            vals=data_df[col_key].values,
            cmap=cmap_rect, norm=norm_rect,
            fmt=fmt, text_size=text_size,
        )

    # ── triangular MAE/RMSE data cells ────────────────────────────────────────
    if has_tri:
        for jt, (key_mae, key_rmse, _, fmt_mae, fmt_rmse) in enumerate(tri_cols):
            draw_triangular_column(
                ax, col_idx=ncols_rect + jt, nrows=nrows,
                vals_lower=df_tri[key_mae].values,
                vals_upper=df_tri[key_rmse].values,
                cmap_lower=mae_cmap, norm_lower=mae_norm,
                cmap_upper=rmse_cmap, norm_upper=rmse_norm,
                fmt_lower=fmt_mae, fmt_upper=fmt_rmse,
                text_size=text_size,
            )
        # dashed separator stops at the grey header (not through it)
        ax.plot([ncols_rect, ncols_rect], [0, nrows],
                color="black", lw=1.5, ls="--")

    # ── frame and labels ──────────────────────────────────────────────────────
    setup_frame(ax, ncols_total, nrows_total)
    setup_ticks_and_labels(
        ax,
        ncols=ncols_total, nrows_data=nrows, nrows_total=nrows_total,
        row_labels=models, col_labels=col_labels_all,
        title=panel_title, title_pad=title_pad, xlabel="",
        extra_row_label="Regime\nfraction" if show_regime_row else "",
        extra_row_index=nrows,
        text_size=text_size,
    )

    # ── colorbars ─────────────────────────────────────────────────────────────
    fig.canvas.draw()
    ax_pos = ax.get_position()
    x0     = ax_pos.x0 + ax_pos.width + 0.025

    cbar_specs = [(cmap_rect, norm_rect, "(%)", "right")]
    if has_tri:
        cbar_specs += [
            (mae_cmap,  mae_norm,  mae_cbar_label,  "right"),
            (rmse_cmap, rmse_norm, rmse_cbar_label, "right"),
        ]

    for k, (cmap_k, norm_k, label_k, side_k) in enumerate(cbar_specs):
        cax = fig.add_axes([
            x0 + k * (cbar_w + cbar_gap),
            ax_pos.y0, cbar_w, ax_pos.height,
        ])
        sm = plt.cm.ScalarMappable(norm=norm_k, cmap=cmap_k)
        cb = fig.colorbar(sm, cax=cax)
        cb.ax.tick_params(labelsize=text_size)
        cb.ax.yaxis.set_ticks_position(side_k)
        cb.ax.yaxis.set_label_position(side_k)
        cb.set_label(label_k, fontsize=text_size, rotation=90, labelpad=cbar_label_pad)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=save_dpi, bbox_inches="tight")
    plt.show()
