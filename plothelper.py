import numpy as np
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
from PIL import Image

colorplates = {
    "3colors_1": ["#0F6E56", "#A32D2D", "#EF9F27"],
}


def showcolors(colors):
    import matplotlib.pyplot as plt
    n = len(colors)
    fig, axes = plt.subplots(1, n, figsize=(n * 1.2, 1.8))
    if n == 1:
        axes = [axes]
    for ax, c in zip(axes, colors):
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.05, 0.3), 0.9, 0.65,
            boxstyle="square,pad=0",
            facecolor=c, edgecolor="none",
            transform=ax.transAxes,
        ))
        ax.text(0.5, 0.15, c, transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color="black")
        ax.set_axis_off()
    fig.subplots_adjust(wspace=0.05)
    return fig


def _read_pixel_size(img):
    """
    Try to extract the physical pixel size (in metres) from TIFF metadata.
 
    Supported formats
    -----------------
    1. Zeiss SEM (tag 34118) — proprietary text block containing the pixel size
       either as a human-readable "AP_PIXEL_SIZE" entry or as a raw float on
       the second line of each channel block (e.g. "3.854261e-009").
    2. Standard TIFF resolution tags (282 = XResolution, 296 = ResolutionUnit).
 
    Returns pixel size in metres, or None if nothing can be parsed.
    """
    tags = img.tag_v2
 
    # ── 1. Zeiss tag 34118 ────────────────────────────────────────────────────
    zeiss_block = tags.get(34118)
    if zeiss_block:
        lines = [l.strip() for l in zeiss_block.replace("\r\n", "\n").split("\n")]
 
        # Method A: "AP_PIXEL_SIZE" label followed by "Pixel Size = 3.854 nm"
        for i, line in enumerate(lines):
            if line == "AP_PIXEL_SIZE" and i + 1 < len(lines):
                parts = lines[i + 1].split("=")
                if len(parts) == 2:
                    val_unit = parts[1].strip().split()    # ["3.854", "nm"]
                    if len(val_unit) == 2:
                        unit_factors = {"m": 1, "cm": 1e-2, "mm": 1e-3,
                                        "µm": 1e-6, "um": 1e-6, "nm": 1e-9}
                        factor = unit_factors.get(val_unit[1])
                        if factor:
                            return float(val_unit[0]) * factor
 
        # Method B: raw scientific-notation float on line index 1
        try:
            px = float(lines[1])
            if 1e-12 < px < 1e-3:    # sanity: between 1 pm and 1 mm
                return px
        except (ValueError, IndexError):
            pass
 
    # ── 2. Standard TIFF resolution tags ─────────────────────────────────────
    try:
        x_res    = tags.get(282)
        res_unit = tags.get(296, 2)
 
        if x_res is not None:
            if hasattr(x_res, "numerator"):
                x_res = float(x_res)
            elif isinstance(x_res, tuple):
                x_res = x_res[0] / x_res[1]
 
            if x_res > 0:
                if res_unit == 2:        # pixels per inch
                    return 0.0254 / x_res
                elif res_unit == 3:      # pixels per cm
                    return 0.01 / x_res
    except Exception:
        pass
 
    return None
 
 
def _auto_unit(img_width_px, pixel_size_m):
    """Return the most readable SI unit for a ~15 % image-width scalebar."""
    target_m = img_width_px * pixel_size_m * 0.15
    if target_m >= 5e-1:
        return "m"
    elif target_m >= 5e-4:
        return "mm"
    elif target_m >= 5e-7:
        return "µm"
    else:
        return "nm"


def _nice_scalebar_length(img_width_px, pixel_size_m, unit, length=None):
    """
    Choose a scalebar length. If length is given (in display unit) use it
    directly; otherwise pick a round number ~15 % of the image width.
    Pass unit=None to auto-select the most readable SI unit.
    Returns (length_px, length_value, unit_label).
    """
    if unit is None:
        unit = _auto_unit(img_width_px, pixel_size_m)

    unit_factors = {"m": 1, "cm": 1e2, "mm": 1e3, "µm": 1e6, "nm": 1e9}
    factor = unit_factors.get(unit, 1e9)

    if length is not None:
        return length / (pixel_size_m * factor), length, unit

    img_width_real = img_width_px * pixel_size_m * factor
    target = img_width_real * 0.15

    magnitude = 10 ** np.floor(np.log10(target))
    for step in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]:
        candidate = step * magnitude
        if candidate >= target:
            nice = candidate
            break
    else:
        nice = target

    length_px = nice / (pixel_size_m * factor)
    return length_px, nice, unit
 
 
def _draw_scalebar(ax, arr, pixel_size_m, unit,
                   color, fontsize, loc, length=None, fullwidth=True):
    """
    Draw a scalebar rectangle + label using axes-fraction coordinates.
    loc      : "lower right" | "lower left" | "upper right" | "upper left"
               | "below left" | "below center" | "below right"
    fullwidth: if True (default), the black box spans the full axes width and
               the bar/label are centered; if False, a small corner box is used.
    """
    img_width_px = arr.shape[1]
    length_px, length_val, unit_label = _nice_scalebar_length(
        img_width_px, pixel_size_m, unit, length
    )

    bar_frac = length_px / img_width_px
    bar_h  = 0.012
    margin = 0.03
    pad    = 0.012
    gap    = 0.008
    label  = f"{length_val:.4g} {unit_label}"
    fig    = ax.get_figure()

    # ── measure text dimensions ───────────────────────────────────────────────
    tmp = ax.text(0.5, 0.5, label, transform=ax.transAxes,
                  ha="center", va="bottom", fontsize=fontsize, fontweight="bold")
    fig.canvas.draw()
    try:
        renderer = fig.canvas.get_renderer()
    except AttributeError:
        renderer = fig.canvas.renderer

    bb     = tmp.get_window_extent(renderer=renderer)
    ax_bb  = ax.get_window_extent(renderer=renderer)
    text_w = bb.width  / ax_bb.width   # axes-fraction
    text_h = bb.height / ax_bb.height  # axes-fraction
    tmp.remove()

    # ── below-image layout ────────────────────────────────────────────────────
    if "below" in loc:
        if "right" in loc:
            x_center = 1 - margin - bar_frac / 2
        elif "left" in loc:
            x_center = margin + bar_frac / 2
        else:
            x_center = 0.5

        box_h  = pad + bar_h + gap + text_h + pad
        box_y0 = -box_h
        x_bar  = x_center - bar_frac / 2
        y_bar  = box_y0 + pad
        y_text = y_bar + bar_h + gap

        ax.add_patch(mpatches.FancyBboxPatch(
            (0, box_y0), 1, box_h,
            boxstyle="square,pad=0", transform=ax.transAxes,
            facecolor="black", edgecolor="none", zorder=4, clip_on=False,
        ))
        ax.add_patch(mpatches.FancyBboxPatch(
            (x_bar, y_bar), bar_frac, bar_h,
            boxstyle="square,pad=0", transform=ax.transAxes,
            color=color, zorder=5, clip_on=False,
        ))
        ax.text(
            x_center, y_text, label,
            transform=ax.transAxes,
            ha="center", va="bottom",
            fontsize=fontsize, color=color,
            fontweight="bold", zorder=5, clip_on=False,
        )
        return

    # ── within-image layout ───────────────────────────────────────────────────
    box_h = bar_h + gap + text_h + 2 * pad

    if fullwidth:
        box_x0   = 0
        box_w    = 1
        box_y0   = 0 if "lower" in loc else (1 - box_h)
        x_center = 0.5
    else:
        inner_w  = max(bar_frac, text_w)
        box_w    = inner_w + 2 * pad
        box_x0   = (1 - margin - box_w) if "right" in loc else margin
        box_y0   = margin if "lower" in loc else (1 - margin - box_h)
        x_center = box_x0 + box_w / 2

    x_bar  = x_center - bar_frac / 2
    y_bar  = box_y0 + pad
    y_text = y_bar + bar_h + gap

    ax.add_patch(mpatches.FancyBboxPatch(
        (box_x0, box_y0), box_w, box_h,
        boxstyle="square,pad=0", transform=ax.transAxes,
        facecolor="black", edgecolor="none", zorder=4, clip_on=False,
    ))
    ax.add_patch(mpatches.FancyBboxPatch(
        (x_bar, y_bar), bar_frac, bar_h,
        boxstyle="square,pad=0", transform=ax.transAxes,
        color=color, zorder=5, clip_on=False,
    ))
    ax.text(
        x_center, y_text, label,
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=fontsize, color=color,
        fontweight="bold", zorder=5,
    )
 
 
def plot_sem(ax, path, cmap="gray", origin="upper", normalize=False,
            cropx=None, cropy=None,
            scalebar=False, unit=None,
            scalebar_color="white", scalebar_fontsize=10,
            scalebar_loc="lower right", scalebar_length=None,
            scalebar_fullwidth=True,
            pixel_size=None, frame_color=None, frame_linewidth=2):
    """
    Plot a SEM .tif image on a given matplotlib Axes object.
 
    Parameters
    ----------
    ax               : matplotlib.axes.Axes
    path             : str or PIL.Image  — file path or already-opened image
    cmap             : str               — colormap (default: "gray")
    origin           : str               — "upper" or "lower"
    normalize        : bool              — stretch intensity to [0, 1]
    cropx            : tuple (x0, x1)   — pixel range to keep in x (columns)
    cropy            : tuple (y0, y1)   — pixel range to keep in y (rows)
    scalebar         : bool              — draw a scalebar (default: False)
    unit             : str               — display unit: "nm","µm","mm","cm","m"
    scalebar_color   : str               — bar + text color (default: "white")
    scalebar_fontsize: int               — label fontsize (default: 10)
    scalebar_loc     : str               — "lower right" | "lower left" |
                                           "upper right" | "upper left"
    scalebar_length  : float or None     — explicit bar length in `unit`.
                                           If None, auto-sized to ~15 % of width.
    pixel_size       : float or None     — physical size of one pixel in metres.
                                           If None, read from TIFF metadata.
    frame_color      : str or None       — draw a frame around the image in this color.
                                           If None, no frame is drawn (default).
    frame_linewidth  : float             — linewidth of the frame (default: 2).
 
    Returns
    -------
    im : AxesImage
    """
    img = Image.open(path) if isinstance(path, str) else path
    arr = np.array(img)
 
    if cropy is not None:
        arr = arr[cropy[0]:cropy[1], :]
    if cropx is not None:
        arr = arr[:, cropx[0]:cropx[1]]
 
    if normalize:
        arr = arr.astype(float)
        arr = (arr - arr.min()) / (arr.max() - arr.min())
 
    im = ax.imshow(arr, cmap=cmap, origin=origin)
    ax.set_aspect('auto')
    ax.set_xlim(-0.5, arr.shape[1] - 0.5)
    if origin == "upper":
        ax.set_ylim(arr.shape[0] - 0.5, -0.5)
    else:
        ax.set_ylim(-0.5, arr.shape[0] - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    if frame_color is not None:
        ax.add_patch(mpatches.FancyBboxPatch(
            (0, 0), 1, 1,
            boxstyle="square,pad=0",
            transform=ax.transAxes,
            fill=False, edgecolor=frame_color, linewidth=frame_linewidth,
            zorder=10, clip_on=False,
        ))
 
    if scalebar:
        px_size = pixel_size if pixel_size is not None else _read_pixel_size(img)
        if px_size is None:
            raise ValueError(
                "Could not read pixel size from TIFF metadata. "
                "Pass it explicitly via pixel_size=<size in metres>."
            )
        _draw_scalebar(ax, arr, px_size, unit,
                       scalebar_color, scalebar_fontsize, scalebar_loc,
                       scalebar_length, scalebar_fullwidth)

    ax.set_aspect('auto')
    return im


def _fit_fontsize(ax, bar_x0, bar_x1, label_strings, max_fs=14, min_fs=4):
    """Return largest integer fontsize where all label_strings fit within bar_x0..bar_x1."""
    fig = ax.figure
    dpi = fig.dpi
    p0 = ax.transData.transform([bar_x0, 0])
    p1 = ax.transData.transform([bar_x1, 0])
    bar_px = abs(p1[0] - p0[0]) * 0.85  # 85 % usable width
    max_chars = max(len(s) for s in label_strings)
    # Bold sans-serif: ~0.65 pt wide per point of font size; 1 pt = dpi/72 px
    px_per_pt = dpi / 72
    for fs in range(int(max_fs), min_fs - 1, -1):
        if max_chars * 0.65 * fs * px_per_pt <= bar_px:
            return fs
    return min_fs


def plot_growth(ax, steps, times, width_factors, temperatures,
                ratios=None,
                colors=None, colors_text=None,
                label_fontsize=14,
                ylim_lower=None, ylim_upper=None):
    """
    Plot a growth sequence diagram on a matplotlib Axes object.

    Parameters
    ----------
    ax            : matplotlib.axes.Axes
    steps         : list of str     — step type per segment, e.g. ["InGaAs", "ramp", "InAs"]
                                      Recognised types: "InGaAs", "InAs", "ramp", "InAlAs"
    times         : array-like      — duration of each step (minutes)
    width_factors : array-like      — scaling factor applied to each step's time for display width
    temperatures  : array-like      — temperature at the end of each step (°C)
    ratios        : list or None    — V-III ratio label per step; use 0 / None entry to skip
                                      that step's label (default: no labels)
    colors        : dict or None    — override fill colours keyed by step name
    colors_text   : dict or None    — override text colours keyed by step name
    label_fontsize: int             — max fontsize for in-plot labels; auto-reduced to fit (default: 14)
    ylim_lower    : float or None   — lower y-limit; auto = min(T) - 20
    ylim_upper    : float or None   — upper y-limit; auto = max(T) + 10
    """
    times = np.asarray(times, dtype=float)
    width_factors = np.asarray(width_factors, dtype=float)
    temperatures = np.asarray(temperatures, dtype=float)

    nsteps = len(steps)

    scaled_times = times * width_factors
    scaled_times = np.concatenate(([0], scaled_times))
    cum_scaled_times = np.cumsum(scaled_times)
    temperatures = np.concatenate(([temperatures[0]], temperatures))

    _colors = {"InGaAs": "#ff9d33", "InAs": "#ffb037", "ramp": "#fbdfb7", "InAlAs": "#0f9dd5"}
    _colors_text = {"InGaAs": "#b56918", "InAs": "#b6730d", "ramp": "#a31f03", "InAlAs": "#085370"}
    if colors is not None:
        _colors.update(colors)
    if colors_text is not None:
        _colors_text.update(colors_text)

    y_lo = (np.min(temperatures) - 20) if ylim_lower is None else ylim_lower
    y_hi = (np.max(temperatures) + 10) if ylim_upper is None else ylim_upper
    ax.set_ylim([y_lo, y_hi])
    ax.set_yticks(np.unique(temperatures))

    x_end = cum_scaled_times[-1]
    ax.set_xlim([-x_end * 0.05, x_end * 1.12])

    ax.set_xlabel("Growth Time")
    ax.set_ylabel("Temperature (°C)")
    ax.set_xticks([])

    ax.spines["left"].set_position(("data", 0))
    for pos in ["top", "right", "bottom", "left"]:
        ax.spines[pos].set_visible(False)

    y_range = y_hi - y_lo
    # Compute arrowhead size in inches then convert to data units per axis,
    # so both arrowheads appear the same physical size regardless of axis scales.
    fig = ax.figure
    ax_pos = ax.get_position()
    ax_w_in = fig.get_figwidth() * ax_pos.width
    ax_h_in = fig.get_figheight() * ax_pos.height
    x_scale = (ax.get_xlim()[1] - ax.get_xlim()[0]) / ax_w_in  # data units per inch, x
    y_scale = y_range / ax_h_in                                  # data units per inch, y
    head_w_in = min(ax_w_in, ax_h_in) * 0.02   # arrowhead width in inches
    head_l_in = head_w_in * 0.75               # arrowhead length in inches
    hw_x = head_w_in * y_scale   # x-arrow: head_width is in y data units
    hl_x = head_l_in * x_scale   # x-arrow: head_length is in x data units
    hw_y = head_w_in * x_scale   # y-arrow: head_width is in x data units
    hl_y = head_l_in * y_scale   # y-arrow: head_length is in y data units

    ax.arrow(0, y_lo + 1, x_end * 1.05, 0,
             length_includes_head=True, color="black",
             head_width=hw_x, head_length=hl_x, zorder=10)
    ax.arrow(0, 0, 0, y_hi,
             length_includes_head=True, color="black",
             head_width=hw_y, head_length=hl_y, zorder=10)

    for i in range(nsteps):
        ax.fill_between(
            [cum_scaled_times[i], cum_scaled_times[i + 1]],
            [temperatures[i], temperatures[i + 1]],
            [y_lo + 1, y_lo + 1],
            color=_colors.get(steps[i], "#cccccc"),
            edgecolor="black",
        )

    ax.tick_params(which="major", right=False, direction="out", left=True, color="black")
    ax.tick_params(which="minor", right=False, direction="out", left=False)

    # Step labels (material name + time) — uniform fontsize, sized to the narrowest bar
    text_y_step = y_lo + y_range * 0.08
    text_y_name = y_lo + y_range * 0.04
    step_fs = label_fontsize
    for i in range(nsteps):
        if steps[i] != "ramp":
            x0 = cum_scaled_times[i]
            x1 = cum_scaled_times[i + 1]
            time_str = f"{int(times[i])}min"
            step_fs = min(step_fs, _fit_fontsize(ax, x0, x1, [steps[i], time_str], max_fs=label_fontsize))
    for i in range(nsteps):
        if steps[i] != "ramp":
            x0 = cum_scaled_times[i]
            x1 = cum_scaled_times[i + 1]
            time_str = f"{int(times[i])}min"
            tc = _colors_text.get(steps[i], "black")
            x_text = x0 + (x1 - x0) * 0.04
            ax.text(x_text, text_y_step, time_str,
                    color=tc, fontsize=step_fs, fontweight="bold")
            ax.text(x_text, text_y_name, steps[i],
                    color=tc, fontsize=step_fs, fontweight="bold")

    # V-III ratio labels — above the bars, autoscaled to the bar's width
    if ratios is not None:
        ratio_y_offset = y_range * 0.02
        for i, r in enumerate(ratios):
            if r and i < nsteps:
                x0 = cum_scaled_times[i]
                x1 = cum_scaled_times[i + 1]
                label = f"$\\mathbf{{R_{{V/III}}}}$={r}"
                bar_top = max(temperatures[i], temperatures[i + 1])
                x_text = x0 + (x1 - x0) * 0.04
                ax.text(x_text, bar_top + ratio_y_offset, label,
                        color="black", fontsize=step_fs, fontweight="bold")


def make_axes(fig, specs, low, top, heights, dys, lefts, rights,
              widths_list, ds_list, fixed_heights=None, fixed_list=None,
              labels=False, label_fontsize=12, xlabels=None, ylabels=None):
    """
    Create all axes for a figure layout and return them as a dict.

    Parameters
    ----------
    fig            : matplotlib.figure.Figure
    specs          : list of lists of name strings — one list per row;
                     specs[0] is the top row, specs[-1] is the bottom row
    low            : float             — bottom edge of the layout (figure fraction)
    top            : float             — top edge of the layout (figure fraction)
    heights        : array-like        — height of each row (figure fraction)
    dys            : array-like        — vertical gaps between rows (n_rows - 1 entries)
    lefts          : array-like        — left edge of each row (figure fraction)
    rights         : array-like        — right edge of each row (figure fraction)
    widths_list    : list of array-like — subplot widths per row
    ds_list        : list of array-like — horizontal gaps per row (n_subplots - 1 entries)
    fixed_heights  : array-like of bool, optional — fixed flags for row heights
    fixed_list     : list of array-like of bool, optional — fixed flags for widths per row
    labels         : bool              — add a), b), c)... labels to each axes (default: False)
    label_fontsize : int               — fontsize for panel labels (default: 12)
    xlabels        : list of str or None, optional — x-axis labels in left-to-right,
                                                     top-to-bottom order; use None entries to skip
    ylabels        : list of str or None, optional — y-axis labels in left-to-right,
                                                     top-to-bottom order; use None entries to skip

    Returns
    -------
    ax : dict — axes keyed by name
    """
    if fixed_list is None:
        fixed_list = [None] * len(specs)

    y_positions, scaled_heights = rescale_heights(low, top, heights, dys, fixed_heights)

    ax = {}
    idx = 0
    for row_idx, row_specs in enumerate(specs):
        positions, scaled_widths = rescale_widths(
            lefts[row_idx], rights[row_idx],
            widths_list[row_idx], ds_list[row_idx],
            fixed_list[row_idx])
        for i, name in enumerate(row_specs):
            ax[name] = fig.add_axes(
                [positions[i], y_positions[row_idx], scaled_widths[i], scaled_heights[row_idx]])
            if labels:
                offset = mtransforms.ScaledTranslation(-4/72, 4/72, fig.dpi_scale_trans)
                ax[name].text(0, 1, f"{chr(ord('a') + idx)})",
                              transform=ax[name].transAxes + offset,
                              ha="right", va="bottom",
                              fontsize=label_fontsize,
                              clip_on=False)
            if xlabels is not None and xlabels[row_idx][i] is not None:
                ax[name].set_xlabel(xlabels[row_idx][i])
            if ylabels is not None and ylabels[row_idx][i] is not None:
                ax[name].set_ylabel(ylabels[row_idx][i])
            idx += 1

    return ax


def rescale_heights(low, top, heights, dys, fixed=None):
    """
    Compute subplot y-positions and heights so that they fit exactly in [low, top].
    Row index 0 is the top row, index -1 is the bottom row.

    Parameters
    ----------
    low     : float        — bottom edge of the total region (axes fraction)
    top     : float        — top edge of the total region (axes fraction)
    heights : array-like   — relative heights of each row (N entries)
    dys     : array-like   — gaps between consecutive rows (N-1 entries)
    fixed   : array-like of bool, optional — if True for entry i, heights[i] is
                                             kept as-is; False entries and all dys
                                             are scaled uniformly (default: all False)

    Returns
    -------
    y_positions    : ndarray — bottom y-coordinate of each row (axes fraction)
    heights        : ndarray — rescaled height of each row (axes fraction)
    """
    heights = np.asarray(heights, dtype=float)
    dys     = np.asarray(dys,     dtype=float)

    if fixed is None:
        fixed = np.zeros(len(heights), dtype=bool)
    else:
        fixed = np.asarray(fixed, dtype=bool)

    scale          = (top - low - heights[fixed].sum()) / (heights[~fixed].sum() + dys.sum())
    scaled_heights = np.where(fixed, heights, heights * scale)
    scaled_dys     = dys * scale

    y_positions = [None] * len(heights)
    y_positions[-1] = low
    for i in range(len(heights) - 2, -1, -1):
        y_positions[i] = y_positions[i + 1] + scaled_heights[i + 1] + scaled_dys[i]

    return np.array(y_positions), scaled_heights


def rescale_widths(left, width, widths, ds, fixed=None):
    """
    Compute subplot positions and widths so that they fit exactly in [left, width].

    Parameters
    ----------
    left   : float        — left edge of the total region (axes fraction)
    width  : float        — right edge of the total region (axes fraction)
    widths : array-like   — relative widths of each subplot (N entries)
    ds     : array-like   — gaps between consecutive subplots (N-1 entries)
    fixed  : array-like of bool, optional — if True for entry i, widths[i] is
                                            kept as-is; False entries and all ds
                                            are scaled uniformly (default: all False)

    Returns
    -------
    positions : ndarray   — left x-coordinate of each subplot (axes fraction)
    widths    : ndarray   — rescaled width of each subplot (axes fraction)
    """
    widths = np.asarray(widths, dtype=float)
    ds     = np.asarray(ds,     dtype=float)

    if fixed is None:
        fixed = np.zeros(len(widths), dtype=bool)
    else:
        fixed = np.asarray(fixed, dtype=bool)

    scale         = (width - left - widths[fixed].sum()) / (widths[~fixed].sum() + ds.sum())
    scaled_widths = np.where(fixed, widths, widths * scale)
    scaled_ds     = ds * scale

    cum_widths = np.concatenate(([0], np.cumsum(scaled_widths[:-1])))
    cum_ds     = np.concatenate(([0], np.cumsum(scaled_ds)))
    positions  = left + cum_widths + cum_ds

    return positions, scaled_widths


def xy_to_hdf5(xy_path, hdf5_path=None):
    """
    Convert an XRD .xy file (two-column: 2theta, intensity) to HDF5.

    Parameters
    ----------
    xy_path  : str  — path to the .xy file
    hdf5_path: str  — output path; defaults to xy_path with .h5 extension

    Returns
    -------
    hdf5_path : str — path of the written file

    HDF5 layout
    -----------
    /two_theta   float64[N]   — 2theta angles in degrees
    /intensity   float64[N]   — measured intensity (counts)

    Attributes on the root group (parsed from filename if possible):
      scan_type, sample, reflection, optics, slit, scan_index, source_file
    """
    import h5py
    import os
    import re

    if hdf5_path is None:
        hdf5_path = os.path.splitext(xy_path)[0] + ".h5"

    # ── parse data ────────────────────────────────────────────────────────────
    two_theta, intensity = [], []
    with open(xy_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            try:
                two_theta.append(float(parts[0]))
                intensity.append(float(parts[1]))
            except ValueError:
                continue

    two_theta = np.array(two_theta, dtype=np.float64)
    intensity  = np.array(intensity,  dtype=np.float64)

    # ── parse metadata from filename ──────────────────────────────────────────
    basename = os.path.basename(xy_path)
    # expected pattern: <scan_type>_<sample>_<reflection>_<optics>_<slit>_<index>.xy
    pattern = r"^([^_]+)_([^_]+)_([^_]+)_([^_]+)_([^_]+)_(\d+)\.xy$"
    m = re.match(pattern, basename, re.IGNORECASE)
    meta = {
        "scan_type":   m.group(1) if m else "",
        "sample":      m.group(2) if m else "",
        "reflection":  m.group(3) if m else "",
        "optics":      m.group(4) if m else "",
        "slit":        m.group(5) if m else "",
        "scan_index":  int(m.group(6)) if m else -1,
        "source_file": basename,
    }

    # ── write HDF5 ────────────────────────────────────────────────────────────
    with h5py.File(hdf5_path, "w") as f:
        f.create_dataset("two_theta", data=two_theta)
        f.create_dataset("intensity",  data=intensity)
        for key, val in meta.items():
            f.attrs[key] = val

    return hdf5_path


def load_xrd_hdf5(hdf5_path):
    """
    Load an XRD HDF5 file written by xy_to_hdf5.

    Returns a SimpleNamespace with:
      .two_theta   float64 array — 2theta angles in degrees
      .intensity   float64 array — intensity (counts)
      .scan_type, .sample, .reflection, .optics, .slit,
      .scan_index, .source_file  — metadata from file attributes
    """
    import h5py
    from types import SimpleNamespace

    with h5py.File(hdf5_path, "r") as f:
        data = SimpleNamespace(
            two_theta = f["two_theta"][:],
            intensity  = f["intensity"][:],
            **{k: v for k, v in f.attrs.items()},
        )

    return data


def moving_average(n, y, x):
    """
    Apply a moving average of window size n to y, with x centered accordingly.

    Parameters
    ----------
    n : int            — window size
    y : array-like     — data to smooth
    x : array-like     — corresponding axis values

    Returns
    -------
    x_avg : ndarray    — x values centered on each window
    y_avg : ndarray    — smoothed y values
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D arrays")
    if len(x) != len(y):
        raise ValueError(f"x and y must have the same length (got {len(x)} and {len(y)})")

    y_avg = np.convolve(y, np.ones(n) / n, mode="valid")

    # Center x on each window: window i spans y[i:i+n], center at i + (n-1)/2
    if n % 2 == 1:
        x_avg = x[(n - 1) // 2 : (n - 1) // 2 + len(y_avg)]
    else:
        lo = n // 2 - 1
        x_avg = (x[lo : lo + len(y_avg)] + x[lo + 1 : lo + 1 + len(y_avg)]) / 2

    return x_avg, y_avg


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog="plothelper")
    sub    = parser.add_subparsers(dest="command")

    p = sub.add_parser("xy_to_hdf5", help="Convert an XRD .xy file to HDF5")
    p.add_argument("xy_path",              help="Input .xy file")
    p.add_argument("hdf5_path", nargs="?", help="Output .h5 file (default: same name as input)")

    args = parser.parse_args()

    if args.command == "xy_to_hdf5":
        print(f"Written to: {xy_to_hdf5(args.xy_path, args.hdf5_path)}")
    else:
        parser.print_help()
