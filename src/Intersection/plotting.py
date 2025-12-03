#!/usr/bin/env python3
"""
Paper figures for Intersection Ablations — polished styling + 99% CIs + robust safety.
Updates per request:
  • Third panel of Fig.2 uses acceleration notation: |a_L − a*_L| (m/s^2)
  • Fig.2 panels labeled with a), b), c) (like Fig.1)
  • Keeps wider uncertainty bands (99% CI) and professional styling

Inputs (current dir):
  • complete_abelation_results.pkl
  • baseline_abelation_results.pkl
  • influence_abelation_results_gamma0.pkl
  • influence_abelation_results_gamma001.pkl
  • influence_abelation_results_gamma01.pkl

Outputs (./figures):
  • fig_intersection_trajectories_grid.png / .pdf
      (1×4) a) Complete information, b) Baseline, c) Influence (γ=0), d) Influence (γ=0.1)
      – Safe vs Unsafe colored; central collision box shown
  • fig_beliefs_and_error.png / .pdf
      (1×3) a) Teacher belief, b) Learner belief, c) |a_L − a*_L| with mean±CI (99%)
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

# ============================ CONFIG / STYLE ============================

# Geometry
R: float = 70.0
W: float = 1.5
L: float = 3.0
D_CROSS: float = R / 2.0
HALF_BOX: float = 2.25  # per complete ablation code

# Confidence interval level (wider band)
CI_LEVEL = 0.99  # 99% CI

# Color-blind friendly palette
SAFE_COLOR   = "#2b8cbe"  # safe trajectories (blue)
UNSAFE_COLOR = "#d7301f"  # unsafe trajectories (red)

# Belief / error series colors
SERIES_STYLES = {
    "baseline": {"label": "Baseline", "color": "#1b9e77"},
    "g1"      : {"label": "γ = 1",    "color": "#d95f02"},
    "g10"    : {"label": "γ = 10", "color": "#7570b3"},
    "g100"     : {"label": "γ = 100",  "color": "#e7298a"},
}

BOX_FACE = "#e6e6e6"
BOX_EDGE = "#9a9a9a"

# Output dir
FIG_DIR = Path("figures2")

# Matplotlib global style (paper-ish)
mpl.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 420,
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "legend.fontsize": 12,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "pdf.fonttype": 42,  # embed fonts
    "ps.fonttype": 42,
    "axes.linewidth": 1.0,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.7,
})

# ============================ IO HELPERS ============================

def load_pkl(path: Path) -> List[Dict[str, Any]]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"{path} did not contain a non-empty list.")
    return data

# ============================ SHAPE / NAN GUARDS ============================

def ensure_states_4xT(states: np.ndarray) -> np.ndarray:
    """Ensure states is (4,T) with rows [d1, v1, d2, v2]; accept (T,4)."""
    arr = np.asarray(states)
    if arr.ndim != 2:
        raise ValueError(f"states must be 2D, got {arr.ndim}D")
    if arr.shape[0] == 4:
        return arr
    if arr.shape[1] == 4:
        return arr.T
    raise ValueError(f"states has invalid shape {arr.shape}; expected (4,T) or (T,4).")

# ============================ SAFETY CHECK (vertex + segment, NaN-aware) ============================

def _segment_intersects_box(x0: float, y0: float, x1: float, y1: float,
                            hx: float, hy: float) -> bool:
    """
    Liang–Barsky slab test for segment-box intersection with axis-aligned box [-hx,hx]×[-hy,hy].
    Inclusive: touching the boundary counts as intersection.
    """
    dx, dy = x1 - x0, y1 - y0

    def interval_1d(p0, dp, h):
        if abs(dp) < 1e-12:
            return (0.0, 1.0) if abs(p0) <= h else (1.0, 0.0)
        t0, t1 = (-h - p0) / dp, (h - p0) / dp
        if t0 > t1:
            t0, t1 = t1, t0
        return max(0.0, t0), min(1.0, t1)

    tx0, tx1 = interval_1d(x0, dx, hx)
    ty0, ty1 = interval_1d(y0, dy, hy)
    t_enter, t_exit = max(tx0, ty0), min(tx1, ty1)
    return t_enter <= t_exit

def traj_is_safe(states: np.ndarray) -> Tuple[bool, np.ndarray, np.ndarray]:
    """
    SAFE iff (a) for all valid k, NOT (|d1-D_CROSS|<HALF_BOX AND |d2-D_CROSS|<HALF_BOX)
    AND (b) no segment between successive valid samples intersects the central box.
    Returns (is_safe, x_xy, y_xy); x/y may contain NaNs (for plotting gaps).
    """
    xs = ensure_states_4xT(states)
    d1 = np.asarray(xs[0], dtype=float)
    d2 = np.asarray(xs[2], dtype=float)

    valid = np.isfinite(d1) & np.isfinite(d2)
    if not np.any(valid):
        return True, np.array([]), np.array([])

    # (a) Vertex test on valid points
    if np.any((np.abs(d1[valid] - D_CROSS) < HALF_BOX) &
              (np.abs(d2[valid] - D_CROSS) < HALF_BOX)):
        return False, d2 - D_CROSS, d1 - D_CROSS

    # (b) Segment intersection on valid-to-valid neighbors
    x = d2 - D_CROSS
    y = d1 - D_CROSS
    for k in range(len(x) - 1):
        if valid[k] and valid[k+1]:
            if _segment_intersects_box(x[k], y[k], x[k+1], y[k+1], HALF_BOX, HALF_BOX):
                return False, x, y

    return True, x, y

# ============================ PLOTTING HELPERS ============================

def add_panel_letter(ax: plt.Axes, letter: str):
    ax.text(0.015, 0.985, letter, transform=ax.transAxes,
            ha="left", va="top", fontsize=16, fontweight="bold")

def prettify_axes(ax: plt.Axes, equal=True):
    if equal:
        ax.set_aspect("equal", adjustable="box")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', length=6)
    ax.tick_params(axis='both', which='minor', length=3)

def plot_trajectories_panel(ax: plt.Axes, results: List[Dict[str, Any]]):
    # Collision box
    ax.add_patch(Rectangle(
        (-HALF_BOX, -HALF_BOX), 2*HALF_BOX, 2*HALF_BOX,
        facecolor=BOX_FACE, edgecolor=BOX_EDGE, linewidth=1.2, zorder=0
    ))

    # Sort: plot SAFE first (lighter), then UNSAFE on top (heavier)
    safe_curves = []
    unsafe_curves = []
    for res in results:
        try:
            is_safe, x_xy, y_xy = traj_is_safe(res["states"])
        except Exception:
            continue
        if x_xy.size == 0:
            continue
        (safe_curves if is_safe else unsafe_curves).append((x_xy, y_xy))

    # Safe (subtle)
    for x, y in safe_curves:
        ax.plot(x, y, color=SAFE_COLOR, alpha=0.35, linewidth=1.0,
                solid_capstyle="round", zorder=2)

    # Unsafe (emphasized with slight outline)
    for x, y in unsafe_curves:
        ax.plot(
            x, y, color=UNSAFE_COLOR, alpha=0.95, linewidth=1.6,
            solid_capstyle="round", zorder=3,
            path_effects=[pe.Stroke(linewidth=2.2, foreground="white", alpha=0.6),
                          pe.Normal()]
        )

    # Axes
    ax.axhline(0, color="#bbbbbb", lw=0.8, zorder=1)
    ax.axvline(0, color="#bbbbbb", lw=0.8, zorder=1)
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.set_xlabel("x = d2 − D_CROSS (m)")
    ax.set_ylabel("y = d1 − D_CROSS (m)")
    prettify_axes(ax, equal=True)

def z_from_ci(ci: float) -> float:
    """Map CI level to a z-score (supports 0.95, 0.99; extend if needed)."""
    if ci >= 0.99:
        return 2.576
    return 1.96

def mean_ci_nanaware(Y: np.ndarray, ci: float = CI_LEVEL) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Y: (N, T) possibly with NaNs → mean, lower, upper (per-time).
    Normal approx with per-time effective N.
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2:
        raise ValueError("Y must be (N,T)")
    mean = np.nanmean(Y, axis=0)
    n_eff = np.sum(np.isfinite(Y), axis=0)
    std = np.nanstd(Y, axis=0, ddof=1)
    std = np.where(n_eff > 1, std, 0.0)
    z = z_from_ci(ci)
    sem = np.where(n_eff > 0, std / np.sqrt(n_eff), np.nan)
    lo, hi = mean - z * sem, mean + z * sem
    return mean, lo, hi

def stack_field(results: List[Dict[str, Any]], key: str) -> np.ndarray:
    """Stack a 1D field of length T into (N,T), trimming to min T across entries."""
    arrs = [np.asarray(r[key], dtype=float) for r in results]
    T = min(a.shape[0] for a in arrs)
    arrs = [a[:T] for a in arrs]
    return np.stack(arrs, axis=0)

def align_by_trials(*lists_of_results: List[List[Dict[str, Any]]]) -> Tuple[int, int]:
    """Return (N_min, T_min) common across all provided result lists."""
    Ns = [len(lst) for lst in lists_of_results]
    Nmin = min(Ns)
    Ts = []
    for lst in lists_of_results:
        T_this = min(np.asarray(r["a1"]).shape[0] for r in lst[:Nmin])
        Ts.append(T_this)
    Tmin = min(Ts)
    return Nmin, Tmin

# ============================ FIGURE BUILDERS ============================

def make_figure_trajectories_grid(
    complete_pkl: Path,
    baseline_pkl: Path,
    g1_pkl: Path,
    g10_pkl: Path,
    out_png: Path,
    out_pdf: Path,
):
    complete = load_pkl(complete_pkl)
    baseline = load_pkl(baseline_pkl)
    g1       = load_pkl(g1_pkl)
    g10      = load_pkl(g10_pkl)

    fig, axes = plt.subplots(1, 4, figsize=(18.0, 4.8), constrained_layout=True)

    # (a) Complete information
    plot_trajectories_panel(axes[0], complete)
    add_panel_letter(axes[0], "a)")
    axes[0].set_title("Complete information", pad=6)

    # (b) Baseline
    plot_trajectories_panel(axes[1], baseline)
    add_panel_letter(axes[1], "b)")
    axes[1].set_title("Baseline", pad=6)

    # (c) Influence (γ = 0)
    plot_trajectories_panel(axes[2], g1)
    add_panel_letter(axes[2], "c)")
    axes[2].set_title("Influence (γ = 0)", pad=6)

    # (d) Influence (γ = 0.1)
    plot_trajectories_panel(axes[3], g10)
    add_panel_letter(axes[3], "d)")
    axes[3].set_title("Influence (γ = 0.1)", pad=6)

    # Shared legend (one for all panels)
    handles = [
        Line2D([0], [0], color=SAFE_COLOR, lw=2, alpha=0.6, label="Safe"),
        Line2D([0], [0], color=UNSAFE_COLOR, lw=2, alpha=0.95, label="Unsafe"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, borderaxespad=0.6)

    fig.savefig(out_png, dpi=420, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

def make_figure_beliefs_and_error(
    complete_pkl: Path,
    baseline_pkl: Path,
    g1_pkl: Path,
    g10_pkl: Path,
    g100_pkl: Path,
    out_png: Path,
    out_pdf: Path,
):
    complete = load_pkl(complete_pkl)
    baseline = load_pkl(baseline_pkl)
    g1       = load_pkl(g1_pkl)
    g10     = load_pkl(g10_pkl)
    g100      = load_pkl(g100_pkl)

    # Align trials/time
    N, T = align_by_trials(complete, baseline, g1, g10, g100)
    complete = complete[:N]
    baseline = baseline[:N]
    g1       = g1[:N]
    g10     = g10[:N]
    g100      = g100[:N]

    dt = 0.1
    ts = np.arange(T) * dt

    # Beliefs
    bel_r_baseline = stack_field(baseline, "belief_r")[:, :T]
    bel_r_g1       = stack_field(g1,       "belief_r")[:, :T]
    bel_r_g10     = stack_field(g10,     "belief_r")[:, :T]
    bel_r_g100      = stack_field(g100,      "belief_r")[:, :T]

    bel_h_baseline = stack_field(baseline, "belief_h")[:, :T]
    bel_h_g1       = stack_field(g1,       "belief_h")[:, :T]
    bel_h_g10     = stack_field(g10,     "belief_h")[:, :T]
    bel_h_g100     = stack_field(g100,      "belief_h")[:, :T]

    # Absolute acceleration error vs complete-info (a1 is acceleration)
    a_star = stack_field(complete, "a1")[:, :T]  # learner/ human accel under complete info
    def abs_err(lst: List[Dict[str, Any]]) -> np.ndarray:
        return np.abs(stack_field(lst, "a1")[:, :T] - a_star)

    pe_baseline = 1*abs_err(baseline)
    pe_g1       = 1*abs_err(g1)
    pe_g10     = 1*abs_err(g10)
    pe_g100      = 1*abs_err(g100)

    # Build figure
    fig, axes = plt.subplots(1, 3, figsize=(17.6, 5.0), constrained_layout=True)

    def plot_mean_ci(ax, ts, Y, style_key: str, lw: float = 2.6):
        mean, lo, hi = mean_ci_nanaware(Y, ci=CI_LEVEL)
        st = SERIES_STYLES[style_key]
        ax.plot(ts, mean, lw=lw, color=st["color"], label=st["label"],
                path_effects=[pe.Stroke(linewidth=lw+0.8, foreground="white", alpha=0.5), pe.Normal()])
        ax.fill_between(ts, lo, hi, alpha=0.30, color=st["color"], linewidth=0)

    # (a) Teacher (robot) belief
    ax0 = axes[0]
    plot_mean_ci(ax0, ts, bel_r_baseline, "baseline")
    plot_mean_ci(ax0, ts, bel_r_g1,       "g1")
    plot_mean_ci(ax0, ts, bel_r_g10,     "g10")
    plot_mean_ci(ax0, ts, bel_r_g100,      "g100")
    ax0.set_xlabel("time (s)")
    ax0.set_ylabel("teacher belief in true $\\theta_L$")
    ax0.set_ylim(-0.05, 1.05)
    add_panel_letter(ax0, "a)")
    prettify_axes(ax0, equal=False)
    ax0.legend(frameon=False, ncol=2, loc="lower right")

    # (b) Learner (human) belief
    ax1 = axes[1]
    plot_mean_ci(ax1, ts, bel_h_baseline, "baseline")
    plot_mean_ci(ax1, ts, bel_h_g1,       "g1")
    plot_mean_ci(ax1, ts, bel_h_g10,     "g10")
    plot_mean_ci(ax1, ts, bel_h_g100,      "g100")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("learner belief in true $\\theta_T$")
    ax1.set_ylim(-0.05, 1.05)
    add_panel_letter(ax1, "b)")
    prettify_axes(ax1, equal=False)

    # (c) |a_L − a*_L|
    ax2 = axes[2]
    plot_mean_ci(ax2, ts, np.log2(1+pe_baseline), "baseline")
    plot_mean_ci(ax2, ts, np.log2(1+pe_g1),       "g1")
    plot_mean_ci(ax2, ts, np.log2(1+pe_g10),     "g10")
    plot_mean_ci(ax2, ts, np.log2(1+pe_g100),      "g100")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("|a$_{L}$ − a$^*_{L}$|  (m/s$^2$)")
    add_panel_letter(ax2, "c)")
    prettify_axes(ax2, equal=False)

    fig.savefig(out_png, dpi=420, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

# ============================ MAIN ============================

if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Inputs
    complete_pkl = Path("complete_abelation_results.pkl")
    baseline_pkl = Path("baseline_abelation_results.pkl")
    g1_pkl       = Path("influence_abelation_results_gamma1.pkl")
    g10_pkl     = Path("influence_abelation_results_gamma10.pkl")
    g100_pkl      = Path("influence_abelation_results_gamma100.pkl")

    # Figure 1: trajectories grid
    make_figure_trajectories_grid(
        complete_pkl=complete_pkl,
        baseline_pkl=baseline_pkl,
        g1_pkl=g1_pkl,
        g10_pkl=g10_pkl,
        out_png=FIG_DIR / "fig_intersection_trajectories_grid.png",
        out_pdf=FIG_DIR / "fig_intersection_trajectories_grid.pdf",
    )

    # Figure 2: beliefs (99% CI) + absolute acceleration error
    make_figure_beliefs_and_error(
        complete_pkl=complete_pkl,
        baseline_pkl=baseline_pkl,
        g1_pkl=g1_pkl,
        g10_pkl=g10_pkl,
        g100_pkl=g100_pkl,
        out_png=FIG_DIR / "fig_beliefs_and_error.png",
        out_pdf=FIG_DIR / "fig_beliefs_and_error.pdf",
    )

    print(f"Saved figures to: {FIG_DIR.resolve()}")
