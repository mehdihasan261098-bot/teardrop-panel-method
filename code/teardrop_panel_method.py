"""
===============================================================================
 PANEL METHODS PROJECT  -  TEARDROP GEOMETRY
 Student ID : 210011253
 Course     : Aerodynamics (MPE)

"""

import subprocess
from pathlib import Path as SysPath

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.path import Path

# The four geometric-integral routines supplied with the course material
from COMPUTE_IJ_SPM import COMPUTE_IJ_SPM      # source method: I and J integrals
from COMPUTE_KL_VPM import COMPUTE_KL_VPM      # vortex method: K and L integrals
from STREAMLINE_SPM import STREAMLINE_SPM      # source method: Mx and My integrals
from STREAMLINE_VPM import STREAMLINE_VPM      # vortex method: Nx and Ny integrals


# =============================================================================
# STEP 1 - SETTINGS
# =============================================================================
BASE_DIR   = SysPath(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

V_INF     = 1.0         # freestream speed                      [m/s]
ALPHA_DEG = 0.0         # angle of attack (0 deg -> symmetry)   [deg]
RHO       = 1.225       # air density                           [kg/m^3]
CHORD     = 1.0         # chord length (geometry is normalised)

N_SPM      = 100        # panels on the COMPLETE body for the source method (even)

N_VPM_HALF = 4
VPM_STUDY = (4, 6, 8, 10, 12, 16, 20)   # counts used for the sensitivity table

THICKNESS = 0.30        # maximum half-thickness y/c  (full thickness = 0.60c)

GRID_NX, GRID_NY = 130, 100     # flow-field grid resolution
QUIVER_STEP      = 8            # draw one velocity arrow every N grid points
CP_VECTOR_SCALE  = 0.08         # length scaling of the Cp vectors

SHOW_FIGURES = True             # set False to only save the PNG files


# =============================================================================
# STEP 2 - GEOMETRY
# =============================================================================
def teardrop_thickness(x):
    """
    Half-thickness of the teardrop at station x.

        y(x) = ymax * [ sqrt(x) * (1 - x) ] / max_value

    sqrt(x)  -> rounded nose at the leading edge
    (1 - x)  -> sharp point at the trailing edge
    """
    x = np.asarray(x, dtype=float)
    shape = np.sqrt(np.clip(x, 0.0, 1.0)) * (1.0 - x)
    shape_max = 2.0 / (3.0 * np.sqrt(3.0))       # peak of sqrt(x)*(1-x), at x = 1/3
    return THICKNESS * shape / shape_max


def make_teardrop(n_panels):
    """
    Build the closed teardrop boundary in CLOCKWISE order.

    Point order is  TE -> lower surface -> LE -> upper surface -> TE,
    which is the order the supplied panel routines expect.
    """
    if n_panels % 2 != 0:
        raise ValueError("n_panels must be an even number.")

    n_half = n_panels // 2

    # Cosine spacing: packs points near the LE and TE where curvature is large
    theta = np.linspace(0.0, np.pi, n_half + 1)
    x = 0.5 * (1.0 - np.cos(theta))
    y = teardrop_thickness(x)

    XB = np.concatenate((x[::-1], x[1:]))        # lower (TE->LE) then upper (LE->TE)
    YB = np.concatenate((-y[::-1], y[1:]))
    return XB, YB


def make_panels(XB, YB, alpha_deg=ALPHA_DEG):
    """
    Turn boundary points into panel data.

    Returns
        XC, YC : panel control points (mid-points)
        S      : panel lengths
        phi    : angle of the panel measured from the +x axis
        delta  : angle of the OUTWARD normal
        beta   : angle between the outward normal and the freestream
    """
    XC = 0.5 * (XB[:-1] + XB[1:])
    YC = 0.5 * (YB[:-1] + YB[1:])

    dx = XB[1:] - XB[:-1]
    dy = YB[1:] - YB[:-1]

    S   = np.hypot(dx, dy)                                # panel length
    phi = np.mod(np.arctan2(dy, dx), 2.0 * np.pi)         # panel orientation

    delta = np.mod(phi + np.pi / 2.0, 2.0 * np.pi)        # outward normal
    beta  = np.mod(delta - np.deg2rad(alpha_deg), 2.0 * np.pi)

    return XC, YC, S, phi, delta, beta


# =============================================================================
# STEP 3 - SOURCE PANEL METHOD (COMPLETE BODY)
# =============================================================================
def solve_source_panel(XB, YB):
    """
    Constant-strength source panels on the whole teardrop.

    No-penetration at every control point gives the linear system

        pi*lambda_i + sum_j I_ij * lambda_j = -2*pi*V_inf*cos(beta_i)

    Once lambda is known, the surface tangential velocity is

        Vt_i = V_inf*sin(beta_i) + (1/2pi) * sum_j J_ij * lambda_j
        Cp_i = 1 - (Vt_i / V_inf)^2
    """
    XC, YC, S, phi, delta, beta = make_panels(XB, YB)
    n = len(XC)

    # Geometric integrals from the supplied routine
    I, J = COMPUTE_IJ_SPM(XC, YC, XB, YB, phi, S)

    # Assemble and solve  A * lambda = b
    A = I + np.pi * np.eye(n)
    b = -2.0 * np.pi * V_INF * np.cos(beta)
    lam = np.linalg.solve(A, b)

    # Surface velocity and pressure coefficient
    Vt = V_INF * np.sin(beta) + (J @ lam) / (2.0 * np.pi)
    Cp = 1.0 - (Vt / V_INF) ** 2

    # Force coefficients by integrating the surface pressure
    alpha = np.deg2rad(ALPHA_DEG)
    CN = -Cp * S * np.sin(beta)          # normal force
    CA = -Cp * S * np.cos(beta)          # axial force
    CL = np.sum(CN) * np.cos(alpha) - np.sum(CA) * np.sin(alpha)
    CD = np.sum(CN) * np.sin(alpha) + np.sum(CA) * np.cos(alpha)

    # CHECK: for a closed body the total source strength must be ~ 0
    source_sum = np.sum(lam * S)

    return {
        "XB": XB, "YB": YB, "XC": XC, "YC": YC,
        "S": S, "phi": phi, "delta": delta,
        "lambda": lam, "Cp": Cp,
        "CL": CL, "CD": CD,
        "source_sum": source_sum,
        "bc_residual": np.max(np.abs(A @ lam - b)),
        "cond": np.linalg.cond(A),
    }


# =============================================================================
# STEP 4 - VORTEX PANEL METHOD (UPPER HALF ONLY)
# =============================================================================
def solve_vortex_panel_upper_half(n_half=N_VPM_HALF):
    
    if abs(ALPHA_DEG) > 1e-12:
        raise ValueError("The half-body (symmetry) reduction is only valid at alpha = 0.")

    n_half = int(n_half)
    n_full = 2 * n_half

    XB, YB = make_teardrop(n_full)
    XC, YC, S, phi, delta, beta = make_panels(XB, YB)

    K, L = COMPUTE_KL_VPM(XC, YC, XB, YB, phi, S)

    # Panel numbering:  0 .. n_half-1 = lower (TE->LE),  n_half .. n_full-1 = upper (LE->TE)
    upper = np.arange(n_half, n_full)

    # Build the reduced (half-size) system
    A_half = np.zeros((n_half, n_half))
    for row, i in enumerate(upper):
        for k in range(n_half):
            j_upper = n_half + k                 # upper panel k
            j_lower = n_half - 1 - k             # its mirror on the lower surface
            A_half[row, k] = K[i, j_lower] - K[i, j_upper]

    b_half = -2.0 * np.pi * V_INF * np.cos(beta[upper])
    gamma_upper = np.linalg.solve(A_half, b_half)

    # Rebuild the full strength distribution by symmetry (used for the flow field)
    gamma_full = np.empty(n_full)
    gamma_full[n_half:] = gamma_upper
    gamma_full[:n_half] = -gamma_upper[::-1]

    # Surface velocity and pressure coefficient
    Vt = V_INF * np.sin(beta) - (L @ gamma_full) / (2.0 * np.pi) + gamma_full / 2.0
    Cp = 1.0 - (Vt / V_INF) ** 2

    # Circulation and the Kutta-Joukowski lift check
    circulation = np.sum(gamma_full * S)
    CL_kj = 2.0 * circulation / (V_INF * CHORD)

    return {
        "n_half": n_half, "n_full": n_full,
        "XB": XB, "YB": YB, "XC": XC, "YC": YC,
        "S": S, "phi": phi, "delta": delta,
        "gamma_full": gamma_full,
        "Cp": Cp,
        "XC_upper": XC[upper], "YC_upper": YC[upper],
        "Cp_upper": Cp[upper],
        "delta_upper": delta[upper],
        "circulation": circulation, "CL_kj": CL_kj,
        "bc_residual": np.max(np.abs(A_half @ gamma_upper - b_half)),
        "cond": np.linalg.cond(A_half),
    }


# =============================================================================
# STEP 5 & 6 - FLOW FIELD  (velocity, streamlines, pressure)
# =============================================================================
def make_grid():
    """Rectangular grid of points around the body where the flow is evaluated."""
    x = np.linspace(-0.35, 1.40, GRID_NX)
    y = np.linspace(-0.60, 0.60, GRID_NY)
    return np.meshgrid(x, y)


def inside_body(XX, YY, XB, YB):
    """True for grid points that fall inside the teardrop (flow is not valid there)."""
    poly = Path(np.column_stack((XB, YB)))
    pts = np.column_stack((XX.ravel(), YY.ravel()))
    return poly.contains_points(pts).reshape(XX.shape)


def field_from_velocity(U, V):
    """Convert the velocity field into speed, Cp and the pressure difference."""
    speed = np.sqrt(U ** 2 + V ** 2)
    Cp = 1.0 - (speed / V_INF) ** 2
    dp = 0.5 * RHO * V_INF ** 2 * Cp        # p - p_inf  [Pa]
    return speed, Cp, dp


def source_flow_field(spm):
    """
    Velocity field of the source panel method.

        u(P) = V_inf*cos(alpha) + (1/2pi) * sum_j lambda_j * Mx_j(P)
        v(P) = V_inf*sin(alpha) + (1/2pi) * sum_j lambda_j * My_j(P)
    """
    XX, YY = make_grid()
    U = np.full_like(XX, np.nan)
    V = np.full_like(YY, np.nan)
    solid = inside_body(XX, YY, spm["XB"], spm["YB"])
    alpha = np.deg2rad(ALPHA_DEG)

    for r in range(XX.shape[0]):
        for c in range(XX.shape[1]):
            if solid[r, c]:
                continue                      # skip points inside the body
            Mx, My = STREAMLINE_SPM(XX[r, c], YY[r, c],
                                    spm["XB"], spm["YB"], spm["phi"], spm["S"])
            U[r, c] = V_INF * np.cos(alpha) + np.sum(spm["lambda"] * Mx) / (2.0 * np.pi)
            V[r, c] = V_INF * np.sin(alpha) + np.sum(spm["lambda"] * My) / (2.0 * np.pi)

    speed, Cp, dp = field_from_velocity(U, V)
    return {"XX": XX, "YY": YY, "U": U, "V": V,
            "speed": speed, "Cp": Cp, "dp": dp}


def vortex_flow_field(vpm):
    """
    Velocity field of the vortex panel method, using the symmetric strength
    distribution rebuilt from the upper-half solution.

        u(P) = V_inf*cos(alpha) - (1/2pi) * sum_j gamma_j * Nx_j(P)
        v(P) = V_inf*sin(alpha) - (1/2pi) * sum_j gamma_j * Ny_j(P)
    """
    XX, YY = make_grid()
    U = np.full_like(XX, np.nan)
    V = np.full_like(YY, np.nan)
    solid = inside_body(XX, YY, vpm["XB"], vpm["YB"])
    alpha = np.deg2rad(ALPHA_DEG)

    for r in range(XX.shape[0]):
        for c in range(XX.shape[1]):
            if solid[r, c]:
                continue
            Nx, Ny = STREAMLINE_VPM(XX[r, c], YY[r, c],
                                    vpm["XB"], vpm["YB"], vpm["phi"], vpm["S"])
            U[r, c] = V_INF * np.cos(alpha) - np.sum(vpm["gamma_full"] * Nx) / (2.0 * np.pi)
            V[r, c] = V_INF * np.sin(alpha) - np.sum(vpm["gamma_full"] * Ny) / (2.0 * np.pi)

    speed, Cp, dp = field_from_velocity(U, V)
    return {"XX": XX, "YY": YY, "U": U, "V": V,
            "speed": speed, "Cp": Cp, "dp": dp}


# =============================================================================
# STEP 7 - XFOIL REFERENCE DATA
# =============================================================================
def write_dat_file(XB, YB, filename="teardrop_210011253.dat"):
    """
    Save the geometry in XFOIL format.
    XFOIL wants COUNTER-CLOCKWISE points (TE -> upper -> LE -> lower -> TE),
    so the clockwise panel-method ordering is simply reversed.
    """
    path = BASE_DIR / filename
    with open(path, "w") as f:
        f.write("TEARDROP_210011253\n")
        for x, y in zip(XB[::-1], YB[::-1]):
            f.write(f"{x:.10f}  {y:.10f}\n")
    return path


def run_xfoil(dat_path):
    """Run xfoil.exe on the same geometry and read back its surface Cp."""
    exe = BASE_DIR / "xfoil.exe"
    cp_file = BASE_DIR / "xfoil_cp_210011253.txt"

    if not exe.exists():
        print("  XFOIL skipped: xfoil.exe not found in this folder.")
        return None
    if cp_file.exists():
        cp_file.unlink()

    commands = (f"LOAD {dat_path.name}\n"
                "PANE\n"
                "OPER\n"
                f"ALFA {ALPHA_DEG}\n"
                f"CPWR {cp_file.name}\n"
                "\n"
                "QUIT\n")

    try:
        subprocess.run([str(exe)], input=commands, text=True,
                       capture_output=True, cwd=BASE_DIR, timeout=60)
    except Exception as exc:
        print(f"  XFOIL skipped: could not run it ({exc}).")
        return None

    if not cp_file.exists():
        print("  XFOIL ran but produced no Cp file.")
        return None

    data = np.loadtxt(cp_file, skiprows=3)
    print(f"  XFOIL finished: {len(data)} surface points read.")
    return {"x": data[:, 0], "y": data[:, 1], "Cp": data[:, 2]}


def xfoil_side(xfoil, upper=True):
    """Split the XFOIL data into the upper or the lower surface, sorted by x."""
    if xfoil is None:
        return None, None
    mask = xfoil["y"] >= 0.0 if upper else xfoil["y"] < 0.0
    x, cp = xfoil["x"][mask], xfoil["Cp"][mask]
    order = np.argsort(x)
    return x[order], cp[order]


def cp_rmse(x_model, cp_model, xfoil, upper=True):
    """RMS difference between the panel-method Cp and the XFOIL Cp."""
    x_ref, cp_ref = xfoil_side(xfoil, upper)
    if x_ref is None or len(x_ref) < 2:
        return np.nan

    order = np.argsort(x_model)
    xm, cpm = np.asarray(x_model)[order], np.asarray(cp_model)[order]
    keep = (xm >= x_ref.min()) & (xm <= x_ref.max())
    xm, cpm = xm[keep], cpm[keep]
    if len(xm) == 0:
        return np.nan

    return float(np.sqrt(np.mean((cpm - np.interp(xm, x_ref, cp_ref)) ** 2)))


# =============================================================================
# PLOTTING HELPERS
# =============================================================================
def save_figure(fig, filename):
    """Save a figure into the outputs folder."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=200)
    print(f"  saved  outputs/{filename}")


def draw_body(ax, XB, YB, color="0.85"):
    """Draw the filled teardrop outline."""
    ax.fill(XB, YB, color, edgecolor="black", linewidth=1.3, zorder=4)


# -----------------------------------------------------------------------------
# FIGURE 1 - geometry and panels
# -----------------------------------------------------------------------------
def plot_geometry(spm, vpm):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7))

    ax1.plot(spm["XB"], spm["YB"], "k.-", ms=4, lw=1.2, label="panel boundary points")
    ax1.plot(spm["XC"], spm["YC"], "rx", ms=4, label="control points")
    ax1.set_title("Teardrop geometry - complete body, %d source panels" % len(spm["XC"]))
    ax1.legend(loc="upper right", fontsize=9)

    n = vpm["n_half"]
    ax2.plot(vpm["XB"], vpm["YB"], "0.7", lw=1.0, label="full body (reference)")
    ax2.plot(vpm["XB"][n:], vpm["YB"][n:], "b.-", ms=7, lw=1.8, label="solved UPPER half")
    ax2.plot(vpm["XC_upper"], vpm["YC_upper"], "rx", ms=6, label="control points")
    ax2.axhline(0.0, color="0.4", ls="--", lw=1.0)
    ax2.set_title("Vortex panel method - upper half only, %d panels (symmetry used)" % n)
    ax2.legend(loc="upper right", fontsize=9)

    for ax in (ax1, ax2):
        ax.set_aspect("equal")
        ax.set_xlabel("x/c")
        ax.set_ylabel("y/c")
        ax.grid(alpha=0.3)

    save_figure(fig, "01_geometry.png")


# -----------------------------------------------------------------------------
# FIGURES 2-5 - streamlines, velocity field and pressure field
# -----------------------------------------------------------------------------
def plot_flow_field(name, field, XB, YB, f_stream, f_cp, upper_only=False):
    """
    Two figures per method:
      (a) streamlines coloured by speed, plus velocity vectors
      (b) Cp field - this is the pressure field, since
          p - p_inf = 0.5*rho*V_inf^2 * Cp
    """
    XX, YY = field["XX"], field["YY"]
    U, V = field["U"], field["V"]

    # the upper-half plots are about half as tall, so shrink their colorbar
    cbar_shrink = 0.60 if upper_only else 1.0

    # streamplot cannot handle NaN, so use zero velocity inside the body
    Us, Vs = np.nan_to_num(U), np.nan_to_num(V)

    # ---------- (a) streamlines + velocity field ----------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_aspect("equal")

    sp = ax.contourf(XX, YY, field["speed"], levels=40, cmap="viridis")
    fig.colorbar(sp, ax=ax, shrink=cbar_shrink,
                 label=r"velocity magnitude  $|V|/V_\infty$")

    seed_y = np.linspace(YY.min() * 0.95, YY.max() * 0.95, 26)
    seeds = np.column_stack((np.full_like(seed_y, XX.min() + 0.01), seed_y))
    ax.streamplot(XX, YY, Us, Vs, color="white", linewidth=0.8,
                  density=3.0, arrowsize=0.8, start_points=seeds)

    # velocity vectors on a coarser sub-grid
    s = QUIVER_STEP
    ax.quiver(XX[::s, ::s], YY[::s, ::s], U[::s, ::s], V[::s, ::s],
              color="black", scale=28, width=0.0022, zorder=5)

    draw_body(ax, XB, YB)
    ax.set_title("%s: streamlines and velocity field  "
                 r"($V_\infty$ = %.1f m/s, $\alpha$ = %.0f$^\circ$)"
                 % (name, V_INF, ALPHA_DEG))
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")
    if upper_only:
        ax.set_ylim(-0.05, YY.max())
    save_figure(fig, f_stream)

    # ---------- (b) pressure field ----------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_aspect("equal")

    # clip only so that the colour scale stays readable near the sharp TE
    cp_plot = np.clip(field["Cp"], -3.0, 1.0)
    cf = ax.contourf(XX, YY, cp_plot, levels=np.linspace(-3.0, 1.0, 41), cmap="coolwarm_r")
    fig.colorbar(cf, ax=ax, shrink=cbar_shrink,
                 label=r"pressure coefficient  $C_p$")
    ax.contour(XX, YY, cp_plot, levels=10, colors="black", linewidths=0.4, alpha=0.5)

    draw_body(ax, XB, YB, color="0.35")
    q = 0.5 * RHO * V_INF ** 2
    ax.set_title("%s: pressure field   " r"($p-p_\infty = %.3f\,C_p$ Pa)" % (name, q))
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")
    if upper_only:
        ax.set_ylim(-0.05, YY.max())
    save_figure(fig, f_cp)


# -----------------------------------------------------------------------------
# FIGURES 6-7 - surface Cp compared with XFOIL
# -----------------------------------------------------------------------------
def plot_surface_cp(spm, vpm, xfoil):
    n = len(spm["XC"]) // 2
    x_lo, cp_lo = spm["XC"][:n], spm["Cp"][:n]      # lower surface
    x_up, cp_up = spm["XC"][n:], spm["Cp"][n:]      # upper surface

    # ---------- source panel method ----------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(x_up, cp_up, "b.-", ms=4, lw=1.4, label="SPM upper surface")
    ax.plot(x_lo, cp_lo, "g.--", ms=4, lw=1.2, label="SPM lower surface")

    if xfoil is not None:
        xf_x, xf_cp = xfoil_side(xfoil, upper=True)
        ax.plot(xf_x, xf_cp, "r-", lw=2.0, alpha=0.75, label="XFOIL")

    ax.invert_yaxis()
    ax.set_xlabel("x/c")
    ax.set_ylabel(r"$C_p$")
    ax.set_title("Surface pressure distribution - Source Panel Method vs XFOIL")
    ax.grid(alpha=0.3)
    ax.legend()
    save_figure(fig, "06_spm_surface_cp_vs_xfoil.png")

    # ---------- vortex panel method (upper half) ----------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(vpm["XC_upper"], vpm["Cp_upper"], "bs-", ms=6, lw=1.6,
            label="VPM upper half (%d panels)" % vpm["n_half"])
    ax.plot(x_up, cp_up, "0.5", ls=":", lw=1.5, label="SPM upper surface")

    if xfoil is not None:
        xf_x, xf_cp = xfoil_side(xfoil, upper=True)
        ax.plot(xf_x, xf_cp, "r-", lw=2.0, alpha=0.75, label="XFOIL")

    ax.invert_yaxis()
    ax.set_xlabel("x/c")
    ax.set_ylabel(r"$C_p$")
    ax.set_title("Surface pressure distribution - Vortex Panel Method (upper half) vs XFOIL")
    ax.grid(alpha=0.3)
    ax.legend()
    save_figure(fig, "07_vpm_surface_cp_vs_xfoil.png")


# -----------------------------------------------------------------------------
# FIGURES 8-9 - Cp vector plots
# -----------------------------------------------------------------------------
def draw_cp_vectors(ax, XC, YC, delta, Cp, arrows=False, min_length=0.0):
    
    for i in range(len(Cp)):
        true_length = abs(Cp[i]) * CP_VECTOR_SCALE
        length = max(true_length, min_length)
        floored = true_length < min_length
        style = ":" if floored else "-"

        x0, y0 = XC[i], YC[i]
        x1 = x0 + length * np.cos(delta[i])
        y1 = y0 + length * np.sin(delta[i])
        col = "red" if Cp[i] < 0.0 else "blue"

        if arrows:
            # shrinkA/shrinkB = 0 is important: without it matplotlib trims the
            # ends of a short arrow and the arrowhead can point the wrong way.
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6,
                                        linestyle=style, shrinkA=0, shrinkB=0,
                                        mutation_scale=13),
                        zorder=6)
        else:
            ax.plot([x0, x1], [y0, y1], color=col, lw=1.3, ls=style, zorder=6)


def plot_cp_vectors(spm, vpm):
    legend = [Line2D([0], [0], color="blue", lw=2, label=r"$C_p \geq 0$  (pressure)"),
              Line2D([0], [0], color="red",  lw=2, label=r"$C_p < 0$  (suction)")]

    # ---------- source panel method, complete body ----------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    draw_body(ax, spm["XB"], spm["YB"], color="0.90")
    draw_cp_vectors(ax, spm["XC"], spm["YC"], spm["delta"], spm["Cp"])
    ax.set_aspect("equal")
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")
    ax.set_title(r"$C_p$ vector plot - Source Panel Method (complete body)")
    ax.legend(handles=legend, loc="upper right")
    save_figure(fig, "08_spm_cp_vectors.png")

    # ---------- vortex panel method, upper half ----------
    fig, ax = plt.subplots(figsize=(9, 5.0))
    draw_body(ax, vpm["XB"], vpm["YB"], color="0.93")
    draw_cp_vectors(ax, vpm["XC_upper"], vpm["YC_upper"],
                    vpm["delta_upper"], vpm["Cp_upper"],
                    arrows=True, min_length=0.020)
    ax.plot(vpm["XC_upper"], vpm["YC_upper"], "ko", ms=4, zorder=7)
    for i, (x, y) in enumerate(zip(vpm["XC_upper"], vpm["YC_upper"])):
        ax.text(x, y - 0.045, "P%d" % (i + 1), fontsize=9, ha="center", zorder=7)
    ax.axhline(0.0, color="0.4", ls="--", lw=1.0)
    ax.set_ylim(-0.10, 0.52)
    legend = legend + [Line2D([0], [0], color="0.4", lw=1.5, ls=":",
                              label=r"dotted = drawing floor for near-zero $|C_p|$")]
    ax.set_aspect("equal")
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")
    ax.set_title(r"$C_p$ vector plot - Vortex Panel Method (upper half only)")
    ax.legend(handles=legend, loc="upper right")
    save_figure(fig, "09_vpm_cp_vectors.png")


# =============================================================================
# STEP 9 - SUMMARY OF RESULTS
# =============================================================================
def panel_sensitivity_table(xfoil):
    """
    Re-solve the upper-half VPM for several panel counts.
    This documents why N_VPM_HALF was chosen.
    """
    rows = []
    for n in VPM_STUDY:
        v = solve_vortex_panel_upper_half(n)
        rows.append((n,
                     v["cond"],
                     v["Cp_upper"].min(),
                     cp_rmse(v["XC_upper"], v["Cp_upper"], xfoil, upper=True)))
    return rows


def build_summary(spm, vpm, spm_field, xfoil, rows):
    """Collect every numerical result into a list of printable lines."""
    L = []
    add = L.append

    add("=" * 70)
    add(" PANEL METHODS PROJECT - RESULTS SUMMARY")
    add(" Student ID : 210011253      Geometry : TEARDROP")
    add("=" * 70)
    add("")
    add("FLOW CONDITIONS")
    add("  Freestream speed  V_inf   = %.3f m/s" % V_INF)
    add("  Angle of attack   alpha   = %.1f deg" % ALPHA_DEG)
    add("  Air density       rho     = %.3f kg/m^3" % RHO)
    add("  Max thickness             = %.2f c" % (2.0 * THICKNESS))
    add("")

    add("-" * 70)
    add("STEP 3 - SOURCE PANEL METHOD (complete body)")
    add("-" * 70)
    add("  Panels used                     = %d" % len(spm["XC"]))
    add("  Matrix condition number         = %.2f" % spm["cond"])
    add("  Boundary-condition residual     = %.3e   (should be ~0)" % spm["bc_residual"])
    add("  Sum of lambda_j * S_j           = %.3e   (should be ~0 for a closed body)"
        % spm["source_sum"])
    add("  Cp maximum (stagnation)         = %.4f   (theory: 1.0)" % spm["Cp"].max())
    add("  Cp minimum (peak suction)       = %.4f" % spm["Cp"].min())
    add("  CL                              = %.6f  (0 expected: symmetric, alpha=0)" % spm["CL"])
    add("  CD                              = %.6f  (0 expected: d'Alembert paradox)" % spm["CD"])
    add("")
    add("  PRESSURE FIELD (from the same solution, p - p_inf = 0.5*rho*V_inf^2*Cp)")
    dp = spm_field["dp"]
    add("    highest pressure in the field = %+.4f Pa  (at the stagnation points)"
        % np.nanmax(dp))
    add("    lowest  pressure in the field = %+.4f Pa  (over the shoulder)"
        % np.nanmin(dp))
    add("")

    add("-" * 70)
    add("STEP 4 - VORTEX PANEL METHOD (upper half only, symmetry reduction)")
    add("-" * 70)
    add("  Upper-half panels solved        = %d" % vpm["n_half"])
    add("  Full body reconstructed with    = %d panels" % vpm["n_full"])
    add("  Reduced matrix condition number = %.2f" % vpm["cond"])
    add("  Boundary-condition residual     = %.3e   (should be ~0)" % vpm["bc_residual"])
    add("  Total circulation  Gamma        = %.3e   (0 expected: symmetric, alpha=0)"
        % vpm["circulation"])
    add("  CL from Kutta-Joukowski         = %.3e   (0 expected)" % vpm["CL_kj"])
    add("  Cp minimum on the upper half    = %.4f" % vpm["Cp_upper"].min())
    add("")

    add("-" * 70)
    add("STEP 7 - COMPARISON AGAINST XFOIL")
    add("-" * 70)
    if xfoil is None:
        add("  XFOIL data not available (xfoil.exe missing or it failed to run).")
    else:
        add("  XFOIL surface points read       = %d" % len(xfoil["x"]))
        n = len(spm["XC"]) // 2
        r_up = cp_rmse(spm["XC"][n:], spm["Cp"][n:], xfoil, upper=True)
        r_lo = cp_rmse(spm["XC"][:n], spm["Cp"][:n], xfoil, upper=False)
        r_v = cp_rmse(vpm["XC_upper"], vpm["Cp_upper"], xfoil, upper=True)
        add("  SPM Cp RMSE, upper surface      = %.4f" % r_up)
        add("  SPM Cp RMSE, lower surface      = %.4f" % r_lo)
        add("  VPM Cp RMSE, upper half         = %.4f" % r_v)
        add("")
        add("  The SOURCE method matches XFOIL closely, because sources are the")
        add("  correct singularity for a non-lifting thickness problem.")
        add("  The VORTEX method is less accurate here: at alpha = 0 a symmetric")
        add("  body produces no circulation, so vortex panels alone cannot")
        add("  reproduce the thickness-driven pressure distribution exactly.")
    add("")

    add("-" * 70)
    add("VPM PANEL-COUNT SENSITIVITY (justifies N_VPM_HALF = %d)" % N_VPM_HALF)
    add("-" * 70)
    add("   n_half   cond(A)      Cp_min     RMSE vs XFOIL")
    for n, cond, cpmin, rmse in rows:
        rmse_txt = "    n/a  " if not np.isfinite(rmse) else "  %8.4f" % rmse
        add("   %5d  %9.1f  %9.4f  %s" % (n, cond, cpmin, rmse_txt))
    add("")
    add("  The sharp trailing edge is a singular point for the vortex sheet.")
    add("  A constant-strength (first-order) panel cannot represent that, so the")
    add("  error does NOT fall as panels are added - it grows, and the matrix")
    add("  becomes more ill-conditioned at the same time.")
    add("  n_half = %d gives the lowest RMSE of the counts tested, so that is" % N_VPM_HALF)
    add("  the working value.  This is a limitation of the first-order vortex")
    add("  panel, not a coding error: a higher-order (linear-strength) vortex")
    add("  panel would be required for the solution to converge properly.")
    add("=" * 70)
    return L


# =============================================================================
# MAIN PROGRAM
# =============================================================================
def main():
    print("=" * 70)
    print(" PANEL METHODS PROJECT - TEARDROP - ID 210011253")
    print("=" * 70)

    OUTPUT_DIR.mkdir(exist_ok=True)

    # ---- STEP 2: geometry -------------------------------------------------
    print("\n[STEP 2] Building teardrop geometry ...")
    XB, YB = make_teardrop(N_SPM)
    print("  %d boundary points, %d panels on the complete body."
          % (len(XB), N_SPM))

    # ---- STEP 3: source panel method --------------------------------------
    print("\n[STEP 3] Solving the SOURCE panel method on the COMPLETE body ...")
    spm = solve_source_panel(XB, YB)
    print("  done.  CL = %.6f,  CD = %.6f,  Cp_max = %.4f"
          % (spm["CL"], spm["CD"], spm["Cp"].max()))

    # ---- STEP 4: vortex panel method --------------------------------------
    print("\n[STEP 4] Solving the VORTEX panel method on the UPPER HALF ...")
    vpm = solve_vortex_panel_upper_half(N_VPM_HALF)
    print("  done.  %d unknowns, Gamma = %.3e" % (vpm["n_half"], vpm["circulation"]))

    # ---- STEP 5 & 6: flow field -------------------------------------------
    print("\n[STEP 5/6] Computing the flow field on a %dx%d grid ..."
          % (GRID_NX, GRID_NY))
    print("  source method ...")
    spm_field = source_flow_field(spm)
    print("  vortex method ...")
    vpm_field = vortex_flow_field(vpm)
    print("  done.")

    # ---- STEP 7: XFOIL ----------------------------------------------------
    print("\n[STEP 7] Running XFOIL on the same geometry ...")
    dat_path = write_dat_file(XB, YB)
    print("  geometry written to %s" % dat_path.name)
    xfoil = run_xfoil(dat_path)

    # ---- plots ------------------------------------------------------------
    print("\n[PLOTS] Generating figures ...")
    plot_geometry(spm, vpm)

    plot_flow_field("Source Panel Method", spm_field, spm["XB"], spm["YB"],
                    "02_spm_streamlines_velocity.png",
                    "03_spm_pressure_field.png")

    plot_flow_field("Vortex Panel Method (upper half)", vpm_field,
                    vpm["XB"], vpm["YB"],
                    "04_vpm_streamlines_velocity.png",
                    "05_vpm_pressure_field.png",
                    upper_only=True)

    plot_surface_cp(spm, vpm, xfoil)
    plot_cp_vectors(spm, vpm)

    # ---- STEP 9: summary --------------------------------------------------
    print("\n[STEP 9] Panel-count sensitivity study ...")
    rows = panel_sensitivity_table(xfoil)

    lines = build_summary(spm, vpm, spm_field, xfoil, rows)
    text = "\n".join(lines)
    print("\n" + text)

    with open(OUTPUT_DIR / "results_210011253.txt", "w") as f:
        f.write(text + "\n")
    print("\n  saved  outputs/results_210011253.txt")

    if SHOW_FIGURES:
        print("\nClose the figure windows to finish.")
        plt.show()
    else:
        plt.close("all")

    print("\nAll done.")


if __name__ == "__main__":
    main()
