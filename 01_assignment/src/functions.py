"""Backend model, optimization, and plotting utilities for MECH559 HW1."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import OptimizeResult, minimize


@dataclass(frozen=True)
class Parameters:
    air_density: float = 1.225
    motor_power: float = 1000.0
    power_efficiency: float = 0.80
    arm_radius: float = 0.005
    arm_density: float = 1600.0
    youngs_modulus: float = 70.0e9
    allowable_stress: float = 250.0e6
    allowable_deflection: float = 0.005
    fixed_mass: float = 1.0
    gravity: float = 9.81
    thrust_margin: float = 1.5
    clearance_angle: float = np.pi / 2.0


CONSTRAINT_NAMES = ("stress", "deflection", "thrust_margin", "clearance")


def rotor_thrust(R, p: Parameters):
    """Maximum thrust of one rotor from actuator-disk theory."""
    induced_power = p.power_efficiency * p.motor_power
    return (induced_power * np.sqrt(2.0 * p.air_density * np.pi * np.asarray(R) ** 2)) ** (2.0 / 3.0)


def arm_mass(L, p: Parameters):
    """Combined mass of the four identical cylindrical arms."""
    return 4.0 * p.arm_density * np.pi * p.arm_radius**2 * np.asarray(L)


def total_mass(L, p: Parameters):
    return p.fixed_mass + arm_mass(L, p)


def maximum_stress(R, L, p: Parameters):
    I = np.pi * p.arm_radius**4 / 4.0
    return rotor_thrust(R, p) * np.asarray(L) * p.arm_radius / I


def tip_deflection(R, L, p: Parameters):
    I = np.pi * p.arm_radius**4 / 4.0
    return rotor_thrust(R, p) * np.asarray(L) ** 3 / (3.0 * p.youngs_modulus * I)


def constraint_values(x: Iterable[float], p: Parameters) -> np.ndarray:
    """Four normalized inequality functions; feasible means every value <= 0."""
    R, L = np.asarray(x, dtype=float)
    T = rotor_thrust(R, p)
    m = total_mass(L, p)
    clearance_limit = L * np.sin(p.clearance_angle / 2.0)
    return np.array([
        maximum_stress(R, L, p) / p.allowable_stress - 1.0,
        tip_deflection(R, L, p) / p.allowable_deflection - 1.0,
        p.thrust_margin * m * p.gravity / (4.0 * T) - 1.0,
        R / clearance_limit - 1.0,
    ])


def objective(x: Iterable[float], p: Parameters) -> float:
    return float(arm_mass(float(np.asarray(x)[1]), p))


def scipy_constraints(p: Parameters) -> list[dict]:
    return [{"type": "ineq", "fun": lambda x, i=i: -constraint_values(x, p)[i]} for i in range(4)]


def solve_optimization(p: Parameters, x0=(0.05, 0.10),
                       bounds=((1e-4, 0.50), (1e-4, 1.0))) -> OptimizeResult:
    result = minimize(objective, np.asarray(x0, dtype=float), args=(p,), method="SLSQP",
                      bounds=bounds, constraints=scipy_constraints(p),
                      options={"ftol": 1e-12, "maxiter": 1000, "disp": False})
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")
    return result


def make_grid(bounds, resolution=300):
    R = np.linspace(bounds[0][0], bounds[0][1], resolution)
    L = np.linspace(bounds[1][0], bounds[1][1], resolution)
    return np.meshgrid(R, L)


def evaluate_grid(R, L, p: Parameters):
    values = {
        "objective": arm_mass(L, p),
        "stress": maximum_stress(R, L, p) / p.allowable_stress - 1.0,
        "deflection": tip_deflection(R, L, p) / p.allowable_deflection - 1.0,
        "thrust_margin": p.thrust_margin * total_mass(L, p) * p.gravity / (4.0 * rotor_thrust(R, p)) - 1.0,
        "clearance": R / (L * np.sin(p.clearance_angle / 2.0)) - 1.0,
    }
    values["feasible"] = np.logical_and.reduce([values[name] <= 0.0 for name in CONSTRAINT_NAMES])
    return values


def plot_contour(R, L, Z, feasible, x0, xopt, title, colorbar_label, path: Path,
                 constraint=False, plot_initial_seed=False, show_title=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.0, 5.5), constrained_layout=True)
    contour = ax.contourf(R, L, Z, levels=24, cmap="viridis")
    fig.colorbar(contour, ax=ax, label=colorbar_label)
    if constraint:
        ax.contour(R, L, Z, levels=[0.0], colors="crimson", linewidths=2.0)
    if plot_initial_seed:
        ax.plot(*x0, "o", color="white", markeredgecolor="black", label="initial seed")
    ax.plot(*xopt, "*", color="red", markersize=12, markeredgecolor="black", label="optimum")
    ax.set(xlabel="Rotor radius, R [m]", ylabel="Arm length, L [m]")
    if show_title:
        ax.set_title(title)
    ax.legend(loc="best")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_design_space(R, L, values, x0, xopt, path: Path,
                      plot_initial_seed=False, show_title=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=(7.0, 5.5), constrained_layout=True)
    ax.contourf(R, L, values["feasible"], levels=[-0.5, 0.5, 1.5], colors=["white", "#b9e3c6"], alpha=0.9)
    constraint_colors = ("#d62728", "#9467bd", "#ff7f0e", "#17becf")
    for name, color in zip(CONSTRAINT_NAMES, constraint_colors):
        ax.contour(R, L, values[name], levels=[0.0], colors=color, linewidths=1.8)
    if plot_initial_seed:
        ax.plot(*x0, "o", color="black", markersize=7, label="initial seed")
    ax.plot(*xopt, "*", color="red", markersize=13, markeredgecolor="black", label="optimal solution")
    ax.set(xlabel="Rotor radius, R [m]", ylabel="Arm length, L [m]")
    if show_title:
        ax.set_title("Constrained design space")
    legend_handles = [
        Patch(facecolor="#b9e3c6", edgecolor="none", label="feasible region"),
        *[Line2D([0], [0], color=color, linewidth=1.8, label=f"{name} constraint")
          for name, color in zip(CONSTRAINT_NAMES, constraint_colors)],
    ]
    if plot_initial_seed:
        legend_handles.append(Line2D([0], [0], marker="o", color="black", linestyle="None",
                                     markersize=7, label="initial seed"))
    legend_handles.append(Line2D([0], [0], marker="*", color="red", markeredgecolor="black",
                                 linestyle="None", markersize=12, label="optimal solution"))
    ax.legend(handles=legend_handles, loc="best")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def result_as_dict(result: OptimizeResult, p: Parameters, x0) -> dict:
    x = np.asarray(result.x, dtype=float)
    c = constraint_values(x, p)
    return {
        "parameters": asdict(p),
        "initial_seed": {"R_m": float(x0[0]), "L_m": float(x0[1])},
        "success": bool(result.success), "message": str(result.message),
        "iterations": int(getattr(result, "nit", -1)),
        "function_evaluations": int(getattr(result, "nfev", -1)),
        "optimal_design": {"R_m": float(x[0]), "L_m": float(x[1])},
        "objective_arm_mass_kg": float(objective(x, p)),
        "total_mass_kg": float(total_mass(x[1], p)),
        "rotor_thrust_N": float(rotor_thrust(x[0], p)),
        "maximum_stress_Pa": float(maximum_stress(x[0], x[1], p)),
        "tip_deflection_m": float(tip_deflection(x[0], x[1], p)),
        "constraint_values_leq_zero": {name: float(value) for name, value in zip(CONSTRAINT_NAMES, c)},
        "feasible": bool(np.all(c <= 1e-8)),
    }


def parameter_sweep(p: Parameters, x0, bounds, motor_powers, thrust_margins) -> dict:
    """Solve the required parameter studies for part (d)."""
    power_results = []
    for power in motor_powers:
        pp = replace(p, motor_power=float(power))
        rr = solve_optimization(pp, x0=x0, bounds=bounds)
        power_results.append({"motor_power_W": float(power), "R_m": float(rr.x[0]),
                             "L_m": float(rr.x[1]), "arm_mass_kg": float(objective(rr.x, pp)),
                             "constraint_values": [float(v) for v in constraint_values(rr.x, pp)]})
    margin_results = []
    for margin in thrust_margins:
        pp = replace(p, thrust_margin=float(margin))
        rr = solve_optimization(pp, x0=x0, bounds=bounds)
        margin_results.append({"thrust_margin": float(margin), "R_m": float(rr.x[0]),
                               "L_m": float(rr.x[1]), "arm_mass_kg": float(objective(rr.x, pp)),
                               "constraint_values": [float(v) for v in constraint_values(rr.x, pp)]})
    return {"motor_power_sweep": power_results, "thrust_margin_sweep": margin_results}


def plot_sweeps(sweeps: dict, path: Path, show_title=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    power = sweeps["motor_power_sweep"]
    margin = sweeps["thrust_margin_sweep"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), constrained_layout=True)
    axes[0].plot([r["motor_power_W"] for r in power], [r["arm_mass_kg"] for r in power], "o-")
    axes[0].set(xlabel="Motor power P0 [W]", ylabel="Optimal arm mass [kg]")
    if show_title:
        axes[0].set_title("Effect of motor power")
    axes[1].plot([r["thrust_margin"] for r in margin], [r["arm_mass_kg"] for r in margin], "o-")
    axes[1].set(xlabel="Required thrust margin k", ylabel="Optimal arm mass [kg]")
    if show_title:
        axes[1].set_title("Effect of thrust margin")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_k_sweep_design_space(R, L, p: Parameters, margin_results, path: Path,
                              show_title=False, active_only=True):
    """Plot optimal (R, L) designs and constraint boundaries for each k."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.colors import Normalize

    margins = np.asarray([row["thrust_margin"] for row in margin_results], dtype=float)
    norm = Normalize(vmin=margins.min(), vmax=margins.max())
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(7.5, 5.8), constrained_layout=True)

    # Re-plot the boundaries for each k. When active_only is true, use the
    # saved normalized constraint values to retain only active constraints.
    for row in margin_results:
        pp = replace(p, thrust_margin=row["thrust_margin"])
        color = cmap(norm(row["thrust_margin"]))
        grid_values = evaluate_grid(R, L, pp)
        active = [abs(value) <= 1e-6 for value in row["constraint_values"]]
        for name, is_active in zip(CONSTRAINT_NAMES, active):
            if (not active_only) or is_active:
                ax.contour(R, L, grid_values[name], levels=[0.0], colors=[color],
                           linewidths=1.2, alpha=0.22,
                           linestyles={
                               "stress": ":",
                               "deflection": "-.",
                               "thrust_margin": "--",
                               "clearance": "-",
                           }[name])

    points = ax.scatter(
        [row["R_m"] for row in margin_results],
        [row["L_m"] for row in margin_results],
        c=margins, cmap=cmap, norm=norm, s=48, edgecolors="black", linewidths=0.5,
        zorder=3,
    )
    fig.colorbar(points, ax=ax, label="Thrust-margin factor, k")
    ax.set(xlabel="Rotor radius, R [m]", ylabel="Arm length, L [m]")
    if show_title:
        ax.set_title("Optimal designs for the thrust-margin sweep")
    legend_lines = [
        Line2D([0], [0], color="black", linewidth=1.2, linestyle=":", alpha=0.35,
               label="stress boundary"),
        Line2D([0], [0], color="black", linewidth=1.2, linestyle="-.", alpha=0.35,
               label="deflection boundary"),
        Line2D([0], [0], color="black", linewidth=1.2, linestyle="--", alpha=0.35,
               label="thrust-margin boundary"),
        Line2D([0], [0], color="black", linewidth=1.2, linestyle="-", alpha=0.35,
               label="clearance boundary"),
    ] if not active_only else [
        Line2D([0], [0], color="black", linewidth=1.2, linestyle="--", alpha=0.35,
               label="thrust-margin boundary"),
        Line2D([0], [0], color="black", linewidth=1.2, linestyle="-", alpha=0.35,
               label="clearance boundary"),
    ]
    ax.legend(handles=[
        *legend_lines,
        Line2D([0], [0], marker="o", color="white", markeredgecolor="black",
               linestyle="None", markersize=7, label="optimal solution"),
    ], loc="best")
    fig.savefig(path, dpi=220)
    plt.close(fig)
