"""Run the MECH559 HW1 optimization and generate figures/results."""

from __future__ import annotations
import json
import argparse
from pathlib import Path
from functions import (CONSTRAINT_NAMES, Parameters, evaluate_grid, make_grid,
                       parameter_sweep, plot_contour, plot_design_space,
                       plot_k_sweep_design_space, plot_sweeps, result_as_dict,
                       solve_optimization)

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"


def main(plot_initial_seed: bool = False, show_titles: bool = False) -> None:
    FIGURES.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    parameters = Parameters()
    initial_seed = (0.05, 0.10)
    bounds = ((0.001, 0.20), (0.001, 0.40))
    result = solve_optimization(parameters, x0=initial_seed, bounds=bounds)
    R, L = make_grid(bounds, resolution=300)
    values = evaluate_grid(R, L, parameters)
    optimum = result.x
    plot_contour(R, L, values["objective"], values["feasible"], initial_seed, optimum,
                 "Objective: combined arm mass", "Arm mass [kg]", FIGURES / "objective_contour.png",
                 plot_initial_seed=plot_initial_seed, show_title=show_titles)
    labels = {
        "stress": ("Constraint 1: normalized bending stress", "sigma/sigma_allow - 1"),
        "deflection": ("Constraint 2: normalized tip deflection", "delta/delta_allow - 1"),
        "thrust_margin": ("Constraint 3: thrust-margin requirement", "k m g/(4T) - 1"),
        "clearance": ("Constraint 4: rotor-clearance requirement", "R/(L sin(phi/2)) - 1"),
    }
    for name in CONSTRAINT_NAMES:
        title, colorbar = labels[name]
        plot_contour(R, L, values[name], values["feasible"], initial_seed, optimum,
                     title, colorbar, FIGURES / f"constraint_{name}_contour.png",
                     constraint=True, plot_initial_seed=plot_initial_seed,
                     show_title=show_titles)
    plot_design_space(R, L, values, initial_seed, optimum,
                      FIGURES / "constrained_design_space.png",
                      plot_initial_seed=plot_initial_seed, show_title=show_titles)
    data = result_as_dict(result, parameters, initial_seed)
    sweeps = parameter_sweep(
        parameters, x0=initial_seed, bounds=bounds,
        motor_powers=(700.0, 1000.0, 1300.0),
        thrust_margins=tuple(float(k) for k in range(1, 11)),
    )
    plot_sweeps(sweeps, FIGURES / "parameter_sweeps.png", show_title=show_titles)
    plot_k_sweep_design_space(
        R, L, parameters, sweeps["thrust_margin_sweep"],
        FIGURES / "k_sweep_design_space.png", show_title=show_titles,
    )
    plot_k_sweep_design_space(
        R, L, parameters, sweeps["thrust_margin_sweep"],
        FIGURES / "k_sweep_design_space_all_constraints.png",
        show_title=show_titles, active_only=False,
    )
    data["parameter_sweeps"] = sweeps
    with (RESULTS / "optimization_results.json").open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the MECH559 HW1 optimization.")
    parser.add_argument("--plot-initial-seed", action="store_true",
                        help="include the initial seed marker in the design-space plots")
    parser.add_argument("--plot-titles", action="store_true",
                        help="include titles on generated plots")
    args = parser.parse_args()
    main(plot_initial_seed=args.plot_initial_seed, show_titles=args.plot_titles)
