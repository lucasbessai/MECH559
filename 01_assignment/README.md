# MECH559 Assignment 1: Simple Drone Design Optimization

This assignment formulates and solves a constrained design optimization problem
for a simple quadrotor. The design variables are the rotor radius (R) and arm
length (L). The Python implementation performs the optimization, parameter
studies, and figure generation. The report is in
[`LucasBessai_hw1_report.tex`](LucasBessai_hw1_report.tex).

## Repository layout

```text
01_assignment/
+-- HW1.pdf                         Assignment statement
+-- LucasBessai_hw1_report.tex      Report source
+-- src/
|   +-- functions.py                Model, constraints, optimization, plotting
|   +-- main.py                     Run script
+-- figures/                        Generated PNG figures
+-- results/                        Generated JSON results
```

## Model

The rotor thrust is calculated using actuator-disk theory:

$$
T(R)=\left(\eta P_0\sqrt{2\rho\pi R^2}\right)^{2/3}.
$$

The objective is the mass of the four arms,

$$
W(L)=4\rho_{\mathrm{mat}}\pi r^2L.
$$

The four inequality constraints are:

1. rotor clearance,
2. allowable bending stress,
3. allowable tip deflection, and
4. the required thrust-margin condition.

The optimizer uses the convention `g_i(R,L) <= 0`. All model parameters are
defined in the `Parameters` dataclass in
[`src/functions.py`](src/functions.py).

## Running the code

The repository uses a virtual environment located one level above this
assignment:

```powershell
cd ..
.\optimization_venv\Scripts\Activate.ps1
cd 01_assignment
python src\main.py
```

The script uses a non-interactive plotting backend and writes figures directly
to disk, so no graphical interface is required.

Optional plotting flags are available:

```powershell
python src\main.py --plot-titles
python src\main.py --plot-initial-seed
python src\main.py --plot-titles --plot-initial-seed
```

By default, plot titles and the initial-seed marker are hidden.

## Generated outputs

The script creates or updates:

- `results/optimization_results.json`: nominal optimum and parameter-sweep data;
- `figures/objective_contour.png`: objective contour;
- `figures/constraint_*.png`: individual constraint contours;
- `figures/constrained_design_space.png`: feasible region and constraint boundaries;
- `figures/parameter_sweeps.png`: motor-power and thrust-margin studies;
- `figures/k_sweep_design_space.png`: optimal designs and active constraints for `k = 1` to `10`;
- `figures/k_sweep_design_space_all_constraints.png`: the same sweep with all constraint boundaries.

## Current qualitative result

For the default parameter set, the optimum is governed by the thrust-margin and
clearance constraints. The stress and deflection constraints remain inactive
over the investigated `k = 1` to `10` range. Increasing motor power reduces
the required rotor size and arm mass, while increasing the thrust-margin factor
increases both.
