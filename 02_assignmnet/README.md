# MECH559 Assignment 2: Data-Based Surrogate Modeling

This assignment compares surrogate models for the lift coefficient $C_L$ of
NACA 4-digit airfoils. The six inputs are maximum camber, camber position,
thickness, angle of attack, Mach number, and $\log_{10}(\mathrm{Re})$.

The completed analysis is in
[`LucasBessai_HW2.ipynb`](LucasBessai_HW2.ipynb), and the report source is in
[`LucasBessai_hw2_report.tex`](LucasBessai_hw2_report.tex).

## Repository layout

```text
02_assignmnet/
+-- HW2.pdf                           Assignment statement
+-- aero_naca4_airfoil_panel.csv      Airfoil dataset and train/test split
+-- LucasBessai_HW2.ipynb             Completed analysis notebook
+-- LucasBessai_hw2_report.tex        Report source
+-- LucasBessai_hw2_report.pdf        Compiled report
+-- figures/                          Report figures exported from the notebook
```

## Models

The notebook evaluates Latin hypercube sampling, affine and quadratic
regression with ridge regularization, Gaussian RBF interpolation, a
single-hidden-layer neural network, and Kriging/Gaussian-process regression.
Model settings are chosen by five-fold CV on the training rows and evaluated
once on the supplied test rows.

## Running the notebook

Use the repository virtual environment and select its `optimization_venv`
kernel in Jupyter:

```powershell
cd ..
.\optimization_venv\Scripts\Activate.ps1
cd 02_assignmnet
jupyter notebook LucasBessai_HW2.ipynb
```

## Current result

The Gaussian process gave the lowest held-out RMSE, $0.000038$. The Gaussian
RBF was also highly accurate, with test RMSE $0.000273$. The GP additionally
provides model-based predictive standard deviations.
