# Singular Spectrum Analysis (SSA) in MATLAB

A step-by-step MATLAB tutorial demonstrating Singular Spectrum Analysis for time series decomposition and forecasting.

## What it does

The script walks through the four classic SSA stages — embedding, SVD decomposition, grouping, and diagonal averaging — then adds two extras:

- **Automatic rank selection** via the Gavish & Donoho (2014) optimal hard threshold, replacing manual elbow-picking
- **Cross-validation** against the covariance-matrix formulation of SSA (Vautard & Ghil, 1989; Groth & Ghil, 2015)

It also includes **Recurrent SSA forecasting** to extend the reconstructed signal out-of-sample.

## Usage

Run `tutorial.m` in MATLAB. The script generates a synthetic time series (trend + seasonal + noise) and produces five figures:

1. Original time series
2. Singular value spectrum with Gavish-Donoho threshold
3. SSA reconstruction vs. original
4. Out-of-sample forecast
5. Eigenvalue spectrum from the covariance-matrix formulation

To use your own data, replace the signal-generation block near the top with `y = yourDataVector;`.

## References

1. Gavish, M. & Donoho, D.L. (2014). The optimal hard threshold for singular values is 4/√3. [arXiv:1305.5870](https://arxiv.org/pdf/1305.5870)
2. Vautard, R. & Ghil, M. (1989). Singular spectrum analysis in nonlinear dynamics.
3. Groth, A. & Ghil, M. (2015). Multivariate singular spectrum analysis.
4. Rodrigues, P.C. — [Hybrid Strategies for Time Series Forecasting (seminar)](https://www.youtube.com/watch?v=muWEfZTrVrE&t=2264s)
5. [SSA Beginner's Guide — MATLAB File Exchange](https://www.mathworks.com/matlabcentral/fileexchange/58967-singular-spectrum-analysis-beginners-guide)
