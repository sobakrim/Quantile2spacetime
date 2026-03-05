# Quantile2SpaceTime
ML quantiles → latent Gaussian fields → coherent spatio-temporal simulation

[![DOI](https://joss.theoj.org/papers/10.21105/joss.09605/status.svg)](https://doi.org/10.21105/joss.09605)

A research codebase for **modeling and simulating spatio-temporal processes** by combining  
**machine-learning quantile regression** with **latent Gaussian random fields (GRFs)**.

---

## Paper

- **JOSS (recommended citation)**: https://doi.org/10.21105/joss.09605  
- **HAL preprint**: https://hal.science/hal-05441043/

---

## Overview

**Quantile2SpaceTime** implements a coherent two-stage framework:

1. **Conditional marginals (data space)**  
   Learn the conditional distribution \(Y \mid X\) using quantile regression:
   - **KNN CDF**
   - **Quantile Regression Forests (QRF)**
   - **Quantile Regression Neural Networks (QRNN)**

2. **Dependence (latent space)**  
   Map observations to a latent Gaussian space and model dependence with a GRF:
   - Transform: \(U = F_{Y|X}(y)\), then \(Z = \Phi^{-1}(U)\)
   - Fit a spatio-temporal GRF for \(Z(s,t)\) (e.g., **Matérn–Gneiting**)
   - Simulate \(Z\) and invert back to \(Y\)

This yields simulations that preserve:
- **non-Gaussian, covariate-dependent marginals**, and
- **spatio-temporal dependence** through the latent GRF.
