# Literature

Reference papers for the theoretical foundations and inference methods used in this project.

## Dynamic Structural Equation Models

**Asparouhov, Hamaker & Muthen (2017)**
*Structural Equation Modeling: A Multidisciplinary Journal*

Introduces Dynamic SEM (DSEM), extending traditional SEM to intensive longitudinal data. Combines multilevel modeling with time-series analysis, allowing latent variables to evolve over time. This is the primary reference model our pipeline aims to match and extend.

- DOI: [10.1080/10705511.2017.1406803](https://doi.org/10.1080/10705511.2017.1406803)

## Causal Identification in Time Series Models

**Jahn, Karnik & Schulman (2025)**
*Proceedings of Machine Learning Research 275:1-15 (CLeaR 2025)*

Analyzes applicability of the Causal Identification algorithm to causal time series graphs with latent confounders. Shows that applying the ID algorithm to a constant-size segment of the time series graph is sufficient to decide identifiability of causal effects, even across unbounded time intervals. Provides bounds depending only on the number of variables per timestep and the maximum time lag.

- arXiv: [2504.20172](https://arxiv.org/abs/2504.20172)

## Particle Gibbs with Ancestor Sampling (PGAS)

**Lindsten, Jordan & Schon (2014)**

Presents a novel Particle MCMC algorithm combining SMC and MCMC. The ancestor sampling procedure enables fast mixing of the PGAS kernel even with few particles, making it well suited for inference in state-space models and models with complex dependencies (non-Markovian, Bayesian nonparametric, general probabilistic graphical models).

- arXiv: [1401.0604](https://arxiv.org/abs/1401.0604)

## Hess-MC^2: Sequential Monte Carlo Squared using Hessian Information

**Murphy, Rosato, Millard, Devlin, Horridge & Maskell (2025)**

Extends SMC^2 with second-order (Hessian) proposals for parameter estimation in state-space models. Uses automatic differentiation to compute curvature-aware proposals that improve concentration and diversity of posterior distributions, especially in non-linear non-Gaussian settings. Relevant to our inference backend selection.

- arXiv: [2507.07461](https://arxiv.org/abs/2507.07461)

## Automated Learning with a Probabilistic Programming Language: Birch

**Murray & Schon (2020)**

Broad perspective on probabilistic modeling and inference via probabilistic programming languages. Focuses on how the *structure* (conditional dependencies) and *form* (mathematical specification) of a model can be revealed by PPLs to automatically match models with appropriate inference methods. Demonstrates with the Birch PPL on a multiple object tracking example.

- arXiv: [1810.01539](https://arxiv.org/abs/1810.01539)
