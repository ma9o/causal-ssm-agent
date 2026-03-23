# Literature

Consolidated bibliography of papers referenced across the documentation. Grouped by topic; each entry notes which doc(s) use it.

## Dynamic Structural Equation Models

**Asparouhov, Hamaker & Muthen (2018).** Dynamic Structural Equation Models. *Structural Equation Modeling*, 25(3), 359-388. DOI: [10.1080/10705511.2017.1406803](https://doi.org/10.1080/10705511.2017.1406803). Primary reference model our pipeline extends. *Used in: assumptions.md (A4, A8), estimation.md.*

**Driver & Voelkle (2018).** Hierarchical Bayesian Continuous Time Dynamic Modeling. *Psychological Methods*. Reference for CT-SDE parameterization and hierarchical panel data. *Used in: estimation.md.*

## Causal Identification

**Jahn, Karnik & Schulman (2025).** Causal Identification in Time Series Models. *PMLR 275:1-15 (CLeaR 2025)*. arXiv: [2504.20172](https://arxiv.org/abs/2504.20172). Proves a constant-size time segment suffices for ID in causal time series graphs. Foundation for A3a (2-timestep unrolling). *Used in: assumptions.md (A3a).*

**Shpitser & Pearl (2006).** Identification of Joint Interventional Distributions in Recursive Semi-Markovian Causal Models. *AAAI*. The ID algorithm applied internally via y0. *Used in: scope.md, assumptions.md (A4).*

**Miao, Geng & Tchetgen Tchetgen (2018).** Identifying causal effects with proxy variables of an unmeasured confounder. *Biometrika*, 105(4), 987-993. *Used in: assumptions.md (A7).*

## Measurement Theory

**Diamantopoulos & Siguaw (2006).** Formative versus reflective indicators in organizational measure development. *British Journal of Management*. Justifies reflective-only constraint. *Used in: assumptions.md (A1).*

**Anderson & Gerbing (1988).** Structural equation modeling in practice: A review and recommended two-step approach. *Psychological Bulletin*, 103(3), 411-423. *Used in: assumptions.md (A7), functional-specification.md.*

**Bollen (1989).** *Structural Equations with Latent Variables*. Wiley. Chapter 7: single-indicator identification. *Used in: assumptions.md (A9).*

## Bayesian Inference Methods

**Lindsten, Jordan & Schon (2014).** Particle Gibbs with Ancestor Sampling. *JMLR*. arXiv: [1401.0604](https://arxiv.org/abs/1401.0604). PGAS kernel for SSM inference. *Used in: inference-routing.md, benchmarks/results.md.*

**Murphy, Rosato, Millard, Devlin, Horridge & Maskell (2025).** Hess-MC^2: Sequential Monte Carlo Squared using Hessian Information. arXiv: [2507.07461](https://arxiv.org/abs/2507.07461). Curvature-aware SMC proposals. *Used in: inference-routing.md.*

**Hoffman & Gelman (2014).** The No-U-Turn Sampler. *JMLR*. NUTS algorithm. *Used in: inference-routing.md.*

**Andrieu, Doucet & Holenstein (2010).** Particle Markov Chain Monte Carlo Methods. *JRSS-B*. Foundational PMCMC framework. *Used in: inference-routing.md.*

**Dau & Chopin (2022).** Waste-Free Sequential Monte Carlo. *JRSS-B*. Adaptive tempering with recycling. *Used in: inference-routing.md.*

**Tanner & Wong (1987).** The Calculation of Posterior Distributions by Data Augmentation. *JASA*. Data augmentation MCMC. *Used in: inference-routing.md.*

**Zhang, Carpenter, Gelman & Vehtari (2022).** Pathfinder: Parallel Quasi-Newton Variational Inference. *JMLR*. *Used in: inference-routing.md.*

**Sarkka (2013).** *Bayesian Filtering and Smoothing*. Cambridge University Press. Reference for Kalman filtering and state-space models. *Used in: estimation.md, inference-routing.md.*

## Bayesian Workflow & Model Validation

**Gelman et al. (2020).** Bayesian Workflow. arXiv: [2011.01808](https://arxiv.org/abs/2011.01808). *Used in: functional-specification.md.*

**Betancourt (2018).** Towards a Principled Bayesian Workflow. *Used in: functional-specification.md.*

**Gabry et al. (2019).** Visualization in Bayesian Workflow. *JRSS-A*, 182(2), 389-402. *Background for: functional-specification.md (model validation).*

## LLM-Assisted Prior Elicitation

**Capstick et al. (2024).** AutoElicit: Using Large Language Models for Expert Prior Elicitation in Predictive Modelling. arXiv: [2411.17284](https://arxiv.org/abs/2411.17284). Paraphrased prompting + mixture aggregation. *Used in: functional-specification.md.*

**Chen et al. (2025).** LLM-BI: Towards Fully Automated Bayesian Inference with Large Language Models. arXiv: [2508.08300](https://arxiv.org/abs/2508.08300). Full model specification from NL. *Used in: functional-specification.md.*

**Huang (2025).** LLM-Prior: A Framework for Knowledge-Driven Prior Elicitation and Aggregation. arXiv: [2508.03766](https://arxiv.org/abs/2508.03766). *Used in: functional-specification.md.*

**Riegler et al. (2025).** Using large language models to suggest informative prior distributions in Bayesian regression analysis. *Scientific Reports*. DOI: [10.1038/s41598-025-18425-9](https://www.nature.com/articles/s41598-025-18425-9). *Background for: functional-specification.md (LLM prior elicitation).*

**Selby et al. (2024).** Had Enough of Experts? Elicitation and Evaluation of Bayesian Priors from Large Language Models. *NeurIPS BDU Workshop*. *Background for: functional-specification.md (LLM prior elicitation).*

## Probabilistic Programming

**Murray & Schon (2020).** Automated Learning with a Probabilistic Programming Language: Birch. arXiv: [1810.01539](https://arxiv.org/abs/1810.01539). Structure-aware inference routing in PPLs. *Background for: inference-routing.md (design motivation).*

## Parametric Identifiability

**Raue et al. (2009).** Structural and practical identifiability analysis of partially observed dynamical models by exploiting the profile likelihood. *Bioinformatics*. Profile likelihood diagnostics. *Used in: functional-specification.md (Stage 4b).*
