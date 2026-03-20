# Causal Inference Pipeline Report

**User ID**: `DEFAULT`
**Generated**: 2026-03-18

---

## Stage 0: Preprocess
> Parses raw data files and prepares them for downstream analysis.

- **Records**: 4,597
- **Columns**: 4
- **Date range**: Jan 1, 2024 - Jun 30, 2024

### Data Sample (first 10 rows)

| datetime | activity_type | content | location |
| --- | --- | --- | --- |
| 2024-01-01T08:12:00+00:00 | search | morning routine optimization |  |
| 2024-01-13T14:05:00+00:00 | search | how to stop procrastinating |  |
| 2024-01-25T09:30:00+00:00 | view | Deep work strategies (25 min) |  |
| 2024-02-06T11:45:00+00:00 | search | pomodoro technique review |  |
| 2024-02-18T16:20:00+00:00 | view | ADHD and procrastination (30 min) |  |
| 2024-03-02T08:00:00+00:00 | search | cognitive behavioral therapy procrastination |  |
| 2024-03-14T13:30:00+00:00 | view | task avoidance psychology (18 min) |  |
| 2024-03-26T10:15:00+00:00 | search | deadline anxiety coping strategies | 37.7749,-122.4194 |
| 2024-04-07T15:00:00+00:00 | search | focus music for studying |  |
| 2024-04-19T09:45:00+00:00 | search | time blocking method |  |

### Column Descriptions

| Column | Type | Description |
| --- | --- | --- |
| datetime | datetime | Timestamp of the activity |
| activity_type | string | Type of activity (search, view, visit, other) |
| content | string | Content or description of the activity |
| location | string | GPS coordinates or null |

---

## Stage 1a: Latent Model
> Proposes a latent causal model based on domain knowledge alone, specifying theoretical constructs and their causal relationships.

- **Outcome**: procrastination
- **Treatments**: stress, sleep_quality, digital_distraction, self_efficacy, workload

### Constructs

| Name | Description | Role | Temporal | Outcome |
| --- | --- | --- | --- | --- |
| procrastination | The tendency to delay or avoid tasks despite knowing it will have negative consequences | endogenous | time_varying | Yes |
| stress | Perceived psychological stress and pressure from responsibilities | endogenous | time_varying | No |
| sleep_quality | Quality and duration of sleep, affecting cognitive resources | endogenous | time_varying | No |
| digital_distraction | Time spent on non-productive digital media consumption | endogenous | time_varying | No |
| self_efficacy | Belief in one's ability to accomplish tasks and goals | endogenous | time_varying | No |
| workload | External task demands and deadlines | exogenous | time_varying | No |

### Causal Edges

| Cause | Effect | Lagged | Description |
| --- | --- | --- | --- |
| stress | procrastination | No | High stress triggers avoidance behavior |
| sleep_quality | stress | Yes | Poor sleep increases next-day stress |
| digital_distraction | procrastination | No | Media consumption displaces productive work |
| self_efficacy | procrastination | Yes | Low self-efficacy leads to task avoidance |
| workload | stress | No | Higher workload increases stress |
| procrastination | stress | Yes | Procrastination increases stress from accumulating tasks |
| stress | sleep_quality | Yes | Stress impairs sleep quality |
| procrastination | self_efficacy | Yes | Chronic procrastination erodes self-efficacy |

---

## Stage 1b: Measurement & Nonparametric Identification
> Maps latent constructs to observable indicators and verifies nonparametric identifiability via do-calculus.

### Non-Identifiable Treatments

- **sleep_quality**: confounded by unobserved_chronotype

### Identifiable Treatments

| Treatment | Method | Estimand |
| --- | --- | --- |
| stress | do_calculus | P(procrastination \| do(stress)) |
| digital_distraction | do_calculus | P(procrastination \| do(digital_distraction)) |
| workload | do_calculus | P(procrastination \| do(workload)) |
| self_efficacy | do_calculus | P(procrastination \| do(self_efficacy)) |

### Indicators

| Indicator | Construct | Type | Aggregation | How to Measure |
| --- | --- | --- | --- | --- |
| youtube_watch_time | digital_distraction | continuous | sum | Total YouTube watch duration in minutes per day |
| search_count | digital_distraction | count | count | Number of non-work-related searches per day |
| late_night_activity | sleep_quality | binary | max | Any digital activity between 11PM-6AM (binary) |
| task_completion_rate | procrastination | continuous | mean | Proportion of planned tasks completed (0-1) |
| deadline_proximity_searches | stress | count | sum | Count of deadline/urgent-related searches |
| productive_hours | self_efficacy | continuous | sum | Hours spent on work-related activities per week |
| workload_events | workload | count | count | Number of calendar events and deadlines per day |

---

## Stage 2: Data Extraction
> Dispatches worker LLMs to extract indicator observations from raw activity data, processing each chunk independently.

- **Workers**: 10 succeeded, 0 failed, 10 total

### Extractions per Indicator

| Indicator | Count |
| --- | --- |
| youtube_watch_time | 182 |
| search_count | 182 |
| late_night_activity | 182 |
| task_completion_rate | 182 |
| deadline_proximity_searches | 182 |
| productive_hours | 26 |
| workload_events | 200 |

### Extractions Sample

| Indicator | Tick | Value |
| --- | --- | --- |
| youtube_watch_time | — | 45.5 |
| youtube_watch_time | — | 30.2 |
| youtube_watch_time | — | 62.1 |
| youtube_watch_time | — | 18.7 |
| youtube_watch_time | — | 55.3 |
| youtube_watch_time | — | 38.9 |
| youtube_watch_time | — | 41 |
| youtube_watch_time | — | 27.4 |
| youtube_watch_time | — | 52.8 |
| youtube_watch_time | — | 35.6 |
| youtube_watch_time | — | 48.2 |
| youtube_watch_time | — | 22.1 |
| youtube_watch_time | — | 59.4 |
| youtube_watch_time | — | 33.7 |
| youtube_watch_time | — | 44.6 |
| youtube_watch_time | — | 29.3 |
| youtube_watch_time | — | 51.1 |
| youtube_watch_time | — | 36.8 |
| youtube_watch_time | — | 42.5 |
| youtube_watch_time | — | 15.9 |

---

## Stage 3: Validation
> Validates extraction quality, checking for missing data, outliers, and consistency across indicators.

> **GATE BLOCKED**: Data validation failed.

---

## Stage 4: Model Specification
> Specifies prior distributions and model parameters using domain knowledge and empirical data.

### State Dynamics

$$
\begin{aligned}
\eta_{\text{procrastination}}(0) &\sim \mathcal{N}(\mu_{0,\text{procrastination}},\; \sigma_{0,\text{procrastination}}^{2}) \\
\eta_{\text{stress}}(0) &\sim \mathcal{N}(\mu_{0,\text{stress}},\; \sigma_{0,\text{stress}}^{2}) \\
\eta_{\text{procrastination}}(t) &= \rho_{\text{procrastination}} \, \eta_{\text{procrastination}}(t\!-\!1) + \beta_{\text{stress} \to \text{procrastination}} \, \eta_{\text{stress}}(t\!-\!1) + \beta_{\text{distraction} \to \text{procrastination}} \, \eta_{\text{distraction}}(t\!-\!1) + \varepsilon_{\text{procrastination}}(t) \\
\eta_{\text{stress}}(t) &= \rho_{\text{stress}} \, \eta_{\text{stress}}(t\!-\!1) + \beta_{\text{sleep} \to \text{stress}} \, \eta_{\text{sleep}}(t\!-\!1) + \beta_{\text{workload} \to \text{stress}} \, \eta_{\text{workload}}(t\!-\!1) + \varepsilon_{\text{stress}}(t) \\
\varepsilon_{\text{procrastination}}(t) &\sim \mathcal{N}(0,\, \sigma_{\text{procrastination}}^2) \\
\varepsilon_{\text{stress}}(t) &\sim \mathcal{N}(0,\, \sigma_{\text{stress}}^2)
\end{aligned}
$$

### Marginalized Confounders

$$
\begin{aligned}
U_{\text{mood}} &\to \{\text{procrastination},\, \text{stress}\} \\
(\varepsilon_{\text{procrastination}},\, \varepsilon_{\text{stress}}) &\sim \mathcal{N}(\mathbf{0},\, \Psi_{\text{mood}}) \\
\psi_{\text{procrastination},\,\text{stress}} &\neq 0
\end{aligned}
$$

### Observation Model

$$
\begin{aligned}
y_{\text{youtube watch time}}(t) &\sim \mathcal{N}(\lambda_{\text{youtube watch time}} \, \eta_{\text{digital distraction}}(t),\; \sigma_{\text{youtube watch time}}^{2}) \\
y_{\text{search count}}(t) &\sim \text{Poisson}(\exp(\lambda_{\text{search count}} \, \eta_{\text{digital distraction}}(t))) \\
y_{\text{late night activity}}(t) &\sim \text{Bernoulli}(\sigma(\lambda_{\text{late night activity}} \, \eta_{\text{sleep quality}}(t))) \\
y_{\text{task completion rate}}(t) &\sim \text{Beta}(\sigma(\lambda_{\text{task completion rate}} \, \eta_{\text{procrastination}}(t))\,\phi,\; (1 - \sigma(\lambda_{\text{task completion rate}} \, \eta_{\text{procrastination}}(t)))\,\phi) \\
y_{\text{deadline proximity searches}}(t) &\sim \text{Poisson}(\exp(\lambda_{\text{deadline proximity searches}} \, \eta_{\text{stress}}(t))) \\
y_{\text{productive hours}}(t) &\sim \mathcal{N}(\lambda_{\text{productive hours}} \, \eta_{\text{self efficacy}}(t),\; \sigma_{\text{productive hours}}^{2}) \\
y_{\text{workload events}}(t) &\sim \text{Poisson}(\exp(\lambda_{\text{workload events}} \, \eta_{\text{workload}}(t)))
\end{aligned}
$$

### Measurement Model

| Variable | Distribution | Link | Reasoning | Sources |
| --- | --- | --- | --- | --- |
| youtube_watch_time | gaussian | identity | Continuous minutes with approximately normal distribution | — |
| search_count | poisson | log | Count data with no upper bound | — |
| late_night_activity | bernoulli | logit | Binary indicator | — |
| task_completion_rate | beta | logit | Proportion bounded between 0 and 1 | — |
| deadline_proximity_searches | poisson | log | Count of search events | — |
| productive_hours | gaussian | identity | Continuous hours, normally distributed | — |
| workload_events | poisson | log | Count of calendar events | — |

### Prior Distributions

$$
\begin{aligned}
\beta_{\text{stress procrastination}} &\sim \mathcal{N}(0.3,\; 0.15) \\
\beta_{\text{distraction procrastination}} &\sim \mathcal{N}(0.25,\; 0.1) \\
\beta_{\text{sleep stress}} &\sim \mathcal{N}(-0.2,\; 0.1) \\
\beta_{\text{workload stress}} &\sim \mathcal{N}(0.35,\; 0.15) \\
\rho_{\text{procrastination}} &\sim \text{Beta}(5,\; 2) \\
\rho_{\text{stress}} &\sim \text{Beta}(4,\; 3) \\
\sigma_{\text{procrastination}} &\sim \text{HalfNormal}(0.5) \\
\sigma_{\text{stress}} &\sim \text{HalfNormal}(0.5) \\
\psi_{\text{procrastination stress}} &\sim \text{Uniform}(-1,\; 1) \\
\mu_{0,\,\text{procrastination}} &\sim \mathcal{N}(0,\; 2) \\
\sigma_{0,\,\text{procrastination}} &\sim \text{HalfNormal}(2) \\
\mu_{0,\,\text{stress}} &\sim \mathcal{N}(0,\; 2) \\
\sigma_{0,\,\text{stress}} &\sim \text{HalfNormal}(2)
\end{aligned}
$$

| Parameter | Prior | Sources | Reasoning |
| --- | --- | --- | --- |
| $\beta_{\text{stress procrastination}}$ | Normal(mu=0.3, sigma=0.15) | [Steel (2007) - Meta-analysis of procrastination](https://doi.org/10.1037/0033-2909.133.1.65); [Sirois & Pychyl (2013) - Procrastination and the priority of short-term mood regulation](https://doi.org/10.1111/spc3.12011) | Moderate positive effect expected based on meta-analytic evidence |
| $\beta_{\text{distraction procrastination}}$ | Normal(mu=0.25, sigma=0.1) | — | Expected positive but weaker effect than stress |
| $\beta_{\text{sleep stress}}$ | Normal(mu=-0.2, sigma=0.1) | [Åkerstedt et al. (2012) - Sleep and stress longitudinal study](https://doi.org/10.1111/j.1365-2869.2012.01033.x) | Better sleep reduces next-day stress; negative lagged effect supported by longitudinal evidence |
| $\beta_{\text{workload stress}}$ | Normal(mu=0.35, sigma=0.15) | [Bowling et al. (2015) - Workload-strain meta-analysis](https://doi.org/10.1177/0149206314559804) | Strong positive contemporaneous effect of workload on stress, well-supported by occupational health literature |
| $\rho_{\text{procrastination}}$ | Beta(alpha=5, beta=2) | — | Procrastination is moderately persistent day-to-day |
| $\rho_{\text{stress}}$ | Beta(alpha=4, beta=3) | — | Stress shows moderate autocorrelation |
| $\sigma_{\text{procrastination}}$ | HalfNormal(sigma=0.5) | — | Weakly informative prior on residual SD |
| $\sigma_{\text{stress}}$ | HalfNormal(sigma=0.5) | — | Weakly informative prior on residual SD |
| $\psi_{\text{procrastination stress}}$ | Uniform(lower=-1, upper=1) | — | Default uniform prior on residual correlation from marginalized confounder (mood) |
| $\mu_{0,\,\text{procrastination}}$ | Normal(mu=0, sigma=2) | — | Default weakly informative prior for the initial state mean of procrastination. |
| $\sigma_{0,\,\text{procrastination}}$ | HalfNormal(sigma=2) | — | Default weakly informative prior for the initial state standard deviation of procrastination. |
| $\mu_{0,\,\text{stress}}$ | Normal(mu=0, sigma=2) | — | Default weakly informative prior for the initial state mean of stress. |
| $\sigma_{0,\,\text{stress}}$ | HalfNormal(sigma=2) | — | Default weakly informative prior for the initial state standard deviation of stress. |

---

## Stage 4b: Parametric Identifiability
> Checks whether the specified model parameters are identifiable from the available data.

### T-Rule

- **Free parameters**: 8
- **Moment conditions**: 15
- **Satisfies**: Yes

### Inference Structure

- **Likelihood path**: composed
- **Auto method**: laplace_em
- **First-pass RB**: active
- **Latents (Kalman)**: stress
- **Latents (Particle)**: procrastination
- **Observed Channels (Kalman-side)**: cortisol_level, sleep_quality
- **Observed Channels (Particle-side)**: task_completion_rate, missed_deadlines

### Parameter Classification

| Parameter | Classification | Contraction Ratio |
| --- | --- | --- |
| beta_sleep_stress | identified | 0.820 |
| beta_workload_stress | identified | 0.910 |
| rho_stress | identified | 0.840 |
| sigma_stress | practically_unidentifiable | 0.450 |

---

## Stage 5a: SVI Preflight
> Fast variational fit as a diagnostic before expensive inference. Shows ELBO convergence and approximate posterior.

### SVI / ELBO Convergence

```
ELBO loss over optimization steps
 2450.30 │•                                                           
         │ •                                                          
         │  ••                                                        
         │    •                                                       
         │     ••                                                     
         │       ••                                                   
         │         ••                                                 
         │           •••                                              
         │              •••                                           
 1652.50 │                 •••                                        
         └────────────────────────────────────────────────────────────
  x: [0.000, 19.000]
  mean=8.802  sd=5.784  mode=0.000
```
Initial loss: 2450.3, Final loss: 1652.5, Improvement: 32.6%, Converged: No

### Posterior Marginals

```
beta_stress  (mean=0.330, sd=0.130, HDI=[0.080, 0.560])
    5.00 │     •                                                      
         │     │•                                                     
         │     │ │                                                    
         │    •  │                                                    
         │    │  │                                                    
         │    │  •                                                    
         │   •    │                                                   
         │   │    │                                                   
         │  •     •                                                   
    0.05 │••       •                                                  
         └────────────────────────────────────────────────────────────
  x: [-0.150, 0.750]
  mean=0.355  sd=0.150  mode=0.350
```

```
beta_distraction  (mean=0.220, sd=0.100, HDI=[0.020, 0.410])
    5.50 │   •                                                        
         │   │•                                                       
         │   │ │                                                      
         │   │ │                                                      
         │   │ │                                                      
         │   │ │                                                      
         │  •  │                                                      
         │  │  •                                                      
         │ •    │                                                     
    0.10 │•     •                                                     
         └────────────────────────────────────────────────────────────
  x: [-0.100, 0.500]
  mean=0.235  sd=0.104  mode=0.200
```

```
sigma_obs  (mean=0.520, sd=0.090, HDI=[0.360, 0.680])
    5.80 │    •                                                       
         │    ││                                                      
         │    ││                                                      
         │   • │                                                      
         │   │ •                                                      
         │   │  │                                                     
         │   │  │                                                     
         │  •   │                                                     
         │  │   •                                                     
    0.05 │••     •                                                    
         └────────────────────────────────────────────────────────────
  x: [0.150, 0.850]
  mean=0.528  sd=0.113  mode=0.550
```

### Posterior Pairs

```
beta_stress vs beta_distraction
   0.26 │   •                                              
        │                                        •         
        │      •                                           
        │                                  •               
        │            •               •                     
        │                        •                         
        │•                           •                     
        │         •        •  •                            
        │               •                                  
        │                               •                  
        │                  •                               
        │                                           •      
        │                                     •            
        │                                                 •
   0.11 │                                              •   
        └──────────────────────────────────────────────────
        0.22                                          0.38
  n=20  x: mean=0.299 sd=0.045  y: mean=0.189 sd=0.042
```
Pearson r: -0.531

*SVI Preflight: svi | 500 samples | 6.1s*

---

## Stage 5b: Inference & Diagnostics
> Fits the Bayesian model via MCMC or SVI and runs convergence and sensitivity diagnostics.

### SVI / ELBO Convergence

```
ELBO loss over optimization steps
 2450.30 │•                                                           
         │ ••                                                         
         │   ••                                                       
         │     •                                                      
         │      •••                                                   
         │         ••                                                 
         │           ••••                                             
         │               •••••                                        
         │                    •••••••••                               
 1483.50 │                             •••••••••••••••••••••          
         └────────────────────────────────────────────────────────────
  x: [0.000, 49.000]
  mean=22.508  sd=14.742  mode=0.000
```
Initial loss: 2450.3, Final loss: 1483.5, Improvement: 39.5%, Converged: Yes

### Posterior Predictive Checks

**Status**: Consistent

| Variable | Check | Result | Value | Message |
| --- | --- | --- | --- | --- |
| youtube_watch_time | calibration | Pass | 0.932 | 95% CI coverage: 93.2% (expected ~95%) |
| youtube_watch_time | autocorrelation | Pass | 0.150 | Residual autocorrelation at lag 1: 0.15 |
| youtube_watch_time | variance | Pass | 0.900 | Predicted variance 14.2 vs observed 15.7 (ratio 0.90) |
| productive_hours | calibration | Pass | 0.961 | 95% CI coverage: 96.1% (expected ~95%) |
| productive_hours | autocorrelation | Pass | 0.220 | Residual autocorrelation at lag 1: 0.22 |
| productive_hours | variance | Pass | 0.890 | Predicted variance 5.1 vs observed 5.7 (ratio 0.89) |

#### Posterior Predictive Overlays

```
youtube_watch_time (• observed, ◦ median)
   55.00 │    •                                                       
         │  • ││      •    •                                          
         │  ││◦│  •   ││ • ││                                         
         │  ◦│││• ◦│  ◦• ◦│◦│                                         
         │• ││││◦│││• │◦│││││◦                                        
         │◦││◦││││││◦││ ││◦ ││                                        
         │ ││• ││◦ ││◦│ ││  ││                                        
         │ ◦   ││  ◦│•  ││  ◦                                         
         │     ◦│  •    ◦                                             
   33.90 │     •                                                      
         └────────────────────────────────────────────────────────────
  • series 1: n=20  mean=44.620  sd=6.200  range=[33.900, 55.000]
  ◦ series 2: n=20  mean=43.490  sd=4.053  range=[36.800, 50.200]
```
95% CI coverage: 100.0% (20/20), RMSE: 2.473, MAE: 2.100, Pearson r: 0.995

```
productive_hours (• observed, ◦ median)
    7.50 │     •            •                                         
         │ •   ◦│ •   •     ◦│                                        
         │ ││• ││ ◦│  ◦│ •  ││                                        
         │ ◦│◦│││ ││• ││ ◦│ ││                                        
         │ │││││• ││◦│││ │• │◦                                        
         │◦ ││││◦││•│││• │◦││                                         
         │  •│││ ◦ ◦ ││◦•│ ││                                         
         │  ◦ ││     •│ ◦  •│                                         
         │    •│     ◦     ◦                                          
    5.00 │    ◦                                                       
         └────────────────────────────────────────────────────────────
  • series 1: n=20  mean=6.420  sd=0.679  range=[5.200, 7.500]
  ◦ series 2: n=20  mean=6.180  sd=0.636  range=[5.000, 7.200]
```
95% CI coverage: 100.0% (20/20), RMSE: 0.245, MAE: 0.240, Pearson r: 0.999

#### Test Statistics

| Variable | Statistic | Observed | p(rep ≥ obs) | Result |
| --- | --- | --- | --- | --- |
| youtube_watch_time | mean | 44.100 | 0.480 | Pass |
| youtube_watch_time | sd | 6.100 | 0.500 | Pass |
| productive_hours | mean | 6.400 | 0.500 | Pass |
| productive_hours | sd | 0.680 | 0.500 | Pass |

### LOO Cross-Validation

- **ELPD**: -38.6
- **p_loo**: 2.8
- **SE**: 4.7
- **Data points**: 50
- **Observation unit**: undefined
- **Bad Pareto k**: 0

#### LOO-PIT

```
LOO-PIT (should be uniform)
    0.16 │ █████████████████████████████████ 5
    0.24 │ ███████████████████████████ 4
    0.31 │ █████████████████████████████████ 5
    0.39 │ ████████████████████████████████████████ 6
    0.47 │ █████████████████████████████████ 5
    0.54 │ ████████████████████████████████████████ 6
    0.62 │ ███████████████████████████ 4
    0.70 │ ████████████████████████████████████████ 6
    0.77 │ █████████████████████████████████ 5
    0.85 │ ███████████████████████████ 4
  n=50  mean=0.504  sd=0.216  median=0.495  range=[0.120, 0.890]
```
Mean: 0.504 (ideal: 0.500), Std: 0.216 (ideal: 0.289), KS stat: 0.130, Calibration: Fair

#### Pareto k Diagnostics

- **k > 0.7 (fail)**: 0
- **0.5 < k ≤ 0.7 (warn)**: 0
- **k ≤ 0.5 (ok)**: 50

```
Pareto k distribution
    0.02 │ ██████████ 2
    0.04 │ █████████████████████████ 5
    0.06 │ █████████████████████████ 5
    0.07 │ ████████████████████████████████████████ 8
    0.09 │ ████████████████████ 4
    0.11 │ █████████████████████████ 5
    0.13 │ ████████████████████ 4
    0.15 │ ████████████████████ 4
    0.16 │ ███████████████ 3
    0.18 │ ███████████████ 3
    0.20 │ █████ 1
    0.22 │ ██████████ 2
    0.24 │ █████ 1
    0.25 │ █████ 1
    0.27 │ ██████████ 2
  n=50  mean=0.115  sd=0.067  median=0.100  range=[0.010, 0.280]
```

### Power Scaling Diagnostics

| Parameter | Diagnosis | Prior Sens. | Likelihood Sens. | PSIS k-hat |
| --- | --- | --- | --- | --- |
| beta_stress_procrastination | well_identified | 0.120 | 0.850 | 0.150 |
| beta_distraction_procrastination | well_identified | 0.180 | 0.790 | 0.220 |
| beta_sleep_stress | prior_dominated | 0.650 | 0.320 | 0.580 |
| rho_procrastination | well_identified | 0.080 | 0.910 | 0.100 |
| sigma_stress | well_identified | 0.150 | 0.820 | 0.180 |

```
Power Scaling (prior vs likelihood sensitivity)
   0.91 │•                                                 
        │                                                  
        │   •                                              
        │      •                                           
        │         •                                        
        │                                                  
        │                                                  
        │                                                  
        │                                                  
        │                                                  
        │                                                  
        │                                                  
        │                                                  
        │                                                  
        │                                                  
        │                                                  
        │                                                  
        │                                                  
        │                                                  
   0.32 │                                                 •
        └──────────────────────────────────────────────────
        0.08                                          0.65
  n=5  x: mean=0.236 sd=0.210  y: mean=0.738 sd=0.213
```

### Posterior Marginals

```
beta_stress  (mean=0.312, sd=0.089, HDI=[0.150, 0.480])
    6.20 │       •                                                    
         │      • •                                                   
         │      │  •                                                  
         │     •    │                                                 
         │     │    │                                                 
         │    •     •                                                 
         │    │      •                                                
         │   •        │                                               
         │  •         ••                                              
    0.05 │••            •••                                           
         └────────────────────────────────────────────────────────────
  x: [-0.100, 0.700]
  mean=0.258  sd=0.126  mode=0.250
```

```
beta_distraction  (mean=0.198, sd=0.072, HDI=[0.060, 0.340])
    6.50 │     •                                                      
         │     ││                                                     
         │    • •                                                     
         │    │  │                                                    
         │    │  •                                                    
         │   •    │                                                   
         │   │    │                                                   
         │  •     •                                                   
         │  │      •                                                  
    0.20 │••        •                                                 
         └────────────────────────────────────────────────────────────
  x: [-0.050, 0.450]
  mean=0.205  sd=0.092  mode=0.200
```

```
sigma_obs  (mean=0.510, sd=0.053, HDI=[0.420, 0.610])
    6.80 │     •                                                      
         │     │•                                                     
         │    •  │                                                    
         │    │  •                                                    
         │    │   │                                                   
         │   •    │                                                   
         │   │    •                                                   
         │  •      │                                                  
         │  │      •                                                  
    0.05 │••        •••                                               
         └────────────────────────────────────────────────────────────
  x: [0.200, 0.800]
  mean=0.473  sd=0.095  mode=0.450
```

### Posterior Pairs

```
beta_stress vs beta_distraction (2 divergent)
   0.26 │   •                                              
        │                                        •         
        │      •                                           
        │                                  •               
        │            •               •                     
        │                        •                         
        │•                           •                     
        │         •        •  •                            
        │               •                                  
        │                               •                  
        │                  •                               
        │                                           •      
        │                                     •            
        │                                                 •
   0.11 │                                              •   
        └──────────────────────────────────────────────────
        0.22                                          0.38
  n=20  x: mean=0.299 sd=0.045  y: mean=0.189 sd=0.042
```
Pearson r: -0.531, Divergent: 2/20 (10.0%)

```
beta_stress vs sigma_obs
   0.57 │   •                                              
        │      •                                           
        │                  •                               
        │               •                                  
        │•           •                                     
        │         •        •                               
        │                     •                            
        │                            •                    •
        │                        •                         
        │                            •  •                  
        │                                  •               
        │                                        •         
        │                                           •      
        │                                     •            
   0.43 │                                              •   
        └──────────────────────────────────────────────────
        0.22                                          0.38
  n=20  x: mean=0.299 sd=0.045  y: mean=0.501 sd=0.039
```
Pearson r: -0.844

*Inference: SVI | 10000 samples | 45.2s*

---

## Stage 6: Treatment Effects
> Computes interventional treatment effects and ranks them by magnitude and certainty.

### Treatment Ranking

| Treatment | τ̂ | 95% CI | P(τ>0) | Identifiable | Status |
| --- | --- | --- | --- | --- | --- |
| stress | 0.312 | [0.161, 0.494] | 97.8% | Yes | ok |
| digital_distraction | 0.198 | [0.055, 0.347] | 94.3% | Yes | ok |
| sleep_quality | -0.156 | [-0.343, 0.029] | 8.7% | Yes | prior-sensitive |
| workload | 0.089 | [-0.023, 0.206] | 82.1% | Yes | ok |

#### Prior Sensitivity Warnings

- **sleep_quality**: Sensitivity warning (mock fixture)

```
Posterior: stress
    0.09 │ ███ 2
    0.12 │  0
    0.15 │ █████ 4
    0.18 │ ███████████ 8
    0.21 │ ███████████████████ 14
    0.24 │ ████████████ 9
    0.27 │ ███████████████████████████████ 23
    0.30 │ ███████████████████████████████████████ 29
    0.33 │ ████████████████████████████████████████ 30
    0.36 │ ███████████████████████████████████████ 29
    0.39 │ ████████████████████████████████ 24
    0.42 │ ████████████ 9
    0.45 │ ████████████ 9
    0.48 │ █████ 4
    0.51 │ ████████ 6
  n=200  mean=0.322  sd=0.083  median=0.325  range=[0.076, 0.522]
```

```
Posterior: digital_distraction
    0.05 │ ██████████ 8
    0.08 │ ██████ 5
    0.10 │ █████████████ 10
    0.13 │ ███████████████████ 15
    0.15 │ ████████████████████████████████████████ 31
    0.18 │ ███████████████████████████████████ 27
    0.21 │ ██████████████████████████████████ 26
    0.23 │ █████████████████████████████████████ 29
    0.26 │ ██████████████████████ 17
    0.29 │ █████████████ 10
    0.31 │ ██████████████ 11
    0.34 │ ████████ 6
    0.37 │ ████ 3
    0.39 │  0
    0.42 │ ███ 2
  n=200  mean=0.201  sd=0.075  median=0.198  range=[0.036, 0.431]
```

```
Posterior: sleep_quality
   -0.39 │ ████ 3
   -0.35 │ ████ 3
   -0.32 │ ██████ 5
   -0.29 │ ██████████████████ 14
   -0.26 │ ██████████████████████ 17
   -0.23 │ ███████████████████ 15
   -0.20 │ ██████████████████████████████ 23
   -0.16 │ ████████████████████████████████████████ 31
   -0.13 │ █████████████████████████████████████ 29
   -0.10 │ █████████████████████ 16
   -0.07 │ █████████████████████████ 19
   -0.04 │ ██████████████ 11
   -0.00 │ ████████ 6
    0.03 │ ████████ 6
    0.06 │ ███ 2
  n=200  mean=-0.160  sd=0.095  median=-0.156  range=[-0.401, 0.075]
```

```
Posterior: workload
   -0.04 │ ███ 3
   -0.02 │ ██████████ 9
    0.01 │ ████████████████ 14
    0.03 │ █████████████████████████████ 26
    0.05 │ ██████████████████████████████ 27
    0.08 │ ████████████████████████████████████████ 36
    0.10 │ ██████████████████████████████████ 31
    0.13 │ █████████████████████████████ 26
    0.15 │ █████████████ 12
    0.18 │ ████████ 7
    0.20 │ ████ 4
    0.23 │ ████ 4
    0.25 │  0
    0.28 │  0
    0.30 │ █ 1
  n=200  mean=0.083  sd=0.059  median=0.081  range=[-0.056, 0.314]
```
