# MeasurementStructure Assumptions

This page collects the assumptions that constrain how constructs are measured.

## A1. Reflective Measurement Structure

**Assumption:** All constructs with indicators use a reflective effect-indicator measurement structure, not a formative causal-indicator model.

**Definition:** In a reflective model, the construct is the common cause of its indicators. Causality flows from construct to indicators:

```text
Construct -> Indicator_1
          -> Indicator_2
          -> Indicator_3
```

In a formative model, indicators cause the construct, as in examples like socioeconomic status being "formed" by income, education, and occupation. This project does not support formative measurement.

**Implications:**

- Indicators of the same construct should be correlated because they share a common cause
- Removing an indicator does not change the definition of the construct
- The construct exists independently of its specific operationalization
- No causal edges run from indicators to their parent construct

**Justification:** Reflective models are standard in psychological and behavioral SEM. They align with classical test theory where observed scores reflect true scores plus error. Formative models require different identification constraints and are conceptually suited to composite indices rather than theoretical constructs[^diamantopoulos2006].

## A6. Measurement Error Handling Depends on Indicator Count

**Assumption:** How measurement error is handled depends on whether a construct has one or multiple indicators.

**Definition:**

- **Multi-indicator constructs (>= 2):** Measurement error is separated from construct variance via factor analysis. The construct is identified through shared variance among indicators.
- **Single-indicator constructs (= 1):** Measurement error is absorbed into the structural residual under [A9](#a9-single-indicator-constructs-absorb-measurement-error). No separation is possible.

**Implications:**

- Multi-indicator constructs yield unattenuated coefficient estimates because measurement error is partitioned out
- Single-indicator constructs may have attenuated coefficients biased toward zero
- Both are "observed" for the purpose of causal identification; the distinction is about precision, not identifiability

**Justification:** Separating measurement error from true construct variance requires multiple indicators to identify the factor structure[^bollen1989]. With a single indicator, the two sources of variance are fundamentally confounded without external reliability information.

## A8. Indicator Residuals Are Temporally Independent

**Assumption:** Measurement error in indicators is iid across time. All temporal dependence in observed indicator series is attributed to the construct's dynamics.

**Definition:** For any indicator `I` of construct `C`:

```text
I_t = lambda * C_t + epsilon_t
epsilon_t ~ N(0, sigma^2), independent across t
```

**Implications:**

- Indicators do not receive AR structure; only constructs do
- Residual autocorrelation in indicators suggests model misspecification
- Possible causes include construct granularity that is too coarse, missing cross-loadings, or systematic measurement dynamics

**Justification:** Separating construct dynamics from indicator dynamics requires strong identification constraints. By attributing all temporal structure to the construct, the framework keeps a clean separation between "what is happening" and "how we see it." This is the default in dynamic SSM implementations. As Asparouhov, Hamaker, and Muthén (2018)[^asparouhov2018] note, measurement errors are usually assumed uncorrelated across time, and serially correlated residuals indicate that the latent variable does not fully account for the observed dynamics.

**Relaxation (not currently supported):** AR in indicator residuals is possible but introduces identification challenges. Mplus allows this via the `RESIDUAL` option. Future versions may support this with appropriate constraints.

## A9. Single-Indicator Constructs Absorb Measurement Error

**Assumption:** When a construct has exactly one indicator, the indicator is treated as identical to the construct. Measurement error is absorbed into the structural residual.

**Definition:** For a single-indicator construct:

```text
Construct_t ≡ Indicator_t
```

Conceptually, this collapses

```text
Construct_t = structural dynamics + structural error
Indicator_t = lambda * Construct_t + measurement error
```

into

```text
Indicator_t = structural dynamics + combined error
```

where `lambda` is fixed to `1` and measurement error merges with structural error.

**Implications:**

- Coefficient estimates may be attenuated if measurement error is substantial
- No separation of true construct variance from measurement noise is available
- This is a pragmatic choice, not an assertion that measurement is perfect

**Justification:** Single-indicator identification of separate measurement and structural variance is impossible without external information such as known reliability coefficients. Bollen (1989)[^bollen1989] describes the standard fallback: fix the loading to unity and the error variance to zero, thereby equating the indicator with the latent variable for modeling purposes.

**Recommendation:** When substantively important, prefer multiple indicators per construct to enable measurement-error separation. Single-indicator constructs are appropriate for well-validated scales with known high reliability or for exploratory analysis where attenuation bias is acceptable.

## Future Considerations (MeasurementStructure-Related)

The following are explicitly not assumed and may be added in future versions:

- **Formative measurement:** Currently only reflective models are supported
- **Indicator AR:** Currently indicator residuals are iid; correlated residuals are not supported

## Boundary

These assumptions shape the `MeasurementStructure` itself. The follow-on assumption that an identified measurement structure permits causal identification lives with the [CausalDesign](../causal-design/identifiability.md).

[^diamantopoulos2006]: Diamantopoulos, A., & Siguaw, J. A. (2006). Formative Versus Reflective Indicators in Organizational Measure Development. *British Journal of Management*, 17(4), 263–282. [Bibliography entry](../bibliography.md)
[^asparouhov2018]: Asparouhov, T., Hamaker, E. L., & Muthén, B. (2018). Dynamic Structural Equation Models. *Structural Equation Modeling*, 25(3), 359–388. [Bibliography entry](../bibliography.md)
[^bollen1989]: Bollen, K. A. (1989). *Structural Equations with Latent Variables*. Wiley. [Bibliography entry](../bibliography.md)
