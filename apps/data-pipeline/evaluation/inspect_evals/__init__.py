"""Inspect-native LLM evals (Target 1a/1b/2 proposal quality).

These are a deliberately *separate execution framework* from the statistical
evaluation registry: Inspect supplies the model providers, parallelism and
logging that LLM evals need, so they are ``@task`` functions — NOT
``RegistryEntry`` rows. The registry (``evaluation.registry``) holds the
deterministic / statistical benchmarks (identification gates, recovery).

They are NOT, however, a parallel *scoring* system. The grading logic is the same
spine code a registry row would use — ``evaluation.scorers.constructs`` and
``evaluation.scorers.measurement`` — imported here, never duplicated. That
single-source rule is what prevents the drift that previously left the
``orchestrator``-based eval scorers silently broken.
"""
