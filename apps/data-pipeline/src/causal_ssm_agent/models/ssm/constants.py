"""Shared constants for state-space model code."""

# Minimum time interval for the first element of dt arrays.
# The first dt is undefined (no previous timepoint), so we use a small positive
# sentinel to avoid division-by-zero in discretization while keeping the
# initial-state contribution negligible.
MIN_DT = 1e-6

# Sites registered by SSMModel.model() that are internal diagnostics,
# not user-facing parameter/latent sites.  All trace consumers that need
# to distinguish "public" from "internal" should use this constant.
INTERNAL_DIAGNOSTIC_SITES: frozenset[str] = frozenset({"log_likelihood", "ll_per_timestep"})
