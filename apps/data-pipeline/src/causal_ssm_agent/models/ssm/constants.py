"""Shared constants for state-space model code."""

# Minimum time interval for the first element of dt arrays.
# The first dt is undefined (no previous timepoint), so we use a small positive
# sentinel to avoid division-by-zero in discretization while keeping the
# initial-state contribution negligible.
MIN_DT = 1e-6
