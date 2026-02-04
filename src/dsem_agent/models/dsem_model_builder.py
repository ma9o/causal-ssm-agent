"""DSEM Model Builder using PyMC ModelBuilder.

Builds Dynamic Structural Equation Models from ModelSpec + priors.
Supports AR(1) dynamics, cross-lagged effects, and multiple distribution families.
"""

from typing import Any

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from pymc_extras.model_builder import ModelBuilder

from dsem_agent.orchestrator.schemas_model import ModelSpec
from dsem_agent.workers.schemas_prior import PriorProposal


class DSEMModelBuilder(ModelBuilder):
    """PyMC ModelBuilder for DSEM models.

    Builds a proper DSEM with:
    - AR(1) dynamics for time-varying constructs
    - Cross-lagged effects between constructs
    - Multiple distribution families for indicators
    - Random effects for hierarchical structure

    The model is built from:
    1. ModelSpec: Specifies likelihoods, parameters, and their roles
    2. Priors: Prior distributions for each parameter

    Data expectations:
    - X should have columns for each indicator
    - Lagged columns named {indicator}_lag1 for AR and cross-lag effects
    - Optional: subject_id for random effects, time_bucket for time index
    """

    _model_type = "DSEM"
    version = "0.1.0"

    def __init__(
        self,
        model_spec: ModelSpec | dict | None = None,
        priors: dict[str, PriorProposal] | dict[str, dict] | None = None,
        model_config: dict | None = None,
        sampler_config: dict | None = None,
    ):
        """Initialize the DSEM model builder.

        Args:
            model_spec: Model specification from orchestrator
            priors: Prior proposals for each parameter
            model_config: Override model configuration
            sampler_config: Override sampler configuration
        """
        if model_spec is not None:
            if isinstance(model_spec, ModelSpec):
                self._model_spec_dict = model_spec.model_dump()
            else:
                self._model_spec_dict = model_spec
        else:
            self._model_spec_dict = {}

        if priors is not None:
            self._priors_dict = {}
            for name, prior in priors.items():
                if isinstance(prior, PriorProposal):
                    self._priors_dict[name] = prior.model_dump()
                else:
                    self._priors_dict[name] = prior
        else:
            self._priors_dict = {}

        if model_config is None:
            model_config = {
                "model_spec": self._model_spec_dict,
                "priors": self._priors_dict,
            }

        super().__init__(
            model_config=model_config,
            sampler_config=sampler_config,
        )

    @staticmethod
    def get_default_model_config() -> dict:
        return {"model_spec": {}, "priors": {}}

    @staticmethod
    def get_default_sampler_config() -> dict:
        return {
            "draws": 1000,
            "tune": 1000,
            "chains": 4,
            "cores": 4,
            "target_accept": 0.9,
        }

    @property
    def output_var(self) -> str:
        likelihoods = self.model_config.get("model_spec", {}).get("likelihoods", [])
        if likelihoods:
            return likelihoods[0]["variable"]
        return "y"

    @property
    def _serializable_model_config(self) -> dict:
        return self.model_config

    def build_model(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray | None = None,
        **kwargs: Any,
    ) -> pm.Model:
        """Build the PyMC model from model spec and priors.

        Args:
            X: Data with indicator columns and lagged columns ({name}_lag1)
            y: Optional target (if not in X)

        Returns:
            The constructed PyMC model
        """
        model_spec = self.model_config.get("model_spec", {})
        priors_config = self.model_config.get("priors", {})

        # Parse ModelSpec structure
        likelihoods = model_spec.get("likelihoods", [])
        parameters = model_spec.get("parameters", [])
        random_effects = model_spec.get("random_effects", [])

        # Index parameters by role for easy lookup
        params_by_role = self._index_parameters_by_role(parameters)

        with pm.Model() as self.model:
            # 1. Store observed data
            data_vars = self._store_data(X, y, likelihoods)

            # 2. Create all prior distributions
            param_vars = self._create_priors(priors_config)

            # 3. Create random effects if specified
            random_effect_vars = self._create_random_effects(random_effects, X)

            # 4. Build likelihood for each observed indicator
            for lik_spec in likelihoods:
                self._build_indicator_likelihood(
                    lik_spec=lik_spec,
                    param_vars=param_vars,
                    params_by_role=params_by_role,
                    random_effect_vars=random_effect_vars,
                    data_vars=data_vars,
                    X=X,
                )

        return self.model

    def _index_parameters_by_role(
        self, parameters: list[dict]
    ) -> dict[str, list[dict]]:
        """Index parameters by their role for easy lookup."""
        by_role: dict[str, list[dict]] = {}
        for param in parameters:
            role = param.get("role", "fixed_effect")
            if role not in by_role:
                by_role[role] = []
            by_role[role].append(param)
        return by_role

    def _store_data(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray | None,
        likelihoods: list[dict],
    ) -> dict[str, Any]:
        """Store predictor data as pm.Data.

        Note: Observed variables (likelihood targets) are stored as raw arrays,
        not pm.Data, to avoid name collisions with the likelihood RVs.
        """
        data_vars = {}

        # Get names of observed variables (these will be likelihood RVs)
        observed_vars = {lik.get("variable") for lik in likelihoods}

        # Store predictor columns as pm.Data (not observed variables)
        for col in X.columns:
            if X[col].dtype in (np.float64, np.float32, np.int64, np.int32, bool):
                if col in observed_vars:
                    # Store as raw array to avoid name collision with likelihood
                    data_vars[col] = X[col].values.astype(float)
                else:
                    data_vars[col] = pm.Data(col, X[col].values.astype(float))

        # Store y if provided separately
        if y is not None:
            if isinstance(y, pd.Series):
                data_vars["y"] = y.values.astype(float)
            else:
                data_vars["y"] = np.asarray(y).astype(float)

        return data_vars

    def _create_priors(self, priors_config: dict) -> dict[str, Any]:
        """Create PyMC distributions for all priors."""
        param_vars = {}

        for name, prior_spec in priors_config.items():
            dist_name = prior_spec.get("distribution", "Normal")
            dist_params = prior_spec.get("params", {})
            param_vars[name] = self._create_distribution(name, dist_name, dist_params)

        return param_vars

    def _create_distribution(self, name: str, dist_name: str, params: dict) -> Any:
        """Create a PyMC distribution from specification."""
        if dist_name == "Normal":
            return pm.Normal(name, mu=params.get("mu", 0), sigma=params.get("sigma", 1))
        elif dist_name == "HalfNormal":
            return pm.HalfNormal(name, sigma=params.get("sigma", 1))
        elif dist_name == "Beta":
            return pm.Beta(name, alpha=params.get("alpha", 2), beta=params.get("beta", 2))
        elif dist_name == "Uniform":
            return pm.Uniform(name, lower=params.get("lower", 0), upper=params.get("upper", 1))
        elif dist_name == "TruncatedNormal":
            return pm.TruncatedNormal(
                name,
                mu=params.get("mu", 0),
                sigma=params.get("sigma", 1),
                lower=params.get("lower"),
                upper=params.get("upper"),
            )
        elif dist_name == "Gamma":
            return pm.Gamma(name, alpha=params.get("alpha", 2), beta=params.get("beta", 1))
        elif dist_name == "HalfCauchy":
            return pm.HalfCauchy(name, beta=params.get("beta", 1))
        elif dist_name == "Exponential":
            return pm.Exponential(name, lam=params.get("lam", 1))
        else:
            return pm.Normal(name, mu=params.get("mu", 0), sigma=params.get("sigma", 1))

    def _create_random_effects(
        self,
        random_effects: list[dict],
        X: pd.DataFrame,
    ) -> dict[str, Any]:
        """Create random effect variables."""
        re_vars = {}

        for re_spec in random_effects:
            grouping = re_spec.get("grouping", "subject")
            effect_type = re_spec.get("effect_type", "intercept")
            applies_to = re_spec.get("applies_to", [])

            # Check if grouping column exists
            if grouping not in X.columns and f"{grouping}_id" not in X.columns:
                continue

            group_col = grouping if grouping in X.columns else f"{grouping}_id"
            n_groups = X[group_col].nunique()

            if n_groups <= 1:
                continue

            # Create random effect SD
            re_sd_name = f"sd_{effect_type}_{grouping}"
            re_sd = pm.HalfNormal(re_sd_name, sigma=1)

            # Create random effects for each construct it applies to
            for construct in applies_to:
                re_name = f"re_{effect_type}_{grouping}_{construct}"
                re_vars[re_name] = pm.Normal(
                    re_name,
                    mu=0,
                    sigma=re_sd,
                    shape=n_groups,
                )
                # Store the group index mapping
                re_vars[f"{re_name}_idx"] = X[group_col].factorize()[0]

        return re_vars

    def _build_indicator_likelihood(
        self,
        lik_spec: dict,
        param_vars: dict[str, Any],
        params_by_role: dict[str, list[dict]],
        random_effect_vars: dict[str, Any],
        data_vars: dict[str, Any],
        X: pd.DataFrame,
    ) -> None:
        """Build the likelihood for a single indicator."""
        indicator = lik_spec.get("variable")
        distribution = lik_spec.get("distribution", "Normal")
        link = lik_spec.get("link", "identity")

        # Build linear predictor
        mu = self._build_linear_predictor(
            indicator=indicator,
            param_vars=param_vars,
            params_by_role=params_by_role,
            random_effect_vars=random_effect_vars,
            data_vars=data_vars,
            X=X,
        )

        # Apply inverse link function
        mu_transformed = self._apply_inverse_link(mu, link)

        # Get observed data
        observed = data_vars.get(indicator)
        if observed is None and indicator in X.columns:
            observed = X[indicator].values

        # Get residual SD if this is a continuous distribution
        sigma = self._get_residual_sd(indicator, param_vars)

        # Create likelihood
        self._create_likelihood(indicator, distribution, mu_transformed, sigma, observed)

    def _build_linear_predictor(
        self,
        indicator: str,
        param_vars: dict[str, Any],
        params_by_role: dict[str, list[dict]],
        random_effect_vars: dict[str, Any],
        data_vars: dict[str, Any],
        X: pd.DataFrame,
    ) -> Any:
        """Build the linear predictor for an indicator.

        Linear predictor = intercept + AR term + cross-lag effects + random effects

        Parameter naming convention:
        - AR coefficient: rho_{indicator} or ar_{indicator}
        - Cross-lag: beta_{target}_{source} where target is this indicator
        - Intercept: intercept_{indicator}
        """
        n_obs = len(X)
        mu = pt.zeros(n_obs)

        # 1. Intercept
        intercept_name = f"intercept_{indicator}"
        if intercept_name in param_vars:
            mu = mu + param_vars[intercept_name]

        # 2. AR(1) term: rho * indicator_{t-1}
        ar_names = [f"rho_{indicator}", f"ar_{indicator}"]
        for ar_name in ar_names:
            if ar_name in param_vars:
                lag_col = f"{indicator}_lag1"
                if lag_col in data_vars:
                    mu = mu + param_vars[ar_name] * data_vars[lag_col]
                elif lag_col in X.columns:
                    mu = mu + param_vars[ar_name] * X[lag_col].values
                break

        # 3. Cross-lag effects: beta_{target}_{source} * source_{t-1}
        for param_name, param_var in param_vars.items():
            if not param_name.startswith("beta_"):
                continue

            # Parse beta_{target}_{source}
            parts = param_name.split("_")
            if len(parts) < 3:
                continue

            # Target is parts[1], source is parts[2] (or rest joined)
            target = parts[1]
            source = "_".join(parts[2:])

            if target != indicator:
                continue

            # Look for source data (lagged or contemporaneous)
            source_lag_col = f"{source}_lag1"
            if source_lag_col in data_vars:
                mu = mu + param_var * data_vars[source_lag_col]
            elif source_lag_col in X.columns:
                mu = mu + param_var * X[source_lag_col].values
            elif source in data_vars:
                # Contemporaneous effect
                mu = mu + param_var * data_vars[source]
            elif source in X.columns:
                mu = mu + param_var * X[source].values

        # 4. Random effects
        for re_name, re_var in random_effect_vars.items():
            if indicator in re_name and not re_name.endswith("_idx"):
                idx_name = f"{re_name}_idx"
                if idx_name in random_effect_vars:
                    idx = random_effect_vars[idx_name]
                    mu = mu + re_var[idx]

        return mu

    def _apply_inverse_link(self, mu: Any, link: str) -> Any:
        """Apply the inverse link function."""
        if link == "identity":
            return mu
        elif link == "log":
            return pt.exp(mu)
        elif link == "logit":
            return pt.sigmoid(mu)
        elif link == "probit":
            # Use normal CDF approximation
            return 0.5 * (1 + pt.erf(mu / pt.sqrt(2)))
        elif link == "softmax":
            return pt.softmax(mu, axis=-1)
        else:
            return mu

    def _get_residual_sd(
        self, indicator: str, param_vars: dict[str, Any]
    ) -> Any:
        """Get the residual SD parameter for an indicator."""
        sd_names = [f"sigma_{indicator}", f"sd_{indicator}", f"residual_sd_{indicator}"]
        for sd_name in sd_names:
            if sd_name in param_vars:
                return param_vars[sd_name]
        return 1.0  # Default if not specified

    def _create_likelihood(
        self,
        name: str,
        distribution: str,
        mu: Any,
        sigma: Any,
        observed: Any,
    ) -> None:
        """Create the likelihood distribution for an indicator."""
        if distribution == "Normal":
            pm.Normal(name, mu=mu, sigma=sigma, observed=observed)

        elif distribution == "Bernoulli":
            # mu should be probability after inverse link
            p = pt.clip(mu, 1e-6, 1 - 1e-6)
            pm.Bernoulli(name, p=p, observed=observed)

        elif distribution == "Poisson":
            # mu should be rate after inverse link (positive)
            rate = pt.maximum(mu, 1e-6)
            pm.Poisson(name, mu=rate, observed=observed)

        elif distribution == "NegativeBinomial":
            rate = pt.maximum(mu, 1e-6)
            pm.NegativeBinomial(name, mu=rate, alpha=sigma, observed=observed)

        elif distribution == "Gamma":
            # Parameterize by mean (mu) and SD (sigma)
            # alpha = (mu/sigma)^2, beta = mu/sigma^2
            mu_pos = pt.maximum(mu, 1e-6)
            alpha = (mu_pos / sigma) ** 2
            beta = mu_pos / (sigma ** 2)
            pm.Gamma(name, alpha=alpha, beta=beta, observed=observed)

        elif distribution == "Beta":
            # mu is mean in (0,1), sigma controls concentration
            mu_clipped = pt.clip(mu, 1e-6, 1 - 1e-6)
            kappa = pt.maximum(1.0 / (sigma ** 2), 2.0)  # concentration
            alpha = mu_clipped * kappa
            beta = (1 - mu_clipped) * kappa
            pm.Beta(name, alpha=alpha, beta=beta, observed=observed)

        elif distribution == "OrderedLogistic":
            # For ordinal data - simplified version
            pm.Normal(name, mu=mu, sigma=sigma, observed=observed)

        else:
            # Default to Normal
            pm.Normal(name, mu=mu, sigma=sigma, observed=observed)

    def _data_setter(self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None) -> None:
        """Update mutable data for prediction."""
        with self.model:
            for col in X.columns:
                if col in self.model.named_vars:
                    pm.set_data({col: X[col].values.astype(float)})

            if y is not None:
                if "y" in self.model.named_vars:
                    y_arr = y.values if isinstance(y, pd.Series) else y
                    pm.set_data({"y": np.asarray(y_arr).astype(float)})

    def _save_input_params(self, idata: Any) -> None:
        """Save additional parameters to inference data."""
        idata.attrs["model_spec"] = str(self.model_config.get("model_spec", {}))
        idata.attrs["priors"] = str(self.model_config.get("priors", {}))

    def sample_prior_predictive(self, samples: int = 500) -> Any:
        """Sample from the prior predictive distribution."""
        if self.model is None:
            raise ValueError("Model must be built before sampling prior predictive")

        with self.model:
            idata = pm.sample_prior_predictive(draws=samples)

        return idata
