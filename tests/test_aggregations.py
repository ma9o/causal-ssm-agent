"""Tests for aggregation registry."""

import pytest
import polars as pl

from causal_agent.utils.aggregations import (
    AGGREGATION_REGISTRY,
    apply_aggregation,
    get_aggregator,
    list_aggregations,
)


class TestAggregationRegistry:
    """Test the aggregation registry."""

    def test_registry_has_expected_keys(self):
        """Registry should have all documented aggregations."""
        expected = {
            "mean", "sum", "min", "max", "std", "var", "first", "last", "count",
            "median", "p10", "p25", "p75", "p90", "p99", "skew", "kurtosis", "iqr",
            "range", "cv",
            "entropy", "instability", "trend", "n_unique",
        }
        assert expected.issubset(set(AGGREGATION_REGISTRY.keys()))

    def test_list_aggregations(self):
        """list_aggregations returns sorted list."""
        result = list_aggregations()
        assert result == sorted(result)
        assert "mean" in result
        assert "sum" in result

    def test_get_aggregator_valid(self):
        """get_aggregator returns callable for valid name."""
        agg = get_aggregator("mean")
        assert callable(agg)

    def test_get_aggregator_invalid(self):
        """get_aggregator raises ValueError for invalid name."""
        with pytest.raises(ValueError, match="Unknown aggregation 'invalid'"):
            get_aggregator("invalid")


class TestAggregatorFunctions:
    """Test individual aggregator functions."""

    @pytest.fixture
    def sample_df(self):
        """Sample DataFrame for testing."""
        return pl.DataFrame({
            "group": ["A", "A", "A", "B", "B", "B"],
            "value": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        })

    def test_mean(self, sample_df):
        """Test mean aggregation."""
        result = sample_df.select(AGGREGATION_REGISTRY["mean"]("value"))
        assert result.item() == pytest.approx(11.0)

    def test_sum(self, sample_df):
        """Test sum aggregation."""
        result = sample_df.select(AGGREGATION_REGISTRY["sum"]("value"))
        assert result.item() == pytest.approx(66.0)

    def test_min(self, sample_df):
        """Test min aggregation."""
        result = sample_df.select(AGGREGATION_REGISTRY["min"]("value"))
        assert result.item() == pytest.approx(1.0)

    def test_max(self, sample_df):
        """Test max aggregation."""
        result = sample_df.select(AGGREGATION_REGISTRY["max"]("value"))
        assert result.item() == pytest.approx(30.0)

    def test_first(self, sample_df):
        """Test first aggregation."""
        result = sample_df.select(AGGREGATION_REGISTRY["first"]("value"))
        assert result.item() == pytest.approx(1.0)

    def test_last(self, sample_df):
        """Test last aggregation."""
        result = sample_df.select(AGGREGATION_REGISTRY["last"]("value"))
        assert result.item() == pytest.approx(30.0)

    def test_count(self, sample_df):
        """Test count aggregation."""
        result = sample_df.select(AGGREGATION_REGISTRY["count"]("value"))
        assert result.item() == 6

    def test_median(self, sample_df):
        """Test median aggregation."""
        result = sample_df.select(AGGREGATION_REGISTRY["median"]("value"))
        assert result.item() == pytest.approx(6.5)

    def test_range(self, sample_df):
        """Test range aggregation."""
        result = sample_df.select(AGGREGATION_REGISTRY["range"]("value"))
        assert result.item() == pytest.approx(29.0)

    def test_n_unique(self, sample_df):
        """Test n_unique aggregation."""
        result = sample_df.select(AGGREGATION_REGISTRY["n_unique"]("group"))
        assert result.item() == 2


class TestApplyAggregation:
    """Test the apply_aggregation helper."""

    @pytest.fixture
    def sample_df(self):
        return pl.DataFrame({
            "group": ["A", "A", "B", "B"],
            "value": [1.0, 3.0, 10.0, 20.0],
        })

    def test_apply_without_groupby(self, sample_df):
        """Apply aggregation to whole column."""
        result = apply_aggregation(sample_df, "value", "mean")
        assert result.shape == (1, 1)
        assert result.item() == pytest.approx(8.5)

    def test_apply_with_groupby(self, sample_df):
        """Apply aggregation with group by."""
        result = apply_aggregation(sample_df, "value", "mean", group_by=["group"])
        assert result.shape == (2, 2)
        # Sort to ensure consistent order
        result = result.sort("group")
        assert result["value_mean"][0] == pytest.approx(2.0)  # A: (1+3)/2
        assert result["value_mean"][1] == pytest.approx(15.0)  # B: (10+20)/2


class TestSchemaAggregationValidation:
    """Test that schema validates aggregation names against registry."""

    def test_dimension_valid_aggregation(self):
        """Dimension accepts valid aggregation name."""
        from causal_agent.orchestrator.schemas import Dimension

        dim = Dimension(
            name="test",
            description="test var",
            time_granularity="daily",
            dtype="continuous",
            role="endogenous",
            aggregation="mean",
        )
        assert dim.aggregation == "mean"

    def test_dimension_invalid_aggregation(self):
        """Dimension rejects invalid aggregation name."""
        from causal_agent.orchestrator.schemas import Dimension

        with pytest.raises(ValueError, match="Unknown aggregation 'invalid'"):
            Dimension(
                name="test",
                description="test var",
                time_granularity="daily",
                dtype="continuous",
                role="endogenous",
                aggregation="invalid",
            )

    def test_edge_valid_aggregation(self):
        """CausalEdge accepts valid aggregation name."""
        from causal_agent.orchestrator.schemas import CausalEdge

        edge = CausalEdge(cause="X", effect="Y", lag=24, aggregation="sum")
        assert edge.aggregation == "sum"

    def test_edge_invalid_aggregation(self):
        """CausalEdge rejects invalid aggregation name."""
        from causal_agent.orchestrator.schemas import CausalEdge

        with pytest.raises(ValueError, match="Unknown aggregation 'bad_agg'"):
            CausalEdge(cause="X", effect="Y", lag=24, aggregation="bad_agg")
