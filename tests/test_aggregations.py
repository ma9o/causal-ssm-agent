"""Tests for aggregation registry."""

import pytest
import polars as pl

from causal_agent.utils.aggregations import (
    AGGREGATION_REGISTRY,
    aggregate_worker_measurements,
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

    def test_indicator_valid_aggregation(self):
        """Indicator accepts valid aggregation name."""
        from causal_agent.orchestrator.schemas import Indicator

        ind = Indicator(
            name="test_indicator",
            construct="test",
            how_to_measure="Extract test from data",
            measurement_granularity="finest",
            measurement_dtype="continuous",
            aggregation="mean",
        )
        assert ind.aggregation == "mean"

    def test_indicator_invalid_aggregation(self):
        """Indicator rejects invalid aggregation name."""
        from causal_agent.orchestrator.schemas import Indicator

        with pytest.raises(ValueError, match="Unknown aggregation 'invalid'"):
            Indicator(
                name="test_indicator",
                construct="test",
                how_to_measure="Extract test from data",
                measurement_granularity="finest",
                measurement_dtype="continuous",
                aggregation="invalid",
            )


class TestAggregateWorkerMeasurements:
    """Test aggregate_worker_measurements function.

    The function returns a dict of DataFrames, one per causal_granularity:
    - Each DataFrame has time_bucket as index and dimensions as columns
    - Dimensions are grouped by their causal_granularity
    """

    @pytest.fixture
    def daily_schema(self):
        """Schema with daily causal_granularity dimensions."""
        return {
            "dimensions": [
                {
                    "name": "temperature",
                    "aggregation": "mean",
                    "observability": "observed",
                    "causal_granularity": "daily",
                },
                {
                    "name": "step_count",
                    "aggregation": "sum",
                    "observability": "observed",
                    "causal_granularity": "daily",
                },
                {
                    "name": "sleep_quality",
                    "observability": "latent",  # Should be skipped
                    "causal_granularity": "daily",
                },
            ]
        }

    @pytest.fixture
    def worker_dataframes(self):
        """Sample worker dataframes with timestamps.

        Data layout:
        - 2024-01-01: temperature=[20, 22], step_count=[5000, 2000]
        - 2024-01-02: temperature=[24], step_count=[3000, 4000]
        """
        df1 = pl.DataFrame({
            "dimension": ["temperature", "temperature", "step_count", "step_count"],
            "value": [20.0, 22.0, 5000, 3000],
            "timestamp": ["2024-01-01 10:00", "2024-01-01 14:00", "2024-01-01 08:00", "2024-01-02 09:00"],
        }, schema={"dimension": pl.Utf8, "value": pl.Object, "timestamp": pl.Utf8})

        df2 = pl.DataFrame({
            "dimension": ["temperature", "step_count", "step_count"],
            "value": [24.0, 2000, 4000],
            "timestamp": ["2024-01-02 12:00", "2024-01-01 20:00", "2024-01-02 18:00"],
        }, schema={"dimension": pl.Utf8, "value": pl.Object, "timestamp": pl.Utf8})

        return [df1, df2]

    def test_empty_dataframes_list(self, daily_schema):
        """Empty list returns empty dict."""
        result = aggregate_worker_measurements([], daily_schema)
        assert result == {}

    def test_returns_dict_by_granularity(self, daily_schema, worker_dataframes):
        """Result is a dict with granularity keys."""
        result = aggregate_worker_measurements(worker_dataframes, daily_schema)

        assert isinstance(result, dict)
        assert "daily" in result
        # Should have time_bucket column and dimension columns
        daily_df = result["daily"]
        assert "time_bucket" in daily_df.columns
        assert "temperature" in daily_df.columns
        assert "step_count" in daily_df.columns
        assert "sleep_quality" not in daily_df.columns  # latent

    def test_aggregates_within_time_buckets(self, daily_schema, worker_dataframes):
        """Values are aggregated within each time bucket using dimension's aggregation."""
        result = aggregate_worker_measurements(worker_dataframes, daily_schema)
        daily_df = result["daily"].sort("time_bucket")

        # 2024-01-01: temperature=[20, 22] -> mean = 21.0
        # 2024-01-01: step_count=[5000, 2000] -> sum = 7000
        jan1 = daily_df.filter(pl.col("time_bucket").dt.day() == 1)
        assert jan1["temperature"][0] == pytest.approx(21.0)
        assert jan1["step_count"][0] == pytest.approx(7000.0)

        # 2024-01-02: temperature=[24] -> mean = 24.0
        # 2024-01-02: step_count=[3000, 4000] -> sum = 7000
        jan2 = daily_df.filter(pl.col("time_bucket").dt.day() == 2)
        assert jan2["temperature"][0] == pytest.approx(24.0)
        assert jan2["step_count"][0] == pytest.approx(7000.0)

    def test_separates_different_granularities(self):
        """Dimensions with different granularities go to separate DataFrames."""
        schema = {
            "dimensions": [
                {
                    "name": "hourly_metric",
                    "aggregation": "mean",
                    "observability": "observed",
                    "causal_granularity": "hourly",
                },
                {
                    "name": "daily_metric",
                    "aggregation": "sum",
                    "observability": "observed",
                    "causal_granularity": "daily",
                },
            ]
        }

        df = pl.DataFrame({
            "dimension": ["hourly_metric", "hourly_metric", "hourly_metric", "daily_metric", "daily_metric"],
            "value": [10.0, 20.0, 30.0, 100.0, 200.0],
            "timestamp": [
                "2024-01-01 10:15",  # hourly bucket: 10:00
                "2024-01-01 10:45",  # hourly bucket: 10:00
                "2024-01-01 11:30",  # hourly bucket: 11:00
                "2024-01-01 10:00",  # daily bucket: 2024-01-01
                "2024-01-01 22:00",  # daily bucket: 2024-01-01
            ],
        }, schema={"dimension": pl.Utf8, "value": pl.Object, "timestamp": pl.Utf8})

        result = aggregate_worker_measurements([df], schema)

        # Should have separate keys for hourly and daily
        assert "hourly" in result
        assert "daily" in result

        # Hourly should have 2 rows (10:00 and 11:00)
        hourly_df = result["hourly"]
        assert hourly_df.height == 2
        assert "hourly_metric" in hourly_df.columns
        assert "daily_metric" not in hourly_df.columns

        # Daily should have 1 row
        daily_df = result["daily"]
        assert daily_df.height == 1
        assert "daily_metric" in daily_df.columns
        assert "hourly_metric" not in daily_df.columns

        # Check hourly aggregation: 10:00 has [10, 20] -> mean = 15
        hourly_df = hourly_df.sort("time_bucket")
        assert hourly_df["hourly_metric"][0] == pytest.approx(15.0)
        # 11:00 has [30] -> mean = 30
        assert hourly_df["hourly_metric"][1] == pytest.approx(30.0)

        # Check daily aggregation: [100, 200] -> sum = 300
        assert daily_df["daily_metric"][0] == pytest.approx(300.0)

    def test_skips_latent_dimensions(self, daily_schema):
        """Latent dimensions are not included in output."""
        df = pl.DataFrame({
            "dimension": ["sleep_quality", "temperature"],
            "value": [8.0, 20.0],
            "timestamp": ["2024-01-01 08:00", "2024-01-01 10:00"],
        }, schema={"dimension": pl.Utf8, "value": pl.Object, "timestamp": pl.Utf8})

        result = aggregate_worker_measurements([df], daily_schema)

        daily_df = result["daily"]
        assert "sleep_quality" not in daily_df.columns
        assert "temperature" in daily_df.columns

    def test_time_invariant_in_separate_key(self):
        """Time-invariant dimensions go to 'time_invariant' key."""
        schema = {
            "dimensions": [
                {
                    "name": "age",
                    "aggregation": "first",
                    "observability": "observed",
                    "causal_granularity": None,  # time-invariant
                },
                {
                    "name": "mood",
                    "aggregation": "mean",
                    "observability": "observed",
                    "causal_granularity": "daily",
                },
            ]
        }

        df = pl.DataFrame({
            "dimension": ["age", "age", "mood"],
            "value": [30, 31, 7.5],
            "timestamp": ["2024-01-01", "2024-01-02", "2024-01-01 10:00"],
        }, schema={"dimension": pl.Utf8, "value": pl.Object, "timestamp": pl.Utf8})

        result = aggregate_worker_measurements([df], schema)

        # Time-invariant should be in separate key
        assert "time_invariant" in result
        assert "daily" in result

        # time_invariant has no time_bucket, just aggregated values
        ti_df = result["time_invariant"]
        assert "age" in ti_df.columns
        assert ti_df.height == 1
        # first aggregation: 30
        assert ti_df["age"][0] == pytest.approx(30.0)

        # daily has mood
        daily_df = result["daily"]
        assert "mood" in daily_df.columns
        assert "age" not in daily_df.columns

    def test_handles_unparseable_timestamps(self, daily_schema):
        """Rows with unparseable timestamps are filtered out."""
        df = pl.DataFrame({
            "dimension": ["temperature", "temperature", "temperature"],
            "value": [20.0, 22.0, 24.0],
            "timestamp": ["2024-01-01 10:00", "invalid-date", "2024-01-01 14:00"],
        }, schema={"dimension": pl.Utf8, "value": pl.Object, "timestamp": pl.Utf8})

        result = aggregate_worker_measurements([df], daily_schema)

        # Should have one row for 2024-01-01 with mean of 20 and 24 = 22
        daily_df = result["daily"]
        assert daily_df.height == 1
        assert daily_df["temperature"][0] == pytest.approx(22.0)

    def test_handles_boolean_values(self):
        """Boolean values are converted to 0/1 for aggregation."""
        schema = {
            "dimensions": [
                {
                    "name": "is_active",
                    "aggregation": "mean",
                    "observability": "observed",
                    "causal_granularity": "daily",
                },
            ]
        }

        df = pl.DataFrame({
            "dimension": ["is_active", "is_active", "is_active", "is_active"],
            "value": [True, False, True, True],
            "timestamp": ["2024-01-01 08:00", "2024-01-01 10:00", "2024-01-01 12:00", "2024-01-01 14:00"],
        }, schema={"dimension": pl.Utf8, "value": pl.Object, "timestamp": pl.Utf8})

        result = aggregate_worker_measurements([df], schema)

        # Mean of [1, 0, 1, 1] = 0.75
        daily_df = result["daily"]
        assert daily_df["is_active"][0] == pytest.approx(0.75)

    def test_outer_join_fills_missing_dimensions(self):
        """Dimensions with no data for a time bucket get null values."""
        schema = {
            "dimensions": [
                {
                    "name": "dim_a",
                    "aggregation": "mean",
                    "observability": "observed",
                    "causal_granularity": "daily",
                },
                {
                    "name": "dim_b",
                    "aggregation": "mean",
                    "observability": "observed",
                    "causal_granularity": "daily",
                },
            ]
        }

        df = pl.DataFrame({
            "dimension": ["dim_a", "dim_a", "dim_b"],
            "value": [10.0, 20.0, 100.0],
            "timestamp": ["2024-01-01 10:00", "2024-01-02 10:00", "2024-01-01 10:00"],
        }, schema={"dimension": pl.Utf8, "value": pl.Object, "timestamp": pl.Utf8})

        result = aggregate_worker_measurements([df], schema)
        daily_df = result["daily"].sort("time_bucket")

        # 2024-01-01 has both dim_a=10 and dim_b=100
        # 2024-01-02 has dim_a=20 but dim_b should be null
        assert daily_df.height == 2
        jan2 = daily_df.filter(pl.col("time_bucket").dt.day() == 2)
        assert jan2["dim_a"][0] == pytest.approx(20.0)
        assert jan2["dim_b"][0] is None

