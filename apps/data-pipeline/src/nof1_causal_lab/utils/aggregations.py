"""Dtype encoding and aggregation helpers for extracted indicator data.

Provides non-continuous dtype encoding (binary, ordinal, categorical -> numeric)
and Polars aggregation expression builders used by the pipeline's stage 2 logic.
"""

import ast
import logging

import numpy as np
import polars as pl

from nof1_causal_lab.json_types import UncheckedJsonObject
from nof1_causal_lab.utils.data import ensure_datetime_column, support_window_tick_frame

logger = logging.getLogger(__name__)

# Aggregations that require map_groups (cannot be expressed as a single Polars expr)
_MAP_GROUPS_AGGREGATIONS = {"trend"}
COMPUTED_RULE_FUNCTIONS = {
    "abs",
    "all",
    "any",
    "coalesce",
    "contains",
    "contains_any",
    "count_non_null",
    "count_true",
    "first",
    "last",
    "lower",
    "max",
    "mean",
    "min",
    "std",
    "sum",
}


def _compile_computed_rule_expr(window_expr: str, *, allowed_names: set[str]) -> pl.Expr:
    """Compile a deterministic computed-rule expression to a Polars expression."""
    try:
        parsed = ast.parse(window_expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid computed_rule.window_expr: {exc.msg}") from exc
    return _compile_computed_rule_node(parsed.body, allowed_names=allowed_names)


def _compile_computed_rule_node(node: ast.AST, *, allowed_names: set[str]) -> pl.Expr:
    """Recursively compile an allowed AST node into a Polars expression."""
    if isinstance(node, ast.Constant):
        return pl.lit(node.value)

    if isinstance(node, ast.Name):
        if node.id not in allowed_names:
            raise ValueError(
                f"computed_rule.window_expr references unknown source column '{node.id}'"
            )
        return pl.col(node.id)

    if isinstance(node, ast.BinOp):
        left = _compile_computed_rule_node(node.left, allowed_names=allowed_names)
        right = _compile_computed_rule_node(node.right, allowed_names=allowed_names)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Mod):
            return left % right
        if isinstance(node.op, ast.Pow):
            return left.pow(right)
        raise ValueError(f"Unsupported computed_rule binary operator: {type(node.op).__name__}")

    if isinstance(node, ast.UnaryOp):
        operand = _compile_computed_rule_node(node.operand, allowed_names=allowed_names)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.Not):
            return ~operand.fill_null(False)
        raise ValueError(f"Unsupported computed_rule unary operator: {type(node.op).__name__}")

    if isinstance(node, ast.BoolOp):
        values = [
            _compile_computed_rule_node(value, allowed_names=allowed_names) for value in node.values
        ]
        if not values:
            raise ValueError("computed_rule.window_expr boolean expressions cannot be empty")
        result = values[0]
        if isinstance(node.op, ast.And):
            for value in values[1:]:
                result = result & value
            return result
        if isinstance(node.op, ast.Or):
            for value in values[1:]:
                result = result | value
            return result
        raise ValueError(f"Unsupported computed_rule boolean operator: {type(node.op).__name__}")

    if isinstance(node, ast.Compare):
        return _compile_computed_rule_compare(node, allowed_names=allowed_names)

    if isinstance(node, ast.IfExp):
        return (
            pl.when(_compile_computed_rule_node(node.test, allowed_names=allowed_names))
            .then(_compile_computed_rule_node(node.body, allowed_names=allowed_names))
            .otherwise(_compile_computed_rule_node(node.orelse, allowed_names=allowed_names))
        )

    if isinstance(node, ast.Call):
        return _compile_computed_rule_call(node, allowed_names=allowed_names)

    raise ValueError(f"Unsupported computed_rule syntax: {type(node).__name__}")


def _compile_computed_rule_compare(node: ast.Compare, *, allowed_names: set[str]) -> pl.Expr:
    """Compile comparison expressions, including chained comparisons."""
    left = _compile_computed_rule_node(node.left, allowed_names=allowed_names)
    result: pl.Expr | None = None
    current_left = left

    for op, comparator_node in zip(node.ops, node.comparators, strict=True):
        if isinstance(op, (ast.In, ast.NotIn)):
            values = _literal_values_from_ast(comparator_node)
            current = current_left.is_in(values)
            if isinstance(op, ast.NotIn):
                current = ~current
        elif isinstance(op, (ast.Is, ast.IsNot)):
            if not _is_none_literal(comparator_node):
                raise ValueError(
                    "computed_rule.window_expr only supports 'is None' and 'is not None'"
                )
            current = (
                current_left.is_null() if isinstance(op, ast.Is) else current_left.is_not_null()
            )
        else:
            comparator = _compile_computed_rule_node(comparator_node, allowed_names=allowed_names)
            if isinstance(op, ast.Eq):
                current = current_left == comparator
            elif isinstance(op, ast.NotEq):
                current = current_left != comparator
            elif isinstance(op, ast.Lt):
                current = current_left < comparator
            elif isinstance(op, ast.LtE):
                current = current_left <= comparator
            elif isinstance(op, ast.Gt):
                current = current_left > comparator
            elif isinstance(op, ast.GtE):
                current = current_left >= comparator
            else:
                raise ValueError(
                    f"Unsupported computed_rule comparison operator: {type(op).__name__}"
                )
            current_left = comparator

        result = current if result is None else result & current

    if result is None:
        raise ValueError("computed_rule.window_expr comparison cannot be empty")
    return result


def _compile_computed_rule_call(node: ast.Call, *, allowed_names: set[str]) -> pl.Expr:
    """Compile supported helper functions used in computed_rule.window_expr."""
    if not isinstance(node.func, ast.Name):
        raise ValueError("computed_rule.window_expr only supports simple function calls")
    name = node.func.id
    if name not in COMPUTED_RULE_FUNCTIONS:
        available = ", ".join(sorted(COMPUTED_RULE_FUNCTIONS))
        raise ValueError(f"Unsupported computed_rule function '{name}'. Available: {available}")
    if node.keywords:
        raise ValueError("computed_rule.window_expr does not support keyword arguments")

    if name == "contains" and len(node.args) != 2:
        raise ValueError("computed_rule function 'contains' expects exactly 2 arguments")
    if name == "contains_any" and len(node.args) != 2:
        raise ValueError("computed_rule function 'contains_any' expects exactly 2 arguments")
    if name == "coalesce" and len(node.args) < 2:
        raise ValueError("computed_rule function 'coalesce' expects at least 2 arguments")

    if name == "contains":
        haystack = _compile_computed_rule_node(node.args[0], allowed_names=allowed_names)
        pattern = _string_literal_from_ast(node.args[1], fn_name="contains")
        return (
            haystack.cast(pl.Utf8, strict=False)
            .str.to_lowercase()
            .str.contains(pattern.lower(), literal=True)
        )
    if name == "contains_any":
        haystack = _compile_computed_rule_node(node.args[0], allowed_names=allowed_names)
        patterns = _string_list_literal_from_ast(node.args[1], fn_name="contains_any")
        if not patterns:
            return pl.lit(False)
        result: pl.Expr | None = None
        haystack = haystack.cast(pl.Utf8, strict=False).str.to_lowercase()
        for pattern in patterns:
            current = haystack.str.contains(pattern.lower(), literal=True)
            result = current if result is None else result | current
        if result is None:
            return pl.lit(False)
        return result

    args = [_compile_computed_rule_node(arg, allowed_names=allowed_names) for arg in node.args]
    if (
        name
        in {
            "abs",
            "all",
            "any",
            "count_non_null",
            "count_true",
            "first",
            "last",
            "lower",
            "max",
            "mean",
            "min",
            "std",
            "sum",
        }
        and len(args) != 1
    ):
        raise ValueError(f"computed_rule function '{name}' expects exactly 1 argument")

    if name == "abs":
        return args[0].abs()
    if name == "all":
        return (
            pl.when(args[0].is_not_null().any())
            .then(args[0].fill_null(False).all())
            .otherwise(None)
        )
    if name == "any":
        return args[0].fill_null(False).any()
    if name == "coalesce":
        return pl.coalesce(args)
    if name == "count_non_null":
        return args[0].is_not_null().cast(pl.Int64).sum()
    if name == "count_true":
        return args[0].fill_null(False).cast(pl.Int64).sum()
    if name == "first":
        return args[0].drop_nulls().first()
    if name == "last":
        return args[0].drop_nulls().last()
    if name == "lower":
        return args[0].cast(pl.Utf8, strict=False).str.to_lowercase()
    if name == "max":
        return args[0].max()
    if name == "mean":
        return args[0].mean()
    if name == "min":
        return args[0].min()
    if name == "std":
        return args[0].std()
    if name == "sum":
        return pl.when(args[0].is_not_null().any()).then(args[0].sum()).otherwise(None)

    raise ValueError(f"Unhandled computed_rule function '{name}'")


def _literal_values_from_ast(node: ast.AST) -> list[object]:
    """Extract a literal list/tuple/set from an AST node."""
    if not isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        raise ValueError(
            "computed_rule.window_expr 'in' comparisons require a literal list/tuple/set"
        )
    values: list[object] = []
    for element in node.elts:
        if not isinstance(element, ast.Constant):
            raise ValueError(
                "computed_rule.window_expr literal collections support only constant values"
            )
        values.append(element.value)
    return values


def _string_literal_from_ast(node: ast.AST, *, fn_name: str) -> str:
    """Extract a literal string from an AST node."""
    if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
        raise ValueError(f"computed_rule function '{fn_name}' requires a literal string pattern")
    return node.value


def _string_list_literal_from_ast(node: ast.AST, *, fn_name: str) -> list[str]:
    """Extract a literal list of strings from an AST node."""
    values = _literal_values_from_ast(node)
    if not all(isinstance(value, str) for value in values):
        raise ValueError(f"computed_rule function '{fn_name}' requires a literal list of strings")
    return [str(value) for value in values]


def _is_none_literal(node: ast.AST) -> bool:
    """Return whether an AST node is the literal None."""
    return isinstance(node, ast.Constant) and node.value is None


def _build_agg_expr(agg_name: str, col_name: str = "value") -> pl.Expr:
    """Map an aggregation name to a Polars expression over a named column.

    Supports 23 of 24 aggregation functions as expressions. The 'trend'
    aggregation requires map_groups and is handled separately via
    _build_map_groups_fn.

    Args:
        agg_name: Name of the aggregation function.
        col_name: Column to aggregate (default: "value").

    Returns:
        Polars expression aliased to "value".
    """
    col = pl.col(col_name)

    simple = {
        "mean": col.mean(),
        "sum": pl.when(col.is_not_null().any()).then(col.sum()).otherwise(None),
        "min": col.min(),
        "max": col.max(),
        "std": col.std(),
        "var": col.var(),
        "last": col.drop_nulls().last(),
        "first": col.drop_nulls().first(),
        "count": col.drop_nulls().count(),
        "median": col.median(),
        "n_unique": col.n_unique(),
        "skew": col.skew(),
        "kurtosis": col.kurtosis(),
        "entropy": col.entropy(),
    }

    if agg_name in simple:
        return simple[agg_name].alias("value")

    # Percentiles
    percentiles = {
        "p10": 0.10,
        "p25": 0.25,
        "p75": 0.75,
        "p90": 0.90,
        "p99": 0.99,
    }
    if agg_name in percentiles:
        q = percentiles[agg_name]
        return col.quantile(q).alias("value")

    # Composite aggregations
    if agg_name == "range":
        return (col.max() - col.min()).alias("value")

    if agg_name == "iqr":
        return (col.quantile(0.75) - col.quantile(0.25)).alias("value")

    if agg_name == "cv":
        return (
            pl.when(col.mean().abs() > 1e-15).then(col.std() / col.mean()).otherwise(None)
        ).alias("value")

    # MSSD: mean squared successive differences
    if agg_name == "instability":
        return (col.diff().pow(2).mean()).alias("value")

    raise ValueError(f"Unknown aggregation function: '{agg_name}'")


def _build_map_groups_fn(agg_name: str):
    """Return a callable for use with group_by().map_groups().

    Used for aggregations that cannot be expressed as a single Polars expression.
    """
    if agg_name == "trend":

        def _ols_slope(df: pl.DataFrame) -> pl.DataFrame:
            values = df["value"].drop_nulls().to_numpy()
            n = len(values)
            if n == 0:
                slope = None
            elif n < 2:
                slope = 0.0
            else:
                x = np.arange(n, dtype=np.float64)
                slope = float(np.polyfit(x, values, 1)[0])
            return df.head(1).with_columns(pl.lit(slope).alias("value"))

        return _ols_slope

    raise ValueError(f"Unknown map_groups aggregation: '{agg_name}'")


def compute_indicators(
    raw_df: pl.DataFrame,
    indicators: list[UncheckedJsonObject],
    model_clock: str,
    time_col: str,
) -> pl.DataFrame:
    """Compute indicator values directly via Polars aggregation.

    For indicators with extraction_mode='computed', applies a deterministic
    support-window computation grouped by each indicator's effective
    observation window (explicit observation_window or fallback model_clock).
    Direct single-column aggregations are supported, along with computed_rule
    expressions that deterministically derive one scalar per support window
    from one or more source columns. Non-numeric direct columns are supported
    for point aggregations (`first`/`last`), and ordinal direct columns are
    converted to their declared integer codes before emission.

    Args:
        raw_df: Raw wide-format DataFrame with actual column names.
        indicators: List of indicator dicts with extraction_mode="computed".
            Each must have exactly one directly aggregated source column.
        model_clock: Global fallback duration string for truncation (e.g., "1d").
        time_col: Name of the datetime column in raw_df.

    Returns:
        Long-format DataFrame with columns: indicator (Utf8), value (Utf8),
        timestamp (Utf8). Matches the schema produced by the semantic path.
    """
    output_schema = {"indicator": pl.Utf8, "value": pl.Utf8, "timestamp": pl.Utf8}
    if not indicators:
        return pl.DataFrame(schema=output_schema)

    df = ensure_datetime_column(raw_df, time_col)

    frames: list[pl.DataFrame] = []
    for ind in indicators:
        name = ind["name"]
        agg_name = ind["aggregation"]
        measurement_dtype = ind.get("measurement_dtype", "continuous")
        observation_window = ind.get("observation_window") or model_clock
        source_columns = list(ind.get("source_columns", []))
        computed_rule = ind.get("computed_rule")

        if not source_columns:
            logger.warning(
                "Computed indicator '%s': no source_columns declared, skipping",
                name,
            )
            continue
        missing_source_cols = [column for column in source_columns if column not in df.columns]
        if missing_source_cols:
            logger.warning(
                "Computed indicator '%s': source columns %s not in DataFrame, skipping",
                name,
                missing_source_cols,
            )
            continue

        if computed_rule:
            tick_frame = support_window_tick_frame(df, observation_window, time_col)
            prepared = _prepare_computed_rule_frame(
                df,
                time_col=time_col,
                source_columns=source_columns,
                observation_window=observation_window,
            )
            prepared = _with_dense_support_rows(prepared, tick_frame)
            expr = _missing_window_guard(
                _compile_computed_rule_expr(
                    computed_rule["window_expr"],
                    allowed_names=set(source_columns),
                )
            )
            agg_df = prepared.group_by("__tick__", maintain_order=True).agg(expr)
        else:
            source_col = source_columns[0]
            tick_frame = support_window_tick_frame(df, observation_window, time_col)
            prepared = _prepare_computed_indicator_frame(
                df,
                time_col=time_col,
                source_col=source_col,
                observation_window=observation_window,
                measurement_dtype=measurement_dtype,
                ordinal_levels=ind.get("ordinal_levels"),
            )
            prepared = _with_dense_support_rows(prepared, tick_frame)

            if agg_name in _MAP_GROUPS_AGGREGATIONS:
                # trend etc: rename source_col → "value" for map_groups function
                fn = _build_map_groups_fn(agg_name)
                agg_df = (
                    prepared.select(
                        "__tick__",
                        pl.col("__value__").cast(pl.Float64, strict=False).alias("value"),
                    )
                    .sort("__tick__")
                    .group_by("__tick__", maintain_order=True)
                    .map_groups(fn)
                )
            else:
                expr = _build_dense_agg_expr(agg_name, "__value__")
                agg_df = prepared.group_by("__tick__", maintain_order=True).agg(expr)

        agg_df = agg_df.select(
            pl.lit(name).alias("indicator"),
            pl.col("value").cast(pl.Utf8).alias("value"),
            pl.col("__tick__").dt.to_string("%Y-%m-%dT%H:%M:%S").alias("timestamp"),
        )
        frames.append(agg_df)

    if not frames:
        return pl.DataFrame(schema=output_schema)

    return pl.concat(frames, how="vertical").sort("timestamp", "indicator")


def _missing_window_guard(expr: pl.Expr) -> pl.Expr:
    """Return null for synthetic windows with no raw rows."""
    return (
        pl.when(pl.col("__observed_row__").fill_null(False).any())
        .then(expr)
        .otherwise(None)
        .alias("value")
    )


def _build_dense_agg_expr(agg_name: str, col_name: str) -> pl.Expr:
    """Build aggregation over a dense support grid with missing-window nulls."""
    if agg_name == "count":
        col = pl.col(col_name)
        return (
            pl.when(pl.col("__observed_row__").fill_null(False).any())
            .then(col.drop_nulls().count())
            .otherwise(None)
            .alias("value")
        )
    return _build_agg_expr(agg_name, col_name)


def _prepare_computed_indicator_frame(
    df: pl.DataFrame,
    *,
    time_col: str,
    source_col: str,
    observation_window: str,
    measurement_dtype: str,
    ordinal_levels: list[str] | None,
) -> pl.DataFrame:
    """Prepare a computed indicator's source values for deterministic aggregation."""
    value_expr = _computed_value_expr(
        source_col,
        measurement_dtype=measurement_dtype,
        ordinal_levels=ordinal_levels,
    ).alias("__value__")
    return df.select(
        pl.col(time_col).dt.truncate(observation_window).alias("__tick__"),
        value_expr,
    )


def _with_dense_support_rows(prepared: pl.DataFrame, tick_frame: pl.DataFrame) -> pl.DataFrame:
    """Add null-valued placeholder rows for support windows with no raw rows."""
    if tick_frame.is_empty():
        return prepared.with_columns(pl.lit(True).alias("__observed_row__"))
    observed = prepared.with_columns(pl.lit(True).alias("__observed_row__"))
    return tick_frame.join(observed, on="__tick__", how="left")


def _prepare_computed_rule_frame(
    df: pl.DataFrame,
    *,
    time_col: str,
    source_columns: list[str],
    observation_window: str,
) -> pl.DataFrame:
    """Prepare source columns for a deterministic support-window computed rule."""
    return df.select(
        pl.col(time_col).dt.truncate(observation_window).alias("__tick__"),
        *[pl.col(column) for column in source_columns],
    )


def _computed_value_expr(
    source_col: str,
    *,
    measurement_dtype: str,
    ordinal_levels: list[str] | None,
) -> pl.Expr:
    """Build the deterministic source-value expression for a computed indicator."""
    source = pl.col(source_col)
    if measurement_dtype == "ordinal":
        max_code = len(ordinal_levels or []) - 1
        label_map = {
            str(level).strip().lower(): idx for idx, level in enumerate(ordinal_levels or [])
        }
        return source.map_elements(
            lambda value, _max=max_code, _label_map=label_map: _coerce_ordinal_code(
                value, _label_map, _max
            ),
            return_dtype=pl.Int64,
        )
    return source


def _coerce_ordinal_code(
    value: object,
    label_map: dict[str, int],
    max_code: int,
) -> int | None:
    """Normalize an ordinal source value to its canonical integer code."""
    if value is None or isinstance(value, bool):
        return None

    code: int | None = None
    if isinstance(value, int):
        code = value
    elif isinstance(value, float):
        if value.is_integer():
            code = int(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            numeric = float(stripped)
        except ValueError:
            code = label_map.get(stripped.lower())
        else:
            if numeric.is_integer():
                code = int(numeric)

    if code is None:
        return None
    if code < 0:
        return None
    if max_code >= 0 and code > max_code:
        return None
    return code


_BINARY_TRUE = {"true", "yes", "1", "1.0", "t", "y"}
_BINARY_FALSE = {"false", "no", "0", "0.0", "f", "n"}


def _encode_non_continuous(
    df: pl.DataFrame,
    dtype_lookup: dict[str, str],
    ordinal_levels_lookup: dict[str, list[str]] | None = None,
) -> pl.DataFrame:
    """Encode non-continuous indicator values to numeric before Float64 cast.

    - binary: map true/false/yes/no/1/0 → 1.0/0.0
    - ordinal: integer label-encode using ordinal_levels order (or sorted fallback)
    - categorical: integer label-encode (sorted categories)
    - continuous/count: no-op (already numeric)

    Modifies the 'value' column in-place per indicator partition.
    """
    if not dtype_lookup:
        return df

    ordinal_levels_lookup = ordinal_levels_lookup or {}

    non_continuous = {
        name: dtype
        for name, dtype in dtype_lookup.items()
        if dtype in ("binary", "ordinal", "categorical")
    }
    if not non_continuous:
        return df

    # Ensure value is Utf8 for string matching
    if df.schema.get("value") != pl.Utf8:
        df = df.with_columns(pl.col("value").cast(pl.Utf8, strict=False))

    frames = []
    remaining_mask = pl.lit(True)

    for name, dtype in non_continuous.items():
        indicator_mask = pl.col("indicator") == name
        subset = df.filter(indicator_mask)
        if subset.is_empty():
            continue

        remaining_mask = remaining_mask & ~indicator_mask

        if dtype == "binary":
            subset = subset.with_columns(
                pl.col("value")
                .str.to_lowercase()
                .map_elements(
                    lambda v: 1.0 if v in _BINARY_TRUE else (0.0 if v in _BINARY_FALSE else None),
                    return_dtype=pl.Float64,
                )
                .alias("value")
            )
            n_null = subset["value"].null_count()
            if n_null > 0:
                logger.warning(
                    "Binary indicator '%s': %d/%d values could not be encoded",
                    name,
                    n_null,
                    len(subset),
                )
        elif dtype == "ordinal":
            explicit_levels = ordinal_levels_lookup.get(name)
            max_code = len(explicit_levels) - 1 if explicit_levels else None
            subset = (
                subset.with_columns(
                    pl.col("value").cast(pl.Float64, strict=False).alias("__ordinal_code")
                )
                .with_columns(
                    pl.when(pl.col("__ordinal_code").is_null())
                    .then(None)
                    .when(pl.col("__ordinal_code") != pl.col("__ordinal_code").round(0))
                    .then(None)
                    .when(pl.col("__ordinal_code") < 0)
                    .then(None)
                    .when(
                        pl.lit(max_code is not None)
                        & (pl.col("__ordinal_code") > pl.lit(max_code or 0))
                    )
                    .then(None)
                    .otherwise(pl.col("__ordinal_code"))
                    .alias("value")
                )
                .drop("__ordinal_code")
            )
            n_null = subset["value"].null_count()
            if n_null > 0:
                logger.warning(
                    "Ordinal indicator '%s': %d/%d values could not be encoded",
                    name,
                    n_null,
                    len(subset),
                )
        else:
            # ordinal/categorical: label encoding
            unique_vals = sorted(v for v in subset["value"].unique().to_list() if v is not None)
            # Normalize for case-insensitive matching (mirrors binary branch)
            label_map = {
                v.strip().lower() if isinstance(v, str) else v: float(i)
                for i, v in enumerate(unique_vals)
            }
            subset = subset.with_columns(
                pl.col("value")
                .str.strip_chars()
                .str.to_lowercase()
                .map_elements(lambda v, _lm=label_map: _lm.get(v), return_dtype=pl.Float64)
                .alias("value")
            )
            logger.info(
                "%s indicator '%s': label-encoded %d categories",
                dtype.capitalize(),
                name,
                len(unique_vals),
            )

        # Cast value back to Utf8 for consistency with remaining data
        subset = subset.with_columns(pl.col("value").cast(pl.Utf8, strict=False))
        frames.append(subset)

    if not frames:
        return df

    remaining = df.filter(remaining_mask)
    return pl.concat([remaining, *frames], how="vertical")
