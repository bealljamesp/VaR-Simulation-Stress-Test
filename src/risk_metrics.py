import polars as pl
from scipy.stats import jarque_bera


def non_normality_check(data: pl.DataFrame) -> pl.DataFrame:
    """
    Pure Polars implementation to verify non-normality across columns using the Jarque-Bera test.
    Explicitly wraps scalar outputs in a pl.Series to guarantee clean schema mapping.
    """
    if not isinstance(data, pl.DataFrame):
        raise TypeError("Engine Exception: Expected input to be a Polars DataFrame")

    # Isolate numerical columns and safely ignore 'date' regardless of casing
    numeric_cols = [
        col
        for col in data.columns
        if data[col].dtype.is_numeric() and col.lower() != "date"
    ]

    # Execute Jarque-Bera checks across all columns in parallel
    # Wrapping the boolean in pl.Series([ ... ]) completely bypasses the scalar UDF constraint
    return data.select(
        [
            pl.col(col)
            .map_batches(
                lambda s: pl.Series([jarque_bera(s.drop_nulls())[1] < 0.05]),
                return_dtype=pl.Boolean,
            )
            .alias(col)
            for col in numeric_cols
        ]
    )
