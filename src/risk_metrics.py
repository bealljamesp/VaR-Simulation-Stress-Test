import polars as pl


# Vectorized verification of non-normality using Polars expressions
def non_normality_check(df_clean):
    return df_clean.select(
        [
            pl.col("Date"),
            # Calculate skewness and excess kurtosis using parallelized expressions
            pl.all().exclude("Date").skew().alias("skewness"),
            pl.all().exclude("Date").kurtosis().alias("excess_kurtosis"),
        ]
    ).with_columns(
        [
            # Jarque-Bera test formula component: N/6 * [S^2 + (K^2 / 4)]
            (
                (pl.col("Lo 20").count() / 6)
                * (pl.col("skewness") ** 2 + (pl.col("excess_kurtosis") ** 2 / 4))
            ).alias("JB_Statistic")
        ]
    )
