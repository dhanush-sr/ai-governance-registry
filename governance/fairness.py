"""Group-fairness metrics computed with Spark.

The CI gate and the at-scale audit run this same code: one Spark scan
produces counts per group, and the metrics are taken from those counts.
"""
import os

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


def spark(cores="*", app="governance"):
    os.environ.setdefault("PYARROW_IGNORE_TIMEZONE", "1")
    return (SparkSession.builder.master(f"local[{cores}]").appName(app)
            .config("spark.driver.memory", os.environ.get("SPARK_DRIVER_MEMORY", "4g"))
            .config("spark.ui.enabled", os.environ.get("SPARK_UI", "false"))  # SPARK_UI=true to watch jobs at :4040
            .config("spark.hadoop.fs.defaultFS", "file:///")  # ignore any local HADOOP_CONF_DIR
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .getOrCreate())


def group_counts(df: DataFrame, keys: list[str]) -> pd.DataFrame:
    """One pass: confusion-matrix counts for every combination of `keys`."""
    return (df.groupBy(*keys)
              .agg(F.count("*").alias("n"),
                   F.sum("y_pred").alias("selected"),
                   F.sum("y_true").alias("positives"),
                   F.sum(F.col("y_true") * F.col("y_pred")).alias("true_pos"))
              .toPandas())


def metrics(counts: pd.DataFrame, attr: str, min_group_size: int = 0) -> dict:
    """Disparate impact and equal-opportunity gap for one attribute.

    counts may be finer-grained (e.g. per state or intersectional); it is
    summed up to `attr` first. Groups below min_group_size rows (or, for the
    TPR gap, below min_group_size positives) are reported but not gated on,
    because their rates are too noisy.
    """
    g = counts.groupby(attr)[["n", "selected", "positives", "true_pos"]].sum()
    g["selection_rate"] = g.selected / g.n
    g["tpr"] = g.true_pos / g.positives.where(g.positives > 0)
    used = g[g.n >= min_group_size]
    tpr = used.tpr[used.positives >= min_group_size]
    return {
        "disparate_impact": float(used.selection_rate.min() / used.selection_rate.max()),
        "equal_opportunity_diff": float(tpr.max() - tpr.min()),
        "groups": g.reset_index().to_dict("records"),
        "excluded_groups": g.index[g.n < min_group_size].tolist(),
    }
