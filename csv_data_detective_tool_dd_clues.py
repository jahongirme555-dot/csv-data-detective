#!/usr/bin/env python
"""
CSV Data Detective Tool (DD Clues)
--------------------------------
A single-file Python tool to analyze CSV datasets.

Features:
- Load CSV
- Full data preview
- Descriptive statistics (numeric & categorical)
- Anomaly detection (IQR & Z-score)
- Unique counts for categorical columns
- Filtering by conditions
- Group by & aggregation
- Date/time trend analysis
- Visualizations (bar, line, histogram)
- Export transformed data (e.g., anomalies)

Usage:
  python dd_tool.py data.csv

Optional:
  Edit the CONFIG section below to customize behavior.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# =====================
# CONFIGURATION
# =====================
CONFIG = {
    "preview_rows": 50,
    "anomaly_method": "iqr",  # 'iqr' or 'zscore'
    "zscore_threshold": 3.0,
    "output_dir": "outputs"
}

# =====================
# CORE FUNCTIONS
# =====================

def load_csv(path):
    df = pd.read_csv(path)
    print(f"Loaded dataset with shape: {df.shape}")
    return df


def preview_data(df):
    print("\n=== DATA PREVIEW ===")
    print(df.head(CONFIG["preview_rows"]))


def descriptive_stats(df):
    print("\n=== DESCRIPTIVE STATISTICS (NUMERIC) ===")
    print(df.describe())

    print("\n=== DESCRIPTIVE STATISTICS (CATEGORICAL) ===")
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        print(f"\nColumn: {col}")
        print(df[col].describe())


def unique_counts(df):
    print("\n=== UNIQUE COUNTS (CATEGORICAL) ===")
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        print(f"{col}: {df[col].nunique()} unique values")


def find_anomalies(df):
    print("\n=== ANOMALY DETECTION ===")
    numeric_cols = df.select_dtypes(include=np.number).columns
    anomalies = {}

    for col in numeric_cols:
        series = df[col].dropna()
        if CONFIG["anomaly_method"] == "iqr":
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            mask = (df[col] < lower) | (df[col] > upper)
        else:  # z-score
            z = (series - series.mean()) / series.std()
            mask = abs(z) > CONFIG["zscore_threshold"]

        anomalies[col] = df[mask]
        print(f"{col}: {mask.sum()} anomalies")

    return anomalies


def filter_data(df, condition):
    print("\n=== FILTERED DATA ===")
    filtered = df.query(condition)
    print(filtered.head())
    return filtered


def group_and_aggregate(df, group_col, agg_col, agg_func="mean"):
    print("\n=== GROUP BY & AGGREGATE ===")
    result = df.groupby(group_col)[agg_col].agg(agg_func)
    print(result)
    return result


def datetime_analysis(df, date_col, value_col):
    print("\n=== DATE/TIME ANALYSIS ===")
    df[date_col] = pd.to_datetime(df[date_col])
    trend = df.set_index(date_col)[value_col].resample("M").mean()
    print(trend.tail())

    plt.figure()
    trend.plot(title=f"Trend of {value_col} over time")
    plt.tight_layout()
    plt.show()


def visualize(df):
    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        plt.figure()
        df[col].hist(bins=30)
        plt.title(f"Histogram of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()


def export_anomalies(anomalies):
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(exist_ok=True)

    for col, data in anomalies.items():
        if not data.empty:
            path = output_dir / f"anomalies_{col}.csv"
            data.to_csv(path, index=False)
            print(f"Exported anomalies for {col} -> {path}")


# =====================
# MAIN
# =====================

def main():
    if len(sys.argv) < 2:
        print("Usage: python dd_tool.py <data.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    df = load_csv(csv_path)

    preview_data(df)
    descriptive_stats(df)
    unique_counts(df)

    anomalies = find_anomalies(df)
    export_anomalies(anomalies)

    visualize(df)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
