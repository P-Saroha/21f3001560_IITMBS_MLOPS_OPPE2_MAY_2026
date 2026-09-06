import numpy as np
import pandas as pd

from scipy.stats import ks_2samp


TRAINING_FILE = "data/data.csv"
INCOMING_FILE = "D5/random_100_rows.csv"
OUTPUT_FILE = "D7/drift_report.csv"


# ============================================================
# Load datasets
# ============================================================

train_df = pd.read_csv(TRAINING_FILE)
incoming_df = pd.read_csv(INCOMING_FILE)

print("=" * 80)
print("INPUT DRIFT DETECTION")
print("=" * 80)

print(f"Training dataset : {train_df.shape}")
print(f"Incoming dataset : {incoming_df.shape}")


# ============================================================
# Features
# ============================================================

features = [
    "sno",
    "age",
    "gender",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal"
]


# Remove target from training data
train_df = train_df[features]
incoming_df = incoming_df[features]


# ============================================================
# Feature types
# ============================================================

categorical_features = [
    "gender",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal"
]

numerical_features = [
    "sno",
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak"
]


# ============================================================
# PSI for categorical variables
# ============================================================

def categorical_psi(expected, actual):

    expected = pd.Series(expected).astype(str)
    actual = pd.Series(actual).astype(str)

    categories = sorted(
        set(expected.unique()) | set(actual.unique())
    )

    expected_dist = (
        expected.value_counts(normalize=True)
        .reindex(categories, fill_value=0)
    )

    actual_dist = (
        actual.value_counts(normalize=True)
        .reindex(categories, fill_value=0)
    )

    # Prevent division by zero
    expected_dist = np.clip(expected_dist, 0.0001, None)
    actual_dist = np.clip(actual_dist, 0.0001, None)

    psi = np.sum(
        (actual_dist - expected_dist)
        * np.log(actual_dist / expected_dist)
    )

    return float(psi)


# ============================================================
# PSI for numerical variables
# ============================================================

def numerical_psi(expected, actual, bins=10):

    expected = pd.Series(expected).dropna()
    actual = pd.Series(actual).dropna()

    # Quantile-based bins from training data
    quantiles = np.linspace(0, 1, bins + 1)

    edges = np.unique(
        np.quantile(expected, quantiles)
    )

    # Not enough unique values
    if len(edges) < 3:
        return np.nan

    # Make first and last bins open-ended
    edges[0] = -np.inf
    edges[-1] = np.inf

    expected_binned = pd.cut(
        expected,
        bins=edges,
        include_lowest=True
    )

    actual_binned = pd.cut(
        actual,
        bins=edges,
        include_lowest=True
    )

    categories = expected_binned.cat.categories

    expected_dist = (
        expected_binned
        .value_counts(normalize=True)
        .reindex(categories, fill_value=0)
    )

    actual_dist = (
        actual_binned
        .value_counts(normalize=True)
        .reindex(categories, fill_value=0)
    )

    expected_dist = np.clip(expected_dist, 0.0001, None)
    actual_dist = np.clip(actual_dist, 0.0001, None)

    psi = np.sum(
        (actual_dist - expected_dist)
        * np.log(actual_dist / expected_dist)
    )

    return float(psi)


# ============================================================
# Drift calculation
# ============================================================

results = []

for feature in features:

    train_values = train_df[feature]
    incoming_values = incoming_df[feature]

    # --------------------------------------------------------
    # Numerical feature
    # --------------------------------------------------------

    if feature in numerical_features:

        ks_statistic, p_value = ks_2samp(
            train_values.dropna(),
            incoming_values.dropna()
        )

        psi = numerical_psi(
            train_values,
            incoming_values
        )

        drift_detected = (
            p_value < 0.05
            or (
                not np.isnan(psi)
                and psi >= 0.20
            )
        )

        results.append({
            "feature": feature,
            "type": "numerical",
            "ks_statistic": round(float(ks_statistic), 6),
            "p_value": round(float(p_value), 6),
            "psi": round(float(psi), 6)
            if not np.isnan(psi) else np.nan,
            "drift_detected": drift_detected
        })

    # --------------------------------------------------------
    # Categorical feature
    # --------------------------------------------------------

    else:

        psi = categorical_psi(
            train_values,
            incoming_values
        )

        drift_detected = psi >= 0.20

        results.append({
            "feature": feature,
            "type": "categorical",
            "ks_statistic": np.nan,
            "p_value": np.nan,
            "psi": round(float(psi), 6),
            "drift_detected": drift_detected
        })


# ============================================================
# Create report
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# Display report
# ============================================================

print()
print("=" * 80)
print("DRIFT RESULTS")
print("=" * 80)

print(
    results_df.to_string(index=False)
)

print()
print("=" * 80)

drift_count = int(
    results_df["drift_detected"].sum()
)

total_features = len(results_df)

drift_percentage = (
    drift_count / total_features * 100
)

print(
    f"Features with detected drift : "
    f"{drift_count}/{total_features}"
)

print(
    f"Drift percentage              : "
    f"{drift_percentage:.2f}%"
)

print()
print("Thresholds:")
print("KS test       : p-value < 0.05")
print("PSI           : >= 0.20 indicates significant drift")

print()
print(f"Report saved to: {OUTPUT_FILE}")
print("=" * 80)
