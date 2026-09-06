import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression

from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    demographic_parity_difference,
    demographic_parity_ratio
)

# ============================================================
# 1. Load dataset
# ============================================================

df = pd.read_csv("data/data.csv")

print("Original dataset shape:", df.shape)

# Same preprocessing as official notebook
df["gender"] = pd.factorize(df["gender"])[0]

cleaned_df = df.dropna()

print("Cleaned dataset shape:", cleaned_df.shape)

# ============================================================
# 2. Features and target
# ============================================================

X = cleaned_df.drop("target", axis=1)
y = cleaned_df["target"]

# ============================================================
# 3. Train-test split
# ============================================================

np.random.seed(42)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2
)

# Reset indices
X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# ============================================================
# 4. Train Logistic Regression
# ============================================================

log_reg_grid = {
    "C": np.logspace(-4, 4, 20),
    "solver": ["liblinear"]
}

model = RandomizedSearchCV(
    LogisticRegression(),
    param_distributions=log_reg_grid,
    cv=5,
    n_iter=20,
    verbose=True
)

model.fit(X_train, y_train)

best_model = model.best_estimator_

print("\nBest parameters:")
print(model.best_params_)

overall_accuracy = model.score(X_test, y_test)

print("\nOverall test accuracy:")
print(overall_accuracy)

# ============================================================
# 5. Generate predictions
# ============================================================

y_pred = best_model.predict(X_test)

print("\nPrediction counts:")
print(pd.Series(y_pred).value_counts())

# ============================================================
# 6. Sensitive attribute = AGE
# ============================================================

age_groups = pd.cut(
    X_test["age"],
    bins=[0, 40, 50, 60, 100],
    labels=["<=40", "41-50", "51-60", "61+"],
    include_lowest=True
)

age_groups = age_groups.reset_index(drop=True)

print("\nAge group counts:")
print(age_groups.value_counts().sort_index())

# ============================================================
# 7. Fairness metrics
# ============================================================

# The original target uses strings:
# "yes" = positive class
# "no"  = negative class

metrics = {
    "accuracy": lambda y_true, y_pred: np.mean(
        np.asarray(y_true) == np.asarray(y_pred)
    ),

    "selection_rate": lambda y_true, y_pred: selection_rate(
        y_true,
        y_pred,
        pos_label="yes"
    )
}

metric_frame = MetricFrame(
    metrics=metrics,
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=age_groups
)

print("\nFairness metrics by age group:")
print(metric_frame.by_group)

print("\nOverall metrics:")
print(metric_frame.overall)

# ============================================================
# 8. Convert labels to binary for demographic parity
# ============================================================

# Fairlearn version in the environment does not accept
# pos_label directly in demographic_parity_difference().
#
# Therefore convert:
# yes -> 1
# no  -> 0

y_test_binary = (y_test == "yes").astype(int)
y_pred_binary = (y_pred == "yes").astype(int)

# ============================================================
# 9. Demographic parity
# ============================================================

dp_difference = demographic_parity_difference(
    y_test_binary,
    y_pred_binary,
    sensitive_features=age_groups,
    method="between_groups"
)

dp_ratio = demographic_parity_ratio(
    y_test_binary,
    y_pred_binary,
    sensitive_features=age_groups,
    method="between_groups"
)

print("\nDemographic parity difference:")
print(dp_difference)

print("\nDemographic parity ratio:")
print(dp_ratio)

# ============================================================
# 10. Interpretation
# ============================================================

selection_rates = metric_frame.by_group["selection_rate"]

highest_group = selection_rates.idxmax()
lowest_group = selection_rates.idxmin()

highest_rate = selection_rates.max()
lowest_rate = selection_rates.min()

print("\nFairness interpretation:")

print(
    f"Highest positive prediction rate: {highest_group} "
    f"({highest_rate:.4f})"
)

print(
    f"Lowest positive prediction rate: {lowest_group} "
    f"({lowest_rate:.4f})"
)

print(
    f"Difference between highest and lowest selection rates: "
    f"{dp_difference:.4f}"
)

print(
    f"Demographic parity ratio: "
    f"{dp_ratio:.4f}"
)

print(
    "\nA demographic parity difference closer to 0 indicates "
    "smaller disparity between age groups."
)

print(
    "A demographic parity ratio closer to 1 indicates "
    "greater similarity in positive prediction rates."
)

# ============================================================
# 11. Save results
# ============================================================

metric_frame.by_group.to_csv(
    "D3/fairness_by_age_group.csv"
)

with open("D3/fairness_summary.txt", "w") as f:

    f.write("D3 Fairness Analysis\n")
    f.write("====================\n\n")

    f.write("Sensitive attribute: age\n")
    f.write("Positive class: yes\n")
    f.write("Age groups: <=40, 41-50, 51-60, 61+\n\n")

    f.write(f"Best parameters: {model.best_params_}\n")
    f.write(f"Overall test accuracy: {overall_accuracy}\n\n")

    f.write("Age group counts:\n")
    f.write(
        age_groups.value_counts()
        .sort_index()
        .to_string()
    )

    f.write("\n\n")

    f.write("Fairness metrics by age group:\n")
    f.write(metric_frame.by_group.to_string())

    f.write("\n\n")

    f.write(f"Demographic parity difference: {dp_difference}\n")
    f.write(f"Demographic parity ratio: {dp_ratio}\n\n")

    f.write("Interpretation:\n")

    f.write(
        f"The highest positive prediction rate is for the "
        f"{highest_group} age group ({highest_rate:.4f}).\n"
    )

    f.write(
        f"The lowest positive prediction rate is for the "
        f"{lowest_group} age group ({lowest_rate:.4f}).\n"
    )

    f.write(
        f"The demographic parity difference is {dp_difference:.4f}. "
        f"A value closer to 0 indicates smaller disparity.\n"
    )

    f.write(
        f"The demographic parity ratio is {dp_ratio:.4f}. "
        f"A value closer to 1 indicates greater parity.\n"
    )

print("\nSaved:")
print("D3/fairness_by_age_group.csv")
print("D3/fairness_summary.txt")
