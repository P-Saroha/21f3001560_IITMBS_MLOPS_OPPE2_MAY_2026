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

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv("data/data.csv")

print("Original dataset shape:", df.shape)

# Same preprocessing as official notebook
df["gender"] = pd.factorize(df["gender"])[0]

cleaned_df = df.dropna()

print("Cleaned dataset shape:", cleaned_df.shape)

# -----------------------------
# 2. Features and target
# -----------------------------
X = cleaned_df.drop("target", axis=1)
y = cleaned_df["target"]

# -----------------------------
# 3. Train-test split
# -----------------------------
np.random.seed(42)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2
)

# -----------------------------
# 4. Train Logistic Regression
# -----------------------------
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

# -----------------------------
# 5. Predictions
# -----------------------------
y_pred = best_model.predict(X_test)

# Sensitive attribute = age
sensitive_age = X_test["age"]

# Create age groups
age_groups = pd.cut(
    sensitive_age,
    bins=[0, 40, 50, 60, 100],
    labels=["<=40", "41-50", "51-60", "61+"]
)

print("\nAge group counts:")
print(age_groups.value_counts().sort_index())

# -----------------------------
# 6. Fairness metrics
# -----------------------------
metrics = {
    "accuracy": lambda y_true, y_pred: np.mean(y_true == y_pred),
    "selection_rate": selection_rate
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

# -----------------------------
# 7. Demographic parity
# -----------------------------
dp_difference = demographic_parity_difference(
    y_test,
    y_pred,
    sensitive_features=age_groups,
    method="between_groups"
)

dp_ratio = demographic_parity_ratio(
    y_test,
    y_pred,
    sensitive_features=age_groups,
    method="between_groups"
)

print("\nDemographic parity difference:")
print(dp_difference)

print("\nDemographic parity ratio:")
print(dp_ratio)

# -----------------------------
# 8. Interpretation
# -----------------------------
print("\nFairness interpretation:")

if dp_difference == 0:
    print("Perfect demographic parity across age groups.")
else:
    print(
        "There is a difference in selection rates between age groups. "
        "A value closer to 0 for demographic parity difference indicates "
        "smaller disparity."
    )

if dp_ratio == 1:
    print("The demographic parity ratio is 1, indicating equal selection rates.")
else:
    print(
        "A demographic parity ratio closer to 1 indicates more similar "
        "selection rates across age groups."
    )

# -----------------------------
# 9. Save results
# -----------------------------
metric_frame.by_group.to_csv("D3/fairness_by_age_group.csv")

with open("D3/fairness_summary.txt", "w") as f:

    f.write("D3 Fairness Analysis\n")
    f.write("====================\n\n")

    f.write(f"Sensitive attribute: age\n")
    f.write("Age groups: <=40, 41-50, 51-60, 61+\n\n")

    f.write(f"Best parameters: {model.best_params_}\n")
    f.write(f"Overall test accuracy: {overall_accuracy}\n\n")

    f.write("Age group counts:\n")
    f.write(age_groups.value_counts().sort_index().to_string())
    f.write("\n\n")

    f.write("Fairness metrics by age group:\n")
    f.write(metric_frame.by_group.to_string())
    f.write("\n\n")

    f.write(f"Demographic parity difference: {dp_difference}\n")
    f.write(f"Demographic parity ratio: {dp_ratio}\n\n")

    f.write("Interpretation:\n")

    if dp_difference == 0:
        f.write("Perfect demographic parity across age groups.\n")
    else:
        f.write(
            "There is a difference in selection rates between age groups. "
            "Smaller demographic parity difference indicates smaller disparity.\n"
        )

    if dp_ratio == 1:
        f.write(
            "The demographic parity ratio is 1, indicating equal selection rates.\n"
        )
    else:
        f.write(
            "A demographic parity ratio closer to 1 indicates more similar "
            "selection rates across age groups.\n"
        )

print("\nSaved:")
print("D3/fairness_by_age_group.csv")
print("D3/fairness_summary.txt")
