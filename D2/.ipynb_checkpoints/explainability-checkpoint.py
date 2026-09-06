import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------
# 1. Load the official dataset
# ---------------------------------------------------------
df = pd.read_csv("data/data.csv")

print("Original dataset shape:", df.shape)


# ---------------------------------------------------------
# 2. Same preprocessing as the official notebook
# ---------------------------------------------------------
df["gender"] = pd.factorize(df["gender"])[0]

cleaned_df = df.dropna()

print("Cleaned dataset shape:", cleaned_df.shape)

X = cleaned_df.drop("target", axis=1)
y = cleaned_df["target"]


# ---------------------------------------------------------
# 3. Same train/test split as the official notebook
# ---------------------------------------------------------
np.random.seed(42)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2
)


# ---------------------------------------------------------
# 4. Same Logistic Regression + RandomizedSearchCV
# ---------------------------------------------------------
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

print("\nBest parameters:")
print(model.best_params_)

print("\nTest accuracy:")
print(model.score(X_test, y_test))


# ---------------------------------------------------------
# 5. SHAP explainability
# ---------------------------------------------------------
# RandomizedSearchCV exposes the best fitted model through best_estimator_
best_model = model.best_estimator_

explainer = shap.LinearExplainer(
    best_model,
    shap.maskers.Independent(X_train, max_samples=len(X_train))
)

shap_values = explainer(X_test)


# ---------------------------------------------------------
# 6. Calculate mean absolute SHAP importance
# ---------------------------------------------------------
importance = np.abs(shap_values.values).mean(axis=0)

importance_df = pd.DataFrame({
    "feature": X_test.columns,
    "mean_abs_shap": importance
})

importance_df = importance_df.sort_values(
    "mean_abs_shap",
    ascending=True
)

print("\nSHAP feature importance:")
print(importance_df.to_string(index=False))


# ---------------------------------------------------------
# 7. Identify least impactful features
# ---------------------------------------------------------
least_impactful = importance_df.head(5)

print("\nLeast impactful features:")
print(least_impactful.to_string(index=False))


# ---------------------------------------------------------
# 8. Save importance results
# ---------------------------------------------------------
importance_df.to_csv(
    "D2/shap_feature_importance.csv",
    index=False
)


# ---------------------------------------------------------
# 9. Generate SHAP bar plot
# ---------------------------------------------------------
plt.figure()

shap.summary_plot(
    shap_values,
    X_test,
    plot_type="bar",
    show=False
)

plt.tight_layout()
plt.savefig(
    "D2/shap_summary.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("\nSaved:")
print("D2/shap_feature_importance.csv")
print("D2/shap_summary.png")
