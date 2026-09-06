# D2 - Model Explainability

## Method

SHAP (SHapley Additive exPlanations) was used to explain the Logistic Regression model trained on the provided heart disease dataset.

The dataset contains 303 observations and 15 columns. After removing missing values, 293 observations remained.

The model was trained using the same preprocessing and Logistic Regression pipeline provided in the assignment notebook.

## Model Performance

- Best solver: `liblinear`
- Best C: `0.615848211066026`
- Test accuracy: `0.9830508474576272`

## Least Impactful Features

Based on mean absolute SHAP values, the least impactful features were:

1. `exang` - 0.006731
2. `thal` - 0.007527
3. `fbs` - 0.012344
4. `gender` - 0.012903
5. `ca` - 0.141317

## Plain-English Interpretation

The SHAP analysis shows that `exang`, `thal`, `fbs`, and `gender` contribute the least to the model's predictions among the provided features.

In particular, `exang` has the smallest average absolute SHAP value, meaning that changes in this feature generally have the smallest effect on the model's prediction compared with the other features.

The feature `sno` has a very large SHAP value. However, `sno` is a serial/record identifier and is not a clinical characteristic. Its high importance should therefore be considered a modeling artifact of the provided training pipeline rather than evidence that patient serial number is medically predictive.

## Output

The analysis generated:

- `D2/shap_feature_importance.csv`
- `D2/shap_summary.png`
