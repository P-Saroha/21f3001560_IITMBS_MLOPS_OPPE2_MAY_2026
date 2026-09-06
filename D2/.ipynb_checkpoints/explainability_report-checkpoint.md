# D2 - Model Explainability

## Model

The provided heart disease dataset was preprocessed using the
same approach as the supplied training notebook. Logistic Regression
with RandomizedSearchCV was used for model training.

## Explainability Method

SHAP (SHapley Additive exPlanations) was used to measure the
contribution of each feature to model predictions.

Mean absolute SHAP values were used to rank feature importance.

## Least Impactful Features

The features with the smallest mean absolute SHAP values were:

1. exang
2. thal
3. fbs
4. gender
5. ca

## Plain-English Interpretation

These features had the smallest average contribution to the
model's predictions in this experiment. A smaller absolute SHAP
value indicates that the feature generally changed the model's
prediction less than features with larger SHAP values.

Therefore, among the supplied features, exang, thal, fbs, gender,
and ca had the least impact according to this SHAP analysis.

## Important Note

`sno` is an observation/index field. Although it produced a very
large SHAP value in this experiment, it is not a clinical factor.
The supplied notebook includes `sno` as a model feature, so it was
retained to reproduce the provided pipeline.