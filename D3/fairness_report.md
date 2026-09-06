# D3 - Fairness Analysis

## Objective

Fairness of the heart disease prediction model was evaluated using the Fairlearn library.

The sensitive attribute used for the analysis is **age**.

## Dataset

- Original dataset: 303 rows
- Cleaned dataset: 293 rows
- Test set: 59 samples
- Sensitive attribute: `age`

The test samples were divided into four age groups:

- `<=40`
- `41-50`
- `51-60`
- `61+`

## Model

The same Logistic Regression pipeline from the provided assignment notebook was used.

Best parameters:

- Solver: `liblinear`
- C: `0.615848211066026`

Overall test accuracy:

`0.9830508474576272`

## Fairness Metrics

| Age Group | Samples | Accuracy | Selection Rate |
|-----------|---------|----------|----------------|
| <=40 | 5 | 1.000000 | 0.800000 |
| 41-50 | 17 | 1.000000 | 0.647059 |
| 51-60 | 23 | 0.956522 | 0.608696 |
| 61+ | 14 | 1.000000 | 0.428571 |

### Demographic Parity

- Demographic parity difference: `0.3714285714285715`
- Demographic parity ratio: `0.5357142857142857`

## Plain-English Interpretation

The model's positive prediction rate varies across age groups.

The `<=40` age group has the highest positive prediction rate at **80%**, while the `61+` group has the lowest positive prediction rate at approximately **42.86%**.

The difference between the highest and lowest selection rates is approximately **37.14 percentage points**, resulting in a demographic parity difference of `0.3714`.

The demographic parity ratio is approximately `0.5357`. A value closer to 1 indicates more similar positive prediction rates across groups.

Therefore, the model shows **some disparity in prediction outcomes across age groups** in this test set.

However, the age groups have relatively small sample sizes, particularly the `<=40` group with only 5 samples. Therefore, the fairness results should be interpreted cautiously and should ideally be validated using a larger and more representative dataset.

## Conclusion

Fairlearn successfully measured fairness using `age` as the sensitive attribute.

The results indicate that positive prediction rates differ between age groups. The model therefore does not demonstrate perfect demographic parity on this test set.

Further fairness evaluation on larger datasets and potentially additional fairness metrics would be useful before making conclusions about real-world model fairness.
