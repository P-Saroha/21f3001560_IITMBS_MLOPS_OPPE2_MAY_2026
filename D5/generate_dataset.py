import numpy as np
import pandas as pd

# Reproducible random dataset
np.random.seed(42)

n = 100

df = pd.DataFrame({
    "sno": np.random.randint(1, 304, n),
    "age": np.random.randint(29, 78, n),
    "gender": np.random.choice(["male", "female"], n),
    "cp": np.random.choice([0, 1, 2, 3], n),
    "trestbps": np.random.randint(90, 201, n),
    "chol": np.random.randint(120, 401, n),
    "fbs": np.random.choice([0, 1], n),
    "restecg": np.random.choice([0, 1, 2], n),
    "thalach": np.random.randint(70, 201, n),
    "exang": np.random.choice([0, 1], n),
    "oldpeak": np.round(np.random.uniform(0, 6.5, n), 1),
    "slope": np.random.choice([0, 1, 2], n),
    "ca": np.random.choice([0, 1, 2, 3, 4], n),
    "thal": np.random.choice([0, 1, 2, 3], n)
})

output_file = "D5/random_100_rows.csv"

df.to_csv(output_file, index=False)

print(f"Generated {len(df)} rows")
print(f"Saved to: {output_file}")
print()
print(df.head())
print()
print("Shape:", df.shape)
