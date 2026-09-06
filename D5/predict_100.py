import json
import time

import pandas as pd
import requests


API_URL = "http://104.198.219.81/predict"
INPUT_FILE = "D5/random_100_rows.csv"
OUTPUT_FILE = "D5/predictions_100.csv"


df = pd.read_csv(INPUT_FILE)

results = []

print(f"Sending {len(df)} individual requests...")
print()

for index, row in df.iterrows():

    payload = row.to_dict()

    response = requests.post(
        API_URL,
        json=payload,
        timeout=30
    )

    response.raise_for_status()

    result = response.json()

    results.append({
        "row_number": index + 1,
        "prediction": result["prediction"],
        "probability_no": result["probabilities"].get("no"),
        "probability_yes": result["probabilities"].get("yes")
    })

    print(
        f"Row {index + 1:3d}/100 "
        f"-> prediction={result['prediction']}"
    )

    # Small delay to avoid unnecessary burst
    time.sleep(0.05)


results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("========================================")
print("D5 prediction completed")
print("========================================")
print(f"Input rows       : {len(df)}")
print(f"Predictions made : {len(results_df)}")
print(f"Output file      : {OUTPUT_FILE}")
print()
print(results_df.head())
