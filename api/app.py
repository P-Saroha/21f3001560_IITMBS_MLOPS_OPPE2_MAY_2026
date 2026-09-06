import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Dockerized heart disease prediction API for MLOps OPPE-2",
    version="1.0.0"
)


# ============================================================
# Train model
# ============================================================

def train_model():

    df = pd.read_csv("data/data.csv")

    # Same preprocessing as official notebook
    df["gender"] = pd.factorize(df["gender"])[0]

    cleaned_df = df.dropna()

    X = cleaned_df.drop("target", axis=1)
    y = cleaned_df["target"]

    np.random.seed(42)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2
    )

    log_reg_grid = {
        "C": np.logspace(-4, 4, 20),
        "solver": ["liblinear"]
    }

    model = RandomizedSearchCV(
        LogisticRegression(),
        param_distributions=log_reg_grid,
        cv=5,
        n_iter=20,
        verbose=False
    )

    model.fit(X_train, y_train)

    return model


model = train_model()


# ============================================================
# Input schema
# ============================================================

class HeartDiseaseInput(BaseModel):

    sno: int
    age: int
    gender: str
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


# ============================================================
# Root endpoint
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Heart Disease Prediction API",
        "status": "running"
    }


# ============================================================
# Health endpoint
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ============================================================
# Prediction endpoint
# ============================================================

@app.post("/predict")
def predict(data: HeartDiseaseInput):

    try:

        # Convert request to dictionary
        input_data = data.model_dump()

        # Convert request to DataFrame
        row = pd.DataFrame([input_data])

        # Apply EXACT same gender preprocessing
        #
        # Training notebook:
        # df["gender"] = pd.factorize(df["gender"])[0]
        #
        # Dataset mapping:
        # female -> 0
        # male   -> 1
        #
        gender_mapping = {
            "female": 0,
            "male": 1
        }

        gender = row["gender"].iloc[0].lower()

        if gender not in gender_mapping:
            raise ValueError(
                "gender must be either 'male' or 'female'"
            )

        row["gender"] = gender_mapping[gender]

        # Ensure exact feature order used during training
        feature_columns = [
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

        row = row[feature_columns]

        # Prediction
        prediction = model.predict(row)[0]

        # Probability of each class
        probabilities = model.predict_proba(row)[0]

        classes = model.classes_

        probability_map = {
            str(cls): float(prob)
            for cls, prob in zip(classes, probabilities)
        }

        # ====================================================
        # D5: Per-sample prediction logging
        # ====================================================

        prediction_log = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_features": input_data,
            "predicted_output": str(prediction),
            "probabilities": probability_map
        }

        # Print structured JSON to stdout.
        # GKE automatically collects container stdout
        # and makes it available in Google Cloud Logging.
        print(
            json.dumps(prediction_log),
            flush=True
        )

        # ====================================================
        # API response
        # ====================================================

        return {
            "prediction": str(prediction),
            "probabilities": probability_map
        }

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )