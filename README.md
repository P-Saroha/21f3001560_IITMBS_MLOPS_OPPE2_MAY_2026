# OPPE-2 MLOps — Heart Disease Prediction Deployment

## IITM BS MLOps OPPE-2 — May 2026

This project implements an end-to-end MLOps pipeline for deploying a Heart Disease Prediction model as a production-style API on Google Cloud Platform (GCP).

The solution covers:

- Model explainability using SHAP
- Fairness analysis using Fairlearn
- Dockerized API deployment
- Google Kubernetes Engine (GKE)
- Kubernetes autoscaling
- GitHub Actions CI/CD
- Per-sample prediction logging
- Google Cloud Logging
- High-concurrency stress testing using `wrk`
- Input data drift detection using KS test and PSI

---

## 1. Problem Statement

The objective of this OPPE-2 assignment is to build a production-ready, explainable, observable, scalable, and maintainable deployment for a heart disease prediction model.

The original model is trained using the provided heart disease dataset and deployed as a FastAPI application inside a Docker container.

The complete production flow is:

```text
Training Dataset
       |
       v
Model Training
       |
       +-------------------+
       |                   |
       v                   v
     SHAP              Fairlearn
Explainability         Fairness
       |                   |
       +---------+---------+
                 |
                 v
            FastAPI API
                 |
                 v
              Docker
                 |
                 v
        Artifact Registry
                 |
                 v
               GKE
          +------+------+
          |             |
       Service         HPA
          |             |
          |          1 -> 3 Pods
          |
          v
       /predict
          |
    +-----+------+----------------+
    |            |                |
    v            v                v
100 Samples  Cloud Logging       wrk
                             > 2000 Connections
    |
    v
Drift Detection
```

---

## 2. OPPE-2 Deliverables

| Deliverable | Description | Marks | Status |
|---|---|---|---|
| D1 | Private GitHub Repository | Mandatory | Completed |
| D2 | Model Explainability using SHAP | 10 | Completed |
| D3 | Fairness Testing using Fairlearn | 10 | Completed |
| D4 | Dockerized API Deployment on GKE + CI/CD | 40 | Completed |
| D5 | 100 Predictions + Logging + Observability | 20 | Completed |
| D6 | wrk Stress Testing | 10 | Completed |
| D7 | Input Drift Detection | 10 | Completed |
| **Total** | | **100** | **Completed** |

---

## 3. Repository Information

GitHub repository:

```text
21f3001560_IITMBS_MLOPS_OPPE2_MAY_2026
```

Repository branch:

```text
main
```

The repository is private.

Required MLOps collaborator was added and verified to have repository access.

---

## 4. Project Folder Structure

```text
21f3001560_MLOPS_OPPE2/
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml
│
├── api/
│   └── app.py
│
├── data/
│   └── data.csv
│
├── D2/
│   └── ...
│
├── D3/
│   └── ...
│
├── D5/
│   ├── generate_dataset.py
│   ├── predict_100.py
│   ├── random_100_rows.csv
│   └── predictions_100.csv
│
├── D6/
│   ├── wrk_post.lua
│   └── wrk_2001_results.txt
│
├── D7/
│   ├── drift_detection.py
│   └── drift_report.csv
│
├── HeartDiseaseTrainingAndPrediction.ipynb
├── HeartDiseaseTrainingAndPrediction.py
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 5. Environment

The project was executed in a Python virtual environment.

Activate the environment:

```bash
source ~/21f3001560_MLOPS_WEEKLY_ASSIGNMENT/.env/bin/activate
```

Verify Python:

```bash
python --version
```

---

## 6. Dataset

The project uses the provided heart disease dataset:

```text
data/data.csv
```

Dataset dimensions:

```text
303 rows
15 columns
```

Columns:

```text
sno
age
gender
cp
trestbps
chol
fbs
restecg
thalach
exang
oldpeak
slope
ca
thal
target
```

The target variable is:

```text
target
```

with values:

```text
yes
no
```

The model uses 14 input features:

```text
sno
age
gender
cp
trestbps
chol
fbs
restecg
thalach
exang
oldpeak
slope
ca
thal
```

---

## D1 — Private GitHub Repository

### Objective

Create a private GitHub repository using the required naming convention:

```text
<IITM_BS_ROLL_NUMBER>_IITMBS_MLOPS_OPPE2_MAY_2026
```

Repository created:

```text
21f3001560_IITMBS_MLOPS_OPPE2_MAY_2026
```

The repository is private.

The required MLOps collaborator was added and verified with repository access.

### Git Commands

Check repository:

```bash
git remote -v
```

Check branch:

```bash
git branch
```

Check repository status:

```bash
git status
```

Check commit history:

```bash
git log --oneline -10
```

Push changes:

```bash
git add .
git commit -m "commit message"
git push origin main
```

Final repository state:

```text
main
origin/main
```

The working tree was verified as clean.

---

## D2 — Model Explainability using SHAP

### Objective

Use an explainability tool to identify and explain the factors having the least impact on heart disease prediction.

SHAP was used to calculate feature importance.

### SHAP Results

The mean absolute SHAP importance values were:

| Feature | Mean Absolute SHAP |
|---|---|
| exang | 0.006731 |
| thal | 0.007527 |
| fbs | 0.012344 |
| gender | 0.012903 |
| ca | 0.141317 |
| slope | 0.247860 |
| restecg | 0.256742 |
| cp | 0.362295 |
| oldpeak | 1.003745 |
| chol | 1.084741 |
| age | 2.113868 |
| trestbps | 2.243699 |
| thalach | 3.638289 |
| sno | 28.577987 |

### Least Impactful Features

The features with the lowest impact were:

- exang
- thal
- fbs
- gender
- ca

### Plain-English Interpretation

The SHAP analysis shows that exang, thal, fbs, and gender have the smallest influence on the model's predictions compared with the other features.

ca also has relatively low importance compared with the strongest features.

Features such as thalach, trestbps, age, chol, and oldpeak had much larger SHAP importance values.

---

## D3 — Fairness Testing using Fairlearn

### Objective

Evaluate model fairness using Fairlearn.

The sensitive attribute required for this deliverable is:

```text
age
```

Age was divided into four groups:

```text
<=40
41-50
51-60
61+
```

### Model Accuracy

```text
Accuracy = 0.9830508474576272
```

### Fairness Results

#### Prediction Counts

```text
yes = 35
no  = 24
```

#### Age Group Counts

```text
<=40   = 5
41-50  = 17
51-60  = 23
61+    = 14
```

#### Selection Rates

| Age Group | Accuracy | Selection Rate |
|---|---|---|
| <=40 | 1.000000 | 0.800000 |
| 41-50 | 1.000000 | 0.647059 |
| 51-60 | 0.956522 | 0.608696 |
| 61+ | 1.000000 | 0.428571 |

Overall selection rate:

```text
0.593220
```

Demographic parity difference:

```text
0.3714285714285715
```

Demographic parity ratio:

```text
0.5357142857142857
```

### Interpretation

The results indicate disparity in positive prediction rates across age groups.

The <=40 group has the highest positive prediction rate:

```text
0.80
```

while the 61+ group has the lowest:

```text
0.428571
```

Therefore, the demographic parity metrics indicate potential fairness disparity across age groups.

The smaller sample size of the <=40 group should also be considered while interpreting the result.

---

## D4 — Dockerized API Deployment on GKE

### Objective

Convert the provided model into a Dockerized API and deploy it on Google Kubernetes Engine.

The deployment also includes:

- Kubernetes Service
- Kubernetes HPA
- Maximum 3 pods
- GitHub Actions CI/CD
- Google Artifact Registry

### 4.1 FastAPI Application

The API is implemented in:

```text
api/app.py
```

The application provides the following endpoints:

```text
GET /
GET /health
POST /predict
```

#### Root Endpoint

`/`

Returns:

```json
{
  "message": "Heart Disease Prediction API",
  "status": "running"
}
```

#### Health Endpoint

`/health`

Returns:

```json
{
  "status": "healthy"
}
```

#### Prediction Endpoint

`/predict`

The API accepts the 14 model features and returns:

```json
{
  "prediction": "yes",
  "probabilities": {
    "no": 0.01,
    "yes": 0.99
  }
}
```

### 4.2 Model Preprocessing

The API follows the preprocessing used by the provided notebook.

Gender is converted using:

```python
gender_mapping = {
    "female": 0,
    "male": 1
}
```

The model uses:

```text
sno
age
gender
cp
trestbps
chol
fbs
restecg
thalach
exang
oldpeak
slope
ca
thal
```

### 4.3 Python Dependencies

`requirements.txt` contains:

```text
fastapi==0.141.1
uvicorn==0.52.1
pydantic==2.13.4
numpy
pandas
scikit-learn
```

### 4.4 Docker

The application was containerized using Python 3.12 slim.

#### Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY data/ ./data/

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Build Docker Image

```bash
docker build -t heart-disease-api:v1 .
```

#### Run Locally

```bash
docker run -p 8000:8000 heart-disease-api:v1
```

Test:

```bash
curl http://localhost:8000/
```

Health check:

```bash
curl http://localhost:8000/health
```

### 4.5 Google Cloud Configuration

GCP project:

```text
project-a74bda7f-3a9e-4ff2-a23
```

Set the project:

```bash
gcloud config set project project-a74bda7f-3a9e-4ff2-a23
```

Verify:

```bash
gcloud config get-value project
```

### 4.6 Artifact Registry

Artifact Registry repository:

```text
iris-repo
```

Region:

```text
us-central1
```

Docker image:

```text
us-central1-docker.pkg.dev/project-a74bda7f-3a9e-4ff2-a23/iris-repo/heart-disease-api:v1
```

Configure Docker authentication:

```bash
gcloud auth configure-docker us-central1-docker.pkg.dev
```

Build:

```bash
docker build \
  -t us-central1-docker.pkg.dev/project-a74bda7f-3a9e-4ff2-a23/iris-repo/heart-disease-api:v1 \
  .
```

Push:

```bash
docker push \
  us-central1-docker.pkg.dev/project-a74bda7f-3a9e-4ff2-a23/iris-repo/heart-disease-api:v1
```

### 4.7 GKE Cluster

GKE cluster:

```text
oppe2-cluster
```

Zone:

```text
us-central1-a
```

Machine type:

```text
e2-standard-2
```

Get cluster credentials:

```bash
gcloud container clusters get-credentials \
  oppe2-cluster \
  --zone us-central1-a \
  --project project-a74bda7f-3a9e-4ff2-a23
```

Check nodes:

```bash
kubectl get nodes
```

### 4.8 Kubernetes Deployment

The deployment is named:

```text
heart-disease-api
```

Important configuration:

```text
replicas: 1
```

Container port:

```text
8000
```

CPU request:

```text
250m
```

CPU limit:

```text
1000m
```

Memory request:

```text
512Mi
```

Memory limit:

```text
1Gi
```

Readiness and liveness probes use:

```text
/health
```

Check deployment:

```bash
kubectl get deployment heart-disease-api
```

Check pods:

```bash
kubectl get pods -l app=heart-disease-api
```

### 4.9 Kubernetes Service

The API is exposed using a Kubernetes LoadBalancer.

Service:

```text
heart-disease-api
```

Check service:

```bash
kubectl get service heart-disease-api
```

External IP obtained:

```text
104.198.219.81
```

API URL:

```text
http://104.198.219.81
```

Prediction endpoint:

```text
http://104.198.219.81/predict
```

### 4.10 Kubernetes HPA

Horizontal Pod Autoscaler was configured using:

```text
autoscaling/v2
```

Configuration:

```text
minReplicas: 1
maxReplicas: 3
```

CPU target:

```text
60%
```

Therefore, the deployment can automatically scale:

```text
1 Pod
  |
  | CPU increases
  v
2 Pods
  |
  | CPU increases
  v
3 Pods
```

Maximum replicas are explicitly limited to:

```text
3
```

Check HPA:

```bash
kubectl get hpa heart-disease-api-hpa
```

Detailed information:

```bash
kubectl describe hpa heart-disease-api-hpa
```

During stress testing, HPA was observed scaling the application from:

```text
1 -> 3 pods
```

### 4.11 GitHub Actions CI/CD

Workflow:

```text
.github/workflows/ci-cd.yml
```

The workflow performs:

```text
GitHub Push
     |
     v
Checkout
     |
     v
Setup Python
     |
     v
Install Dependencies
     |
     v
Python Compilation/Test
     |
     v
Docker Build
     |
     v
Push Image to Artifact Registry
     |
     v
Get GKE Credentials
     |
     v
Update Kubernetes Deployment
     |
     v
Rollout Status
     |
     v
Verify Deployment / Pods / HPA
```

The workflow triggers on:

```yaml
push:
  branches:
    - main

pull_request:
  branches:
    - main
```

For deployment, the workflow uses the GitHub secret:

```text
GCP_SA_KEY
```

The workflow successfully completed the CI/CD pipeline.

---

## D5 — Per-Sample Prediction, Logging and Observability

### Objective

Generate a random 100-row dataset and send each row individually to the deployed API.

### 5.1 Generate 100-Row Dataset

Script:

```text
D5/generate_dataset.py
```

Run:

```bash
python D5/generate_dataset.py
```

Generated file:

```text
D5/random_100_rows.csv
```

The dataset contains 100 rows and the 14 API input features.

### 5.2 Send Individual Predictions

Script:

```text
D5/predict_100.py
```

The script sends each row as a separate HTTP POST request to:

```text
http://104.198.219.81/predict
```

Run:

```bash
python D5/predict_100.py
```

The results are stored in:

```text
D5/predictions_100.csv
```

Verify the number of rows:

```bash
wc -l D5/predictions_100.csv
```

Expected:

```text
101
```

because:

```text
1 header + 100 predictions = 101 lines
```

### 5.3 Per-Sample Logging

Each API prediction creates an individual structured log containing:

```text
timestamp
input_features
predicted_output
probabilities
```

Example structure:

```json
{
  "timestamp": "2026-09-06T...",
  "input_features": {
    "sno": 103,
    "age": 54,
    "gender": "male",
    "cp": 2,
    "trestbps": 130,
    "chol": 246,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.2,
    "slope": 1,
    "ca": 0,
    "thal": 2
  },
  "predicted_output": "yes",
  "probabilities": {
    "no": 0.01,
    "yes": 0.99
  }
}
```

Logging is implemented inside the FastAPI `/predict` endpoint.

### 5.4 Google Cloud Logging

Logs can be queried using:

```bash
gcloud logging read \
'resource.type="k8s_container" AND labels.k8s-pod/app="heart-disease-api" AND jsonPayload.predicted_output:*' \
--limit=100 \
--format="table(timestamp,jsonPayload.predicted_output)"
```

The D5 execution was verified with 100 individual prediction logs.

Structured logs contain:

```text
input_features
predicted_output
timestamp
probabilities
```

This demonstrates observability using GCP Cloud Logging.

---

## D6 — Performance Monitoring and Stress Testing

### Objective

Stress test the deployed API using `wrk` with more than 2,000 concurrent connections.

### 6.1 Install wrk

Verify installation:

```bash
wrk --version
```

The installed version was:

```text
wrk 4.1.0
```

### 6.2 wrk POST Configuration

File:

```text
D6/wrk_post.lua
```

The Lua script configures:

```text
HTTP Method = POST
Content-Type = application/json
```

and sends a valid heart disease prediction request to:

```text
/predict
```

### 6.3 Sanity Test

A smaller test was first performed:

```bash
wrk -t2 -c10 -d10s --latency \
-s D6/wrk_post.lua \
http://104.198.219.81/predict
```

Result:

```text
Requests/sec: 142.17
Average latency: 70.32 ms
P50: 66.35 ms
P75: 77.07 ms
P90: 87.01 ms
P99: 141.90 ms
Timeouts: 0
```

This verified that the API could handle normal concurrent traffic.

### 6.4 Required High-Concurrency Test

The final required test used:

```bash
wrk -t4 -c2001 -d30s --latency \
-s D6/wrk_post.lua \
http://104.198.219.81/predict
```

Important parameter:

```text
-c2001
```

This means:

```text
2001 concurrent connections
```

which satisfies the requirement of greater than 2,000 concurrent connections.

### 6.5 Final Stress Test Results

Results were saved to:

```text
D6/wrk_2001_results.txt
```

Final results:

```text
Duration:        30.05 seconds
Connections:     2001
Requests:        5160
Requests/sec:    171.72
Average latency: 1.45 seconds
P50:             1.31 seconds
P75:             1.79 seconds
P90:             1.81 seconds
P99:             1.82 seconds
Maximum latency: 1.83 seconds
Timeouts:        4960
```

### 6.6 Stress Test Analysis

#### Throughput

The API achieved:

```text
171.72 requests/sec
```

during the final high-concurrency test.

#### Latency

The latency distribution was:

```text
P50 = 1.31 sec
P75 = 1.79 sec
P90 = 1.81 sec
P99 = 1.82 sec
```

The high percentile latency indicates increased response time under extreme concurrency.

#### Timeout Behavior

The test recorded:

```text
4960 timeouts
```

This indicates that the deployment reached its capacity under the extreme 2,001-connection workload.

#### Autoscaling Behavior

During the load test, CPU utilization increased significantly and the Kubernetes HPA scaled the deployment:

```text
1 Pod -> 3 Pods
```

This demonstrates that Kubernetes autoscaling was active and responded to increased workload.

### Overall Conclusion

The deployment handled normal traffic successfully but experienced significant latency and timeouts under extreme concurrency.

The stress test demonstrates both the scalability mechanism and the capacity limitations of the current one-node GKE deployment.

---

## D7 — Input Drift Detection

### Objective

Compare the training data distribution against the 100-row generated prediction dataset.

Training data:

```text
data/data.csv
```

Incoming prediction data:

```text
D5/random_100_rows.csv
```

Script:

```text
D7/drift_detection.py
```

Output:

```text
D7/drift_report.csv
```

### 7.1 Drift Detection Methods

Two methods were used.

**Kolmogorov-Smirnov Test**

For numerical features:

```text
KS p-value < 0.05
```

was treated as evidence of drift.

**Population Stability Index**

PSI threshold:

```text
PSI >= 0.20
```

was treated as evidence of drift.

For categorical features, PSI was calculated using category distributions.

### 7.2 Drift Results

| Feature | Type | KS Statistic | P-value | PSI | Drift |
|---|---|---|---|---|---|
| sno | Numerical | 0.060462 | 0.927278 | 0.071056 | No |
| age | Numerical | 0.191485 | 0.006756 | 0.851672 | Yes |
| gender | Categorical | - | - | 0.207259 | Yes |
| cp | Categorical | - | - | 0.437249 | Yes |
| trestbps | Numerical | 0.332609 | 0.000000 | 1.679942 | Yes |
| chol | Numerical | 0.291060 | 0.000004 | 0.878157 | Yes |
| fbs | Categorical | - | - | 0.412220 | Yes |
| restecg | Categorical | - | - | 0.826611 | Yes |
| thalach | Numerical | 0.285705 | 0.000007 | 0.491867 | Yes |
| exang | Categorical | - | - | 0.098544 | No |
| oldpeak | Numerical | 0.494983 | 0.000000 | 2.082070 | Yes |
| slope | Categorical | - | - | 0.464339 | Yes |
| ca | Categorical | - | - | 1.044806 | Yes |
| thal | Categorical | - | - | 2.067904 | Yes |

### 7.3 Overall Drift Result

Total features:

```text
14
```

Features with detected drift:

```text
12
```

Drift percentage:

```text
85.71%
```

No detected drift:

```text
sno
exang
```

### 7.4 Strongest Drift

The largest PSI values were:

```text
oldpeak = 2.082070
thal    = 2.067904
trestbps = 1.679942
ca      = 1.044806
chol    = 0.878157
age     = 0.851672
restecg = 0.826611
```

These features show substantial distribution differences between the training data and the generated incoming dataset.

### 7.5 Drift Interpretation

The generated 100-row dataset was created using random valid ranges, so its empirical distribution can differ considerably from the original training distribution.

The analysis detected drift in 12 out of 14 input features.

This indicates that the generated incoming data is substantially different from the training distribution and demonstrates how input drift can be monitored in a production ML system.

---

## 8. Kubernetes Useful Commands

Get all resources:

```bash
kubectl get all
```

Get deployments:

```bash
kubectl get deployments
```

Get pods:

```bash
kubectl get pods
```

Get service:

```bash
kubectl get svc
```

Get HPA:

```bash
kubectl get hpa
```

Watch pods:

```bash
kubectl get pods -w
```

Describe deployment:

```bash
kubectl describe deployment heart-disease-api
```

Describe HPA:

```bash
kubectl describe hpa heart-disease-api-hpa
```

Check rollout:

```bash
kubectl rollout status deployment/heart-disease-api
```

Check logs:

```bash
kubectl logs -l app=heart-disease-api
```

---

## 9. GCP Useful Commands

Check active project:

```bash
gcloud config get-value project
```

Set project:

```bash
gcloud config set project project-a74bda7f-3a9e-4ff2-a23
```

List GKE clusters:

```bash
gcloud container clusters list
```

Get GKE credentials:

```bash
gcloud container clusters get-credentials \
  oppe2-cluster \
  --zone us-central1-a \
  --project project-a74bda7f-3a9e-4ff2-a23
```

Query Cloud Logging:

```bash
gcloud logging read \
'resource.type="k8s_container" AND labels.k8s-pod/app="heart-disease-api" AND jsonPayload.predicted_output:*' \
--limit=100
```

---

## 10. Git Useful Commands

Check status:

```bash
git status
```

Check branch:

```bash
git branch
```

Check remote:

```bash
git remote -v
```

View history:

```bash
git log --oneline -10
```

Add changes:

```bash
git add .
```

Commit:

```bash
git commit -m "commit message"
```

Push:

```bash
git push origin main
```

Verify clean repository:

```bash
git status
```

Expected:

```text
nothing to commit, working tree clean
```

---

## 11. Important Git Commit History

The project was developed incrementally.

Important commits include:

```text
ae706a9  Complete D2: SHAP model explainability
7aeb7a4  Complete D3: Fairlearn fairness analysis
401759a  Complete D4: Dockerized API deployment on GKE with CI-CD
05d49b7  Configure GCP authentication for CI-CD
4d1917c  Add per-sample prediction logging
f43d0e6  Complete D5: Per-sample predictions and logging
471d699  Complete D6 stress testing and D7 input drift detection
```

Final commit:

```text
471d699
```

The final branch is:

```text
main
```

and is synchronized with:

```text
origin/main
```

---

## 12. Security

Sensitive credentials are excluded from Git.

`.gitignore` includes:

```text
.env/
.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
github-actions-key.json
```

The GCP service account key was used as a GitHub Actions secret:

```text
GCP_SA_KEY
```

The local key file is excluded from version control.

---

## 13. End-to-End MLOps Workflow

The complete implementation can be summarized as:

```text
                HEART DISEASE DATA
                       |
                       v
             MODEL TRAINING
                       |
          +------------+------------+
          |                         |
          v                         v
       SHAP                    FAIRLEARN
   Explainability               Fairness
          |                         |
          +------------+------------+
                       |
                       v
                  FASTAPI
                       |
                       v
                    DOCKER
                       |
                       v
              ARTIFACT REGISTRY
                       |
                       v
                GITHUB ACTIONS
                       |
                       v
                     GKE
                       |
              +--------+--------+
              |                 |
              v                 v
          LoadBalancer         HPA
              |              max = 3
              |
              v
           /predict
              |
      +-------+--------+
      |       |        |
      v       v        v
   D5 Logs   D6      D7 Drift
   100 rows  wrk     Detection
             2001
```

---

## 14. Final Results Summary

### Explainability

Least impactful features:

```text
exang
thal
fbs
gender
ca
```

### Fairness

Sensitive attribute:

```text
age
```

Demographic parity difference:

```text
0.37143
```

Demographic parity ratio:

```text
0.53571
```

### Deployment

```text
Platform: GKE
API: FastAPI
Container: Docker
Registry: Artifact Registry
Service: LoadBalancer
HPA: 1-3 pods
CPU target: 60%
```

### Observability

```text
100 individual prediction requests
Structured Cloud Logging
Input features logged
Prediction logged
Timestamp logged
```

### Stress Testing

```text
Concurrency: 2001
Throughput: 171.72 requests/sec
P50: 1.31 sec
P99: 1.82 sec
Timeouts: 4960
```

### Drift Detection

```text
Features analyzed: 14
Features with drift: 12
Drift percentage: 85.71%
```

---

## 15. Final Checklist

Before submission, verify:

- [x] Private GitHub repository
- [x] Correct repository naming
- [x] Required collaborator added and accepted
- [x] D2 SHAP explainability completed
- [x] Least impactful features identified
- [x] D3 Fairlearn completed
- [x] Age used as sensitive attribute
- [x] FastAPI application created
- [x] Docker image built
- [x] Docker image pushed to Artifact Registry
- [x] GKE deployment running
- [x] LoadBalancer configured
- [x] HPA configured
- [x] Maximum pods = 3
- [x] GitHub Actions CI/CD configured
- [x] GitHub Actions workflow passed
- [x] 100-row dataset generated
- [x] 100 individual predictions sent
- [x] Per-sample logging implemented
- [x] Cloud Logging verified
- [x] wrk installed
- [x] >2000 concurrent connections tested
- [x] Throughput analyzed
- [x] Latency distribution analyzed
- [x] Timeout behavior analyzed
- [x] Input drift calculated
- [x] Training vs incoming data compared
- [x] D6 evidence saved
- [x] D7 drift report saved
- [x] All changes committed
- [x] Changes pushed to main
- [x] Working tree clean

---

## 16. Conclusion

This project demonstrates an end-to-end MLOps deployment of a heart disease prediction model.

The solution integrates:

- Explainability using SHAP
- Fairness using Fairlearn
- Containerization using Docker
- Cloud deployment using GKE
- Autoscaling using Kubernetes HPA
- CI/CD using GitHub Actions
- Observability using Google Cloud Logging
- Performance testing using wrk
- Data drift detection using KS test and PSI

The final system provides a complete workflow from model training and responsible AI analysis to containerized cloud deployment, monitoring, scalability testing, and data-drift monitoring.

### Final Status

**OPPE-2 implementation completed end-to-end.**