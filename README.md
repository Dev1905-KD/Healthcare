# Healthcare Cost Prediction using Machine Learning & FastAPI

A modern, end-to-end Machine Learning web application designed to predict annual healthcare charges for patients based on demographic details, physical metrics, and lifestyle choices. Built with **FastAPI**, **scikit-learn**, **XGBoost**, **Pydantic**, and a minimal, classy **Pastel Web UI**.

---

## 🌟 Architecture & Workflow

```
User (Browser)
     ↓
Pastel Minimalist Web UI (HTML5 / Custom CSS / Vanilla JS)
     ↓
FastAPI Backend (app/main.py)
     ↓
Input Validation & Schemas (Pydantic app/schemas.py)
     ↓
Domain Predictor & Risk Scoring (app/predictor.py)
     ↓
ColumnTransformer + Trained ML Pipeline (model/healthcare_model.pkl)
     ↓
Predicted Cost & Risk Factors Breakdown
```

### ML Training & Hyperparameter Tuning Pipeline
```
data/insurance.csv
     ↓
Exploratory Data Analysis (EDA)
     ↓
Train/Test Split (80 / 20)
     ↓
ColumnTransformer
  ├── Numerical (age, bmi, children) → Median Imputer + StandardScaler
  └── Categorical (sex, smoker, region) → Most Frequent Imputer + OneHotEncoder
     ↓
Candidate Models Evaluated:
  ├── Ridge Regression
  ├── Random Forest Regressor
  └── XGBoost Regressor
     ↓
RandomizedSearchCV (5-Fold Cross Validation)
     ↓
Best Model Selection (Evaluated on R², MAE, RMSE)
     ↓
Save model/healthcare_model.pkl
```

---

## 🚀 Why `RandomizedSearchCV` over `GridSearchCV`?

> **Interview Justification**:
> In real-world machine learning engineering:
> 1. **Efficiency**: `GridSearchCV` exhaustively tests every combination of hyperparameter values, scaling exponentially ($O(\prod N_i)$ computational complexity).
> 2. **Exploration Space**: `RandomizedSearchCV` samples a fixed number of parameter combinations (`n_iter`) from a much broader range of continuous and discrete distributions, allowing us to evaluate wider ranges of learning rates, tree depths, and regularizations without exponential runtimes.
> 3. **Resource Management**: Randomized search achieves comparable or superior model optimization with significantly less compute time.

---

## 📁 Project Structure

```
healthcare-cost-prediction/
│
├── data/
│   └── insurance.csv            # Kaggle Medical Cost Personal dataset
│
├── model/
│   └── healthcare_model.pkl     # Serialized pipeline & metadata
│
├── app/
│   ├── main.py                  # FastAPI application & routes
│   ├── schemas.py               # Pydantic input & output models
│   └── predictor.py             # Model inference & domain risk breakdown
│
├── training/
│   ├── download_data.py         # Dataset downloader / generator
│   └── train.py                 # EDA, ColumnTransformer, RandomizedSearchCV, & saving
│
├── templates/
│   └── index.html               # Modern pastel web dashboard
│
├── static/
│   ├── style.css                # Minimal classy pastel styling system
│   └── script.js                # Async fetch, counter animations, & UI modal
│
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## ⚙️ Setup & Installation

### 1. Clone & Create Virtual Environment
```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Training Pipeline & Hyperparameter Tuning
```bash
python training/train.py
```
*Outputs EDA summary, model leaderboard comparing Ridge, Random Forest, and XGBoost, and saves `model/healthcare_model.pkl`.*

### 4. Start FastAPI Server
```bash
uvicorn app.main:app --reload --port 8000
```

### 5. Access Web Dashboard
Open your browser at: **`http://127.0.0.1:8000`**

---

## 📊 Model Evaluation Results

| Model | Test $R^2$ | MAE ($) | RMSE ($) | Best CV $R^2$ | Tuning Strategy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Ridge Regression | 0.7830 | $4,201.56 | $5,803.99 | 0.7331 | `RandomizedSearchCV` |
| Random Forest Regressor | 0.8786 | $2,480.00 | $4,341.39 | 0.8466 | `RandomizedSearchCV` |
| **XGBoost Regressor** *(Winner)* | **0.8815** | **$2,421.16** | **$4,289.69** | **0.8480** | `RandomizedSearchCV` |

---

## 🛡️ Risk Interpretation Logic

- **Low Risk (< $6,000)**: Non-smoker, healthy BMI (< 25), younger age demographic.
- **Moderate Risk ($6,000 – $14,000)**: Non-smoker with overweight BMI or middle-aged demographic.
- **High Risk ($14,000 – $28,000)**: Smoker or obese patient (BMI > 30) with moderate age.
- **Very High Risk (> $28,000)**: Smoker + Obese BMI combined factor (strongest medical cost driver in insurance datasets).

---

## 📑 API Endpoints

- `GET /`: Serves web interface dashboard.
- `POST /predict`: Predicts annual healthcare expense given `InsuranceInput` JSON.
- `GET /model-info`: Returns model metrics ($R^2$, MAE, RMSE) and tuned hyperparameter choices.
- `GET /health`: Returns service health status.
