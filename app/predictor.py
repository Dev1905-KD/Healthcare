import os
import joblib
import pandas as pd
import numpy as np
from app.schemas import InsuranceInput, PredictionOutput, RiskFactor, ModelInfoOutput

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "healthcare_model.pkl")

class CostPredictor:
    def __init__(self):
        self.model_pkg = None
        self.pipeline = None
        self.model_name = "Unknown"
        self.metrics = {}
        self.dataset_stats = {}
        self._load_model()

    def _load_model(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. Please run 'python training/train.py' first!"
            )
        self.model_pkg = joblib.load(MODEL_PATH)
        self.pipeline = self.model_pkg['pipeline']
        self.model_name = self.model_pkg.get('model_name', 'Tuned ML Regressor')
        self.metrics = self.model_pkg.get('metrics', {})
        self.dataset_stats = self.model_pkg.get('dataset_stats', {
            'mean_charges': 13270.42,
            'median_charges': 9382.03,
            'min_charges': 1121.87,
            'max_charges': 63770.43,
            'std_charges': 12110.01,
            'p25_charges': 4740.29,
            'p75_charges': 16639.94
        })

    def predict(self, data: InsuranceInput) -> PredictionOutput:
        # Convert input Pydantic model to DataFrame for scikit-learn pipeline
        input_dict = {
            'age': [data.age],
            'sex': [data.sex.lower()],
            'bmi': [float(data.bmi)],
            'children': [data.children],
            'smoker': [data.smoker.lower()],
            'region': [data.region.lower()]
        }
        df_input = pd.DataFrame(input_dict)
        
        # Make prediction
        raw_pred = float(self.pipeline.predict(df_input)[0])
        # Ensure non-negative realistic prediction
        predicted_cost = max(1000.0, round(raw_pred, 2))
        formatted_cost = f"${predicted_cost:,.2f}"
        
        # Determine Risk Level & Badge Color
        if predicted_cost < 6000:
            risk_level = "Low Risk"
            risk_badge_color = "success"  # Soft pastel green
            risk_score_pct = round(min(25.0, (predicted_cost / 6000) * 25), 1)
        elif predicted_cost < 14000:
            risk_level = "Moderate Risk"
            risk_badge_color = "info"     # Soft pastel blue/teal
            risk_score_pct = round(25.0 + ((predicted_cost - 6000) / 8000) * 25, 1)
        elif predicted_cost < 28000:
            risk_level = "High Risk"
            risk_badge_color = "warning"  # Soft pastel amber
            risk_score_pct = round(50.0 + ((predicted_cost - 14000) / 14000) * 25, 1)
        else:
            risk_level = "Very High Risk"
            risk_badge_color = "danger"   # Soft pastel rose
            risk_score_pct = round(min(99.0, 75.0 + ((predicted_cost - 28000) / 35000) * 24), 1)

        # Estimate Population Percentile (approximate log-normal fit / empirical CDF)
        p25 = float(self.dataset_stats.get('p25_charges', 4740))
        median = float(self.dataset_stats.get('median_charges', 9382))
        p75 = float(self.dataset_stats.get('p75_charges', 16640))
        
        if predicted_cost <= p25:
            denom = max(1.0, p25)
            cost_percentile = round((predicted_cost / denom) * 25, 1)
        elif predicted_cost <= median:
            denom = max(1.0, median - p25)
            cost_percentile = round(25 + ((predicted_cost - p25) / denom) * 25, 1)
        elif predicted_cost <= p75:
            denom = max(1.0, p75 - median)
            cost_percentile = round(50 + ((predicted_cost - median) / denom) * 25, 1)
        else:
            denom = max(1.0, 63000.0 - p75)
            cost_percentile = round(min(99.9, 75 + ((predicted_cost - p75) / denom) * 24.9), 1)

        # Risk Factors Analysis
        risk_factors = []
        if data.smoker.lower() == 'yes':
            risk_factors.append(RiskFactor(
                factor="Tobacco Usage (Smoker)",
                impact="+$14,000 to +$20,000 / yr",
                severity="critical",
                description="Smoking is the single largest predictor of elevated healthcare costs, increasing risk of cardio-respiratory conditions."
            ))
            
        if data.bmi >= 30.0:
            severity_str = "critical" if data.smoker.lower() == 'yes' else "high"
            risk_factors.append(RiskFactor(
                factor=f"Elevated BMI ({data.bmi} - Obese)",
                impact="+$3,000 to +$10,000 / yr",
                severity=severity_str,
                description="A Body Mass Index ≥ 30 significantly amplifies medical charges due to metabolic and joint stress factors."
            ))
        elif data.bmi >= 25.0:
            risk_factors.append(RiskFactor(
                factor=f"Overweight BMI ({data.bmi})",
                impact="+$1,200 to +$3,000 / yr",
                severity="medium",
                description="BMI is slightly above optimal range (18.5 - 24.9), contributing moderate risk to long-term care expense."
            ))

        if data.age >= 50:
            risk_factors.append(RiskFactor(
                factor=f"Advancing Age ({data.age} yrs)",
                impact="+$4,500 to +$8,000 / yr",
                severity="medium",
                description="Age-related physiological changes increase baseline diagnostic and preventative care utilization."
            ))
        elif data.age >= 35:
            risk_factors.append(RiskFactor(
                factor=f"Mature Demographic ({data.age} yrs)",
                impact="+$2,000 / yr",
                severity="low",
                description="Age baseline steadily increases medical consumption over time."
            ))

        if data.children >= 3:
            risk_factors.append(RiskFactor(
                factor=f"Multiple Dependents ({data.children} Children)",
                impact="+$1,500 / yr",
                severity="low",
                description="Higher dependent count elevates routine family care visits and policy coverage volume."
            ))

        if not risk_factors:
            risk_factors.append(RiskFactor(
                factor="Optimal Health Profile",
                impact="Baseline Savings",
                severity="low",
                description="Non-smoker status, healthy BMI, and young demographic maintain costs well below population average."
            ))

        # Actionable Health Recommendations
        recommendations = []
        if data.smoker.lower() == 'yes':
            recommendations.append("Smoking Cessation Program: Enrolling in a tobacco cessation plan can reduce predicted annual charges by up to 60%.")
        if data.bmi >= 25.0:
            recommendations.append("Targeted BMI Management: Aiming for a target BMI under 25 through structured nutrition and fitness can yield up to $4,000/yr in health cost savings.")
        recommendations.append("Preventative Health Screenings: Annual biometric wellness checks help catch chronic conditions early, keeping out-of-pocket expenses minimal.")
        recommendations.append("High-Deductible + HSA Strategy: Consider pair-funding a Health Savings Account (HSA) for tax-advantaged care coverage.")

        return PredictionOutput(
            predicted_cost=predicted_cost,
            formatted_cost=formatted_cost,
            risk_level=risk_level,
            risk_badge_color=risk_badge_color,
            risk_score_percentage=risk_score_pct,
            cost_percentile=cost_percentile,
            top_risk_factors=risk_factors,
            health_recommendations=recommendations,
            model_used=self.model_name,
            model_r2_score=self.metrics.get('r2', 0.85)
        )

    def get_model_info(self) -> ModelInfoOutput:
        return ModelInfoOutput(
            model_name=self.model_name,
            metrics=self.metrics,
            best_params=self.model_pkg.get('best_params', {}),
            feature_names=self.model_pkg.get('feature_names', []),
            dataset_stats=self.dataset_stats
        )
