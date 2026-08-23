from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class InsuranceInput(BaseModel):
    age: int = Field(..., ge=18, le=100, json_schema_extra={"example": 35}, description="Age of the patient (18 to 100)")
    sex: str = Field(..., json_schema_extra={"example": "male"}, description="Gender of the patient ('male' or 'female')")
    bmi: float = Field(..., ge=10.0, le=60.0, json_schema_extra={"example": 27.5}, description="Body Mass Index (10.0 to 60.0)")
    children: int = Field(..., ge=0, le=10, json_schema_extra={"example": 1}, description="Number of dependent children (0 to 10)")
    smoker: str = Field(..., json_schema_extra={"example": "no"}, description="Smoking status ('yes' or 'no')")
    region: str = Field(..., json_schema_extra={"example": "southwest"}, description="Residential region ('northeast', 'southeast', 'southwest', 'northwest')")

class RiskFactor(BaseModel):
    factor: str
    impact: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str

class PredictionOutput(BaseModel):
    predicted_cost: float
    formatted_cost: str
    risk_level: str  # 'Low Risk', 'Moderate Risk', 'High Risk', 'Very High Risk'
    risk_badge_color: str
    risk_score_percentage: float
    cost_percentile: float
    top_risk_factors: List[RiskFactor]
    health_recommendations: List[str]
    model_used: str
    model_r2_score: float

class ModelInfoOutput(BaseModel):
    model_name: str
    metrics: Dict[str, float]
    best_params: Dict[str, Any]
    feature_names: List[str]
    dataset_stats: Dict[str, float]
