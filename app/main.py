import os

import fastapi
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import InsuranceInput, PredictionOutput, ModelInfoOutput
from app.predictor import CostPredictor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "frontend/static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "frontend/templates")

# Initialize predictor singleton lazily or on startup
predictor_instance = None

def get_predictor() -> CostPredictor:
    global predictor_instance
    if predictor_instance is None:
        try:
            predictor_instance = CostPredictor()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model predictor failed to initialize: {str(e)}"
            )
    return predictor_instance

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("FastAPI server starting up...")
    try:
        get_predictor()
        print("Healthcare Cost Predictor initialized successfully!")
    except Exception as e:
        print(f"Warning on startup: {e}")
    yield
    print("FastAPI server shutting down...")

app = FastAPI(
    title="AuraHealth • Healthcare Cost Prediction API",
    description="FastAPI Machine Learning Service with RandomizedSearchCV model pipeline for predicting annual healthcare charges.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static assets and Jinja2 templates
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    """Serves the main Pastel Web UI dashboard."""
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/predict", response_model=PredictionOutput)
async def predict_cost(data: InsuranceInput):
    """Predict annual healthcare charges based on patient demographics and lifestyle factors."""
    predictor = get_predictor()
    try:
        result = predictor.predict(data)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )

@app.get("/model-info", response_model=ModelInfoOutput)
async def get_model_info():
    """Retrieve metadata, performance scores (R², MAE, RMSE), and tuned hyperparameters."""
    predictor = get_predictor()
    return predictor.get_model_info()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    predictor = get_predictor()
    return {
        "status": "healthy",
        "model_loaded": predictor.pipeline is not None,
        "model_name": predictor.model_name
    }
