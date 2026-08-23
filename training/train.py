import os
import joblib
import numpy as np
import pandas as pd
import xgboost
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    from download_data import download_or_generate_dataset
except ModuleNotFoundError:
    from training.download_data import download_or_generate_dataset


DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "insurance.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "healthcare_model.pkl")

def run_eda(df: pd.DataFrame):
    print("=" * 60)
    print("                     EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nData Types & Non-Null Counts:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nTarget Variable ('charges') Summary Statistics:")
    print(df['charges'].describe())
    
    # Key domain insights
    smoker_charges = df.groupby('smoker')['charges'].agg(['mean', 'median', 'std'])
    print("\nCharges Breakdown by Smoker Status:")
    print(smoker_charges)
    
    df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'], include_lowest=True)
    bmi_smoker_charges = df.groupby(['bmi_category', 'smoker'], observed=False)['charges'].mean().unstack()
    print("\nMean Charges by BMI Category and Smoker Status:")
    print(bmi_smoker_charges)
    df.drop(columns=['bmi_category'], inplace=True, errors='ignore')
    print("=" * 60 + "\n")

def build_preprocessing_pipeline(num_features, cat_features):
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_features),
            ('cat', cat_transformer, cat_features)
        ]
    )
    return preprocessor

def train_and_evaluate():
    # Step 1: Ensure dataset is available
    download_or_generate_dataset()
    
    df = pd.read_csv(DATA_PATH)
    run_eda(df)
    
    # Step 2: Define Features and Target
    X = df.drop(columns=['charges'])
    y = df['charges']
    
    num_features = ['age', 'bmi', 'children']
    cat_features = ['sex', 'smoker', 'region']
    
    # Step 3: Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train set size: {X_train.shape[0]} | Test set size: {X_test.shape[0]}")
    
    # Step 4: Preprocessing Pipeline
    preprocessor = build_preprocessing_pipeline(num_features, cat_features)
    
    # Step 5: Models & Hyperparameter Spaces for RandomizedSearchCV
    model_configs = {
        'Ridge Regression': {
            'model': Ridge(),
            'params': {
                'regressor__alpha': np.logspace(-2, 3, 50)
            },
            'n_iter': 15
        },
        'Random Forest': {
            'model': RandomForestRegressor(random_state=42),
            'params': {
                'regressor__n_estimators': [50, 100, 150, 200, 300],
                'regressor__max_depth': [None, 5, 8, 12, 15, 20],
                'regressor__min_samples_split': [2, 5, 10],
                'regressor__min_samples_leaf': [1, 2, 4],
                'regressor__max_features': ['sqrt', 'log2', 1.0]
            },
            'n_iter': 25
        },
        'XGBoost': {
            'model': XGBRegressor(random_state=42, objective='reg:squarederror'),
            'params': {
                'regressor__n_estimators': [50, 100, 150, 200, 300],
                'regressor__max_depth': [3, 4, 5, 6, 8],
                'regressor__learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
                'regressor__subsample': [0.6, 0.8, 1.0],
                'regressor__colsample_bytree': [0.6, 0.8, 1.0],
                'regressor__reg_alpha': [0, 0.1, 1, 10],
                'regressor__reg_lambda': [0.1, 1, 10]
            },
            'n_iter': 25
        }
    }
    
    results = []
    best_overall_score = -np.inf
    best_overall_model_pkg = None
    
    print("\n" + "=" * 60)
    print("           MODEL TUNING WITH RandomizedSearchCV")
    print("=" * 60)
    
    for model_name, config in model_configs.items():
        print(f"\n---> Tuning {model_name}...")
        
        # Build full pipeline with placeholder regressor
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', config['model'])
        ])
        
        # RandomizedSearchCV
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=config['params'],
            n_iter=config['n_iter'],
            cv=5,
            scoring='r2',
            random_state=42,
            n_jobs=-1
        )
        
        search.fit(X_train, y_train)
        best_pipeline = search.best_estimator_
        
        # Predictions on test set
        y_pred = best_pipeline.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"[{model_name}] Best CV R2: {search.best_score_:.4f}")
        print(f"[{model_name}] Test R2: {r2:.4f} | MAE: ${mae:.2f} | RMSE: ${rmse:.2f}")
        print(f"[{model_name}] Best Hyperparameters: {search.best_params_}")
        
        results.append({
            'Model': model_name,
            'Test R2': r2,
            'MAE ($)': mae,
            'RMSE ($)': rmse,
            'Best CV R2': search.best_score_,
            'Best Params': search.best_params_,
            'Pipeline': best_pipeline
        })
        
        if r2 > best_overall_score:
            best_overall_score = r2
            
            # Extract feature names after OneHotEncoder
            ohe = best_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
            cat_encoded_names = ohe.get_feature_names_out(cat_features).tolist()
            feature_names = num_features + cat_encoded_names
            
            # Extract feature importances if available
            reg = best_pipeline.named_steps['regressor']
            importances = None
            if hasattr(reg, 'feature_importances_'):
                importances = dict(zip(feature_names, [float(v) for v in reg.feature_importances_]))
            elif hasattr(reg, 'coef_'):
                coefs = np.ravel(reg.coef_)
                importances = dict(zip(feature_names, [float(v) for v in coefs]))
                
            best_overall_model_pkg = {
                'pipeline': best_pipeline,
                'model_name': model_name,
                'best_params': search.best_params_,
                'metrics': {
                    'r2': round(float(r2), 4),
                    'mae': round(float(mae), 2),
                    'rmse': round(float(rmse), 2),
                    'best_cv_r2': round(float(search.best_score_), 4)
                },
                'feature_names': feature_names,
                'feature_importances': importances,
                'dataset_stats': {
                    'mean_charges': round(float(y.mean()), 2),
                    'median_charges': round(float(y.median()), 2),
                    'min_charges': round(float(y.min()), 2),
                    'max_charges': round(float(y.max()), 2),
                    'std_charges': round(float(y.std()), 2),
                    'p25_charges': round(float(y.quantile(0.25)), 2),
                    'p75_charges': round(float(y.quantile(0.75)), 2)
                }
            }
            
    # Step 6: Display Leaderboard
    leaderboard = pd.DataFrame(results)[['Model', 'Test R2', 'MAE ($)', 'RMSE ($)', 'Best CV R2']]
    print("\n" + "=" * 60)
    print("                     FINAL MODEL LEADERBOARD")
    print("=" * 60)
    print(leaderboard.to_string(index=False))
    if best_overall_model_pkg:
        print(f"\nWinning Model: {best_overall_model_pkg['model_name']} with Test R2 = {best_overall_score:.4f}")
    else:
        print("\nNo model succeeded during tuning.")
    
    # Step 7: Save Best Model Artifact
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_overall_model_pkg, MODEL_SAVE_PATH)
    print(f"\nSuccessfully saved model package to: {MODEL_SAVE_PATH}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    train_and_evaluate()
