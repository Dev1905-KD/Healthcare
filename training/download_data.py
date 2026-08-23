import urllib.request
import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CSV_PATH = os.path.join(DATA_DIR, "insurance.csv")

URL = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"

def download_or_generate_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(CSV_PATH):
        print(f"Dataset already exists at {CSV_PATH}")
        return

    print("Attempting to download Kaggle insurance dataset...")
    try:
        urllib.request.urlretrieve(URL, CSV_PATH)
        df = pd.read_csv(CSV_PATH)
        print(f"Successfully downloaded dataset with {len(df)} rows and columns: {list(df.columns)}")
    except Exception as e:
        print(f"Download failed ({e}). Generating standard Kaggle-compatible insurance dataset...")
        np.random.seed(42)
        n_samples = 1338
        
        age = np.random.randint(18, 65, size=n_samples)
        sex = np.random.choice(["female", "male"], size=n_samples)
        bmi = np.round(np.random.normal(30.6, 6.1, size=n_samples), 2)
        bmi = np.clip(bmi, 15.0, 53.0)
        children = np.random.choice([0, 1, 2, 3, 4, 5], p=[0.43, 0.24, 0.18, 0.12, 0.02, 0.01], size=n_samples)
        smoker = np.random.choice(["yes", "no"], p=[0.205, 0.795], size=n_samples)
        region = np.random.choice(["southwest", "southeast", "northwest", "northeast"], size=n_samples)
        
        # Realistic charges formula matching Kaggle distribution
        charges = (
            2000 
            + age * 250 
            + (bmi - 25).clip(0) * 350 
            + (smoker == "yes") * 14000 
            + ((smoker == "yes") & (bmi > 30)) * 19000 
            + children * 450 
            + np.random.normal(0, 1200, size=n_samples)
        )
        charges = np.round(np.clip(charges, 1121.87, 63770.43), 2)
        
        df = pd.DataFrame({
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": region,
            "charges": charges
        })
        df.to_csv(CSV_PATH, index=False)
        print(f"Generated synthetic Kaggle-compatible dataset at {CSV_PATH}")

if __name__ == "__main__":
    download_or_generate_dataset()
