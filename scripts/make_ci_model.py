"""Train a throwaway model on entirely artificial test records for Docker CI.

Never use this model to report model accuracy or make real business decisions.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
from scripts.train_model import ROOT, train

def make_ci_model():
    rng = np.random.default_rng(123)
    n = 120
    age = rng.integers(18, 71, size=n)
    bmi = rng.uniform(18, 39, size=n).round(1)
    smoker = np.where(rng.random(n) < .28, "yes", "no")
    data = pd.DataFrame({
        "age": age,
        "sex": rng.choice(["male", "female"], size=n),
        "bmi": bmi,
        "children": rng.integers(0, 5, size=n),
        "smoker": smoker,
        "region": rng.choice(["northeast", "northwest", "southeast", "southwest"], size=n),
        "charges": 3000 + 180*age + 200*bmi + 23000*(smoker == "yes") + rng.normal(0,1200,n),
    })
    with TemporaryDirectory() as tmp:
        dataset = Path(tmp)/"fictional.csv"
        data.to_csv(dataset,index=False)
        train(dataset, ROOT/"models"/"best_model.joblib")
    print("CI fixture model created from fictional, generated data; NEVER for production.")

if __name__ == "__main__":
    make_ci_model()
