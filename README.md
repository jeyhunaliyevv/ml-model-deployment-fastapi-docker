# Medical Insurance Risk — FastAPI Model Deployment

![API Tests and Docker Build](https://github.com/jeyhunaliyevv/ml-model-deployment-fastapi-docker/actions/workflows/ci.yml/badge.svg)

An **independent ML engineering portfolio project** showing how the selected CatBoost classifier from the [Medical Insurance Cost Risk Prediction](https://github.com/jeyhunaliyevv/medical-insurance-cost-risk-prediction) project can be trained, validated at the API boundary, and exposed as a JSON prediction service.

**Status:** FastAPI endpoints and pytest tests have been verified locally with the legitimate source dataset; Docker is configured but no local Docker executable was available for direct build verification. CI uses a fictional model to test actual container inference once pushed to GitHub. The repository includes Docker configuration and GitHub Actions, but **cloud deployment is not claimed** until a real AWS endpoint is provisioned and tested. The GitHub badge only becomes meaningful after pushing this project and running CI.

## Architecture

```text
Client JSON --> Pydantic validation --> FastAPI /predict --> feature engineering
                                                       --> fitted preprocessing
                                                       --> CatBoost classifier
                                                       --> probability + risk flag
```

`GET /health` returns **200** if the model is available, otherwise **503**. `POST /predict` validates raw input and returns **200**, **422** for invalid data, or **503** when the model has not been trained. Interactive API documentation: `/docs`.

## Dataset and model provenance

- Uses the same six predictor columns as the source project: `age`, `sex`, `bmi`, `children`, `smoker`, `region`.
- High-cost label: `charges >= $20,000`, a fixed **illustrative** threshold.
- The *source repository* selected CatBoost using training-fold CV. This deployment project **re-trains that selected algorithm on the available source data** for serving; it does not re-run model selection or claim a new holdout score.
- Dataset `data/insurance.csv` is **not distributed** here. See [`data/README.md`](data/README.md) for its upstream location. No employer data are used.
- Local `.joblib` artifacts are untrusted pickle-derived content if obtained externally. **Never load model artifacts from unknown sources**. The model generated with our own training script stays gitignored.

## Quick start (Python 3.11)

From the repository root:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
```

1. Independently download the public educational CSV from the [upstream dataset link](https://github.com/stedy/Machine-Learning-with-R-datasets/blob/master/insurance.csv) and save it as `data/insurance.csv`.
2. Train the selected model to create the ignored artifact:

```bash
python -m scripts.train_model --data data/insurance.csv
```

3. Start the service:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

4. Open http://127.0.0.1:8000/docs or make a request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":35,"sex":"female","bmi":27.1,"children":1,"smoker":"no","region":"northeast"}'
```

On PowerShell, use `curl.exe` or the interactive `/docs` interface. Response shape (illustrative; **not an actual score**):

```json
{
  "high_cost_probability": 0.87,
  "predicted_high_cost": true,
  "threshold_usd": 20000.0,
  "decision_threshold": 0.5,
  "disclaimer": "Educational prediction only; not a medical or underwriting decision."
}
```

To run API tests:

```bash
python -m pytest tests/ -v
```

## Docker

**Docker with a real trained model (local only):** First run `python -m scripts.train_model`, so `models/best_model.joblib` exists. Then:

```bash
docker build -t medical-risk-api .
docker run --rm -p 8000:8000 medical-risk-api
```

The regular Dockerfile copies the locally trained artifact into the image; **never publish an image containing a model without reviewing the dataset and redistribution rights first**. In CI, a temporary model is trained from entirely generated fictional data before building the same Dockerfile. That image is exercised with API smoke tests and **never published**. This tests the deployment plumbing, not real-world model accuracy or AWS hosting.

## GitHub Actions

`.github/workflows/ci.yml` runs API tests, generates a **throwaway model using fully fictional test records**, builds a Docker image with that CI-only fixture, and verifies live `/health` and `/predict` responses. This checks container serving without distributing the actual source dataset or production model. CI images are **never published**, and synthetic test results are **not model performance benchmarks**. Real AWS deployment is not claimed.

## AWS deployment — future deployment checklist (not deployed)

1. Train a legitimate model locally and validate `/predict` and `/health`.
2. Review data/model licensing and artifact provenance before packaging.
3. Choose AWS App Runner/ECS Fargate/ECR or other suitable deployment; examine cost, IAM permissions and secret handling.
4. Build a model-equipped image and push to a **private** ECR repository only after reviewing permissions.
5. Deploy behind HTTPS and confirm the actual live `/health` and `/predict` responses.
6. Configure logging, latency/error metrics, safeguards on requests, and budget alerts. Avoid logging personal medical inputs.
7. Only then add a public live URL and verifiable AWS deployment claim.

## Limitations and responsible use

This is an educational classification demo using a small public sample dataset, not claims history. No medical, clinical, individualized premium or underwriting decisions should be made from it. Risk scores are not proved to be well calibrated for deployment. The model may reflect source-data demographic biases; fairness and privacy assessment would be required for real-world use. No measured business cost savings or real AWS hosting are claimed.

## License

MIT license covers original source code and documentation only; not the third-party dataset or third-party model artifacts.
