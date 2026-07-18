
import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(title="Steel Industry Energy Dashboard")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load the trained pipeline (preprocessing + Random Forest) and feature metadata once at startup
model = joblib.load("model.joblib")
metadata = joblib.load("model_metadata.joblib")

NUMERIC_FEATURES = metadata["numeric_features"]
CATEGORICAL_FEATURES = metadata["categorical_features"]
CATEGORICAL_OPTIONS = metadata["categories"]


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "home.html", {})


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(request, "dashboard.html", {})


@app.get("/predict", response_class=HTMLResponse)
async def predict_form(request: Request):
    return templates.TemplateResponse(
        request,
        "predict.html",
        {
            "numeric_features": NUMERIC_FEATURES,
            "categorical_options": CATEGORICAL_OPTIONS,
            "form_values": {},
            "prediction": None,
            "error": None,
        },
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict_submit(request: Request):
    form = await request.form()
    form_values = dict(form)

    prediction = None
    error = None

    try:
        row = {}
        for feat in NUMERIC_FEATURES:
            row[feat] = float(form_values[feat])
        for feat in CATEGORICAL_FEATURES:
            row[feat] = form_values[feat]

        input_df = pd.DataFrame([row])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        pred_value = model.predict(input_df)[0]
        prediction = round(float(pred_value), 2)
    except Exception as exc:  # noqa: BLE001 - surface any input/prediction error to the user
        error = f"Could not generate a prediction: {exc}"

    return templates.TemplateResponse(
        request,
        "predict.html",
        {
            "numeric_features": NUMERIC_FEATURES,
            "categorical_options": CATEGORICAL_OPTIONS,
            "form_values": form_values,
            "prediction": prediction,
            "error": error,
        },
    )
