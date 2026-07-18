

## 📁 Repository Structure

```
.
├── data/
│   └── Steel_industry_data_engineered.csv   # Feature-engineered dataset carried over from Week 2
├── notebooks/
│   └── week3_pca.ipynb                       # Part 1: PCA analysis, all cells run, outputs visible
├── outputs/                                  # Saved PCA charts (scree plot, cumulative variance, etc.)
├── static/                                   # CSS + dashboard PNGs served by FastAPI
│   ├── style.css
│   ├── avg_usage_by_hour.png
│   ├── avg_usage_by_loadtype.png
│   └── correlation_heatmap.png
├── templates/                                # Jinja2 HTML templates
│   ├── home.html
│   ├── dashboard.html
│   └── predict.html
├── main.py                                   # FastAPI application
├── model.joblib                              # Saved production pipeline (preprocessing + Random Forest)
├── model_metadata.joblib                     # Feature names + categorical options, used to build the form
├── requirements.txt
└── README.md
```

## Part 1 — PCA (`notebooks/week3_pca.ipynb`)

- Recreated the exact Week 2 feature set and one-hot encoding (20 total encoded features),
  and the exact 80/20 split (`random_state=42`)
- **StandardScaler and PCA fit on the training set only**, then used to transform both train
  and test sets (no data leakage from the test set into scaling/PCA parameters)
- Full PCA (`n_components=20`) to inspect variance explained per component
- Scree plot (bar chart) of explained variance ratio per component
- Cumulative explained variance curve with a 95% threshold line — **10 of 20 components**
  are needed to reach 95% variance
- Retrained Random Forest on **3 PCA components** and on the **10 components (95% variance)**
- Loading heatmap: original features × first 3 principal components
- Full written Dimensionality Reduction Report (see notebook, section 10)

### Results

| Version | # Features | MAE | RMSE | R² |
|---|---|---|---|---|
| Original (Week 2, all features) | 20 | 0.351 | 1.045 | 0.9990 |
| 3-component PCA (55.7% variance) | 3 | 3.437 | 7.152 | 0.9550 |
| 10-component PCA (95% variance) | 10 | 1.615 | 3.212 | 0.9909 |

**Key finding:** accuracy drop is real and scales with compression — the 3-component model is
~7x worse on RMSE than the original; even the 95%-variance model is ~3x worse. Random Forest
already does implicit feature selection via its splits, so PCA repackages information the model
could use more directly, at a real accuracy cost. **PCA is not recommended for this specific
model** unless a memory/latency constraint makes the trade-off worthwhile — it would pay off
more clearly for linear models. Full reasoning is in the notebook's Dimensionality Reduction
Report.

**Production model note:** the original unconstrained Random Forest (200 trees, no depth limit)
serializes to ~380 MB — too large for GitHub (100 MB limit) and unnecessarily heavy for a demo
dashboard. The deployed `model.joblib` uses `max_depth=12, min_samples_leaf=3, n_estimators=100`
— a deliberate size/accuracy trade-off (MAE 0.634, RMSE 1.533, R² 0.998, ~18 MB) documented in
the notebook. This is the model served by the FastAPI app below.

## Part 2 — FastAPI Dashboard (`main.py`)

- **`GET /`** — welcome page with a navigation bar (Home / Dashboard / Predict)
- **`GET /dashboard`** — renders 3 Week 2 EDA visualizations as static images: average usage
  by hour, average usage by load type, and the full correlation heatmap
- **`GET /predict`** — renders an HTML form with one input field per feature the model needs
  (11 numeric fields + 3 categorical dropdowns, built dynamically from `model_metadata.joblib`
  so the form always matches the deployed model)
- **`POST /predict`** — reads the form, builds a one-row DataFrame, runs it through
  `model.joblib`, and re-renders the same page with the predicted `Usage_kWh`
- All routes tested locally and confirmed working (200 OK on every GET route, and a real
  end-to-end prediction returned successfully on POST) before this was submitted

## 🚀 How to Run
uvicorn main:app --reload
```

Then visit:
- `http://127.0.0.1:8000/` — home page
- `http://127.0.0.1:8000/dashboard` — EDA dashboard
- `http://127.0.0.1:8000/predict` — live prediction form

To regenerate the PCA analysis and the production `model.joblib` / `model_metadata.joblib` from
scratch, run `notebooks/week3_pca.ipynb` top to bottom (requires `data/Steel_industry_data_engineered.csv`,
which is Week 2's engineered output, already included here).

## 🔜 Next Steps

- Standardize features and compare a feature-scaled Ridge/Lasso against PCA-reduced inputs,
  where PCA is expected to help more than it did for Random Forest here
- Add input validation ranges to the prediction form based on the training data's min/max per
  feature, to guard against wildly out-of-distribution inputs
- Containerize the FastAPI app (Dockerfile) for easier deployment
- Add a `/api/predict` JSON endpoint alongside the HTML form for programmatic access

## 📄 License

Dataset licensed under CC BY 4.0 (V E, S., Shin, C., & Cho, Y., 2021, UCI ML Repository).
