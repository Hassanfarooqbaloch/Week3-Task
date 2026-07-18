# ⚡ Steel Industry Energy Consumption Prediction

A Machine Learning and FastAPI project developed for Week 3 that explores **Principal Component Analysis (PCA)** for dimensionality reduction and deploys a **Random Forest regression model** through an interactive FastAPI web application.

---

# 📌 Project Overview

The project consists of two major components:

- **PCA Analysis**
  - Evaluate dimensionality reduction.
  - Compare model performance before and after PCA.
  - Visualize explained variance and principal components.

- **FastAPI Web Dashboard**
  - Display exploratory data analysis (EDA) charts.
  - Allow users to predict electricity consumption through a web interface.

---

# 📂 Project Layout

```
.
├── data/
│   └── Steel_industry_data_engineered.csv
├── notebooks/
│   └── week3_pca.ipynb
├── outputs/
├── static/
│   ├── style.css
│   ├── avg_usage_by_hour.png
│   ├── avg_usage_by_loadtype.png
│   └── correlation_heatmap.png
├── templates/
│   ├── home.html
│   ├── dashboard.html
│   └── predict.html
├── main.py
├── model.joblib
├── model_metadata.joblib
├── requirements.txt
└── README.md
```

---

# 📊 PCA Analysis

The notebook follows the same preprocessing pipeline developed in Week 2.

### Workflow

- Imported the engineered dataset
- Applied one-hot encoding (20 total features)
- Used an 80/20 train-test split (`random_state=42`)
- Standardized only the training data to prevent data leakage
- Applied PCA with all available components
- Generated Scree Plot and Cumulative Variance Plot
- Identified the number of components needed to preserve 95% of the variance
- Retrained Random Forest models using reduced feature spaces
- Compared model performance with the original feature set
- Produced a loading heatmap for the first three principal components

---

# 📈 Model Performance

| Model | Features | MAE | RMSE | R² |
|------|---------:|----:|-----:|----:|
| Original Random Forest | 20 | 0.351 | 1.045 | 0.9990 |
| PCA (3 Components) | 3 | 3.437 | 7.152 | 0.9550 |
| PCA (10 Components) | 10 | 1.615 | 3.212 | 0.9909 |

---

# 🔍 Findings

The experiments indicate that PCA decreases prediction accuracy for this regression task.

Although reducing the feature space to 10 principal components retains approximately 95% of the total variance, the Random Forest trained on the original features consistently achieves better performance.

Because Random Forest inherently performs feature selection while constructing decision trees, PCA provides little benefit for this model and instead removes information useful for prediction.

---

# 🤖 Production Model

For deployment, the original model was optimized to reduce storage requirements.

### Original Model

- 200 trees
- Unlimited depth
- Approximate size: **380 MB**

### Deployment Model

- 100 trees
- `max_depth = 12`
- `min_samples_leaf = 3`

Performance:

| Metric | Value |
|--------|------:|
| MAE | 0.634 |
| RMSE | 1.533 |
| R² | 0.998 |
| Model Size | ~18 MB |

This optimized model is stored as **model.joblib** and is used by the FastAPI application.

---

# 🌐 FastAPI Application

The web application provides three main pages.

## Home

```
GET /
```

Landing page with navigation links.

---

## Dashboard

```
GET /dashboard
```

Displays the EDA visualizations:

- Average Usage by Hour
- Average Usage by Load Type
- Correlation Heatmap

---

## Prediction

```
GET /predict
```

Displays a dynamic prediction form generated from `model_metadata.joblib`.

The form contains:

- 11 numerical inputs
- 3 categorical dropdown menus

---

## Prediction Endpoint

```
POST /predict
```

Processes user input, creates a DataFrame, predicts electricity usage using the trained model, and returns the estimated **Usage_kWh**.

---

# ▶️ Running the Application

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the server:

```bash
uvicorn main:app --reload
```

Open the application in your browser:

| Page | URL |
|------|-----|
| Home | http://127.0.0.1:8000/ |
| Dashboard | http://127.0.0.1:8000/dashboard |
| Predict | http://127.0.0.1:8000/predict |

---

# 📒 Reproducing the PCA Analysis

Run every cell in:

```
notebooks/week3_pca.ipynb
```

Required dataset:

```
data/Steel_industry_data_engineered.csv
```

The notebook regenerates:

- PCA visualizations
- Random Forest model
- model.joblib
- model_metadata.joblib

---

# 🚀 Future Enhancements

- Compare PCA with Ridge and Lasso Regression
- Improve prediction form validation
- Dockerize the FastAPI application
- Create a REST API endpoint for JSON predictions
- Deploy the application to a cloud platform

---

# 📚 Technologies Used

- Python
- FastAPI
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Jinja2
- Joblib

---

# 📄 License

Dataset:

**Steel Industry Energy Consumption Dataset**

Licensed under **CC BY 4.0**

Authors:

- V. E. S.
- C. Shin
- Y. Cho

Available through the UCI Machine Learning Repository.
