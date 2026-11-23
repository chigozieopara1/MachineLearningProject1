# Concrete Compressive Strength Prediction & Cost Optimization

https://machinelearningproject1-uhnlfbns7p7shekg4fxb9p.streamlit.app/

<img width="960" height="423" alt="image" src="https://github.com/user-attachments/assets/79e0f49e-5d27-4d9d-8977-b6f6381ede1b" />

An end-to-end machine learning pipeline and interactive **Streamlit** web application that predicts concrete compressive strength (MPa) and identifies the lowest-cost, structurally safe mix designs.

---

## Overview

Concrete mix design requires balancing structural strength requirements against raw material costs and curing timelines. Traditional physical compression testing takes days or months to yield results, slowing down recipe iteration.

This project solves that problem by combining civil engineering domain ratios with ensemble machine learning (`ExtraTreesRegressor`) to:
1. **Predict compressive strength** instantly from raw ingredient weights and curing age.
2. **Enforce structural safety margins** using a residual error buffer (`+1 RMSE`).
3. **Optimize material costs** by ranking safe formulations by cost per cubic meter (`$/m³`).

---

## Key Findings & Business Impact

* **Ratios Outperform Raw Ingredients:** Exploratory correlation analysis showed no single ingredient exceeds `r ≈ 0.50` against compressive strength (`Cement ≈ 0.50`, `Superplasticizer ≈ 0.37`, `Water ≈ -0.29`). Engineering chemical ratios—such as **Water-to-Binder** and **Superplasticizer-to-Binder**—substantially improved predictive power over linear baselines.
* **Risk-Mitigated Safety Thresholding:** Error analysis revealed higher variance in high supplementary binder blends (`>40–50%` slag/fly ash) and extreme curing windows (`1–3` days or `180–365` days). To protect against under-design on a **40 MPa** target, the system enforces a **+5.2 MPa (`+1 RMSE`) safety buffer**, requiring a predicted strength of **≥ 45.2 MPa**.
* **Cost-Safe Mix Discovery:** Applying per-kilogram material pricing identified a **50/50 Cement-Slag mix** (`190 kg/m³` cement, `190 kg/m³` slag, `0` superplasticizer) that safely reaches **46.9 MPa at just \$58.69/m³** (at 180 days), while mapping cost trade-offs for faster 28-day curing deadlines.

---

## Model & Pipeline Architecture

* **Custom Transformer (`ConcreteFeatureEngineer`):** Automatically computes domain-specific features inside a reproducible `scikit-learn` pipeline:
  * `total_binder` (`cement + blast_furnace_slag + fly_ash`)
  * `water_cement_ratio`
  * `water_binder_ratio`
  * `superplasticizer_binder_ratio`
* **Model Performance (5-Fold Cross-Validation):**
  * **Algorithm:** `ExtraTreesRegressor`
  * **R² Score:** `~0.87`
  * **RMSE:** `~5.2–5.9 MPa`

---

## Web Application Features

The serialized `.joblib` pipeline is deployed via **Streamlit** (`app.py`), enabling engineers to:
* **Single-Mix Simulator:** Adjust ingredient sliders (cement, slag, fly ash, water, superplasticizer, aggregates, and curing age) to get real-time strength predictions, estimated cost (`$/m³`), and pass/fail safety status.
* **Batch CSV Scoring:** Upload a CSV of candidate concrete formulations to score, filter by safety threshold, and sort by lowest cost.

---

## Project Structure

```text
├── data/                   # Concrete compressive strength dataset (UCI)
├── notebooks/              # EDA, correlation analysis, cross-validation & error diagnostics
├── models/                 # Serialized scikit-learn pipeline (.joblib)
├── app.py                  # Streamlit interactive web application
├── features.py             # Custom ConcreteFeatureEngineer transformer class
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit App
```bash
streamlit run app.py
```

---

## Tech Stack

* **Language:** Python
* **Machine Learning:** scikit-learn (Pipelines, Custom Transformers, `ExtraTreesRegressor`)
* **Data & Visualization:** Pandas, NumPy, Matplotlib, Seaborn
* **Deployment:** Streamlit, Joblib

---

## Dataset

Based on the (https://1drv.ms/b/c/334fe14fb6e58b3c/IQC8DvOQOuSlRrt8gS1gy0gmAV19-ca2njtzALm4nOS0sqk?e=LiMnFU), containing 1,030 concrete formulations across 8 quantitative input features (kg/m³ and curing age in days) and compressive strength (MPa).
