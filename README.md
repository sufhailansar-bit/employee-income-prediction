# Employee Income Prediction

Predicts whether an individual's income is above or below $50K from
census-style demographic and employment attributes, using the UCI
Adult / Census Income dataset.

## Files

- `train_model.py` — cleans the data, trains and compares Logistic
  Regression, Random Forest and Gradient Boosting, and saves the best
  model.
- `app.py` — Streamlit app (Home, Data Overview, Model Comparison,
  Predict Income, About).
- `adult.csv` — dataset (add this file yourself; see below).
- `requirements.txt` — Python dependencies.
- `.streamlit/config.toml` — dark theme.

## Run locally

```bash
pip install -r requirements.txt
python train_model.py     # creates model.pkl, encoders.pkl, results.json
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

## Deploy for free (Streamlit Community Cloud)

1. Push this folder (including `adult.csv`, and the `model.pkl` /
   `encoders.pkl` / `results.json` files created by `train_model.py`)
   to a GitHub repository.
2. Go to **share.streamlit.io**, sign in with GitHub.
3. Click **New app**, pick this repo and branch, set the main file to
   `app.py`, and click **Deploy**.
4. Streamlit gives you a live link like
   `https://<your-app-name>.streamlit.app` — use that as your
   deployment link.

## Dataset

Uses the UCI Adult / Census Income dataset (Dua, D. & Graff, C.,
2019). Place `adult.csv` in this same folder before running
`train_model.py`.
