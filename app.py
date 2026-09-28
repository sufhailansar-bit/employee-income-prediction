"""
Employee Income Prediction — Streamlit app.

Loads the model trained by train_model.py and lets a user either
browse the model comparison / confusion matrix results, or enter an
individual's attributes to get a live income-bracket prediction.

Run locally:
    streamlit run app.py
"""

import json

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Employee Income Predictor", page_icon="💰", layout="wide")


@st.cache_resource
def load_artifacts():
    model_bundle = joblib.load("model.pkl")
    encoders = joblib.load("encoders.pkl")
    with open("results.json") as f:
        results = json.load(f)
    return model_bundle, encoders, results


try:
    model_bundle, encoders, results = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files not found. Run `python train_model.py` first "
        "(with adult.csv in the same folder) to generate model.pkl, "
        "encoders.pkl and results.json."
    )
    st.stop()

model = model_bundle["model"]
model_name = model_bundle["model_name"]
feature_order = model_bundle["feature_order"]

st.sidebar.title("Income Predictor")
page = st.sidebar.radio("Navigate", ["Home", "Data Overview", "Model Comparison", "Predict Income", "About"])

# ----------------------------------------------------------------- Home
if page == "Home":
    st.title("Employee Income Prediction")
    st.caption("Predicting whether an individual's income is above or below $50K")

    c1, c2, c3 = st.columns(3)
    c1.metric("Best Model", model_name)
    c2.metric("Best Accuracy", f'{results["results"][model_name]:.1f}%')
    c3.metric("Test Records", f'{results["n_test"]:,}')

    st.markdown(
        """
This app is trained on the **UCI Adult / Census Income** dataset.
Use the sidebar to:

- **Data Overview** — see dataset size and class balance
- **Model Comparison** — compare Logistic Regression, Random Forest and Gradient Boosting
- **Predict Income** — enter an individual's attributes and get a live prediction
        """
    )

# --------------------------------------------------------- Data Overview
elif page == "Data Overview":
    st.title("Data Overview")
    st.write(f'Total cleaned records: **{results["n_total"]:,}**')
    st.write(f'Training records: **{results["n_train"]:,}**  ·  Test records: **{results["n_test"]:,}**')

    if results.get("feature_importance"):
        fi = pd.Series(results["feature_importance"]).sort_values()
        fig = px.bar(fi, orientation="h", labels={"value": "Relative importance", "index": "Feature"},
                     title="Feature importance (Random Forest / Gradient Boosting)")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------- Model Comparison
elif page == "Model Comparison":
    st.title("Model Comparison")
    st.caption("Test-set accuracy across three trained algorithms")

    res = results["results"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Best Model", results["best_name"])
    c2.metric("Best Accuracy", f'{res[results["best_name"]]:.1f}%')
    c3.metric("Test Records", f'{results["n_test"]:,}')

    fig = px.bar(x=list(res.keys()), y=list(res.values()),
                 labels={"x": "", "y": "Test Accuracy (%)"},
                 text=[f"{v:.1f}%" for v in res.values()])
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis_range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"Confusion Matrix — {results['best_name']}")
    cm = results["cm"]
    labels = ["\u2264$50K", ">$50K"]
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm, x=labels, y=labels, colorscale="Blues", showscale=False,
        text=cm, texttemplate="%{text:,}", textfont={"size": 20},
    ))
    fig_cm.update_layout(xaxis_title="Predicted", yaxis_title="Actual", height=400)
    st.plotly_chart(fig_cm, use_container_width=True)

    acc = (cm[0][0] + cm[1][1]) / results["n_test"] * 100
    prec = cm[1][1] / (cm[1][1] + cm[0][1]) * 100
    rec = cm[1][1] / (cm[1][1] + cm[1][0]) * 100
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{acc:.1f}%")
    c2.metric("Precision (>$50K)", f"{prec:.1f}%")
    c3.metric("Recall (>$50K)", f"{rec:.1f}%")

# ------------------------------------------------------------- Predict
elif page == "Predict Income":
    st.title("Predict Income")
    st.caption(f"Model: {model_name}")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        age = st.number_input("Age", 17, 90, 37)
        workclass = st.selectbox("Workclass", encoders["workclass"].classes_)
        education = st.selectbox("Education", encoders["education"].classes_)
    with col2:
        educational_num = st.slider("Education Number", 1, 16, 10)
        marital_status = st.selectbox("Marital Status", encoders["marital-status"].classes_)
        occupation = st.selectbox("Occupation", encoders["occupation"].classes_)
    with col3:
        relationship = st.selectbox("Relationship", encoders["relationship"].classes_)
        race = st.selectbox("Race", encoders["race"].classes_)
        gender = st.selectbox("Gender", encoders["gender"].classes_)
    with col4:
        capital_gain = st.number_input("Capital Gain", 0, 99999, 0)
        capital_loss = st.number_input("Capital Loss", 0, 5000, 0)
        hours_per_week = st.slider("Hours per Week", 1, 99, 40)
    native_country = st.selectbox("Native Country", encoders["native-country"].classes_)

    if st.button("Predict", type="primary"):
        row = {
            "age": age,
            "workclass": encoders["workclass"].transform([workclass])[0],
            "education": encoders["education"].transform([education])[0],
            "educational-num": educational_num,
            "marital-status": encoders["marital-status"].transform([marital_status])[0],
            "occupation": encoders["occupation"].transform([occupation])[0],
            "relationship": encoders["relationship"].transform([relationship])[0],
            "race": encoders["race"].transform([race])[0],
            "gender": encoders["gender"].transform([gender])[0],
            "capital-gain": capital_gain,
            "capital-loss": capital_loss,
            "hours-per-week": hours_per_week,
            "native-country": encoders["native-country"].transform([native_country])[0],
        }
        X_input = pd.DataFrame([row])[feature_order]
        pred = model.predict(X_input)[0]
        proba = model.predict_proba(X_input)[0][pred]

        label = ">$50K" if pred == 1 else "\u2264$50K"
        if pred == 1:
            st.success(f"**PREDICTED INCOME: {label}**  ·  confidence {proba * 100:.1f}%")
        else:
            st.info(f"**PREDICTED INCOME: {label}**  ·  confidence {proba * 100:.1f}%")

# --------------------------------------------------------------- About
elif page == "About":
    st.title("About")
    st.markdown(
        f"""
**Employee Income Prediction** is a capstone project that predicts whether
an individual's income is above or below $50K from census-style
demographic and employment attributes.

- **Dataset:** UCI Adult / Census Income ({results['n_total']:,} cleaned records)
- **Models compared:** Logistic Regression, Random Forest, Gradient Boosting
- **Best model:** {results['best_name']} ({results['results'][results['best_name']]:.1f}% test accuracy)
- **Author:** Mohammad Sohail Akhtar

This is an academic prototype trained on historical census data. It should
support, not replace, human review of income-sensitive decisions.
        """
    )
