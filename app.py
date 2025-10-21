import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import plotly.express as px

# Load Excel file
xls = pd.read_excel("Rec_waters_clean.xlsx", sheet_name=None, engine="openpyxl")

# Combine sheets
data_frames = []
for site, df in xls.items():
    df = df[["cfu/100ml", "rainfall 3 days prior"]].dropna()
    df["site"] = site
    data_frames.append(df)
combined_df = pd.concat(data_frames, ignore_index=True)

# Auto-retrain if model missing
model_path = "site_specific_model.pkl"
if not os.path.exists(model_path):
    X = combined_df[["rainfall 3 days prior", "site"]]
    y = combined_df["cfu/100ml"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
    preprocessor = ColumnTransformer([("site", OneHotEncoder(handle_unknown="ignore"), ["site"])], remainder="passthrough")
    model = Pipeline([("preprocessor", preprocessor), ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))])
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)

# Load model
model = joblib.load(model_path)

# UI
st.title("Enterococci Predictor (Site-Specific)")
site = st.sidebar.selectbox("Select Site", list(xls.keys()))
rainfall = st.sidebar.number_input("Rainfall in last 3 days (mm)", min_value=0.0, max_value=200.0, value=10.0)

# Prediction
X_new = pd.DataFrame({"rainfall 3 days prior": [rainfall], "site": [site]})
prediction = model.predict(X_new)[0]
st.subheader("Prediction")
st.write(f"**Site:** {site}")
st.write(f"**Rainfall:** {rainfall} mm")
st.write(f"**Predicted Enterococci Level:** {prediction:.2f} cfu/100ml")

# Warning chart
st.subheader("Rainfall-Based Warning Decision Chart")
thresholds = [0, 10, 20, 30, 40]
results = []
df_site = xls[site].dropna(subset=["cfu/100ml", "rainfall 3 days prior"])
for threshold in thresholds:
    filtered = df_site[df_site["rainfall 3 days prior"] > threshold]
    prob = (filtered["cfu/100ml"] > 50).mean() if len(filtered) > 0 else None
    results.append({"Rainfall Threshold (mm)": threshold, "Probability of Enterococci > 50": prob})

chart_df = pd.DataFrame(results)
fig = px.bar(chart_df, x="Rainfall Threshold (mm)", y="Probability of Enterococci > 50",
             title=f"Warning Probabilities for {site}")
st.plotly_chart(fig)
