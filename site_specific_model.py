
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

xls = pd.read_excel("Rec_waters_clean.xlsx", sheet_name=None, engine="openpyxl")
data_frames = []
for site, df in xls.items():
    df = df[["cfu/100ml", "rainfall 3 days prior"]].dropna()
    df["site"] = site
    data_frames.append(df)
combined_df = pd.concat(data_frames)

X = combined_df[["rainfall 3 days prior", "site"]]
y = combined_df["cfu/100ml"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
preprocessor = ColumnTransformer([("site", OneHotEncoder(handle_unknown="ignore"), ["site"])], remainder="passthrough")
model = Pipeline([("preprocessor", preprocessor), ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))])
model.fit(X_train, y_train)

joblib.dump(model, "site_specific_model.pkl")
print("Model saved as site_specific_model.pkl")
