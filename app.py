import os
if not os.path.exists('site_specific_model.pkl'):
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    import joblib

    # Load dataset
    xls = pd.read_excel('Rec_waters_clean.xlsx', sheet_name=None)
    data_frames = []
    for site, df in xls.items():
        df = df[['cfu/100ml', 'rainfall 3 days prior']].dropna()
        df['site'] = site
        data_frames.append(df)
    combined_df = pd.concat(data_frames)

    X = combined_df[['rainfall 3 days prior', 'site']]
    y = combined_df['cfu/100ml']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer([('site', OneHotEncoder(handle_unknown='ignore'), ['site'])], remainder='passthrough')
    model = Pipeline([('preprocessor', preprocessor), ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])
    model.fit(X_train, y_train)

    joblib.dump(model, 'site_specific_model.pkl')
    
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Load site-specific model
model = joblib.load('site_specific_model.pkl')

# Load dataset for site-specific charts
xls = pd.read_excel('Rec_waters_clean.xlsx', sheet_name=None)

# Sidebar: Site selection
site = st.sidebar.selectbox("Select Site", list(xls.keys()))

# Sidebar: Rainfall input
rainfall = st.sidebar.number_input("Enter Rainfall (mm, last 3 days)", min_value=0.0, max_value=200.0, value=10.0)

# Prediction using site-specific model
X_new = pd.DataFrame({'rainfall 3 days prior': [rainfall], 'site': [site]})
prediction = model.predict(X_new)[0]

st.title("Enterococci Prediction Tool (Site-Specific)")
st.write(f"**Site:** {site}")
st.write(f"**Rainfall:** {rainfall} mm")
st.write(f"**Predicted Enterococci Level:** {prediction:.2f} cfu/100ml")

# Display Rainfall-Based Warning Decision Chart
st.subheader("Rainfall-Based Warning Decision Chart")
rainfall_thresholds = [0, 10, 20, 30, 40]
results = []
df = xls[site].dropna(subset=['cfu/100ml', 'rainfall 3 days prior'])
for threshold in rainfall_thresholds:
    filtered = df[df['rainfall 3 days prior'] > threshold]
    prob = (filtered['cfu/100ml'] > 50).mean() if len(filtered) > 0 else None
    results.append({'Threshold': threshold, 'Probability': prob})

chart_df = pd.DataFrame(results)
fig = px.bar(chart_df, x='Threshold', y='Probability', title=f'Warning Probabilities for {site}',
             labels={'Threshold': 'Rainfall Threshold (mm)', 'Probability': 'Probability of Enterococci > 50'})
st.plotly_chart(fig)
