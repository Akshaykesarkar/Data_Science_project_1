import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns


st.write("""
# Boston House Price Prediction App

This app predicts the **Boston House Price**!
""")
st.write('---')

california = fetch_california_housing()
X = pd.DataFrame(california.data, columns=california.feature_names)
Y = pd.DataFrame(california.target, columns=["MedHouseVal"])

st.sidebar.header('Specify Input Parameters')

st.write(X.head())

def user_input_features():
    MedInc = st.sidebar.slider('MedInc', X.MedInc.min(), X.MedInc.max(), X.MedInc.mean())
    HouseAge = st.sidebar.slider('HouseAge', X.HouseAge.min(), X.HouseAge.max(), X.HouseAge.mean())
    AveRooms = st.sidebar.slider('AveRooms', X.AveRooms.min(), X.AveRooms.max(), X.AveRooms.mean())
    AveBedrms = st.sidebar.slider('AveBedrms', X.AveBedrms.min(), X.AveBedrms.max(), X.AveBedrms.mean())
    Population = st.sidebar.slider('Population', X.Population.min(), X.Population.max(), X.Population.mean())
    AveOccup = st.sidebar.slider('AveOccup', X.AveOccup.min(), X.AveOccup.max(), X.AveOccup.mean())
    Latitude = st.sidebar.slider('Latitude', X.Latitude.min(), X.Latitude.max(), X.Latitude.mean())
    Longitude = st.sidebar.slider('Longitude', X.Longitude.min(), X.Longitude.max(), X.Longitude.mean())

    data = {
        'MedInc': MedInc,
        'HouseAge': HouseAge,
        'AveRooms': AveRooms,
        'AveBedrms': AveBedrms,
        'Population': Population,
        'AveOccup': AveOccup,
        'Latitude': Latitude,
        'Longitude': Longitude
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()
st.write("User Input Features:", df)

model = RandomForestRegressor()
model.fit(X,Y)
prediction = model.predict(df)

st.header("Predicted Price")
st.write(prediction)
st.write("---")

feature_importance = model.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

st.header("Feature Importance")
st.bar_chart(importance_df.set_index("Feature"))


st.header("Predicted Price Location")

# Convert data into a format suitable for `st.map`
map_data = pd.DataFrame({"lat": [df.Latitude[0]], "lon": [df.Longitude[0]]})

st.map(map_data)  # Display location on map

st.write(f"📍 **Prediction Location:** ({df.Latitude[0]}, {df.Longitude[0]})")