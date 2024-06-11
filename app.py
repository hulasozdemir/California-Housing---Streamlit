import streamlit as st
import numpy as np
import pickle


from sklearn.datasets import fetch_california_housing

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression


california_housing = fetch_california_housing()

X = california_housing.data
y = california_housing.target
columns = california_housing.feature_names



scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



model = LinearRegression()
model.fit(X_scaled, y)

with open('model.pkl', 'wb') as file:
    pickle.dump((scaler, model), file)


with open('model.pkl', 'rb') as file:
    scaler, model = pickle.load(file)

st.title('California Housing Price Prediction')
st.write('Enter the details of the house to predict the price.')

MedInc = st.number_input('MedInc')
HouseAge = st.slider('HouseAge',float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()))
AveRooms = st.slider('AveRooms',float(X[:, 2].min()), float(X[:, 2].max()), float(X[:, 2].mean()))
AveBedrms = st.slider('AveBedrms',float(X[:, 3].min()), float(X[:, 3].max()), float(X[:, 3].mean()))
Population = st.number_input('Population',float(X[:, 4].min()), float(X[:, 4].max()), float(X[:, 4].mean()))
AveOccup = st.number_input('AveOccup',float(X[:, 5].min()), float(X[:, 5].max()), float(X[:, 5].mean()))
Latitude = st.number_input('Latitude',float(X[:, 6].min()), float(X[:, 6].max()), float(X[:, 6].mean()))
Longitude = st.number_input('Longitude',float(X[:, 7].min()), float(X[:, 7].max()), float(X[:, 7].mean()))


input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])
input_data = scaler.transform(input_data)
prediction = model.predict(input_data)

st.write(f'Predicted Price: ${prediction[0]*1e5:,}')
