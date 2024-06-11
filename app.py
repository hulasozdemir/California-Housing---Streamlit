import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns


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

# Plotting
st.header('Exploratory Data Analysis')

# Histogram of target variable (house prices)
st.subheader('Histogram of House Prices')
fig, ax = plt.subplots()
ax.hist(y, bins=30, edgecolor='k', alpha=0.7)
ax.set_xlabel('Price ($100,000s)')
ax.set_ylabel('Frequency')
st.pyplot(fig)

# Pair plot of some features against the target variable
# st.subheader('Pair Plot')
# data = pd.DataFrame(X, columns=columns)
# data['Price'] = y
# selected_features = st.multiselect('Select features for pair plot', columns, default=columns[:4])
# if selected_features:
#     pair_plot_data = data[selected_features + ['Price']]
#     pair_plot = sns.pairplot(pair_plot_data)
#     st.pyplot(pair_plot)

# Feature Importance Plot (if using a model that supports feature importances)
# Example with a RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor

# Train a RandomForestRegressor
rf_model = RandomForestRegressor()
rf_model.fit(X_scaled, y)
feature_importances = rf_model.feature_importances_

# Plot feature importances
st.subheader('Feature Importance')
fig, ax = plt.subplots()
ax.barh(columns, feature_importances, align='center')
ax.set_xlabel('Feature Importance')
ax.set_ylabel('Feature')
st.pyplot(fig)