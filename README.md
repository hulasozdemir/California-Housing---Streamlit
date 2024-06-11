## California Housing Price Prediction

### Overview
This project aims to build a machine learning model to predict house prices using the California Housing dataset. The model is deployed using a Streamlit app to provide an interactive interface for users to input house features and get price predictions.

### Project Structure
- `app.py`: The main script for the Streamlit app.
- `data/`: Directory containing the dataset.
- `model/`: Directory to save and load the trained machine learning model and scaler.
- `requirements.txt`: List of dependencies required for the project.
- `README.md`: Project description and setup instructions.

### Dataset
The California Housing dataset contains information collected during the 1990 California census. The dataset includes features such as the median income, house age, average number of rooms, and house value for various districts.

### Features
- `MedInc`: Median income in block group
- `HouseAge`: Median house age in block group
- `AveRooms`: Average number of rooms per household
- `AveBedrms`: Average number of bedrooms per household
- `Population`: Block group population
- `AveOccup`: Average number of household members
- `Latitude`: Block group latitude
- `Longitude`: Block group longitude

### Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/california-housing-prediction.git
   cd california-housing-prediction
   ```

2. **Install required libraries:**
   ```bash
   pip install -r requirements.txt
   ```

### Usage
1. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

2. **Interact with the app:**
   Use the sliders in the Streamlit interface to input the house features and get the predicted house price.

### Model
The model is a linear regression model trained on the standardized features of the California Housing dataset. The dataset is split into training and testing sets, and the features are scaled using `StandardScaler` from scikit-learn.

### Deployment
The Streamlit app provides an interactive web interface for the model. The app allows users to input house features through sliders and displays the predicted house price.

### Live Demo
Check out the live demo of the Streamlit app [here](https://california-housing---app-257sxrlhtxbavrrpgrcdzu.streamlit.app).

### File Descriptions
- `app.py`: Contains the Streamlit app code, including data loading, feature scaling, and price prediction.
- `model.pkl`: Pickle file containing the trained model and scaler.
- `requirements.txt`: Contains the list of dependencies required to run the project.
- `README.md`: Project description and setup instructions.

