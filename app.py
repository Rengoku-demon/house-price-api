import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

# ✅ Load the trained model from the file
model = joblib.load("house_price_model.pkl")

# Define the input data structure (all 8 features)
class HouseFeatures(BaseModel):
    medinc: float
    house_age: int
    ave_rooms: float
    ave_bedrms: float  
    population: int
    ave_occup: float  
    latitude: float
    longitude: float

# Create FastAPI app
app = FastAPI()

@app.get("/")
def home():
    return {"message": "House Price Prediction API is running!"}

@app.post("/predict")
def predict(features: HouseFeatures):
    try:
        # Convert input to NumPy array
        input_data = np.array([[features.medinc, features.house_age, features.ave_rooms, 
                                features.ave_bedrms, features.population, features.ave_occup, 
                                features.latitude, features.longitude]])

        # Make prediction
        prediction = model.predict(input_data)[0]
        return {"Predicted Price": float(prediction)}

    except Exception as e:
        return {"error": str(e)}
