import pandas as pd
import pickle
import os

MODEL_PATH = "models/xgboost_model.pkl"

# Encodings that were generated during label encoding in training
weather_map  = {"Sunny": 5, "Cloudy": 0, "Windy": 6, "Fog": 1, "Sandstorms": 3, "Stormy": 4}
traffic_map  = {"Low": 2, "Medium": 3, "High": 1, "Jam": 0}
order_map    = {"Buffet": 0, "Drinks": 1, "Meal": 2, "Snack": 3}
vehicle_map  = {"Bicycle": 0, "Electric Scooter": 1, "Motorcycle": 2, "Scooter": 3}
festival_map = {"No": 0, "Yes": 1}
city_map     = {"Metropolitian": 0, "Urban": 2, "Semi-Urban": 1}

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please run train_model.py first.")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def predict_delivery_time(age, rating, distance, weather, traffic, order_type, vehicle, multiple, festival, city, order_hour):
    """
    Takes user inputs, applies the mappings, and returns the predicted delivery time.
    """
    model = load_model()
    
    input_data = pd.DataFrame([{
        'Delivery_person_Age':     age,
        'Delivery_person_Ratings': rating,
        'distance_km':             distance,
        'Weatherconditions':       weather_map.get(weather, 0),
        'Road_traffic_density':    traffic_map.get(traffic, 0),
        'Type_of_order':           order_map.get(order_type, 0),
        'Type_of_vehicle':         vehicle_map.get(vehicle, 0),
        'multiple_deliveries':     multiple,
        'Festival':                festival_map.get(festival, 0),
        'City':                    city_map.get(city, 0),
        'order_hour':              order_hour,
        'order_day':               15, # Arbitrary default for inference
        'order_month':             6,
        'order_dayofweek':         4
    }])
    
    prediction = model.predict(input_data)[0]
    return float(prediction)
