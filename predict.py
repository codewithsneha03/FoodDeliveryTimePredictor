import numpy as np
import pandas as pd
import pickle
import os

MODEL_PATH = "models/xgboost_model.pkl"

# Encodings matched to actual LabelEncoder output from training
weather_map  = {"Sunny": 5, "Cloudy": 0, "Windy": 6, "Fog": 1, "Sandstorms": 3, "Stormy": 4}
traffic_map  = {"High": 0, "Jam": 1, "Low": 2, "Medium": 3}
order_map    = {"Buffet": 0, "Drinks": 1, "Meal": 2, "Snack": 3}
vehicle_map  = {"Bicycle": 0, "Electric Scooter": 1, "Motorcycle": 2, "Scooter": 3}
festival_map = {"No": 1, "Yes": 2}
city_map     = {"Metropolitian": 0, "Urban": 3, "Semi-Urban": 2}

# Average speeds (km/h) per vehicle in Indian city conditions
vehicle_speed = {
    "Bicycle":          10,
    "Electric Scooter": 20,
    "Motorcycle":       25,
    "Scooter":          20,
}

# Traffic multiplier on travel time
traffic_multiplier = {
    "Low":    1.0,
    "Medium": 1.3,
    "High":   1.6,
    "Jam":    2.2,
}

# Weather adds extra delay (minutes)
weather_delay = {
    "Sunny":      0,
    "Cloudy":     1,
    "Windy":      2,
    "Fog":        4,
    "Sandstorms": 6,
    "Stormy":     8,
}

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please run train_model.py first.")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def predict_delivery_time(age, rating, distance, weather, traffic, order_type, vehicle, multiple, festival, city, order_hour):
    """
    Hybrid prediction:
    - XGBoost handles condition-based factors (traffic, weather, ratings, festival, city)
    - Physics-based calculation ensures realistic distance-to-time scaling
    """
    model = load_model()

    # Clamp distance to model's training range for ML prediction
    clamped_distance = min(distance, 20.0)
    traffic_encoded = traffic_map.get(traffic, 0)

    input_data = pd.DataFrame([{
        'Delivery_person_Age':     age,
        'Delivery_person_Ratings': rating,
        'distance_km':             clamped_distance,
        'distance_sq':             clamped_distance ** 2,
        'distance_log':            np.log1p(clamped_distance),
        'distance_per_rating':     clamped_distance / max(rating, 1.0),
        'distance_x_traffic':     clamped_distance * traffic_encoded,
        'Weatherconditions':       weather_map.get(weather, 0),
        'Road_traffic_density':    traffic_encoded,
        'Type_of_order':           order_map.get(order_type, 0),
        'Type_of_vehicle':         vehicle_map.get(vehicle, 0),
        'multiple_deliveries':     multiple,
        'Festival':                festival_map.get(festival, 0),
        'City':                    city_map.get(city, 0),
        'order_hour':              order_hour,
        'order_day':               15,
        'order_month':             6,
        'order_dayofweek':         4
    }])

    # ML base prediction (captures ratings, festival, city, order type effects)
    ml_base = float(model.predict(input_data)[0])

    # Physics-based travel time: distance / speed * 60 (hours to minutes)
    speed = vehicle_speed.get(vehicle, 20)
    travel_time = (distance / speed) * 60

    # Traffic slows down travel
    travel_time *= traffic_multiplier.get(traffic, 1.0)

    # Weather adds extra delay
    travel_time += weather_delay.get(weather, 0)

    # Combine: ML handles base prep/wait time, physics handles actual travel
    base_prep = ml_base * 0.55  # ~55% of ML prediction is prep + wait time

    # Final = prep time + realistic travel time
    prediction = base_prep + travel_time

    # Multiple deliveries add time
    if multiple > 0:
        prediction += multiple * 5

    return round(prediction, 1)
