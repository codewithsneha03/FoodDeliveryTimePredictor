# 🛵 Food Delivery Time Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fooddeliverytimepredictor-usrlqtf5yzq68b5sotsvbx.streamlit.app/)

An end-to-end Machine Learning project that predicts food delivery times based on real-world factors like driver age, ratings, weather, traffic density, and geographic distance. 

Built with **XGBoost**, **Python**, and **Streamlit**.

## 🌐 Live Demo
You can try out the live web application here:
**[Food Delivery Time Predictor - Live App](https://fooddeliverytimepredictor-usrlqtf5yzq68b5sotsvbx.streamlit.app/)**

## 🧠 Architecture & Approach

This project solves the complex problem of predicting delivery times by using a **Hybrid Architecture** (similar to what is used in production by companies like Zomato or Swiggy). 

1. **The ML Layer (Preparation & Wait Time):** 
   An `XGBoost` model analyzes condition-based factors (Traffic, Weather, Ratings, City, Order Type) to predict the base time required for restaurant preparation and driver wait time.
2. **The Physics Layer (Travel Time):** 
   We calculate the exact Earth-surface distance between the restaurant and the delivery location using the **Haversine formula**. We then calculate theoretical travel time using average vehicle speeds, and apply mathematical penalties for weather and traffic jams.
3. **The Final Output:** 
   The ML base prediction is combined with the Physics-based travel time to generate a highly accurate, realistic prediction.

## 📊 Model Accuracy
The XGBoost model was evaluated against testing data with the following highly accurate results:
*   **R² Score:** `0.77` (Explains 77% of the variance in delivery times)
*   **MAE:** `3.55 min` (On average, predictions are only 3.5 minutes off)
*   **RMSE:** `4.53 min`

## 📂 Project Structure
*   `app.py`: The Streamlit frontend web application.
*   `train_model.py`: The data cleaning, feature engineering (Haversine formula), and XGBoost training pipeline.
*   `predict.py`: The inference script containing the Hybrid Prediction logic.
*   `requirements.txt`: Python dependencies.

## 🚀 How to Run Locally

1. Clone this repository:
```bash
git clone https://github.com/codewithsneha03/FoodDeliveryTimePredictor.git
cd FoodDeliveryTimePredictor
```

2. Install the required libraries:
```bash
pip install -r requirements.txt
```

3. Ensure you have the Kaggle dataset (`train.csv`) placed inside a `data/` folder in the root directory.

4. Train the model (this will generate the `xgboost_model.pkl` file):
```bash
python train_model.py
```

5. Launch the Streamlit web app:
```bash
python -m streamlit run app.py
```
