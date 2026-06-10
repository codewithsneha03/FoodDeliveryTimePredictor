import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from geopy.distance import geodesic
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import warnings
import pickle
import os

warnings.filterwarnings('ignore')

print("✅ Libraries imported successfully")

# Ensure directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("notebooks", exist_ok=True)
os.makedirs("data", exist_ok=True)

# ── LOAD DATASET ──────────────────────────────────────
try:
    df = pd.read_csv("data/train.csv")
    print(f"📦 Dataset loaded: {df.shape}")
except FileNotFoundError:
    print("❌ Error: 'data/train.csv' not found. Please download the dataset and place it in the 'data' folder.")
    exit(1)

# ── DATA CLEANING ─────────────────────────────────────
df.columns = df.columns.str.strip()

for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip()

df['Delivery_person_Age'] = pd.to_numeric(df['Delivery_person_Age'], errors='coerce')
df['Delivery_person_Ratings'] = pd.to_numeric(df['Delivery_person_Ratings'], errors='coerce')
df['multiple_deliveries'] = pd.to_numeric(df['multiple_deliveries'], errors='coerce')

df['Time_taken(min)'] = df['Time_taken(min)'].str.extract(r'(\d+)').astype(float)

df['Delivery_person_Age'].fillna(df['Delivery_person_Age'].median(), inplace=True)
df['Delivery_person_Ratings'].fillna(df['Delivery_person_Ratings'].median(), inplace=True)
df['multiple_deliveries'].fillna(df['multiple_deliveries'].median(), inplace=True)

for col in ['Weatherconditions', 'Road_traffic_density', 'City', 'Festival']:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("✅ Data cleaning completed")

# ── FEATURE ENGINEERING ──────────────────────────────
def calculate_distance(row):
    try:
        restaurant = (row['Restaurant_latitude'], row['Restaurant_longitude'])
        delivery = (row['Delivery_location_latitude'], row['Delivery_location_longitude'])
        return geodesic(restaurant, delivery).km
    except:
        return np.nan

df['distance_km'] = df.apply(calculate_distance, axis=1)
df['distance_km'].fillna(df['distance_km'].median(), inplace=True)

df['Order_Date'] = pd.to_datetime(df['Order_Date'], format='%d-%m-%Y', errors='coerce')
df['order_day'] = df['Order_Date'].dt.day
df['order_month'] = df['Order_Date'].dt.month
df['order_dayofweek'] = df['Order_Date'].dt.dayofweek
df['order_hour'] = pd.to_datetime(df['Time_Orderd'], errors='coerce').dt.hour
df['order_hour'].fillna(df['order_hour'].median(), inplace=True)

print("✅ Feature engineering completed")

# ── LABEL ENCODING ───────────────────────────────────
le = LabelEncoder()
categorical_cols = [
    'Weatherconditions', 'Road_traffic_density', 'Type_of_order', 
    'Type_of_vehicle', 'Festival', 'City'
]

for col in categorical_cols:
    df[col] = le.fit_transform(df[col].astype(str))

print("✅ Label encoding completed")

# ── MODELLING ────────────────────────────────────────
features = [
    'Delivery_person_Age', 'Delivery_person_Ratings', 'distance_km',
    'Weatherconditions', 'Road_traffic_density', 'Type_of_order',
    'Type_of_vehicle', 'multiple_deliveries', 'Festival', 'City',
    'order_hour', 'order_day', 'order_month', 'order_dayofweek'
]
target = 'Time_taken(min)'

X = df[features]
y = df[target]

mask = y.notna()
X = X[mask]
y = y[mask]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"🤖 Training XGBoost model on {X_train.shape[0]} rows...")
model = xgb.XGBRegressor(
    n_estimators=200, learning_rate=0.1, max_depth=6, 
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
)
model.fit(X_train, y_train)

# Save model
model_path = "models/xgboost_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)
print(f"✅ Model saved to {model_path}")

# ── EVALUATION ───────────────────────────────────────
y_pred = model.predict(X_test)
print("\n📊 MODEL PERFORMANCE:")
print(f"   MAE:  {mean_absolute_error(y_test, y_pred):.2f} min")
print(f"   RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f} min")
print(f"   R²:   {r2_score(y_test, y_pred):.4f}")

# ── VISUALIZATIONS ───────────────────────────────────
plt.figure(figsize=(10, 6))
importances = model.feature_importances_
feat_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values('Importance', ascending=False)
sns.barplot(x='Importance', y='Feature', data=feat_df, palette='viridis')
plt.title('XGBoost Feature Importance')
plt.tight_layout()
plt.savefig('notebooks/feature_importance.png', dpi=150)
plt.close()
print("✅ Feature importance plot saved to notebooks/feature_importance.png")
