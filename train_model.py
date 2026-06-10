import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import warnings
import pickle
import os

warnings.filterwarnings('ignore')

print("Libraries imported successfully")

# Ensure directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("notebooks", exist_ok=True)
os.makedirs("data", exist_ok=True)

# ── HAVERSINE FORMULA ─────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points
    on Earth using the Haversine formula.
    Returns distance in kilometers.
    """
    R = 6371  # Earth's radius in km

    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2) ** 2 + \
        np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2

    c = 2 * np.arcsin(np.sqrt(a))

    return R * c

# ── LOAD DATASET ──────────────────────────────────────
try:
    df = pd.read_csv("data/train.csv")
    print(f"Dataset loaded: {df.shape}")
except FileNotFoundError:
    print("Error: 'data/train.csv' not found. Please download the dataset and place it in the 'data' folder.")
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

print("Data cleaning completed")

# ── FEATURE ENGINEERING (HAVERSINE DISTANCE) ─────────
df['distance_km'] = haversine(
    df['Restaurant_latitude'].values,
    df['Restaurant_longitude'].values,
    df['Delivery_location_latitude'].values,
    df['Delivery_location_longitude'].values
)

# Replace invalid distances (NaN, inf, negative) with median
df['distance_km'] = df['distance_km'].replace([np.inf, -np.inf], np.nan)
df['distance_km'].fillna(df['distance_km'].median(), inplace=True)

# Filter out unrealistic distance outliers (dataset max is ~21km for valid data)
print(f"Before distance filter: {df.shape[0]} rows")
df = df[df['distance_km'] <= 21]
df = df[df['distance_km'] > 0.1]  # Remove near-zero distances too
print(f"After distance filter:  {df.shape[0]} rows")

# ── DISTANCE-DERIVED FEATURES (to boost distance importance) ──
df['distance_sq']  = df['distance_km'] ** 2                                    # Captures non-linear scaling
df['distance_log'] = np.log1p(df['distance_km'])                               # Captures diminishing returns
df['distance_per_rating'] = df['distance_km'] / df['Delivery_person_Ratings'].clip(lower=1)  # Interaction: worse rating + far = slow

# ── OTHER FEATURES ───────────────────────────────────
df['Order_Date'] = pd.to_datetime(df['Order_Date'], format='%d-%m-%Y', errors='coerce')
df['order_day'] = df['Order_Date'].dt.day
df['order_month'] = df['Order_Date'].dt.month
df['order_dayofweek'] = df['Order_Date'].dt.dayofweek
df['order_hour'] = pd.to_datetime(df['Time_Orderd'], errors='coerce').dt.hour
df['order_hour'].fillna(df['order_hour'].median(), inplace=True)

print("Feature engineering completed (Haversine + distance features)")

# ── LABEL ENCODING ───────────────────────────────────
le = LabelEncoder()
categorical_cols = [
    'Weatherconditions', 'Road_traffic_density', 'Type_of_order', 
    'Type_of_vehicle', 'Festival', 'City'
]

# Store label mappings for reference
label_maps = {}
for col in categorical_cols:
    le.fit(df[col].astype(str))
    label_maps[col] = dict(zip(le.classes_, le.transform(le.classes_)))
    df[col] = le.transform(df[col].astype(str))

print("Label encoding completed")
print("Label mappings:")
for col, mapping in label_maps.items():
    print(f"  {col}: {mapping}")

# ── DISTANCE x TRAFFIC INTERACTION ───────────────────
# This is the key feature: longer distance in worse traffic = much more time
df['distance_x_traffic'] = df['distance_km'] * df['Road_traffic_density']

# ── MODELLING ────────────────────────────────────────
features = [
    'Delivery_person_Age', 'Delivery_person_Ratings',
    'distance_km', 'distance_sq', 'distance_log', 'distance_per_rating', 'distance_x_traffic',
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

print(f"Training XGBoost model on {X_train.shape[0]} rows with {len(features)} features...")
model = xgb.XGBRegressor(
    n_estimators=300, learning_rate=0.08, max_depth=7,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
)
model.fit(X_train, y_train)

# Save model
model_path = "models/xgboost_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)
print(f"Model saved to {model_path}")

# ── EVALUATION ───────────────────────────────────────
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nMODEL PERFORMANCE:")
print(f"   MAE:  {mae:.2f} min")
print(f"   RMSE: {rmse:.2f} min")
print(f"   R2:   {r2:.4f}")

# ── FEATURE IMPORTANCE ───────────────────────────────
importances = model.feature_importances_
feat_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values('Importance', ascending=False)

print(f"\nFEATURE IMPORTANCE RANKING:")
for i, row in feat_df.iterrows():
    marker = " <-- DISTANCE" if "distance" in row['Feature'] else ""
    print(f"   {row['Feature']:25s}: {row['Importance']:.4f}  ({row['Importance']*100:.1f}%){marker}")

# Calculate total distance contribution
dist_features = [f for f in features if 'distance' in f]
dist_importance = feat_df[feat_df['Feature'].isin(dist_features)]['Importance'].sum()
print(f"\n   TOTAL DISTANCE CONTRIBUTION: {dist_importance*100:.1f}%")

plt.figure(figsize=(10, 7))
colors = ['#FF6B35' if 'distance' in f else '#2196F3' for f in feat_df['Feature']]
sns.barplot(x='Importance', y='Feature', data=feat_df, palette=colors)
plt.title('XGBoost Feature Importance\n(Orange = Distance-related features)', fontsize=14)
plt.tight_layout()
plt.savefig('notebooks/feature_importance.png', dpi=150)
plt.close()
print("Feature importance plot saved to notebooks/feature_importance.png")
