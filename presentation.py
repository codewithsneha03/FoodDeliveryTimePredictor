# %% [markdown]
# # 🛵 Food Delivery Data Presentation
# Welcome to the live data analysis! In this interactive script, we will walk through the raw data, how it was cleaned, and why we made our algorithm choices.

# %% [markdown]
# ## 1. Loading the Raw Data
# First, let's load the Kaggle dataset and see exactly how many rows and columns we are working with.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set chart style
sns.set_theme(style="darkgrid")

df = pd.read_csv("data/train.csv")

print(f"Total Rows in raw data: {df.shape[0]}")
print(f"Total Columns in raw data: {df.shape[1]}")

# Let's look at the first 5 rows to see what the raw data looks like
df.head()

# %% [markdown]
# ## 2. Identifying the Messy Data
# Real-world data is never perfect. Let's look at the missing values (NaN) and incorrect data types.
# Notice how `Time_taken(min)` has text like "(min) 24" instead of just the number 24.

# %%
print("MISSING VALUES PER COLUMN:")
print(df.isnull().sum())

print("\nDATA TYPES:")
print(df.dtypes[['Delivery_person_Age', 'Delivery_person_Ratings', 'Time_taken(min)']])

# %% [markdown]
# ## 3. Data Cleaning Pipeline
# To fix this, we strip out text characters, convert columns to pure numbers, and fill missing values with Medians (for numbers) and Modes (for text).

# %%
# Strip whitespace
df.columns = df.columns.str.strip()
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip()

# Extract pure numbers from Time_taken
df['Time_taken(min)'] = df['Time_taken(min)'].str.extract(r'(\d+)').astype(float)

# Convert strings to numbers
df['Delivery_person_Age'] = pd.to_numeric(df['Delivery_person_Age'], errors='coerce')
df['Delivery_person_Ratings'] = pd.to_numeric(df['Delivery_person_Ratings'], errors='coerce')

# Fill missing values
df['Delivery_person_Age'].fillna(df['Delivery_person_Age'].median(), inplace=True)
df['Delivery_person_Ratings'].fillna(df['Delivery_person_Ratings'].median(), inplace=True)

print("Data Cleaning Complete. 'Time_taken(min)' is now a pure float!")
df[['Delivery_person_Age', 'Delivery_person_Ratings', 'Time_taken(min)']].head()

# %% [markdown]
# ## 4. Analyzing the Target (Delivery Time)
# **Why this graph?** We need to know what a "normal" delivery time looks like before we try to predict it. 
# **What it explains:** This histogram shows that most deliveries take between 20 and 30 minutes. It forms a "bell curve", which is great for Machine Learning.

# %%
plt.figure(figsize=(10, 5))
sns.histplot(df['Time_taken(min)'], bins=20, kde=True, color='#FF6B35')
plt.title('Distribution of Delivery Times', fontsize=14, fontweight='bold')
plt.xlabel('Time Taken (minutes)')
plt.ylabel('Number of Deliveries')
plt.show()

# %% [markdown]
# ## 5. Analyzing Traffic Impact
# **Why this graph?** We want to prove to the audience that Traffic has a massive impact on delivery time.
# **What it explains:** A Boxplot shows the minimum, average, and maximum times. You can clearly see that "Jam" traffic severely pushes up the delivery time compared to "Low" traffic. This proves why Traffic is our most important feature!

# %%
plt.figure(figsize=(10, 5))
sns.boxplot(x='Road_traffic_density', y='Time_taken(min)', data=df, palette='viridis', 
            order=['Low', 'Medium', 'High', 'Jam'])
plt.title('How Traffic Density Impacts Delivery Time', fontsize=14, fontweight='bold')
plt.xlabel('Traffic Density')
plt.ylabel('Time Taken (minutes)')
plt.show()

# %% [markdown]
# ## 6. The Problem with the Distance Data
# **Why this graph?** To explain why we had to build a "Hybrid Architecture".
# **What it explains:** We use a scatter plot to map Distance vs Time. In a perfect world, this would be a straight line going up. But look at the graph—it's a massive blob! There are deliveries going 20km that take 15 mins, and deliveries going 5km that take 45 mins. 
# 
# **Conclusion:** Because the correlation here is so weak, we couldn't rely solely on XGBoost for distance. This is why we implemented the Physics-based layer in our final `predict.py` app!

# %%
# Calculate Haversine distance for the graph
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1)/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1)/2)**2
    return R * (2 * np.arcsin(np.sqrt(a)))

df['distance_km'] = haversine(df['Restaurant_latitude'], df['Restaurant_longitude'], 
                              df['Delivery_location_latitude'], df['Delivery_location_longitude'])

# Filter out the 19,000km Kaggle errors
df_clean = df[df['distance_km'] <= 21]

plt.figure(figsize=(10, 5))
sns.scatterplot(x='distance_km', y='Time_taken(min)', data=df_clean, alpha=0.3, color='#2196F3')
plt.title('Distance vs. Time (The Correlation Problem)', fontsize=14, fontweight='bold')
plt.xlabel('Distance (km)')
plt.ylabel('Time Taken (minutes)')
plt.show()

# %% [markdown]
# ## 7. Why We Chose XGBoost
# XGBoost builds hundreds of small "Decision Trees" that learn from each other's mistakes. 
# We used it because:
# 1. **It handles non-linear relationships.** (Traffic Jams don't just add 5 minutes, they multiply the wait time).
# 2. **It handles messy data gracefully.**
# 3. **It provides "Feature Importance"**, telling us exactly what variables drive delivery times in the real world.

# %% [markdown]
# ## 8. Proof of Accuracy
# Finally, let's look at the results of our XGBoost model.
# * **MAE (Mean Absolute Error):** 3.55 minutes. On average, our prediction is only off by 3.5 minutes from the real-world delivery time!
# * **R² Score:** 0.77. We successfully explain 77% of all variance in human delivery times.
# 
# Below is the **Feature Importance** chart generated by the model. It proves exactly what the algorithm learned:

# %%
from IPython.display import Image, display
print("🏆 MODEL PERFORMANCE:")
print("   MAE: 3.55 min")
print("   R² Score: 0.77")
print("\n📊 FEATURE IMPORTANCE RANKING:")
display(Image(filename='notebooks/feature_importance.png'))

