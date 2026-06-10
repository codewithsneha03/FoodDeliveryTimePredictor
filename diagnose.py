import pandas as pd
import numpy as np
from geopy.distance import geodesic

df = pd.read_csv('data/train.csv')
df.columns = df.columns.str.strip()
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip()

target_col = 'Time_taken(min)'
df[target_col] = df[target_col].str.extract(r'(\d+)').astype(float)

df['distance_km'] = df.apply(lambda row: geodesic(
    (row['Restaurant_latitude'], row['Restaurant_longitude']),
    (row['Delivery_location_latitude'], row['Delivery_location_longitude'])
).km, axis=1)

df = df[df['distance_km'] <= 30]

print('=== DISTANCE DISTRIBUTION ===')
print(df['distance_km'].describe())
print()

print('=== DISTANCE BINS vs AVG TIME ===')
bins = [0, 5, 10, 15, 20, 25, 30]
df['dist_bin'] = pd.cut(df['distance_km'], bins=bins)
print(df.groupby('dist_bin')[target_col].agg(['mean','count']))
print()

print('=== CORRELATION: distance vs time ===')
corr = df['distance_km'].corr(df[target_col])
print(f'Pearson correlation: {corr:.4f}')
