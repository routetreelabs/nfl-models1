#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("nfl_cleaned_for_modeling_2015-2024-Copy1.csv")

# Debug: Show shape and head
st.write("CSV Loaded - Shape:", df.shape)
st.write("First 5 rows:", df.head())

# Enforce deterministic sort
df = df.sort_values(by=['Season', 'Tm_Name']).reset_index(drop=True)

# Preprocessing
df['True_Total'] = df['Tm_Pts'] + df['Opp_Pts']
df['Over'] = np.where(df['True_Total'] > df['Total'], 1, 0)
df['Under'] = np.where(df['True_Total'] < df['Total'], 1, 0)
df['Push'] = np.where(df['True_Total'] == df['Total'], 1, 0)
df = df.query('Home == 1').reset_index(drop=True)

# Train set
train_df = df.query('Season < 2025 or (Season == 2025 and Week < 1)')
train_df = train_df.sort_values(['Season', 'Week', 'Tm_Name']).reset_index(drop=True)

# Hash check
st.write("SHA256 of train_df:", pd.util.hash_pandas_object(train_df).sum())

X_train = train_df[['Spread', 'Total']]
y_train = train_df['Under']

# 2025 Week 1
week1_games = [['Cowboys @ Eagles', -6.5, 46.5]]
X_new = pd.DataFrame(week1_games, columns=['Game', 'Spread', 'Total'])
X_new[['Spread', 'Total']] = X_new[['Spread', 'Total']].astype(float)
X_new_features = X_new[['Spread', 'Total']]

# Hash of test input
st.write("SHA256 of X_new_features:", pd.util.hash_pandas_object(X_new_features).sum())

# Model
model = KNeighborsClassifier(n_neighbors=7)
clf = model.fit(X_train, y_train)
X_new['Prediction'] = ['Under' if p == 1 else 'Over' for p in clf.predict(X_new_features)]

# Neighbors
distances, indices = clf.kneighbors(X_new_features)

# Print neighbors of Game 0
st.write("Neighbors for Game 0:")
st.dataframe(train_df.iloc[indices[0]][['Season', 'Week', 'Spread', 'Total', 'Under']])

# Final Output
st.write("Prediction:", X_new[['Game', 'Spread', 'Total', 'Prediction']])

