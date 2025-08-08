#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Load historical data
df = pd.read_csv("nfl_cleaned_for_modeling_2015-2024-Copy1.csv")

# Enforce deterministic row order BEFORE filtering or training
df = df.sort_values(by=['Season', 'Tm_Name']).reset_index(drop=True)

# Preprocessing
df['True_Total'] = df['Tm_Pts'] + df['Opp_Pts']
df['Over'] = np.where(df['True_Total'] > df['Total'], 1, 0)
df['Under'] = np.where(df['True_Total'] < df['Total'], 1, 0)
df['Push'] = np.where(df['True_Total'] == df['Total'], 1, 0)
df = df.query('Home == 1').reset_index(drop=True)

# Train on all games BEFORE 2025 Week 1
train_df = df.query('Season < 2025 or (Season == 2025 and Week < 1)')

# Sort train_df for KNN tie-breaking consistency
train_df = train_df.sort_values(['Season', 'Week', 'Tm_Name']).reset_index(drop=True)

features = ['Spread', 'Total']
X_train = train_df[features]
y_train = train_df['Under']

# 2025 Week 1 matchups
week1_games = [
    ['Cowboys @ Eagles', -6.5, 46.5],
    ['Chiefs @ Chargers', 2.5, 45.5],
    ['Giants @ Commanders', -6.5, 45.5],
    ['Panthers @ Jaguars', -2.5, 46.5],
    ['Steelers @ Jets', 3.0, 38.5],
    ['Raiders @ Patriots', -2.5, 42.5],
    ['Cardinals @ Saints', 5.5, 41.5],
    ['Bengals @ Browns', 5.5, 45.5],
    ['Dolphins @ Colts', -1.5, 46.5],
    ['Buccaneers @ Falcons', 1.5, 48.5],
    ['Titans @ Broncos', -7.5, 41.5],
    ['49ers @ Seahawks', 2.5, 45.5],
    ['Lions @ Packers', -1.5, 49.5],
    ['Texans @ Rams', 2.5, 45.5],
    ['Ravens @ Bills', -1.5, 51.5],
    ['Vikings @ Bears', -1.5, 43.5]
]
X_new = pd.DataFrame(week1_games, columns=['Game', 'Spread', 'Total'])

# Make sure all numeric columns are float type for consistency
X_new[['Spread', 'Total']] = X_new[['Spread', 'Total']].astype(float)

# Train and predict
model = KNeighborsClassifier(n_neighbors=7)
clf = model.fit(X_train, y_train)
X_new_features = X_new[['Spread', 'Total']]
raw_preds = clf.predict(X_new_features)
X_new['Prediction'] = ['Under' if p == 1 else 'Over' for p in raw_preds]

# Get neighbors
distances, indices = clf.kneighbors(X_new_features)

# Analyze neighbors
confidence_percents = []
avg_distances = []
confidence_scores = []

for i in range(len(X_new)):
    neighbor_idxs = indices[i]
    neighbor_dists = distances[i]
    neighbor_labels = y_train.iloc[neighbor_idxs].values

    prediction_label = 1 if X_new.loc[i, 'Prediction'] == 'Under' else 0
    agreeing = np.sum(neighbor_labels == prediction_label)
    confidence_percent = agreeing / len(neighbor_labels)
    avg_distance = np.mean(neighbor_dists)

    # Scaled score (0–100)
    confidence_score = (confidence_percent * 100) * (1 - avg_distance)

    confidence_percents.append(round(confidence_percent, 3))
    avg_distances.append(round(avg_distance, 3))
    confidence_scores.append(round(confidence_score, 1))  # 1 decimal for clarity

# Attach scores to X_new
X_new['ConfidencePercent'] = confidence_percents
X_new['AvgDistance'] = avg_distances
X_new['ConfidenceScore'] = confidence_scores
X_new['Neighbors'] = [
    train_df.iloc[idx][['Season', 'Week', 'Spread', 'Total', 'Under']].to_dict('records')
    for idx in indices
]

# Optional: show SHA of train_df to debug if needed
# st.write("SHA256 of train_df:", pd.util.hash_pandas_object(train_df).sum())

# Display
st.title("NFL Over/Under Predictions – 2025 Week 1")

for _, row in X_new.iterrows():
    st.markdown(f"### {row['Game']}")

    # Prediction summary
    st.write(
        f"**Spread:** {row['Spread']} | **Total:** {row['Total']} | **Prediction:** `{row['Prediction']}`"
    )

    # Confidence metrics
    st.write(
        f"Confidence: **{row['ConfidencePercent'] * 100:.1f}%** | "
        f"Avg Distance: **{row['AvgDistance']}** | "
        f"Score: **{row['ConfidenceScore']:.3f}**"
    )

    # Neighbor breakdown
    with st.expander("Similar Matchups & Results"):
        st.dataframe(pd.DataFrame(row['Neighbors']))
