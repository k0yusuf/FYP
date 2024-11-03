import streamlit as st
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
import joblib
import numpy as np

# Load necessary components (dataset, model, scaler)
df = pd.read_csv('https://raw.githubusercontent.com/k0yusuf/FYP/refs/heads/master/df_2024.csv').drop(columns=['Unnamed: 0'], errors='ignore')
player_names = df['Player'].unique()
SVM_model = joblib.load('svm_model (1).joblib')
scaler = joblib.load('scaler.joblib')

# Select players and calculate average stats
selected_players = st.multiselect('Select between 10 and 15 Players:', options=player_names)
selected_players_df = df[df['Player'].isin(selected_players)]
average_stats = selected_players_df.mean(numeric_only=True).drop(['Season', 'Season Outcome'], errors='ignore')
average_stats_df = pd.DataFrame(average_stats).T
scaled_average_stats = scaler.transform(average_stats_df)

# SHAP Explainer
explainer = shap.KernelExplainer(SVM_model.predict_proba, scaler.transform(df.drop(['Player', 'Season', 'Season Outcome', 'Team','Offense Position', 'Offensive Archetype', 'Defensive Role', 'Stable Avg 2PT Shot Distance','Multiple Teams'], axis=1).values))
shap_values = explainer.shap_values(scaled_average_stats)

# Class index for the most probable predicted outcome
class_index = np.argmax(SVM_model.predict_proba(scaled_average_stats))

# Identify strong and weak features
shap_feature_impact = shap_values[class_index][0]
top_positive_features = np.argsort(shap_feature_impact)[-5:]  # Top 5 strengths
top_negative_features = np.argsort(shap_feature_impact)[:5]   # Top 5 weaknesses

# Display strengths
st.write("### Strengths of Your Team")
for feature_idx in top_positive_features:
    feature_name = average_stats_df.columns[feature_idx]
    contributing_players = selected_players_df.nlargest(3, feature_name)['Player']
    st.write(f"- **{feature_name.capitalize()}** is strong due to players: {', '.join(contributing_players)}")

# Display weaknesses and suggest players to improve them
st.write("### Suggested Improvements for Your Team")
for feature_idx in top_negative_features:
    feature_name = average_stats_df.columns[feature_idx]
    suggested_players = df.nlargest(3, feature_name)['Player']
    st.write(f"- **{feature_name.capitalize()}** needs improvement. Suggested players: {', '.join(suggested_players)}")
