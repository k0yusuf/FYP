import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import random
import warnings

# Load the dataset containing player stats
df = pd.read_csv('https://raw.githubusercontent.com/k0yusuf/FYP/refs/heads/master/df_2024.csv').drop(columns=['Unnamed: 0'], errors='ignore')
player_names = df['Player'].unique()

# Set custom NBA-themed page layout
st.set_page_config(page_title="NBA Season Outcome Predictor", page_icon="🏀", layout="wide")

# NBA-themed title and header
st.markdown(
    """
    <style>
    .main-title {
        font-size:50px;
        font-weight:bold;
        color:#1d428a;
        text-align:center;
    }
    .sub-title {
        font-size:30px;
        font-weight:bold;
        color:#c8102e;
        text-align:center;
    }
    .header {
        background-color: #f5f5f5;
        padding: 10px;
        border-radius: 5px;
    }
    .success-text {
        font-size:18px;
        color: green;
    }
    </style>
    """, unsafe_allow_html=True
)
st.markdown('<h1 class="main-title">🏀 NBA Season Outcome Prediction</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="sub-title">Create Your Dream Roster!</h2>', unsafe_allow_html=True)

# Instructions for the user
st.markdown(
    """
    <div class="header">
    <h3>Instructions:</h3>
    <p>Select between 10 and 15 players to predict the season outcome based on their average stats.</p>
    </div>
    """, unsafe_allow_html=True
)

# Multiselect widget to allow users to select players
selected_players = st.multiselect(
    'Select between 10 and 15 Players:',
    options=player_names,
    default=[],
    max_selections=15,
    help='You must select between 10 and 15 players.'
)

# Load model and scaler
SVM_model = joblib.load('rf_model.joblib')
# scaler = joblib.load('scaler.joblib') # Uncomment if you have a scaler

# Check for player selection limits
if len(selected_players) < 10:
    st.error("Please select at least 10 players.")
elif len(selected_players) > 15:
    st.error("You have selected more than 15 players. Please deselect to make it between 10 and 15.")
else:
    st.markdown('<p class="success-text">You have selected your roster! 🏀</p>', unsafe_allow_html=True)

    # Filter selected player stats and calculate average
    selected_players_df = df[df['Player'].isin(selected_players)]
    average_stats = selected_players_df.mean(numeric_only=True).drop(['Season', 'Season Outcome'], errors='ignore')
    average_stats_df = pd.DataFrame(average_stats).T  # Convert Series to DataFrame with one row

    # Display average stats
    st.write("### Average Stats for Selected Players:")
    st.dataframe(average_stats)

    # Scale the features
    # scaled_average_stats = scaler.transform(average_stats_df)  # Uncomment if you have a scaler


    average_stats_df = average_stats_df.fillna(average_stats_df.mean())
    if st.button('Predict Season Outcome'):
        # Predict the season outcome
        prediction = SVM_model.predict(average_stats_df)
        prediction_proba = SVM_model.predict_proba(average_stats_df)

        # Display prediction possibilities with confidence
        outcome_df = pd.DataFrame({
            'Outcome': SVM_model.classes_,
            'Confidence (%)': prediction_proba[0] * 100
        })
        st.write("### Prediction Possibilities and Confidence:")
        st.table(outcome_df)

        # Show final prediction outcome
        st.markdown('<h2 class="sub-title">🏆 Predicted Season Outcome</h2>', unsafe_allow_html=True)
        st.write(f"### Predicted Outcome: **{prediction[0]}**")
        st.write(f"### Confidence for Outcome: **{np.max(prediction_proba) * 100:.2f}%**")

    if st.button('Show Suggestions'):
        # Check which columns are present in the DataFrame
        columns_to_drop = ['Player', 'Season', 'Season Outcome', 'Team', 'Offense Position', 
                           'Offensive Archetype', 'Defensive Role', 'Stable Avg 2PT Shot Distance', 
                           'Multiple Teams']
        
        # Drop only columns that are present in the DataFrame
        df_filtered = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
        
        # SHAP Explainer
        explainer = shap.KernelExplainer(SVM_model.predict_proba, df_filtered.values)
        shap_values = explainer.shap_values(average_stats_df)

        # Class index for the most probable predicted outcome
        class_index = np.argmax(SVM_model.predict_proba(average_stats_df))

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
