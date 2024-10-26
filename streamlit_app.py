import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the dataset containing player stats
df = pd.read_csv('https://raw.githubusercontent.com/k0yusuf/FYP/refs/heads/master/df_2024.csv')
player_names = df['Player'].unique()

# Set custom NBA-themed page layout
st.set_page_config(page_title="NBA Season Outcome Predictor", page_icon="🏀", layout="wide")

# NBA-themed title and header
st.markdown('<h1 class="main-title">🏀 NBA Season Outcome Prediction</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="sub-title">Create Your Dream 15-Man Roster!</h2>', unsafe_allow_html=True)

# Multiselect widget for player selection
selected_players = st.multiselect(
    'Select 15 Players:', options=player_names, max_selections=15,
    help='Select exactly 15 players.'
)

# Validate player count
if len(selected_players) == 15:
    st.markdown('<p class="success-text">You have selected your 15-man roster! 🏀</p>', unsafe_allow_html=True)

    # Display selected players' stats and average stats
    selected_players_df = df[df['Player'].isin(selected_players)]
    average_stats = selected_players_df.mean(numeric_only=True).drop(['Season', 'Season Outcome'], errors='ignore')
    st.dataframe(selected_players_df)
    st.write("### Average Stats for Selected Players")
    st.dataframe(average_stats)

    # Load the scaler and model
    scaler = joblib.load('scaler.joblib')
    model = joblib.load('svm_model.joblib')

    # Scale the average stats to match the model’s expectations
    scaled_average_stats = scaler.transform([average_stats])

    # Predict season outcome and probability
    prediction = model.predict(scaled_average_stats)
    prediction_proba = model.predict_proba(scaled_average_stats)

    # Display prediction and confidence
    st.markdown('<h2 class="sub-title">🏆 Predicted Season Outcome</h2>', unsafe_allow_html=True)
    st.write(f"Predicted Season Outcome: **{prediction[0]}**")
    st.write(f"Prediction Confidence: **{np.max(prediction_proba) * 100:.2f}%**")

else:
    st.error("Please select exactly 15 players.")
