import streamlit as st
import pandas as pd
import joblib  # To load the pre-trained model
import numpy as np
import pickle

# Load the dataset containing player stats
df = pd.read_csv('https://raw.githubusercontent.com/k0yusuf/FYP/refs/heads/master/df_2024.csv')

player_names = df['Player'].unique()

# Set custom NBA-themed page layout
st.set_page_config(page_title="NBA Season Outcome Predictor", page_icon="🏀", layout="wide")

# NBA themed title and header
st.markdown(
    """
    <style>
    .main-title {
        font-size:50px;
        font-weight:bold;
        color:#1d428a; /* NBA blue color */
        text-align:center;
    }
    .sub-title {
        font-size:30px;
        font-weight:bold;
        color:#c8102e; /* NBA red color */
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
st.markdown('<h2 class="sub-title">Create Your Dream 15-Man Roster!</h2>', unsafe_allow_html=True)

# Instructions for the user with a header
st.markdown(
    """
    <div class="header">
    <h3>Instructions:</h3>
    <p>Enter your 15-man roster by typing the player's name in the search box. Player suggestions will appear as you type.</p>
    </div>
    """, unsafe_allow_html=True
)

# Multiselect widget to allow users to select 15 players
selected_players = st.multiselect(
    'Select 15 Players:',
    options=player_names,  # Provide player names as options
    default=[],  # No default selection
    max_selections=15,  # Limit to 15 players
    help='Start typing to search and select players. You must select exactly 15 players.'
)

# Error message if the user selects fewer than or more than 15 players
if len(selected_players) < 15:
    st.error("Please select exactly 15 players.")
elif len(selected_players) > 15:
    st.error("You have selected more than 15 players. Please deselect to make it 15.")
else:
    st.markdown('<p class="success-text">You have selected your 15-man roster! 🏀</p>', unsafe_allow_html=True)

    # Create two columns for better display
    col1, col2 = st.columns(2)

    with col1:
        st.write("### Selected Players' Stats:")
        # Filter the dataframe to get the stats for the selected players
        selected_players_df = df[df['Player'].isin(selected_players)]
        # Display the selected players' stats
        st.dataframe(selected_players_df)

    with col2:
        st.write("### Average Stats for the Selected 15 Players:")
        # Calculate average stats of the selected players
        average_stats = selected_players_df.mean(numeric_only=True)
        if 'Season' in average_stats.index:
            average_stats = average_stats.drop('Season')
        if 'Season Outcome' in average_stats.index:
            average_stats = average_stats.drop('Season Outcome')
        # Display average stats
        st.dataframe(average_stats)

    # Load the SVM model
    SVM_model = joblib.load('nba_ann_model.h5')

    # Prediction of the season outcome using the pre-trained model
    prediction = SVM_model.predict([average_stats])
    prediction_proba = SVM_model.predict_proba([average_stats])

    # Display the prediction with NBA-themed results
    st.markdown("<hr>", unsafe_allow_html=True)  # Horizontal line for separation
    st.markdown('<h2 class="sub-title">🏆 Predicted Season Outcome</h2>', unsafe_allow_html=True)
    st.write(f"### Predicted Season Outcome: **{prediction[0]}**")
    st.write(f"### Prediction Confidence: **{np.max(prediction_proba) * 100:.2f}%**")
