import streamlit as st
import pandas as pd
import numpy as np
import pickle
files.download('svm_model.pkl')

# Load the dataset containing player stats
df = pd.read_csv('https://raw.githubusercontent.com/k0yusuf/FYP/refs/heads/master/df_2024.csv')

# Load the pre-trained model
@st.cache(allow_output_mutation=True)
def load_model():
    return joblib.load('svm_model.pkl')  # Make sure to update this path if you're using a hosted model

SVM_model = load_model()
# Load scaler if applicable (uncomment if you used a scaler during training)
# scaler = pickle.load(open('scaler.pkl', 'rb'))

# Unique player names for selection
player_names = df['Player'].unique()

# Title of the app
st.title('NBA Season Outcome Prediction')

# Instructions for the user
st.write("""
### Enter your 15-man roster by typing the player's name. Suggestions will appear as you type.
""")

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
    st.success("You have selected your 15-man roster!")

    # Filter the dataframe to get the stats for the selected players
    selected_players_df = df[df['Player'].isin(selected_players)]

    # Display the selected players' stats
    st.write("### Selected Players' Stats:")
    st.dataframe(selected_players_df)

    # Calculate average stats of the selected players
    average_stats = selected_players_df.mean(numeric_only=True)
    
    # Drop columns that should not be included in the prediction
    average_stats = average_stats.drop(['Season', 'Season Outcome'], errors='ignore')

    # Reshape the average_stats to 2D array (1 sample, n features)
    average_stats_array = average_stats.values.reshape(1, -1)

    # Scale the stats if a scaler was used during training
    # average_stats_array = scaler.transform(average_stats_array)
    average_stats = selected_players_df.mean(numeric_only=True)
    average_stats = average_stats.drop(['Season', 'Season Outcome'], errors='ignore')

    # Reshape average stats for prediction
    average_stats_array = average_stats.values.reshape(1, -1)

    # Predict the season outcome using the pre-trained model
    prediction = SVM_model.predict(average_stats_array)
    prediction_proba = SVM_model.predict_proba(average_stats_array)

    # Display prediction and probability
    st.write(f"### Predicted Season Outcome: {prediction[0]}")
    st.write(f"### Prediction Probability: {prediction_proba}")
