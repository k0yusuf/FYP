import streamlit as st
import pandas as pd
import joblib  # To load the pre-trained model
import numpy as np
import pickle

# Load the dataset containing player stats
df = pd.read_csv('https://raw.githubusercontent.com/k0yusuf/FYP/refs/heads/master/df_2024.csv')

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
    if 'Season' in average_stats.index:
        average_stats = average_stats.drop('Season')
    if 'Season Outcome' in average_stats.index:
        average_stats = average_stats.drop('Season Outcome')
    st.write("### Average Stats for the Selected 15 Players:")
    st.dataframe(average_stats)

    SVM_model = pickle.load(open('svm_model.sav', 'rb'))


# Function to get prediction and probability
def predict_season_outcome(model, data):
    # Get prediction
    prediction = model.predict(data)
    # Get prediction probabilities (assuming binary classification or multiclass model)
    prediction_prob = model.predict_proba(data)
    
    return prediction, prediction_prob

# Button to predict the season outcome
if st.button('Predict Season Outcome'):
    # Get prediction and its probability
    prediction, prediction_prob = predict_season_outcome(model, average_stats)
    
    # Display the prediction
    st.write(f"### Predicted Season Outcome: {prediction[0]}")
    
    # Display the prediction probability
    st.write("### Prediction Probability:")
    for i, prob in enumerate(prediction_prob[0]):
        st.write(f"Class {i}: {prob:.4f}")

