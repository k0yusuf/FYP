import streamlit as st
import pandas as pd
import joblib  # To load the pre-trained model
import numpy as np

# Load the dataset containing player stats
df = pd.read_excel('/mnt/data/NBA Dataset Final.xlsx')

# Load the pre-trained model (make sure to have your model saved as 'model.pkl')
model = joblib.load('model.pkl')

# Get player stats from dataset based on input names
def get_player_stats(player_name, df):
    player_data = df[df['Player'] == player_name]
    if not player_data.empty:
        return player_data.iloc[0]  # Return the first match (if multiple entries)
    else:
        return None

# Average the stats of selected players
def average_player_stats(players, df):
    stats = []
    for player in players:
        player_stats = get_player_stats(player, df)
        if player_stats is not None:
            stats.append(player_stats.drop('Player'))  # Remove 'Player' column for numerical stats
        else:
            st.warning(f"Player '{player}' not found in the dataset!")
    
    if stats:
        avg_stats = np.mean(stats, axis=0)  # Average the stats
        return pd.DataFrame([avg_stats], columns=stats[0].index)
    else:
        return None

# Streamlit app title
st.title("NBA Season Outcome Prediction")

# Input: Enter 15 player names
players = []
for i in range(1, 16):
    player_name = st.text_input(f"Enter Player {i} Name", "")
    if player_name:
        players.append(player_name)

# When 15 players are entered, calculate the average stats and make a prediction
if len(players) == 15:
    st.write("### Selected Players:")
    st.write(players)
    
    # Get average stats
    avg_stats_df = average_player_stats(players, df)
    
    if avg_stats_df is not None:
        st.write("### Average Player Stats:")
        st.dataframe(avg_stats_df)

        # Use the model to predict the season outcome
        prediction = model.predict(avg_stats_df)
        
        # Display the prediction result
        st.write("### Predicted Season Outcome:")
        st.write(prediction[0])
