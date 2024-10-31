import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
from sklearn.preprocessing import StandardScaler

# Load the dataset containing player stats
df = pd.read_csv('https://raw.githubusercontent.com/k0yusuf/FYP/refs/heads/master/df_2024.csv').drop(columns=['Unnamed: 0'], errors='ignore')
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

# Check for player selection limits
if len(selected_players) < 10:
    st.error("Please select at least 10 players.")
elif len(selected_players) > 15:
    st.error("You have selected more than 15 players. Please deselect to make it between 10 and 15.")
else:
    st.markdown('<p class="success-text">You have selected your roster! 🏀</p>', unsafe_allow_html=True)

    # Filter and display selected player stats
    selected_players_df = df[df['Player'].isin(selected_players)]

    # Calculate average stats and convert to DataFrame
    average_stats = selected_players_df.mean(numeric_only=True).drop(['Season', 'Season Outcome'], errors='ignore')
    average_stats_df = pd.DataFrame(average_stats).T  # Convert Series to DataFrame with one row

    st.write("### Average Stats for Selected Players:")
    st.dataframe(average_stats)

    # Load the SVM model and scaler
    SVM_model = joblib.load('svm_model (1).joblib')
    scaler = joblib.load('scaler.joblib')

    # Scale the features using the loaded scaler
    scaled_average_stats = scaler.transform(average_stats_df)

    # Predict the season outcome
    prediction = SVM_model.predict(scaled_average_stats)
    prediction_proba = SVM_model.predict_proba(scaled_average_stats)

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

    # Button to show SHAP waterfall plot explanation
    if st.button("Show SHAP Waterfall Explanation"):
        st.markdown('<h2 class="sub-title">📊 Model Explanation with SHAP</h2>', unsafe_allow_html=True)

        # SHAP Kernel Explainer
        explainer = shap.KernelExplainer(SVM_model.predict_proba, scaler.transform(df.drop(['Player', 'Season', 'Season Outcome', 'Team','Offense Position', 'Offensive Archetype', 'Defensive Role', 'Stable Avg 2PT Shot Distance','Multiple Teams'], axis=1).values))
        
        # Generate SHAP values for the selected prediction instance
        shap_values = explainer.shap_values(scaled_average_stats)

        # SHAP waterfall plot for the class with the highest probability
        st.write("### SHAP Waterfall Plot")
        shap_class_index = np.argmax(prediction_proba[0])
    
    # Limit to the top 20 most important features
        shap_values_selected = shap_values[shap_class_index][0]
        top_features = np.argsort(np.abs(shap_values_selected))[-20:]
        shap_explanation = shap.Explanation(
            values=shap_values_selected[top_features], 
            base_values=explainer.expected_value[shap_class_index], 
            feature_names=average_stats_df.columns[top_features]
    )

    # Render the waterfall plot for the top features
        shap.waterfall_plot(shap_explanation)
        st.pyplot()

