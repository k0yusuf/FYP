import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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
    
    # Add SHAP explanation
    st.markdown("### 📊 Model Explanation using SHAP")
    explainer = shap.KernelExplainer(SVM_model.predict_proba, shap.sample(scaled_average_stats, 100))
    shap_values = explainer.shap_values(scaled_average_stats)
    
    # Plot SHAP values
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values[prediction[0]], scaled_average_stats, 
                     feature_names=average_stats_df.columns,
                     show=False)
    st.pyplot(plt)
    plt.clf()
    
    # Add LIME explanation
    st.markdown("### 🔍 Model Explanation using LIME")
    # Prepare training data for LIME
    X_train = scaler.transform(df.drop(['Player', 'Season', 'Season Outcome'], axis=1, errors='ignore'))
    feature_names = df.drop(['Player', 'Season', 'Season Outcome'], axis=1, errors='ignore').columns.tolist()
    
    # Create LIME explainer
    lime_explainer = LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=['0', '1', '2', '3', '4', '5'],
        mode='classification'
    )
    
    # Generate LIME explanation
    explanation = lime_explainer.explain_instance(
        scaled_average_stats[0], 
        SVM_model.predict_proba,
        num_features=10
    )
    
    # Plot LIME explanation
    plt.figure(figsize=(10, 6))
    explanation.as_pyplot_figure()
    st.pyplot(plt)
    
    # Display feature importance summary
    st.markdown("### 🏆 Key Factors Influencing the Prediction")
    feature_importance = pd.DataFrame(
        explanation.as_list(),
        columns=['Feature', 'Impact']
    ).sort_values('Impact', key=abs, ascending=False)
    
    st.write("Top influential features:")
    st.dataframe(feature_importance)
