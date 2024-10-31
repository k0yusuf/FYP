import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt

# Load the dataset containing player stats
df = pd.read_csv('https://raw.githubusercontent.com/k0yusuf/FYP/refs/heads/master/df_2024.csv').drop(columns=['Unnamed: 0'], errors='ignore')

# Define the expected feature names (these should match your model's training features exactly)
expected_features = [
    'Age', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA',
    '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK',
    'TOV', 'PF', 'PTS'
]

# Filter columns to include only expected features
df_filtered = df[['Player'] + expected_features]
player_names = df_filtered['Player'].unique()

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
    selected_players_df = df_filtered[df_filtered['Player'].isin(selected_players)]
    
    # Calculate average stats and convert to DataFrame
    average_stats = selected_players_df[expected_features].mean()
    average_stats_df = pd.DataFrame(average_stats).T
    
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
    
    try:
        # Create background data for SHAP
        background_data = df_filtered[expected_features].sample(n=100)
        background_data_scaled = scaler.transform(background_data)
        
        # Initialize the SHAP explainer
        explainer = shap.KernelExplainer(
            lambda x: SVM_model.predict_proba(x)[:, prediction[0]],
            background_data_scaled
        )
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(scaled_average_stats)
        
        # Create SHAP force plot
        st.write("SHAP Force Plot (Feature Impact):")
        force_plot = shap.force_plot(
            explainer.expected_value,
            shap_values,
            scaled_average_stats,
            feature_names=expected_features,
            matplotlib=True,
            show=False
        )
        plt.tight_layout()
        st.pyplot(plt)
        plt.clf()
        
        # Create SHAP waterfall plot
        st.write("SHAP Waterfall Plot (Feature Contribution):")
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=scaled_average_stats[0],
                feature_names=expected_features
            ),
            show=False
        )
        st.pyplot(plt)
        plt.clf()
        
    except Exception as e:
        st.error(f"SHAP visualization error: {str(e)}")
    
    # Add LIME explanation
    st.markdown("### 🔍 Model Explanation using LIME")
    try:
        # Prepare training data for LIME
        X_train = scaler.transform(df_filtered[expected_features])
        
        # Create LIME explainer
        lime_explainer = LimeTabularExplainer(
            X_train,
            feature_names=expected_features,
            class_names=[str(i) for i in range(6)],
            mode='classification'
        )
        
        # Generate LIME explanation
        explanation = lime_explainer.explain_instance(
            scaled_average_stats[0], 
            SVM_model.predict_proba,
            num_features=10
        )
        
        # Plot LIME explanation
        st.write("LIME Feature Importance Plot:")
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
        
    except Exception as e:
        st.error(f"LIME visualization error: {str(e)}")
