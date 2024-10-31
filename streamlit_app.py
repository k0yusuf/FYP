import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.preprocessing import StandardScaler

# Load the dataset containing player stats
# Load the dataset containing player stats
df = pd.read_csv('https://raw.githubusercontent.com/k0yusuf/FYP/refs/heads/master/df_2024.csv')
df = df.drop(columns=['Unnamed: 0'], errors='ignore')

# Remove commas from numeric columns and convert them to floats

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

# Button to show explanation
    if st.button("Show Prediction Explanation"):
    # Separate page for explanation
        st.markdown('<h2 class="sub-title">📊 Model Explanation with SHAP and LIME</h2>', unsafe_allow_html=True)

    # SHAP explanation
        st.write("### SHAP Explanation")

    # Define a prediction function for SHAP
        def predict_proba_shap(X):
            return SVM_model.predict_proba(X)

    # Use KernelExplainer for SHAP
        shap_explainer = shap.KernelExplainer(predict_proba_shap, shap.kmeans(scaler.transform(df.drop(['Player', 'Season', 'Season Outcome', 'Team','Offense Position', 'Offensive Archetype', 'Defensive Role', 'Stable Avg 2PT Shot Distance','Multiple Teams'], axis=1).values), 10))
        shap_values = shap_explainer.shap_values(scaled_average_stats)

    #Plot SHAP values
        shap.summary_plot(shap_values, scaled_average_stats, plot_type="bar", class_names=SVM_model.classes_)
        st.pyplot(bbox_inches='tight')


        #shap_explainer = shap.KernelExplainer(SVM_model.predict_proba, scaler.transform(df.drop(['Player', 'Season', 'Season Outcome'], axis=1).values))
        #shap_values = shap_explainer.shap_values(scaled_average_stats)

        # Plot SHAP values
        #st.set_option('deprecation.showPyplotGlobalUse', False)
        #shap.summary_plot(shap_values, scaled_average_stats, plot_type="bar", class_names=SVM_model.classes_)
        #st.pyplot(bbox_inches='tight')

        # LIME explanation
        #st.write("### LIME Explanation")
        #lime_explainer = LimeTabularExplainer(
            #scaler.transform(df.drop(['Player', 'Season', 'Season Outcome'], axis=1).values),
            #feature_names=average_stats.index,
            #class_names=SVM_model.classes_,
            #discretize_continuous=True
        #)

        #lime_exp = lime_explainer.explain_instance(
         #   data_row=scaled_average_stats[0],
          #  predict_fn=SVM_model.predict_proba
        #)

        # Display LIME explanation as HTML
        #st.write(lime_exp.as_html(), unsafe_allow_html=True)
