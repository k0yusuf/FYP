import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import lime
from sklearn.preprocessing import StandardScaler
from lime.lime_tabular import LimeTabularExplainer
import re
import plotly.express as px
import plotly.graph_objects as go

# Load the dataset containing player stats
df = pd.read_csv('https://raw.githubusercontent.com/k0yusuf/FYP/refs/heads/master/df_2024.csv').drop(columns=['Unnamed: 0'], errors='ignore')
player_names = df['Player'].unique()

# Set custom NBA-themed page layout
st.set_page_config(page_title="NBA Season Outcome Predictor", page_icon="🏀", layout="wide")

# NBA themed title and header
st.markdown(
    """
    <style>
    .main-title { font-size:50px; font-weight:bold; color:#1d428a; text-align:center; }
    .sub-title { font-size:30px; font-weight:bold; color:#c8102e; text-align:center; }
    .header { background-color: #f5f5f5; padding: 10px; border-radius: 5px; }
    .success-text { font-size:18px; color: green; }
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
SVM_model = joblib.load('svm_model (1).joblib')
scaler = joblib.load('scaler.joblib')
X_train = joblib.load('training_data.joblib')

# Check for player selection limits
if len(selected_players) < 10:
    st.error("Please select at least 10 players.")
elif len(selected_players) > 15:
    st.error("You have selected more than 15 players. Please deselect to make it between 10 and 15.")
else:
    st.markdown('<p class="success-text">You have selected your roster! 🏀</p>', unsafe_allow_html=True)

    # Filter selected player stats and calculate average
    selected_players_df = df[df['Player'].isin(selected_players)]
    average_stats = selected_players_df.mean(numeric_only=True).drop(['Season', 'Season Outcome'], errors='ignore').values.reshape(1, -1) 
    average_stats_df = pd.DataFrame(average_stats).T
    
    # Display average stats
    st.write("### Average Stats for Selected Players:")
    st.dataframe(average_stats)

    # Scale the features
    scaled_average_stats = scaler.transform(average_stats)

    # Button for Prediction
    if st.button('Predict Season Outcome'):
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
    def create_feature_importance_plot(exp_list):
        """Create a horizontal bar chart of feature importance using Plotly."""
        df = pd.DataFrame(exp_list, columns=['Feature', 'Impact'])
        df = df.sort_values('Impact')
        
        fig = go.Figure(go.Bar(
            x=df['Impact'],
            y=df['Feature'],
            orientation='h',
            marker_color=['red' if x < 0 else 'green' for x in df['Impact']]
        ))
        
        fig.update_layout(
            title='Feature Importance Analysis',
            xaxis_title='Impact on Prediction',
            yaxis_title='Features',
            height=400
        )
        return fig
    
    def analyze_player_contributions(selected_players_df, feature_name, ascending=False):
        """Analyze individual player contributions for a specific feature."""
        if feature_name in selected_players_df.columns:
            player_stats = selected_players_df[['Player', feature_name]].sort_values(
                by=feature_name, 
                ascending=ascending
            )
            return player_stats
        return None
    
    def generate_roster_recommendations(df, selected_players, weak_features, top_n=5):
        """Generate intelligent roster recommendations based on weak features."""
        recommendations = {}
        available_players = df[~df['Player'].isin(selected_players)]
        
        for feature in weak_features:
            feature_name = re.sub(r" > .+| < .+", "", feature).strip()
            if feature_name in df.columns:
                # Calculate z-scores for the feature
                z_scores = (available_players[feature_name] - available_players[feature_name].mean()) / available_players[feature_name].std()
                
                # Get players with high z-scores (above 1 standard deviation)
                strong_players = available_players[z_scores > 1]['Player'].tolist()[:top_n]
                
                if strong_players:
                    recommendations[feature] = {
                        'players': strong_players,
                        'average_improvement': (
                            available_players[available_players['Player'].isin(strong_players)][feature_name].mean() -
                            selected_players_df[feature_name].mean()
                        )
                    }
        
        return recommendations
    
    if st.button('Generate Detailed Prediction Explanation'):
        with st.spinner('Analyzing your roster...'):
            # Create LIME explainer
            explainer = LimeTabularExplainer(
                training_data=np.array(X_train),
                feature_names=df.columns.drop(['Player', 'Season', 'Season Outcome']),
                class_names=['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5'],
                mode='classification'
            )
            
            exp = explainer.explain_instance(
                scaled_average_stats[0],
                SVM_model.predict_proba,
                num_features=10
            )
            
            # Extract and process feature impacts
            exp_list = exp.as_list()
            feature_importance_plot = create_feature_importance_plot(exp_list)
            
            # Display overall feature importance visualization
            st.plotly_chart(feature_importance_plot)
            
            # Separate positive and negative features
            positive_features = [x for x in exp_list if x[1] > 0]
            negative_features = [x for x in exp_list if x[1] < 0]
            
            # Create tabs for different analysis sections
            tabs = st.tabs(["Strengths", "Weaknesses", "Recommendations", "Player Details"])
            
            # Strengths Analysis Tab
            with tabs[0]:
                st.subheader("Team Strengths Analysis")
                for feature, impact in positive_features:
                    raw_feature = re.sub(r" > .+| < .+", "", feature).strip()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**{feature}**")
                        st.write(f"Impact Score: +{impact:.3f}")
                    
                    with col2:
                        player_contributions = analyze_player_contributions(selected_players_df, raw_feature)
                        if player_contributions is not None:
                            fig = px.bar(player_contributions, x='Player', y=raw_feature,
                                       title=f'Player Contributions - {raw_feature}')
                            st.plotly_chart(fig)
            
            # Weaknesses Analysis Tab
            with tabs[1]:
                st.subheader("Areas for Improvement")
                for feature, impact in negative_features:
                    raw_feature = re.sub(r" > .+| < .+", "", feature).strip()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**{feature}**")
                        st.write(f"Impact Score: {impact:.3f}")
                    
                    with col2:
                        player_contributions = analyze_player_contributions(selected_players_df, raw_feature)
                        if player_contributions is not None:
                            fig = px.bar(player_contributions, x='Player', y=raw_feature,
                                       title=f'Player Performance - {raw_feature}')
                            st.plotly_chart(fig)
            
            # Recommendations Tab
            with tabs[2]:
                st.subheader("Roster Improvement Recommendations")
                weak_features = [x[0] for x in negative_features]
                recommendations = generate_roster_recommendations(df, selected_players, weak_features)
                
                for feature, rec in recommendations.items():
                    with st.expander(f"Recommendations for {feature}"):
                        st.write("**Recommended Players:**")
                        for player in rec['players']:
                            st.write(f"- {player}")
                        st.write(f"**Expected Improvement:** {rec['average_improvement']:.2f} units")
            
