import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import lime
from sklearn.preprocessing import StandardScaler
from lime.lime_tabular import LimeTabularExplainer
import plotly.express as px
import plotly.graph_objects as go
import re

# Load the default dataset
df = pd.read_csv('https://raw.githubusercontent.com/k0yusuf/FYP/refs/heads/master/df_2024.csv').drop(columns=['Unnamed: 0'], errors='ignore')
player_names = df['Player'].unique()

# NBA themed styling
st.set_page_config(page_title="NBA Season Outcome Predictor", page_icon="🏀", layout="wide")

st.markdown("""
    <style>
    .main-title { 
        font-size:50px; 
        font-weight:bold; 
        color:#1d428a; 
        text-align:center; 
        margin-bottom:30px;
    }
    .sub-title { 
        font-size:30px; 
        font-weight:bold; 
        color:#c8102e; 
        text-align:center; 
        margin-bottom:20px;
    }
    .header { 
        background-color: #f5f5f5; 
        padding: 20px; 
        border-radius: 10px; 
        margin-bottom:20px;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<h1 class="main-title">🏀 NBA Season Outcome Prediction</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="sub-title">Create Your Dream Roster!</h2>', unsafe_allow_html=True)

# File Upload Method Selection
upload_method = st.radio(
    "Choose how to select players:",
    ["Manual Selection", "Upload Text File"]
)

# Function to process text file upload
def process_text_file(uploaded_file):
    try:
        # Read the text file
        file_content = uploaded_file.getvalue().decode('utf-8')
        
        # Split the content into lines and strip whitespace
        uploaded_players = [line.strip() for line in file_content.split('\n') if line.strip()]
        
        return uploaded_players
    except Exception as e:
        st.error(f"Error processing the file: {e}")
        return []

# Player Selection Logic
if upload_method == "Manual Selection":
    # Original multiselect method
    selected_players = st.multiselect(
        'Select between 10 and 15 Players:',
        options=player_names,
        default=[],
        max_selections=15,
        help='You must select between 10 and 15 players.'
    )
else:
    # File upload for players
    uploaded_file = st.file_uploader(
        "Upload a text file with player names (one name per line)", 
        type=['txt'], 
        help="Upload a text file with each player's name on a separate line"
    )
    
    # If file is uploaded
    if uploaded_file is not None:
        # Process the uploaded file
        uploaded_players = process_text_file(uploaded_file)
        
        # Find matching players in the dataset
        matched_players = [player for player in uploaded_players if player in player_names]
        unmatched_players = [player for player in uploaded_players if player not in player_names]
        
        # Display matching and unmatched players
        if unmatched_players:
            st.warning(f"Some players could not be found in the dataset: {', '.join(unmatched_players)}")
        
        # Allow user to select from matched players
        selected_players = st.multiselect(
            'Select players from your uploaded list:',
            options=matched_players,
            default=matched_players,
            max_selections=15,
            help='Select between 10 and 15 players found in the dataset.'
        )

# Rest of the previous script remains the same...
# (The entire prediction logic, visualization, etc. from the previous script would be here)

# Validation checks for player selection
if len(selected_players) < 10:
    st.error("Please select at least 10 players.")
elif len(selected_players) > 15:
    st.error("You have selected more than 15 players. Please reduce your selection.")
else:
    st.markdown('<p class="success-text">✅ Valid roster selected!</p>', unsafe_allow_html=True)

# Helper Functions for Prediction Section
def create_prediction_gauge(probability, prediction):
    """Create a gauge chart showing prediction confidence."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [None, 100]},
            'steps': [
                {'range': [0, 30], 'color': "lightgray"},
                {'range': [30, 70], 'color': "gray"},
                {'range': [70, 100], 'color': "darkblue"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        },
        title={'text': f"Confidence for {prediction}"}
    ))
    fig.update_layout(height=300)
    return fig

def create_probability_distribution(probabilities, classes):
    """Create a bar chart showing probability distribution across all outcomes."""
    fig = go.Figure(data=[
        go.Bar(
            x=classes,
            y=probabilities * 100,
            marker_color=['lightblue' if p != max(probabilities) else 'darkblue' 
                         for p in probabilities],
            text=[f'{p:.1f}%' for p in probabilities * 100],
            textposition='auto',
        )
    ])
    fig.update_layout(
        title='Probability Distribution Across Outcomes',
        xaxis_title='Possible Outcomes',
        yaxis_title='Probability (%)',
        yaxis_range=[0, 100],
        height=400
    )
    return fig

def calculate_confidence_metrics(prediction_proba):
    """Calculate additional confidence metrics."""
    sorted_probs = np.sort(prediction_proba[0])
    return {
        'highest_confidence': sorted_probs[-1],
        'second_highest_confidence': sorted_probs[-2],
        'margin': sorted_probs[-1] - sorted_probs[-2],
        'entropy': -np.sum(prediction_proba[0] * np.log2(prediction_proba[0] + 1e-10))
    }

# Helper Functions for Interpretability Section
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
        return selected_players_df[['Player', feature_name]].sort_values(
            by=feature_name, 
            ascending=ascending
        )
    return None

def generate_roster_recommendations(df, selected_players, weak_features, top_n=5):
    """Generate intelligent roster recommendations based on weak features."""
    recommendations = {}
    available_players = df[~df['Player'].isin(selected_players)]
    
    for feature in weak_features:
        feature_name = re.sub(r" > .+| < .+", "", feature).strip()
        if feature_name in df.columns:
            z_scores = (available_players[feature_name] - available_players[feature_name].mean()) / available_players[feature_name].std()
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

# Main Application Logic
# Instructions
st.markdown("""
    <div class="header">
    <h3>Instructions:</h3>
    <p>1. Select between 10 and 15 players to create your roster</p>
    <p>2. Click 'Predict Season Outcome' to see the team's projected performance</p>
    <p>3. Generate detailed explanations to understand the prediction</p>
    <p>4. Explore recommendations for roster improvements</p>
    </div>
    """, unsafe_allow_html=True)

# Player Selection
selected_players = st.multiselect(
    'Select between 10 and 15 Players:',
    options=player_names,
    default=[],
    max_selections=15,
    help='You must select between 10 and 15 players.'
)

# Load Models
@st.cache_resource
def load_models():
    return {
        'model': joblib.load('svm_model (1).joblib'),
        'scaler': joblib.load('scaler.joblib'),
        'X_train': joblib.load('training_data.joblib')
    }

models = load_models()
SVM_model = models['model']
scaler = models['scaler']
X_train = models['X_train']

# Process Selection
if len(selected_players) < 10:
    st.error("Please select at least 10 players.")
elif len(selected_players) > 15:
    st.error("You have selected more than 15 players. Please reduce your selection.")
else:
    st.markdown('<p class="success-text">✅ Valid roster selected!</p>', unsafe_allow_html=True)

    # Calculate team stats
# Filter data for selected players
    selected_players_df = df[df['Player'].isin(selected_players)]
    
    # Calculate average stats only for numeric columns, ignoring non-numeric ones
    numeric_columns = selected_players_df.select_dtypes(include=[np.number]).columns
    average_stats_to_display = selected_players_df[numeric_columns].mean().values
    average_stats = selected_players_df.mean(numeric_only=True).drop(['Season', 'Season Outcome'], errors='ignore').values.reshape(1, -1) 
    # Display average stats in an attractive format
    st.markdown("### 📊 Team Average Statistics")
    cols = st.columns(4)
    
    # Loop through numeric columns and corresponding values
    for i, (stat, value) in enumerate(zip(numeric_columns, average_stats_to_display)):
        with cols[i % 4]:
            st.markdown(f"""
                <div class="stat-card">
                <h4>{stat}</h4>
                <p style="font-size: 20px; font-weight: bold;">{value:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
    # Scale the features
    scaled_average_stats = scaler.transform(average_stats)

    # Prediction Section
    if st.button('🎯 Predict Season Outcome'):
        with st.spinner('Analyzing team composition and generating predictions...'):
            prediction = SVM_model.predict(scaled_average_stats)
            prediction_proba = SVM_model.predict_proba(scaled_average_stats)
            confidence_metrics = calculate_confidence_metrics(prediction_proba)
            
            pred_tabs = st.tabs(["Main Prediction", "Probability Analysis"])
            
            # Main Prediction Tab
            with pred_tabs[0]:
                col1, col2 = st.columns(2)
                
                with col1:
                    prediction_labels = ['Not Make the Playoffs', 'First Round Exit', '2nd Round Exit', 'Conference Finals', 'Finals', 'Champions']
                    st.markdown(f"### 🏆 Predicted Outcome: {prediction_labels[prediction[0]]}")
                    gauge_chart = create_prediction_gauge(
                        confidence_metrics['highest_confidence'],
                        prediction_labels[prediction[0]]
                    )
                    st.plotly_chart(gauge_chart)
                
                with col2:
                    st.markdown("### 📈 Confidence Metrics")
                    metrics_cols = st.columns(2)
                    with metrics_cols[0]:
                        st.metric("Primary Confidence", f"{confidence_metrics['highest_confidence']*100:.1f}%")
                        st.metric("Decision Margin", f"{confidence_metrics['margin']*100:.1f}%")
                    with metrics_cols[1]:
                        st.metric("Secondary Outcome", f"{confidence_metrics['second_highest_confidence']*100:.1f}%")
                        st.metric("Uncertainty", f"{confidence_metrics['entropy']:.2f}")
            
            # Probability Analysis Tab
            with pred_tabs[1]:
                prob_dist = create_probability_distribution(
                    prediction_proba[0],
                    SVM_model.classes_
                )
                st.plotly_chart(prob_dist)
                
                prob_df = pd.DataFrame({
                    'Outcome': ['Not Make the Playoffs', 'First Round Exit', '2nd Round Exit', 'Conference Finals', 'Finals', 'Champions'],
                    'Probability (%)': prediction_proba[0] * 100
                }).sort_values('Probability (%)', ascending=False)
                st.table(prob_df)

    # Interpretability Section
    if st.button('🔍 Generate Detailed Explanation'):
        with st.spinner('Analyzing roster strengths and weaknesses...'):
            explainer = LimeTabularExplainer(
                training_data=np.array(X_train),
                feature_names=df.columns.drop(['Player', 'Season', 'Season Outcome']),
                class_names=['Not Make the Playoffs', 'First Round Exit', '2nd Round Exit', 'Conference Finals', 'Finals', 'Champions'],
                mode='classification'
            )
            
            exp = explainer.explain_instance(
                scaled_average_stats[0],
                SVM_model.predict_proba,
                num_features=10
            )
            
            exp_list = exp.as_list()
            feature_importance_plot = create_feature_importance_plot(exp_list)
            
            analysis_tabs = st.tabs(["Strengths", "Weaknesses", "Recommendations"])
            
            # Strengths Analysis Tab
            with analysis_tabs[0]:
                st.plotly_chart(feature_importance_plot)
                positive_features = [x for x in exp_list if x[1] > 0]
                
                for feature, impact in positive_features:
                    raw_feature = re.sub(r" > .+| < .+", "", feature).strip()
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"### {feature}")
                        st.metric("Impact Score", f"+{impact:.3f}")
                    
                    with col2:
                        contributions = analyze_player_contributions(selected_players_df, raw_feature)
                        if contributions is not None:
                            fig = px.bar(contributions, x='Player', y=raw_feature,
                                       title=f'Player Contributions - {raw_feature}')
                            st.plotly_chart(fig)
            
            # Weaknesses Analysis Tab
            with analysis_tabs[1]:
                negative_features = [x for x in exp_list if x[1] < 0]
                
                for feature, impact in negative_features:
                    raw_feature = re.sub(r" > .+| < .+", "", feature).strip()
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"### {feature}")
                        st.metric("Impact Score", f"{impact:.3f}")
                    
                    with col2:
                        contributions = analyze_player_contributions(selected_players_df, raw_feature)
                        if contributions is not None:
                            fig = px.bar(contributions, x='Player', y=raw_feature,
                                       title=f'Player Performance - {raw_feature}')
                            st.plotly_chart(fig)
            
            # Recommendations Tab
            with analysis_tabs[2]:
                st.markdown("### 🎯 Roster Improvement Recommendations")
                
                # Get list of features that need improvement
                weak_features = [x[0] for x in exp_list if x[1] < 0]
                recommendations = generate_roster_recommendations(df, selected_players, weak_features)
                
                if recommendations:
                    for feature, data in recommendations.items():
                        with st.expander(f"Improve {feature}"):
                            st.markdown("#### Recommended Players:")
                            for player in data['players']:
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.write(f"🏀 {player}")
                                with col2:
                                    st.metric(
                                        "Potential Improvement",
                                        f"+{data['average_improvement']:.2f}"
                                    )
                            
                            # Show comparison visualization
                            feature_name = re.sub(r" > .+| < .+", "", feature).strip()
                            if feature_name in df.columns:
                                recommended_stats = df[df['Player'].isin(data['players'])][feature_name]
                                current_stats = selected_players_df[feature_name]
                                
                                fig = go.Figure()
                                fig.add_box(y=current_stats, name="Current Roster")
                                fig.add_box(y=recommended_stats, name="Recommended Players")
                                fig.update_layout(
                                    title=f"Statistical Comparison - {feature_name}",
                                    yaxis_title=feature_name,
                                    showlegend=True,
                                    height=400
                                )
                                st.plotly_chart(fig)
