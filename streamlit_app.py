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
df = pd.read_csv('https://raw.githubusercontent.com/k0yusuf/FYP/refs/heads/master/df_2024%20(1).csv').drop(columns=['Unnamed: 0'], errors='ignore')
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
    ["Manual Selection", "Upload Text File"],
    key="upload_method_radio"
)

# Function to process text file upload
def process_text_file(uploaded_file):
    try:
        # Read the text file
        file_content = uploaded_file.getvalue().decode('utf-8')
        
        # Split the content into lines, strip whitespace, and convert to lowercase
        uploaded_players = [line.strip().lower() for line in file_content.split('\n') if line.strip()]
        
        return uploaded_players
    except Exception as e:
        st.error(f"Error processing the file: {e}")
        return []

# Initialize selected_players with an empty list
selected_players = []

# Player Selection Logic
if upload_method == "Manual Selection":
    # Original multiselect method with a unique key
    selected_players = st.multiselect(
        'Select between 10 and 15 Players:',
        options=player_names,
        default=[],
        max_selections=15,
        help='You must select between 10 and 15 players.',
        key="manual_player_selection"
    )
else:
    # File upload for players
    uploaded_file = st.file_uploader(
        "Upload a text file with player names (one name per line)", 
        type=['txt'], 
        help="Upload a text file with each player's name on a separate line",
        key="player_file_upload"
    )
    
    # If file is uploaded
    if uploaded_file is not None:
        # Process the uploaded file
        uploaded_players = process_text_file(uploaded_file)
        
        # Convert player names in the dataset to lowercase for case-insensitive matching
        player_names_lower = [name.lower() for name in player_names]
        
        # Find matching players in the dataset (case-insensitive)
        matched_players = [player for player in uploaded_players if player in player_names_lower]
        unmatched_players = [player for player in uploaded_players if player not in player_names_lower]
        
        # Display matching and unmatched players
        if unmatched_players:
            st.warning(f"Some players could not be found in the dataset: {', '.join(unmatched_players)}")
        
        # Allow user to select from matched players with a unique key
        selected_players = st.multiselect(
            'Select players from your uploaded list:',
            options=[name.title() for name in matched_players],  # Convert back to title case for display
            default=[name.title() for name in matched_players],  # Convert back to title case for display
            max_selections=15,
            help='Select between 10 and 15 players found in the dataset.',
            key="uploaded_player_selection"
        )

# Handle the case where too many players are selected
if len(selected_players) > 15:
    st.error("❌ You have selected more than 15 players. Please reduce your selection to a maximum of 15 players.")
    # Reset the selection to the first 15 players
    selected_players = selected_players[:15]
    st.info(f"Your selection has been trimmed to the first 15 players: {', '.join(selected_players)}")

# Validation checks for player selection
if not selected_players:
    st.error("No players selected. Please choose players either manually or by uploading a file.")
elif len(selected_players) < 10:
    st.error(f"You have selected {len(selected_players)} players. Please select at least 10 players.")
else:
    st.markdown('<p class="success-text">✅ Valid roster selected!</p>', unsafe_allow_html=True)

# Load Models
@st.cache_resource
def load_models():
    return {
        'model': joblib.load('best_svm_model.joblib'),
        'scaler': joblib.load('scaler (1).joblib'),
        'X_train': joblib.load('X_train.joblib')
    }

models = load_models()
SVM_model = models['model']
scaler = models['scaler']
X_train = models['X_train']

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
if len(selected_players) < 10:
    st.warning("⚠️ Please select at least 10 players to form a team.")
elif len(selected_players) > 15:
    st.error("❌ You have selected more than 15 players. Please reduce your selection to a maximum of 15 players.")
else:
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
                    st.markdown(f"### 🏆 Predicted Outcome: {prediction_labels[int(prediction[0])]}")
                    gauge_chart = create_prediction_gauge(
                        confidence_metrics['highest_confidence'],
                        prediction_labels[int(prediction[0])]
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
                            # Recommended Additions
                            st.markdown("#### 📈 Recommended Additions:")
                            for player in data['players_to_add']:
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.write(f"🏀 {player}")
                                with col2:
                                    st.metric(
                                        "Potential Improvement",
                                        f"+{data['average_improvement']:.2f}"
                                    )
                            
                            # Players to Reconsider
                            if not data['players_to_reconsider'].empty:
                                st.markdown("#### 📉 Players to Reconsider:")
                                for _, row in data['players_to_reconsider'].iterrows():
                                    col1, col2 = st.columns([3, 1])
                                    with col1:
                                        st.write(f"🏀 {row['Player']}")
                                    with col2:
                                        st.metric(
                                            "Performance",
                                            f"{row['z_score']:.2f} σ"
                                        )
                            
                            # Show comparison visualization
                            feature_name = re.sub(r" > .+| < .+", "", feature).strip()
                            if feature_name in df.columns:
                                # Create visualization comparing current roster, recommended additions, and players to reconsider
                                fig = go.Figure()
                                
                                # Current roster stats
                                current_stats = selected_players_df[feature_name]
                                fig.add_box(y=current_stats, name="Current Roster", marker_color='gray')
                                
                                # Recommended players stats
                                if data['players_to_add']:
                                    recommended_stats = df[df['Player'].isin(data['players_to_add'])][feature_name]
                                    fig.add_box(y=recommended_stats, name="Recommended Additions", marker_color='green')
                                
                                # Players to reconsider stats
                                if not data['players_to_reconsider'].empty:
                                    reconsider_stats = df[df['Player'].isin(data['players_to_reconsider']['Player'])][feature_name]
                                    fig.add_box(y=reconsider_stats, name="Players to Reconsider", marker_color='red')
                                
                                fig.update_layout(
                                    title=f"Statistical Comparison - {feature_name}",
                                    yaxis_title=feature_name,
                                    showlegend=True,
                                    height=400
                                )
                                st.plotly_chart(fig)
