import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
from sklearn.preprocessing import StandardScaler
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt

# [Previous imports and initial setup remain the same until the prediction button]

def analyze_team_with_lime(model, feature_names, average_stats_df, training_data):
    explainer = LimeTabularExplainer(
        training_data,
        feature_names=feature_names,
        class_names=model.classes_,
        mode='classification'
    )
    
    exp = explainer.explain_instance(
        average_stats_df.iloc[0],
        model.predict_proba,
        num_features=len(feature_names)
    )
    
    # Get feature importance scores
    feature_importance = dict(exp.as_list())
    return feature_importance

def analyze_team_with_shap(model, average_stats_df):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(average_stats_df)
    
    if isinstance(shap_values, list):
        # Classification model
        shap_values = np.abs(np.array(shap_values)).mean(axis=1)[0]
    else:
        # Regression model
        shap_values = np.abs(shap_values).mean(axis=0)
    
    feature_importance = dict(zip(average_stats_df.columns, shap_values))
    return feature_importance

def find_recommended_players(df, weak_features, current_players, top_n=3):
    recommendations = {}
    
    for feature in weak_features:
        # Sort players by the weak feature, excluding current roster
        available_players = df[~df['Player'].isin(current_players)].copy()
        available_players = available_players.sort_values(by=feature, ascending=False)
        
        # Get top players for this feature
        top_players = available_players.head(top_n)[['Player', feature]]
        recommendations[feature] = top_players
    
    return recommendations

# [Previous code remains the same until the prediction button]

if st.button('Predict Season Outcome'):
    # [Previous prediction code remains the same]
    
    # After showing the prediction, add the analysis section
    st.markdown('<h2 class="sub-title">📊 Team Analysis</h2>', unsafe_allow_html=True)
    
    # Prepare feature names and training data
    feature_names = average_stats_df.columns.tolist()
    training_data = df.drop(['Player', 'Season', 'Season Outcome'], axis=1, errors='ignore')
    
    # Get feature importance from both LIME and SHAP
    lime_importance = analyze_team_with_lime(SVM_model, feature_names, average_stats_df, training_data.values)
    shap_importance = analyze_team_with_shap(SVM_model, average_stats_df)
    
    # Combine and average the importance scores
    combined_importance = {}
    for feature in feature_names:
        lime_score = lime_importance.get(feature, 0)
        shap_score = shap_importance.get(feature, 0)
        combined_importance[feature] = (abs(lime_score) + abs(shap_score)) / 2
    
    # Sort features by importance
    sorted_features = sorted(combined_importance.items(), key=lambda x: abs(x[1]), reverse=True)
    
    # Identify strengths and weaknesses
    team_stats = average_stats_df.iloc[0]
    league_avg = df.mean(numeric_only=True)
    
    strengths = []
    weaknesses = []
    
    for feature, importance in sorted_features[:5]:  # Look at top 5 most important features
        if team_stats[feature] > league_avg[feature]:
            strengths.append((feature, team_stats[feature], league_avg[feature]))
        else:
            weaknesses.append((feature, team_stats[feature], league_avg[feature]))
    
    # Display team strengths
    st.write("### 💪 Team Strengths")
    for feature, team_val, league_val in strengths:
        st.write(f"**{feature}**: {team_val:.2f} (League avg: {league_val:.2f})")
        players_strong = selected_players_df.nlargest(3, feature)[['Player', feature]]
        st.write("Top contributors:")
        st.dataframe(players_strong)
    
    # Display team weaknesses and recommendations
    st.write("### 🎯 Areas for Improvement")
    weak_features = [w[0] for w in weaknesses]
    recommendations = find_recommended_players(df, weak_features, selected_players)
    
    for feature, team_val, league_val in weaknesses:
        st.write(f"**{feature}**: {team_val:.2f} (League avg: {league_val:.2f})")
        st.write("Recommended players to target:")
        st.dataframe(recommendations[feature])
    
    # Visualize feature importance
    st.write("### 📈 Feature Importance Overview")
    fig, ax = plt.subplots(figsize=(10, 6))
    features, importance = zip(*sorted_features[:10])  # Top 10 features
    ax.barh(features, importance)
    ax.set_xlabel('Feature Importance')
    plt.tight_layout()
    st.pyplot(fig)
