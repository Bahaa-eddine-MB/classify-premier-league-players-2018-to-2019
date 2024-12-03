import streamlit as st
import pandas as pd
import joblib

# Load the model, scaler, and cluster names
kmeans_model = joblib.load('kmeans_model.pkl')
scaler_model = joblib.load('scaler.pkl')
cluster_names = joblib.load('cluster_names.pkl')

# Features for clustering
features_for_clustering = [
    'minutes_played_overall', 'goals_overall', 'assists_overall',
    'appearances_overall', 'yellow_cards_overall',
    'red_cards_overall', 'clean_sheets_overall'
]

# Streamlit app
st.title("Player Clustering Prediction")

st.write("""
This app predicts which cluster a player belongs to based on their statistics.
The clusters are defined as:
- Least Contributed Players
- Moderate Players
- Most Valuable Players
- Defensive Players
- Highly Offensive Players
""")

# Input form
with st.form("player_stats_form"):
    st.write("Enter Player Statistics:")
    
    minutes_played_overall = st.number_input("Minutes Played Overall", min_value=0, step=1)
    goals_overall = st.number_input("Goals Overall", min_value=0, step=1)
    assists_overall = st.number_input("Assists Overall", min_value=0, step=1)
    appearances_overall = st.number_input("Appearances Overall", min_value=0, step=1)
    yellow_cards_overall = st.number_input("Yellow Cards Overall", min_value=0, step=1)
    red_cards_overall = st.number_input("Red Cards Overall", min_value=0, step=1)
    clean_sheets_overall = st.number_input("Clean Sheets Overall", min_value=0, step=1)
    
    # Submit button
    submitted = st.form_submit_button("Predict")

if submitted:
    # Prepare the input data
    player_data = {
        'minutes_played_overall': minutes_played_overall,
        'goals_overall': goals_overall,
        'assists_overall': assists_overall,
        'appearances_overall': appearances_overall,
        'yellow_cards_overall': yellow_cards_overall,
        'red_cards_overall': red_cards_overall,
        'clean_sheets_overall': clean_sheets_overall
    }
    player_df = pd.DataFrame([player_data])
    player_scaled = scaler_model.transform(player_df)
    
    # Predict the cluster
    cluster = kmeans_model.predict(player_scaled)[0]
    cluster_name = cluster_names[cluster]
    
    # Display results
    st.write("### Prediction Results")
    st.write(f"**Cluster ID**: {cluster}")
    st.write(f"**This is**: {cluster_name}")
