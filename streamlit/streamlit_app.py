import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import os
import plotly.express as px

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
unsupervised_dir = os.path.join(parent_dir, "Unsupervised Learning")

# load the unsupervised models
kmeans = joblib.load(os.path.join(unsupervised_dir, "kmeans_model.joblib"))
cluster_data = pd.read_csv(os.path.join(unsupervised_dir, "visiualize_df.csv"))

url_prediction = "http://127.0.0.1:8000/predict"

def visualize_cluster():
    st.title("Visualize Cluster Based on Appearance and Goals")
    fig = px.scatter(
    cluster_data,
    x="appearance",
    y="goals",
    color="clusters",
    title="Scatter Plot Colored by Cluster",
    labels={"Cluster": "Cluster Label"}
)
    st.plotly_chart(fig)


def test_model():
 
 st.title("Try our clustering model")
 # get input from user
 appearance = st.number_input("Appearance", min_value=0.0, value=0.0)
 goals = st.number_input("Goals", min_value=0.0, value=0.0)
 assists = st.number_input("Assists", min_value=0.0, value=0.0)
 minutes_played = st.number_input("Minutes Played", min_value=0.0, value=0.0)
 # initialize the state of button
 if "predict_clicked" not in st.session_state:
   st.session_state.predict_clicked = False
 # update the state of button once it is clicked
 st.button("Predict",  key="predict_button",on_click=lambda: st.session_state.update({"predict_clicked": True}))

 # make prediction
 if st.session_state.predict_clicked:
    input_data = {"appearance": appearance, "goals": goals, "assists": assists, "minutes_played": minutes_played}
    response = requests.post(url_prediction, json=input_data)
    prediction = response.json()
    st.write("Predicted Cluster:", prediction.get("predition"))


visualize_cluster()

test_model()