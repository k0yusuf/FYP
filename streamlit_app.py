import streamlit as st
import pandas as pd
import joblib  # To load the pre-trained model
import numpy as np

# Load the dataset containing player stats
df = pd.read_csv('https://raw.githubusercontent.com/k0yusuf/FYP/refs/heads/master/NBA%20Dataset%20Final%20CSV.csv')

df
