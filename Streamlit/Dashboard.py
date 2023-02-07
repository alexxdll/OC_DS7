import streamlit as st
import pickle
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import shap
import json
import requests as re
import datetime
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import os

# Dashboard interactif permettant :
#  - aux chargés de relation client d'expliquer de façon la plus transparente possible les décisions d’octroi de crédit
#  - à leurs clients de disposer de leurs informations personnelles et de les explorer facilement.

# Celui-ci contient les fonctionnalités suivantes :
#  - Permettre de visualiser le score et l’interprétation de ce score pour chaque client de façon intelligible pour
#    une personne non experte en data science.
#  - Permettre de visualiser des informations descriptives relatives à un client (via un système de filtre).
#  - Permettre de comparer les informations descriptives relatives à un client à l’ensemble des clients ou à un groupe
#    de clients similaires.

# st.set_page_config(
#     page_title="Loan Prediction App",
#     page_icon="./loan.png"
# )

PREDICTION_API_URL = "http://localhost:5000"
# PREDICTION_API_URL = "http://192.168.1.71:5000"

DATEORIGIN = datetime.datetime(2018, 5, 17)  # Date de la compétition Kaggle
N_CRIT = 15
N_CAT_MAX = 10

IMG_PATH = Path(__file__).parent
print(IMG_PATH)

# Suppression des marges par défaut
padding = 1
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)

st.title("Prêt à dépenser - tableau de bord")
st.markdown("""---""")
st.write(
    "Application web prédisant le risque de défaut de paiement d'un client. L'application utilise un modèle de Machine Learning, servi par une API.")
st.markdown("""---""")

# Récupération des paramètres d'affichage

# Panneau latéral
st.sidebar.title("Informations client")
st.session_state.client_container = st.sidebar

st.sidebar.button("Référence client")

# Chargement des données
data = pd.read_csv('dataset_final.csv',
                   index_col=0)
data.to_pickle('data.pkl')
data = pd.read_pickle('data.pkl')

# CHOIX DU CLIENT

with st.container():
    col1, col2 = st.columns([1, 3])
    with col1:
        st.write("")
        col1.header("**ID Client**")
        client_id = col1.selectbox('Sélectionnez un client :',
                                   data['SK_ID_CURR'].unique())
    with col2:
        # Infos principales client
        # st.write("*Traits stricts*")
        client_info = data[data['SK_ID_CURR'] == client_id].iloc[:, :]
        client_info.set_index('SK_ID_CURR', inplace=True)
        st.table(client_info)
        # Infos principales sur la demande de prêt
        # st.write("*Demande de prêt*")
        client_pret = data[data['SK_ID_CURR'] == client_id].iloc[:, :]
        client_pret.set_index('SK_ID_CURR', inplace=True)
        st.table(client_pret)