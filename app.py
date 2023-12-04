#!/usr/bin/env python
# coding: utf-8

# Comparison
# Receiver Operating Characterisc curve

# Credits to Sarvesh Krishan for the initial development of the streamlit website.
# I debugged the error of the wensite constantly displaying the same answer by applying MinMaxScaler to the user's input.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import roc_curve, auc
from sklearn.neural_network import MLPClassifier

import streamlit as st

# Load the data
dataset = pd.read_csv('riceclass.csv')
X = dataset.drop(['Class', 'id'], axis=1)
y = dataset['Class']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

# Standardize/Scale the data
scaler_lr = MinMaxScaler(feature_range=(0, 1))
X_train_lr_scaled = scaler_lr.fit_transform(X_train)
X_test_lr_scaled = scaler_lr.transform(X_test)

scaler_knn = MinMaxScaler(feature_range=(0, 1))
X_train_knn_scaled = scaler_knn.fit_transform(X_train)
X_test_knn_scaled = scaler_knn.transform(X_test)

scaler_nn = MinMaxScaler(feature_range=(0, 1))
X_train_nn_scaled = scaler_nn.fit_transform(X_train)
X_test_nn_scaled = scaler_nn.transform(X_test)


# Logistic Regression
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_lr_scaled, y_train)


# K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=65)  # Assuming 65 is the best k value from your tuning
knn.fit(X_train_knn_scaled, y_train)

# Neural Network
mlp = MLPClassifier(solver = 'adam', random_state = 42, activation = 'logistic', learning_rate_init = 0.01, batch_size = 300, hidden_layer_sizes = (12, 24, 48,), max_iter = 1000)
mlp.fit(X_train_nn_scaled, y_train)



st.title('Which species of rice is it? :rice:')
st.markdown('Enter values for the variables below and see what Logistic Regression, K-Nearest Neighbors, and a Neural Network predict.')

area = st.slider('Area', 0,20000)
perimeter = st.text_input('Perimeter')
major_axis = st.text_input('Major Axis Length')
minor_axis = st.text_input('Minor Axis Length')
eccentricity = st.text_input('Eccentricity')
convex_area = st.slider('Convex Area', 0,20000)
extent = st.text_input('Extent')


def gui_predict():
    obs = [int(area), float(perimeter), float(major_axis), float(minor_axis), float(eccentricity), int(convex_area), float(extent)]
    obs = pd.DataFrame([obs])
    X_train_obs_scaled = scaler_nn.transform(obs)
    obs = pd.DataFrame(data = X_train_obs_scaled)
    st.success('Logistic Regression predicted ' + logreg.predict(obs) + ', K-Nearest Neighbors predicted ' + knn.predict(obs) + ', and Neural Network predicted ' + mlp.predict(obs))

if st.button('Predict'):
    gui_predict()
