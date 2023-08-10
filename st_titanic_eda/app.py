import streamlit as st
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# make containers
header = st.container()
datasets = st.container()
features = st.container()
model_training = st.container()

with header:
    st.title('Titanic App')
    st.text('In this project, we will work on Titanic Data')

with datasets:
    st.header('Titanic Sank, how!')
    st.text('We will work with titanic datasets')
    # import data
    df = sns.load_dataset('titanic')
    df = df.dropna()
    st.write(df.head(10))
    st.subheader('How many people were there according to gender?')
    st.bar_chart(df['sex'].value_counts())

    # other plots
    st.subheader('Difference according to class')
    st.bar_chart(df['class'].value_counts())
    st.subheader('Distribution of ages')
    st.bar_chart(df['age'].sample(10))

with features:
    st.header('These are our app features')
    st.text('We will add multiple features, let\'s see')
    st.markdown("1. **Feature 1:** Description of feature 1 goes here.")
    st.markdown("2. **Feature 2:** Description of feature 2 goes here.")

with model_training:
    st.header('What happened to the Titanic? Model Training')
    st.text('Here we will adjust our model parameters.')

    # making columns
    input, display = st.columns(2)

    # first column containing selection points
    max_depth = input.slider('Select max depth for RF:', min_value=10, max_value=100, value=20, step=5)
    
    # n_estimators
    n_estimators = input.selectbox('Select the number of trees in RF:', options=[50, 100, 200, 300, 'No limit'])

    # adding list of features
    input_features = input.selectbox('Select a feature to use for prediction:', df.columns)

    # machine learning model
    if n_estimators == 'No limit':
        model = RandomForestRegressor(max_depth=max_depth)
    else:
        model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)

    # define X and y
    X = df[[input_features]]
    y = df['fare']

    # fit the model
    model.fit(X, y)

    # predict with the model
    pred = model.predict(X)

    # display the results
    display.subheader('Mean absolute error of the model is:')
    display.write(mean_absolute_error(y, pred))
    display.subheader('Mean squared error of the model is:')
    display.write(mean_squared_error(y, pred))
    display.subheader('R-squared score of the model is:')
    display.write(r2_score(y, pred))