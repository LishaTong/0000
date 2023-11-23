import streamlit as st
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score

# Load the model
model = tf.keras.models.load_model('model_1000')

# Load the predictions
@st.cache
def load_predictions():
    return pd.read_csv('predictions.csv')

predictions = load_predictions()

# Dropdown menu
option = st.selectbox(
    'Which group analysis do you want to see?',
    ('Sex', 'Pclass'))

# Calculate accuracy
def calculate_accuracy(group):
    if group == 'Sex':
        categories = ['male', 'female']
    else: # 'Pclass'
        categories = predictions['Pclass'].unique()

    accuracies = []
    for category in categories:
        subset = predictions[predictions[group] == category]
        accuracy = accuracy_score(subset['true_values'], subset['predictions'])
        accuracies.append((category, accuracy))

    # Also calculate total accuracy
    total_accuracy = accuracy_score(predictions['true_values'], predictions['predictions'])
    accuracies.append(('total', total_accuracy))

    return pd.DataFrame(accuracies, columns=[group, 'accuracy'])

# Display accuracy
st.table(calculate_accuracy(option))