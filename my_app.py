# This script runs a small Flask app to serve the model predictions through a GET endpoint that sends back JSON data.

from flask import Flask, request, jsonify
from metaflow import Flow
from metaflow import get_metadata, metadata
import uuid
import time
import requests
import streamlit as st

#### THIS IS GLOBAL, SO OBJECTS LIKE THE MODEL CAN BE RE-USED ACROSS REQUESTS ####

FLOW_NAME = 'SampleRegressionFlow'  # name of the target class that generated the model

def get_latest_successful_run(flow_name: str):
    "Gets the latest successful run using Metaflow API"
    for r in Flow(flow_name).runs():
        if r.successful:
            return r

# get artifacts from the latest run, using Metaflow Client API
latest_run = get_latest_successful_run(FLOW_NAME)
latest_model = latest_run.data.model
# We initialize the Flask object to run the flask app
app = Flask(__name__)


@app.route('/predict', methods=['GET'])
def main():
    """
    This function runs when the endpoint /predict is hit with a GET request.

    It looks for an input x value, runs the model, and returns the prediction.
    """
    start = time.time()
    _x = request.args.get('x')
    val = latest_model.predict([[float(_x)]])
    # debug value in the console
    print(_x, val)
    # returning the response to the client
    response = {
        'metadata': {
            'eventId': str(uuid.uuid4()),
            'serverTimestamp': round(time.time() * 1000),  # epoch time in ms
            'serverProcessingTime': round((time.time() - start) * 1000)  # in ms
        },
        'data': [val[0]]
    }

    return jsonify(response)


if __name__ == '__main__':
    # Run the Flask app to run the server
    app.run(debug=True)

# Front-end tier: Streamlit app
st.title('Model Prediction App')

# Accept numerical input from the user
user_input = st.number_input('Enter a number')

# Perform a GET request to the Flask app with the user input
if st.button('Get Prediction'):
    response = requests.get(f'http://127.0.0.1:5000/predict?x={user_input}')
    prediction = response.json()['data'][0]

    # Display the prediction
    st.write(f'The prediction is: {prediction}')