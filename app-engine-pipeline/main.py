import json
import logging
import os

import pandas as pd
from flask import Flask, request
from clients.vertex_ai import VertexAIClient

app = Flask(__name__)

project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
endpoint_name = os.getenv("VERTEX_AI_MODEL_ENDPOINT")
location = os.getenv("LOCATION")

vertex_ai_client = VertexAIClient(project_id, endpoint_name, location)

def process_test_data(raw_df):
    """
    TODO: Copy your feature engineering code from task 1 here

    :param raw_df: the DataFrame of the raw test data
    :return: a DataFrame with the predictors created
    """
    raise NotImplementedError("To be implemented")


@app.route('/')
def index():
    return "Hello"


@app.route('/predict', methods=['POST'])
def predict():
    raw_data_df = pd.read_json(request.data.decode('utf-8'),
                               convert_dates=["pickup_datetime"])
    predictors_df = process_test_data(raw_data_df)

    # return the predictions in the response in json format
    return json.dumps(vertex_ai_client.predict(predictors_df.values.tolist()))

@app.route('/farePrediction', methods=['POST'])
def fare_prediction():
    pass


@app.route('/speechToText', methods=['POST'])
def speech_to_text():
    pass


@app.route('/textToSpeech', methods=['GET'])
def text_to_speech():
    pass


@app.route('/farePredictionVision', methods=['POST'])
def fare_prediction_vision():
    pass


@app.route('/namedEntities', methods=['GET'])
def named_entities():
    pass


@app.route('/directions', methods=['GET'])
def directions():
    pass


@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    app.run()
