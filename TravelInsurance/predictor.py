# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import io
import json
import os
import pickle
import signal
import sys
import traceback

import flask
import pandas as pd

# Define the prefix path where the model will be stored
prefix = "/opt/ml/"
model_path = os.path.join(prefix, "model")

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

class TravelInsurance(object):
    """A class to handle the travel insurance prediction model."""
    model = None  # Class variable to store the loaded model
    @classmethod
    def get_model(cls):
        """Load and return the trained model.
        
        Returns:
            The loaded model if successful, None otherwise.
        """
        if cls.model is None:
            with open(os.path.join(model_path, "Travel_Insurance_XGBoost_Model.sav"), "rb") as inp:
                cls.model = pickle.load(inp)
        return cls.model
    @classmethod
    def predict(cls, input):
        """Make predictions using the loaded model.
        
        Args:
            input: Input data for which predictions are to be made.
            
        Returns:
            Prediction probabilities for the input data.
        """
        clf = cls.get_model()
        return clf.predict_proba(input)

# Create a Flask application
app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    """Health check endpoint to verify the model is loaded.
    
    Returns:
        Response: Empty response with status code indicating health (200) or failure (404).
    """
    health = TravelInsurance.get_model() is not None # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response="\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def transformation():
    """Endpoint to make predictions on incoming data.
    
    Returns:
        Response: CSV formatted predictions or error message if input format is invalid.
    """
    data = None

    
    if flask.request.content_type != "text/csv":
        return flask.Response(
            response="This predictor only supports CSV data", status=415, mimetype="text/plain"
        )

    # Convert from CSV to pandas
    data = flask.request.data.decode("utf-8")
    s = io.StringIO(data)
    data = pd.read_csv(s)
    print(f"Invoked with {data.shape[0]} records")

    # Do the prediction
    data['propensity_output'] = TravelInsurance.predict(data)[:,1]

    # Convert from numpy back to CSV
    out = io.StringIO()
    data.to_csv(out, header=True, index=False, sep='\t', encoding='utf-8')
    result = out.getvalue()

    return flask.Response(response=result, status=200, mimetype="text/csv")