import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.model_selection import train_test_split
import pickle

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("labour.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("index.html", prediction_text = "labour Work Prediction {}".format(prediction))

if __name__ == "__main__":
    flask_app.run(debug=True)
