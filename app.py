import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)
# Create model that will open the pkl file (read binary)
#   A Python data object can be "pickled" as itself, 
#       which then can be directly loaded ("unpickled") as such at a later point; 
#           the process is also known as "object serialization".
model = pickle.load(open("model.pkl", "rb"))

# Main Route
@flask_app.route("/")
def Home():
    return render_template("index.html")

# Displays Predicted Text
@flask_app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("index.html", prediction_text = "The flower species is {}".format(prediction))

# Run the app
if __name__ == "__main__":
    flask_app.run(debug=True)