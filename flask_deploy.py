import pickle
import numpy as np
from flask import Flask
from flask import request
from flask import jsonify

model = pickle.load(open("model.pkl","rb"))

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
	data = request.get_json(force=True)
	features = request.get_json()["feature_array"]
	feature_array = np.asarray([features])
	feature_array = feature_array.reshape((1, 28, 28, 1))
	predictions = model.predict(feature_array).tolist()
	print(len(predictions))
	return jsonify(predictions)

@app.route("/hello", methods=["GET"])
def hello():
	return "Hello World!"

app.run()