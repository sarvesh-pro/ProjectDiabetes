import pickle

import numpy as np
from flask import Flask,request,jsonify,render_template

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict", methods = ["POST","GET"])
def predict():
    features = [float(x) for x in request.form.values()]
    final = [np.array(features)]
    prediction = model.predict(final)
    if prediction > 0.5:
        return render_template("index.html",pred="you have diabetes")
    else:
        return render_template("index.html",pred="you dont have diabetes")

if __name__ == "__main__":
    app.run(debug=True)

