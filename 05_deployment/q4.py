import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

import pickle

from flask import Flask
from flask import request
from flask import jsonify

app = Flask('churn')

dv_path = 'dv.bin'
model_path = 'model1.bin'

with open(dv_path, 'rb') as f_in:
    dv = pickle.load(f_in)

with open(model_path, 'rb') as f_in:
    model = pickle.load(f_in)

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0,1]
    churn = y_pred >= 0.5
    result = {
        "churn_probability": float(y_pred),
        "churn": bool(churn)
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)