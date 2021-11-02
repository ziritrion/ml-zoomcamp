from sklearn.feature_extraction import DictVectorizer

import xgboost as xgb

import pickle

from flask import Flask
from flask import request
from flask import jsonify

app = Flask('match')

dv_path = 'dv.bin'
model_path = 'model.bin'

with open(dv_path, 'rb') as f_in:
    dv = pickle.load(f_in)

with open(model_path, 'rb') as f_in:
    model = pickle.load(f_in)

@app.route('/predict', methods=['POST'])
def predict():
    user = request.get_json()
    X = dv.transform([user])
    dx = xgb.DMatrix(X, feature_names=dv.get_feature_names())
    y_pred = model.predict(dx)
    match = y_pred >= 0.5
    result = {
        "match_probability": float(y_pred),
        "match": bool(match)
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)