#import pandas as pd
import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

import pickle

dv_path = 'dv.bin'
model_path = 'model1.bin'

with open(dv_path, 'rb') as f_in:
    dv = pickle.load(f_in)

with open(model_path, 'rb') as f_in:
    model = pickle.load(f_in)

customer = {"contract": "two_year", "tenure": 12, "monthlycharges": 19.7}
# We pass an index to the dataframe constructor
#df = pd.DataFrame(customer, index=[0])

X = dv.transform([customer])

y_pred = model.predict_proba(X)[0,1]
print(y_pred)