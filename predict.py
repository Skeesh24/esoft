import json

import joblib
import numpy as np

model = joblib.load("trained_model.pkl")

with open("input.json", "r") as f:
    input_data = json.load(f)

input_data = np.array(input_data)

predictions = model.predict([input_data])

print(predictions)
