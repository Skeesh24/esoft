import json

import joblib
import numpy as np

# Загрузка модели из файла
model = joblib.load("trained_model.pkl")

# Загрузка входных данных для предсказаний из JSON файла
with open("input.json", "r") as f:
    input_data = json.load(f)

# Преобразование данных в формат, который может быть использован моделью
input_data = np.array(input_data)

# Выполнение предсказания
predictions = model.predict([input_data])

# Печать предсказанных значений
print(predictions)
