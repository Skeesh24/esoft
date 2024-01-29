import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer


def train_model(input_file, model_output_file):
    data = pd.read_csv(input_file, sep=";", low_memory=False)

    # target column
    X = data.drop(columns=["price"], axis=1)
    y = data["price"]

    # model init
    pipe = Pipeline(
        [
            ("scaler", QuantileTransformer()),
            ("model", KNeighborsClassifier()),
        ]
    )

    model = GridSearchCV(
        estimator=pipe,
        param_grid={"model__n_neighbors": [1, 2, 3, 4, 5, 6, 7]},
        scoring="accuracy",
        cv=3,
    )

    # train
    model.fit(X, y)

    # accurancy rate
    accuracy = model.score(X, y)
    print(f"Model Accuracy: {accuracy}")

    # save
    joblib.dump(model, model_output_file)


if __name__ == "__main__":
    input_file = "preprocessed_data.csv"
    model_output_file = "trained_model.pkl"
    train_model(input_file, model_output_file)
