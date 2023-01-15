from BoW_preprocessing import preprocessor
import pandas as pd
import joblib


def save_predictions(y_pred, final_path, initial_path="../Data/test.csv"):
    submission = pd.read_csv(initial_path, usecols=['id'])
    submission['target'] = y_pred
    submission.to_csv(final_path, index=False)
    return print("success !")


if __name__ == '__main__':

    filename = '../Models/Ridge_BoW_model.sav'
    model = joblib.load(filename)

    X_test, _ = preprocessor(path = "../Data/test.csv")

    y_pred = model.predict(X_test)
    save_predictions(y_pred, final_path="../Data/submission_1.csv")