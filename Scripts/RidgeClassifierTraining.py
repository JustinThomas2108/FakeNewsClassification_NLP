from BoW_preprocessing import preprocessor
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
import joblib


if __name__ == '__main__':

    X, y = preprocessor(path = "../Data/train.csv")
    X_test, _ = preprocessor(path = "../Data/test.csv")

    final_model = make_pipeline(TfidfVectorizer(ngram_range=(1, 1)),
                                SelectKBest(chi2, k=2100),
                                RidgeClassifier(random_state=0, alpha=0.8))
    final_model.fit(X, y)

    # save the model
    filename = '../Models/Ridge_BoW_model.sav'
    joblib.dump(final_model, filename)

