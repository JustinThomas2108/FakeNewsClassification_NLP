import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from BoW_preprocessing import preprocessor
import time
from tqdm import tqdm


def lazy_evaluation(models, X_train, y_train, X_test, y_test, sort_by = 'accuracy'):
    metrics_names = ["accuracy", "balanced_accuracy", "ROC AUC", "F1 score", "Time taken"]
    hyperparameters_names = [key for key in models[0][0].keys()]
    results = []
    for hyperparameters, model in tqdm(models):
        t1 = time.time()
        model.fit(X_train, y_train)
        time_taken = round((time.time() - t1)/60, 2)
        y_pred = model.predict(X_test)
        metrics = [accuracy_score(y_test, y_pred), balanced_accuracy_score(y_test, y_pred),
                   roc_auc_score(y_test, y_pred), f1_score(y_test, y_pred), time_taken]
        hyperparameters_values = [value for value in hyperparameters.values()]
        results.append(metrics + hyperparameters_values)
    df_results = pd.DataFrame(data = results, columns = metrics_names + hyperparameters_names)
    return df_results.sort_values(by = sort_by, ascending = False)


if __name__ == '__main__':

    path = "../Data/train.csv"
    X, y = preprocessor(path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.7, random_state=0)

    hyperparameters = {'n_features':list(np.geomspace(10 ** 2, 5 * 10 ** 3, 10)),
                       'alpha':list(np.geomspace(10**-3, 10**3, 30))}

    models = [
        [{'n_features':n_feat, 'alpha':a},
         make_pipeline(TfidfVectorizer(ngram_range=(1, 1)),
                       SelectKBest(chi2, k=int(n_feat)),
                       RidgeClassifier(random_state=0, alpha=a))]
        for n_feat in hyperparameters['n_features']
        for a in hyperparameters['alpha']
            ]

    print(lazy_evaluation(models, X_train, y_train, X_test, y_test, sort_by="F1 score").head(20))


