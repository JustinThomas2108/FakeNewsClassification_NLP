from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from BoW_preprocessing import preprocessor
from sklearn.metrics import f1_score


if __name__ == '__main__':
    path = "../Data/train.csv"
    X, y = preprocessor(path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.7, random_state=0)
    vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_features=int(len(X_train)/10))
    vectorizer.fit(X_train)
    X_train = vectorizer.transform(X_train).toarray()
    X_test = vectorizer.transform(X_test).toarray()
    classifier = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=f1_score)
    models, predictions = classifier.fit(X_train, X_test, y_train, y_test)
    print(models)

    #RidgeClassifier obtains the best results in terms of f1-score one the positive label class.