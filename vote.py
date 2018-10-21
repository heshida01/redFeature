import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
import sklearn.metrics.accuracy_score
from sklearn.model_selection import cross_val_predict

def read_data(file):
    dataset=pd.read_csv(file).dropna(axis=1)
    feature_name = dataset.columns.values.tolist()
    dataset=np.array(dataset)

    return dataset
def classifier(data):
    X = data[:,1:]
    y = data[:,0]
    clf = RandomForestClassifier(random_state=1, n_estimators=100)
    clf.fit(X, y)
    cv_results = cross_validate(clf, X, y, return_train_score=False, cv=10, n_jobs=-1)

    #np.set_printoptions(threshold=np.inf)
    print(clf.predict_proba(X))
    ypred = cross_val_predict(clf, X, y, n_jobs=-1, cv=10)
    print(ypred)
    acc = sklearn.metrics.accuracy_score(y,ypred)

if __name__ == '__main__':
    data = read_data('AAC.csv')

    classifier(data)