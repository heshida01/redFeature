import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
import sklearn.metrics
from sklearn.model_selection import cross_val_predict
import csv

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
    print(cv_results)
    #np.set_printoptions(threshold=np.inf)
    yproba=clf.predict_proba(X)
    ypred = cross_val_predict(clf, X, y, n_jobs=-1, cv=10)
    #print(ypred)
    acc = sklearn.metrics.accuracy_score(y,ypred)

    Result_csv('toy.csv',y,ypred,yproba)

    svmclf = SVC()

def Result_csv(filepath,yactual,ypred,yproba):
    with open(filepath,'a') as f:
        data = {}
        writer = csv.DictWriter(f,fieldnames=['inst#','actual','predicted','error','prediction'])
        writer.writeheader()
        for i in range(1,len(yactual)+1):
            if ypred[i-1]!=yactual[i-1]:
                data['error'] = '+'
            else:
                data['error'] = ' '
            data['inst#'] = str(i)
            data['actual'] = '1'+':'+str(int(yactual[i-1]))
            data['predicted'] = '2'+':'+str(int(ypred[i - 1]))
            data['prediction'] = max(yproba[i-1])
            writer.writerow(data)

if __name__ == '__main__':
    data = read_data('AAC.csv')

    classifier(data)