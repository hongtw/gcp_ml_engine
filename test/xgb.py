from xgboost import XGBClassifier
import xgboost as xgb
import sys, os, time
import numpy as np
# from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ParameterGrid

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

def inputData(trainFile):
    f = open(sys.argv[1])
    data = np.loadtxt(f)
    X, Y = data[:, :-1], data[:, -1]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
    return  X, Y, X_train, X_test, Y_train, Y_test

def scikitAPI(trainFile):
    X, Y, X_train, X_test, Y_train, Y_test = inputData(trainFile)

    '''
    Y = transLabel(Y)[1]
    Y_train = transLabel(Y_train)[1]
    Y_test = transLabel(Y_test)[1]
    '''

    start = time.time()
    print("class: {0}".format(np.unique(Y_train)))
    print("train data shape: %r, train target shape: %r" % (X_train.shape, Y_train.shape))
    print("test data shape: %r, test target shape: %r" % (X_test.shape, Y_test.shape)) 

    #params = {"n_estimators":300, "num_leaves":128, "learning_rate":0.1}
    params = {"n_estimators":50, "learning_rate":0.1}
    
    print("{:-<50}".format(""))
    print("params", params)
    #clf_test = LGBMClassifier(n_estimators=200)

    # clf_test = XGBClassifier(**params)
    # clf_test.fit(X_train, Y_train)

    # print('Training time:{0}'.format(time.time()-start))


    # print("clf class: ",clf_test.classes_)
    # #pred = clf_test.predict(X_test)
    # #print(accuracy_score(pred, y_test))
    # print("Traing Acc: ", clf_test.score(X_train, Y_train), np.shape(X_train))
    # print("Test Acc: ", clf_test.score(X_test, Y_test), np.shape(X_test))
    # print("Total Acc: :", clf_test.score(X, Y), np.shape(X))

    
    xgtrain = xgb.DMatrix(X, Y)
    clf = XGBClassifier(**params).fit(X, Y)
    model_path = 'xgb/model.joblib'
    # joblib.dump(clf, model_path, compress=1)
    #clf.save_model("lgbm_model.ml")

    model_path = 'xgb/model.bst'
    clf.save_model(model_path) 

    # clf.dump_model('xgb/dump.raw.txt', 'xgb/featmap.txt')
    # print("ACC: ", load_model(model_path).score(X_test, Y_test))
    print("model save to {}".format(model_path))
    print("model.ml size: {:.3f} KB".format(os.path.getsize(model_path)/1024))
    clf = load_model(model_path, Y)
    print(clf.predict([np.arange(51)]))

def load_model(model_path, y_test):
    clf = xgb.XGBClassifier()
    clf.load_model(model_path)
    le = LabelEncoder().fit(y_test)
    clf._le = le
    return clf

def gridSearch(train_file):
    X, Y, X_train, X_test, Y_train, Y_test = inputData(train_file)
    params = dict(
        learning_rate = [0.01, 0.001],
        n_estimators = list(range(100, 600, 100)),
        subsample = [0.8, 0.9, 1],
        colsample_bytree = np.arange(0.3, 0.9, 0.1),
        gamma = [0, 1, 5]
    )

    best_score = 0
    best_params = None
    for param in ParameterGrid(params):
        clf = XGBClassifier(n_job=-1, **param)
        scores = cross_val_score(clf, X, Y, cv=3)
        print('Params: {0}, Scores: {1}'.format(param, scores))
        avg_scores = np.mean(scores)
        if avg_scores > best_score:
            best_score = avg_scores
            best_params = param
            print("Best Score:{0}, Best Parmas:{1}".format(best_score, best_params))

if __name__ == "__main__":
    train_file = sys.argv[1]
    # scikitAPI(train_file)
    gridSearch(train_file)