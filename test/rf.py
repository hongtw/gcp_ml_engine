import datetime
import os
import time
import numpy as np
import subprocess
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
# import joblib
import pickle

model_filename = 'model.pkl'
model_filename = 'model.joblib'
# BUCKET_NAME = 'ancient-snow-224803-ff'
# BUCKET_NAME = 'py3_sklearn_pkl'
BUCKET_NAME = 'testinghaha'

print(sys.argv)
traindense = sys.argv[1]
sk_version = sys.argv[2]

MAX = None
data = np.loadtxt(traindense)
X, y = data[:MAX, :-1], data[:MAX, -1]

print("Total data: {0}".format(X.shape[0]))
params = {
        'n_estimators': np.array(range(150, 160, 3)),
}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print("class: {0}".format(np.unique(y_train)))
print("train data shape: %r, train target shape: %r" % (X_train.shape, y_train.shape)) # train data shape: (1437, 64), train target shape: (1437,)
print("test data shape: %r, test target shape: %r" % (X_test.shape, y_test.shape)) # test data shape: (360, 64), test target shape: (360,)

start = time.time()
#clf = RandomForestClassifier(n_estimators=best_params['n_estimators']).fit(X, y)
estimators = 43
print("estimators: {0}".format(estimators))
clf = RandomForestClassifier(n_estimators=estimators, n_jobs=-1).fit(X, y)
end = time.time()
print('Training time:{0}'.format(end-start))
print(clf.score(X_train, y_train), np.shape(X_train))
print(clf.score(X_test, y_test), np.shape(X_test))
print(clf.score(X, y), np.shape(X))

model_dir = "model"
if not(os.path.exists(model_dir) and os.path.isdir(model_dir)):
    os.makedirs(model_dir)

if model_filename == 'model.pkl':
    with open(model_filename, 'wb') as f:
        pickle.dump(clf, f)
elif model_filename == 'model.joblib':
    joblib.dump(clf, model_filename, compress=1)
    # joblib.dump(clf, model_filename)
else:
    print('check file name')

# Upload the saved model file to Cloud Storage
# gcs_model_path = os.path.join('gs://', BUCKET_NAME,
    # datetime.datetime.now().strftime('ff_%Y%m%d_%H%M%S'), model_filename)

gcs_model_path = os.path.join('gs://', BUCKET_NAME, sk_version, model_filename)

subprocess.check_call(['gsutil', 'cp', model_filename, gcs_model_path],
    stderr=sys.stdout)