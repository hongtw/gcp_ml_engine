import argparse
import joblib
# from sklearn.externals import joblib
import json
import os
import sys
import subprocess
import numpy as np
import trainer.model as model
from trainer.util import load_data
import time
import sklearn
import datetime
# from google.cloud import storage
import scipy
import pickle

def gs_copy(src, dst):
    subprocess.check_call(
        ['gsutil', 'cp', src, dst],
        stderr=sys.stdout)
    print("Copy from {} to {}".format(src, dst))

def show_lib_version():
    print('Numpy Version:{0}'.format(np.__version__))
    print('Scikit-Learn Version:{0}'.format(sklearn.__version__))
    print('Joblib Version:{0}'.format(joblib.__version__))
    print('Scipy Version:{0}'.format(scipy.__version__))
    print('pickle Version:{0}'.format(pickle.__version__))

def get_version(module):
    return (module.__name__.capitalize(), module.__version__)

def gen_meta(now):
    versions = dict([
        get_version(module) for module in [np, sklearn, joblib, scipy]
    ])
    meta = {
        'Versions':versions,
        'Build_Time': now
    }
    with open('meta.json', 'w') as f:
        json.dump(meta, f)

def upload_to_gs(model_name, bucket_name):
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    gs_model_loc = os.path.join('gs://', bucket_name, 'job', now, model_name)
    gs_meta_loc = os.path.join('gs://', bucket_name, 'job', now, 'meta.json')

    subprocess.check_call(
        ['gsutil', 'cp', model_name, gs_model_loc],
        stderr=sys.stdout)
    print("Upload {} to {}".format(model_name, gs_model_loc))
    
    gen_meta(now)
    subprocess.check_call(
        ['gsutil', 'cp', 'meta.json', gs_meta_loc],
        stderr=sys.stdout)
    print("Upload {} to {}".format('meta.json', gs_meta_loc))

    # bucket = storage.Client().bucket(bucket_name)
    # blob = bucket.blob('job/{}/{}'.format(
    #     now,
    #     model_name))
    # blob.upload_from_filename(model_name)

    # blob = bucket.blob('job/{}/meta.json'.format(now))
    # blob.upload_from_filename('meta.json')

def train_and_evaluate(args):
    show_lib_version()
    train_filename = args['train_filename']
    bucket_name = args['bucket_name']
    data_loc = os.path.join('gs://', bucket_name, 'data', train_filename)
    # data_loc = 'gs://ancient-snow-224803-ff/data/train.dense'
    print('data_loc:{}, train_filename:{}'.format(data_loc, train_filename))

    # gsutil outputs everything to stderr so we need to divert it to stdout.
    subprocess.check_call(['gsutil', 'cp', data_loc, train_filename], stderr=sys.stdout)
    config = {
        "params":dict(
            n_estimators=50,
        )
    }    

    x, y = load_data(train_filename)
    clf = model.build_estimator(config)
    clf.fit(x, y)

    model_name = 'model.joblib'
    joblib.dump(clf, model_name, compress=3)

    print("Save model to {0}".format(model_name))
    upload_to_gs(model_name, bucket_name)
    

    try:
        print(subprocess.check_output(
            ['pip freeze'],
            stdout=sys.stdout,
            stderr=sys.stderr
        ))
    except:
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
        '--train-filename',
        help='GCS file or local paths to training data',
        # nargs='+',
        default='train.dense')
    parser.add_argument(
        '--bucket-name',
        help='Buckekt name on Cloud Storage'
    )
    parser.add_argument(
        '--eval-files',
        help='GCS file or local paths to evaluation data',
        nargs='+',
        default='gs://cloud-samples-data/ml-engine/census/data/adult.test.csv')
    parser.add_argument(
        '--model-dir',
        help='GCS location to write checkpoints and export models',
        default='/tmp/census-estimator')
    parser.add_argument(
        '--verbosity',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
        default='INFO')

    args, _ = parser.parse_known_args()
    args = vars(args)
    print(args)

    # Run the training job
    train_and_evaluate(args)