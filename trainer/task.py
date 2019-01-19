import argparse
import json
import os
import sys
import subprocess
import tensorflow as tf
import numpy as np
import trainer.model as model
from trainer.util import load_data
import time

def train_and_evaluate(args):
    # data_loc = args.train_files
    data_loc = 'gs://ancient-snow-224803-ff/data/train.dense'
    data_filename = "train.dense"
    print('data_loc', data_loc, 'data_filename', data_filename)
    # gsutil outputs everything to stderr so we need to divert it to stdout.
    subprocess.check_call(['gsutil', 'cp', data_loc, data_filename], stderr=sys.stdout)
    
    config = {
        "params":dict(
            n_estimators=50,
            learning_rate=0.1
        )
    }
    
    clf = model.build_estimator(config)

    x, y = load_data(data_filename)

    clf.fit(x, y)
    clf.save_model(args.model_dir) 
    print("Save model to {0}".format(args.model_dir))
    clf.save_model('model.bst') 
    print("Save model to {0}".format('model.bst'))
    subprocess.check_call(
        ['gsutil', 'cp', 'model.bst', 'gs://ancient-snow-224803-ff/job/{}/model.bst'.format(time.strftime("%Y%m%d"))],
        stderr=sys.stdout)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
        '--train-files',
        help='GCS file or local paths to training data',
        nargs='+',
        default='gs://ancient-snow-224803-ff/data/train.dense')
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


    # Run the training job
    train_and_evaluate(args)
