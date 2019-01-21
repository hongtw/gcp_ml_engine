import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SERVICE_ACCOUNT = os.path.join(PROJECT_DIR, 'service_account.json')

GCP_PROJECT_ID = "hip-runner-229302"
# BUCKET_NAME = '{}-ff'.format(GCP_PROJECT_ID)
BUCKET_NAME = 'ff-predictor'
TRAIN_FILE = 'gs://{0}/data/train.dense'.format(BUCKET_NAME)
JOB_DIR = 'gs://{0}/job'.format(BUCKET_NAME)
TRAIN_FILENAME = 'train.dense'
# REGION = 'us-central1'
REGION = 'asia-east1'


## Training Machine Type
'''
* standard	
A basic machine configuration suitable for training simple models with small to moderate datasets.

Compute Engine machine name: n1-standard-4

* large_model	
A machine with a lot of memory, specially suited for parameter servers when your model is large (having many hidden layers or layers with very large numbers of nodes).

Compute Engine machine name: n1-highmem-8

* complex_model_s	
A machine suitable for the master and workers of the cluster when your model requires more computation than the standard machine can handle satisfactorily.

Compute Engine machine name: n1-highcpu-8

* complex_model_m	
A machine with roughly twice the number of cores and roughly double the memory of complex_model_s.

Compute Engine machine name: n1-highcpu-16

* complex_model_l	
A machine with roughly twice the number of cores and roughly double the memory of complex_model_m.

Compute Engine machine name: n1-highcpu-32
'''