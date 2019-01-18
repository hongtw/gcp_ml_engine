from googleapiclient import discovery
from googleapiclient import errors
import logging
import os
import time
from setting import GCP_PROJECT_ID, SERVICE_ACCOUNT, BUCKET_NAME, TRAIN_FILE, JOB_DIR

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]= SERVICE_ACCOUNT

training_inputs = {'scaleTier': 'CUSTOM',
    'masterType': 'complex_model_m',
    'packageUris': ['gs://{0}/trainer/trainer-0.1.tar.gz'.format(BUCKET_NAME)],
    'pythonModule': 'trainer.task',
    'args': ['--train-files', TRAIN_FILE, '--arg2', 'value2'],
    'region': 'us-central1',
    'jobDir': JOB_DIR,
    'runtimeVersion': '1.12',
    'pythonVersion': '3.5'}


job_spec = {
    'jobId': "HH_is_handsome_{}".format(time.strftime("%Y%m%d_%H%M%S")), 
    'trainingInput': training_inputs}

project_id = 'projects/{}'.format(GCP_PROJECT_ID)

# Get a Python representation of the Cloud ML Engine services:
cloudml = discovery.build('ml', 'v1')

# Form your request and send it:
request = cloudml.projects().jobs().create(body=job_spec, parent=project_id)
# response = request.execute()

try:
    response = request.execute()
    # You can put your code for handling success (if any) here.

except errors.HttpError as err:
    # Do whatever error response is appropriate for your application.
    # For this example, just send some text to the logs.
    # You need to import logging for this to work.
    print('There was an error creating the training job.'
                    ' Check the details:')
    print(err._get_reason())