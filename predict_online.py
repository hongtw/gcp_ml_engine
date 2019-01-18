import googleapiclient.discovery
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from sklearn.preprocessing import LabelEncoder

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]= "/home/johnlee/ML_engine/rf/service_account.json"


class Predictor(object):
    def __init__(self, project, model, version):
        self.service = googleapiclient.discovery.build('ml', 'v1')
        self.name = 'projects/{}/models/{}'.format(project, model)
        if version is not None:
            self.name += '/versions/{}'.format(version)

        self.classes_ = [-12, -11, -10, 0, 10, 11, 12, 13, 14, 15]
        self._le = LabelEncoder().fit(self.classes_)

    def predict(self, instances):
        response = self.service.projects().predict(
            name=self.name,
            body={'instances': instances}
        ).execute()

        if 'error' in response:
            raise RuntimeError(response['error'])
        
        preds = response['predictions']
        return [self._decode(pred) for pred in preds]

    def _decode(self, pred):
        idx = np.argmax(pred) 
        return self._le.inverse_transform(idx)

def predict_json(project, model, instances, version=None):
    """Send json data to a deployed model for prediction.
    Args:
        project (str): project where the Cloud ML Engine Model is deployed.
        model (str): model name.
        instances ([[float]]): List of input instances, where each input
           instance is a list of floats.
        version: str, version of the model to target.
    Returns:
        Mapping[str: any]: dictionary of prediction results defined by the
            model.
    """
    # Create the ML Engine service object.
    # To authenticate set the environment variable
    # GOOGLE_APPLICATION_CREDENTIALS=<path_to_service_account_file>
    service = googleapiclient.discovery.build('ml', 'v1')
    name = 'projects/{}/models/{}'.format(project, model)

    if version is not None:
        name += '/versions/{}'.format(version)

    response = service.projects().predict(
        name=name,
        body={'instances': instances}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    return response['predictions']


def explicit():
    from google.cloud import storage

    # Explicitly use service account credentials by specifying the private key
    # file.
    storage_client = storage.Client.from_service_account_json(
        'service_account.json')

    # Make an authenticated API request
    buckets = list(storage_client.list_buckets())
    print(buckets)

# explicit()

project = 'ancient-snow-224803'
model = 'xgb'
version = 'xgb_ff'
instances = [np.arange(51).tolist()]




res = predict_json(project, model, instances, version)
print(res)

predictor = Predictor(project, model, version)

# with ThreadPoolExecutor(max_workers=5) as executor:
#     future_to_query = {executor.submit(predictor.predict, [instances.tolist()]): instances.tolist()
#         for instances in np.random.randint(0,100,(10,51))}

#     for future in concurrent.futures.as_completed(future_to_query):
#         query = future_to_query[future]
#         try:
#             data = future.result()
#         except Exception as exc:
#             print('%r generated an exception: %s' % (query, exc))
#         else:
#             print('%r page length is %d' % (query, len(data)))


import time
for instances in np.random.randint(0,100,(10,51)):
    start = time.time()
    print predictor.predict([instances.tolist()]), 'Spent:{}'.format(time.time() - start)

