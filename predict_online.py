import googleapiclient.discovery
import numpy as np
import os, signal
import traceback
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from sklearn.preprocessing import LabelEncoder
import warnings
import threading
import time
import sys
warnings.filterwarnings("ignore", category=DeprecationWarning) 

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
        # return [self._decode(pred) for pred in preds]
        return preds

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

def instance_generator():
    while True:
        yield np.random.randint(0,100,(51,)).tolist()

# explicit()
class Watcher:  
    """this class solves two problems with multithreaded 
    programs in Python, (1) a signal might be delivered 
    to any thread (which is just a malfeature) and (2) if 
    the thread that gets the signal is waiting, the signal 
    is ignored (which is a bug). 

    The watcher is a concurrent process (not thread) that 
    waits for a signal and the process that contains the 
    threads.  See Appendix A of The Little Book of Semaphores. 
    http://greenteapress.com/semaphores/ 

    I have only tested this on Linux.  I would expect it to 
    work on the Macintosh and not work on Windows. 
    """  

    def __init__(self):  
        """ Creates a child thread, which returns.  The parent 
            thread waits for a KeyboardInterrupt and then kills 
            the child thread. 
        """  
        self.child = os.fork()  
        if self.child == 0:  
            return  
        else:  
            print('I am watching you')
            self.watch()  

    def watch(self):  
        start = time.time()
        try:  
            os.wait()  
        except KeyboardInterrupt:  
            # I put the capital B in KeyBoardInterrupt so I can  
            # tell when the Watcher gets the SIGINT  
            print 'KeyBoardInterrupt'  
            self.kill()  
        print('Total Spent: {}'.format(time.time() -start) )
        sys.exit()  

    def kill(self):  
        try:  
            os.kill(self.child, signal.SIGKILL)  
        except OSError: pass  

class request_thread(threading.Thread):
    def __init__(self, count, project, model, version):
        threading.Thread.__init__(self)
        self.predictor = Predictor(project, model, version)
        self.thread_name = '{}-Thread'.format(count)
        self.instance_generator = instance_generator()
        print("Initializing... {}".format(self.thread_name))

    def run(self):
        count = 0
        while True:
            try:
                start = time.time()
                res = self.predictor.predict([next(self.instance_generator)])
                count += 1
                print('Thread:{} and Return:{}, Spent:{:.4f}, Count:{}'.format(
                    self.thread_name, res, time.time() - start, count))
                global TOTAL
                TOTAL += 1
            except Exception as e:
                print(traceback.format_exc())


project = 'ancient-snow-224803'
model = 'sklearn_test'
version = 'py2_numpy_146_sklearn_0_20_0_compressed'
version = 'py2_numpy_141_sk0200_4core'
version = 'py2_numpy_146_sk0191_4core'
version = 'py2_numpy_146_sk0191_4core_10'
# version = 'aisa_4core'
instances = [np.arange(51).tolist()]




res = predict_json(project, model, instances, version)
print(res)

predictor = Predictor(project, model, version)
# predictors = [Predictor(project, model, version) for _ in range(5)]

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



# sequential
# import time
# for instances in np.random.randint(0,100,(20, 10, 51)):
#     start = time.time()
#     if instances.ndim == 1:
#         print predictor.predict([instances.tolist()]), 'Spent:{}'.format(time.time() - start)
#     else:
#         print predictor.predict(instances.tolist()), 'Spent:{}'.format(time.time() - start)



# thread
thread_num = 10
threads = [request_thread(idx, project, model, version) for idx in range(thread_num)]
Watcher() # accept ctrl+C
TOTAL = 0

a = [t.start() for t in threads]
print(a)
print("Total Count:{0}".format(TOTAL))