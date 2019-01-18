import time
from setting import JOB_DIR

VERSION_NAME = 'ff_xgb_{0}'.format(time.strftime("%Y%m%d"))
GS_BUCKET = 'gs://ancient-snow-224803-ff'

payload = {
    "name": VERSION_NAME,
    "deploymentUri": JOB_DIR,
    "runtimeVersion": "1.12",
    "framework": "XGBOOST",
    "pythonVersion": "3.5"
}