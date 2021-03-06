"""
Simple pipeline to train an Scikit-Learn model everyday and push the model to production.
"""

import datetime
import os

from airflow import models
import mlengine_operator

BASE_DIR = 'gs://ff-predictor/trainer'
TRAINER_BIN = os.path.join(BASE_DIR, 'trainer-0.1.tar.gz')
TRAINER_MODULE = 'trainer.task'
RUNTIME_VERSION = '1.9'
PROJECT_ID = 'hip-runner-229302'
MODEL_NAME = 'ff_test'

yesterday = datetime.datetime.combine(
    datetime.datetime.today() - datetime.timedelta(1),
    datetime.datetime.min.time())

default_dag_args = {
    'owner': 'mlengine-pipelines-prod',
    'depends_on_past': False,
    # Setting start date as yesterday starts the DAG immediately when it is
    # detected in the Cloud Storage bucket.
    'start_date': yesterday,
    # To email on failure or retry set 'email' arg to your email and enable
    # emailing here.
    'email_on_failure': False,
    'email_on_retry': False,
    # Retry once, after waiting for 5 minutes for any failures.
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5)
}

with models.DAG(
    'ff_daily_training',
    # Continue to run DAG once per day
    schedule_interval=datetime.timedelta(days=1),
    default_args=default_dag_args) as dag:

    # Using custom ts, since {{ ts }} has incompatible characters to ML Engine.
    # Also adding uuid afterwards to allow retry.
    date_nospecial = '{{ execution_date.strftime("%Y%m%d") }}'
    uuid = '{{ macros.uuid.uuid4().hex[:8] }}'

    training_op = mlengine_operator.MLEngineTrainingOperator(
        task_id='training',
        project_id=PROJECT_ID,
        job_id='ff_daily_{}_{}_{}'.format(date_nospecial, 'training', uuid),
        package_uris=[os.path.join(TRAINER_BIN)],
        training_python_module=TRAINER_MODULE,
        training_args=[
            '--base-dir={}'.format(BASE_DIR),
            '--event-date={}'.format(date_nospecial),
            '--train_filename={}'.format('train.dense'),
            '--bucket-name={}'.format('ff-predictor')
        ],
        region='asia-east1',
        runtime_version=RUNTIME_VERSION)

    export_uri = os.path.join(BASE_DIR, 'models', date_nospecial)
    create_version_op = mlengine_operator.MLEngineVersionOperator(
        task_id='create_version',
        project_id=PROJECT_ID,
        model_name=MODEL_NAME,
        version={
            'name': 'version_{}_{}'.format(date_nospecial, uuid),
            'deploymentUri': export_uri,
            'runtimeVersion': RUNTIME_VERSION,
            'framework': 'SCIKIT_LEARN',
        },
        operation='create')

    (training_op >> create_version_op)