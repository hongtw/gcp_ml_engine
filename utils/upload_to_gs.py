import os
import sys
import subprocess
from google.cloud import storage
from argparse import ArgumentParser

def upload_blob(credential, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    if credential:
        storage_client = storage.Client.from_service_account_json(credential)
    else:
        storage_client = storage.Client()
    bucket_name, blob_name = destination_blob_name[5:].split('/', 1)
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    a = blob.upload_from_filename(source_file_name)

    print('File {} uploaded to {}.'.format(
        source_file_name,
        destination_blob_name))

def main(args):
    credential = args["input_cred"]
    src = args['src']
    dst = args['dst']

    upload_blob(credential, src, dst)
    # subprocess.check_call(['gsutil', 'cp', src, dst],
        # stderr=sys.stdout)

if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        '-i', '--input-cred',
        help='Credential file',
        default='./service_account.json'
    )
    ap.add_argument(
        '--src',
        help='Source which will be uploaded to gs',
        required=True
    )
    ap.add_argument(
        '--dst',
        help='Destination where on gs (ex: gs://test_bucket/123.file)',
        required=True
    )
    args = vars(ap.parse_args())
    # print(args)
    main(args)
