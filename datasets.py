"""
Loading/saving datasets.
"""
import logging, os

from util import download_dir, parse_s3_uri

def load_dataset(s3_uri, cache_dir='/tmp'):
    _, s3_dir = parse_s3_uri(s3_uri)
    dataset_path = os.path.join(cache_dir, s3_dir)

    # Ensure we have the dataset downloaded locally
    if os.path.exists(dataset_path):
        logging.info('Dataset %s exists, using cache' % s3_dir)
    else:
        logging.info('Downloading dataset %s' % s3_dir)
        download_dir(s3_uri, cache_dir)

    # Load the dataset from the local directory
