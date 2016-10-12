from eventlet import *
patcher.monkey_patch(all=True)

import logging, os, sys, tempfile, time
from boto.s3.bucket import Bucket
from boto.s3.connection import S3Connection
from boto.s3.key import Key

logger = logging.getLogger(__name__)


def aws_credentials():
    """
    Get aws (key, secret)
    """
    return (
        os.environ['AWS_ACCESS_KEY_ID'],
        os.environ['AWS_SECRET_ACCESS_KEY'],
    )


def parse_s3_uri(s3_uri):
    """
    Parse a s3 uri into (bucket, key).

    @param s3_uri - formatted s3://bucket/key/path
    @return - (bucket, key)
    """
    assert s3_uri.startswith('s3://')
    return s3_uri.split('s3://')[-1].split('/', 1)


def upload_file(local_file, s3_uri):
    """
    Upload a local file to s3.

    @param local_file - local path to file to upload
    @param s3_uri - formatted s3://bucket/key/path
    """
    s3_bucket, s3_key = parse_s3_uri(s3_uri)
    conn = S3Connection(*aws_credentials())
    bucket = Bucket(connection=conn, name=s3_bucket)
    k = Key(bucket)
    k.key = s3_key
    k.set_contents_from_filename(local_file)


def download_file(s3_bucket, s3_key, out_path):
    """
    Download a file from s3.

    @param s3_bucket - s3 bucket name
    @param s3_key - s3 key's path
    @param out_path - local download location
    """
    # Its important to download the key from a new connection
    conn = S3Connection(*aws_credentials())
    bucket = Bucket(connection=conn, name=s3_bucket)
    key = bucket.get_key(s3_key)

    try:
        logger.info("Downloading s3 key " + s3_key)
        res = key.get_contents_to_filename(out_path)
    except Exception as e:
        print e
        logger.info(key.name + ": FAILED")


def download_dir(s3_uri, local_path):
    """
    Copy all files in an s3 path to a local directory.

    @param s3_uri - formatted s3://bucket/key/path
    @param local_path - local path to copy to
    """
    s3_bucket, s3_prefix = parse_s3_uri(s3_uri)
    logger.info("Copying s3 directory %s to %s" % (s3_uri, local_path))

    conn = S3Connection(*aws_credentials())
    bucket = Bucket(connection=conn, name=s3_bucket)
    bucket_list = bucket.list(prefix=s3_prefix)
    pool = GreenPool(size=20)

    for key in bucket.list():
        key_path = key.key.split(s3_prefix)[-1][1:]
        out_path = os.path.join(local_path, key_path)

        # make sure the path exists
        try: os.makedirs(os.path.dirname(out_path))
        except: pass

        pool.spawn_n(download_file, s3_bucket, key.key, out_path)

    pool.waitall()

    logger.info("Done copying " + s3_prefix)
