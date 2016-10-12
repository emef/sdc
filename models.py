"""
Creating tensorflow models.
"""
import logging, os, tempfile

from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.models import Sequential
from keras.models import load_model as keras_load_model
from keras.optimizers import SGD
from keras.regularizers import l2
import numpy

from util import download_file, parse_s3_uri, upload_file

logger = logging.getLogger(__name__)


class SampleModel(object):
    TYPE = 'sample'

    def __init__(self, model_config):
        self.model_config = model_config
        self.tf_model = download_model(model_config['model_path'])

    @classmethod
    def create(cls, model_path):
        """
        Create and upload an untrained sample model.

        @param model_path - path to upload tf model to in s3.
        @return - model_config dict compatible with SampleModel.
        """
        # fix random seed for reproducibility
        seed = 7
        numpy.random.seed(seed)

        # Create the model
        model = Sequential()
        model.add(Convolution2D(
            nb_filter=32,
            nb_row=3,
            nb_col=3,
            input_shape=(3, 160, 160),
            init= "glorot_uniform",
            activation='relu',
            border_mode='same',
            W_regularizer=l2(0.01),
            bias=True,
            subsample=(1,1)))

        model.add(Dropout(0.5))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Flatten())
        model.add(Dense(
            output_dim=1231,
            init='glorot_uniform',
            activation='relu',
            bias=True,
            W_regularizer=l2(0.1)))
        model.add(Dropout(0.5))
        model.add(Dense(
            output_dim=1,
            init='glorot_uniform',
            activation='relu',
            bias=True,
            W_regularizer=l2(0.1)))

        # Upload the model to designated path
        upload_model(model, model_path)

        # Return model_config params compatible with constructor
        return {'model_path': model_path}


_MODEL_CLASS_BY_TYPE = {
    SampleModel.TYPE: SampleModel
}


def load_from_config(model_type, model_config):
    return _MODEL_CLASS_BY_TYPE[model_type](model_config)


def upload_model(model, s3_uri):
    """
    Upload a keras model to s3.

    @param model - keras model.
    @param s3_uri - formatted s3://bucket/key/path
    """
    _, tmp_path = tempfile.mkstemp()
    try:
        logger.info("Uploading model to " + s3_uri)
        model.save(tmp_path)
        upload_file(tmp_path, s3_uri)
    finally:
        os.remove(tmp_path)


def download_model(s3_uri, skip_cache=False, cache_dir='/tmp'):
    """
    Download and deserialize a keras model.

    @param s3_uri - formatted s3://bucket/key/path
    @param skip_cache - skip local file cache
    @param cache_dir - where to cache model files
    """
    bucket, key = parse_s3_uri(s3_uri)
    model_path = os.path.join(cache_dir, key)
    if not skip_cache and os.path.exists(model_path):
        logger.info("Using cached model at " + model_path)
    else:
        logger.info("Downloading model from " + s3_uri)
        download_file(bucket, key, model_path)

    return keras_load_model(model_path)
