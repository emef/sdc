"""
Creating tensorflow models.
"""
import logging, os

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


class BaseModel(object):
    """
    Abstraction over model which allows fit/evaluate/predict
    given input data.
    """
    def fit(self, dataset, training_args):
        """
        Fit the model with given dataset/args.

        @param dataset - a Dataset
        @param training_args - dict of training args
        """
        raise NotImplemented

    def evaluate(self, dataset):
        """
        Evaluate the model given dataset.

        @param dataset - a Dataset
        @return - evaluation results (type specific to each model)
        """
        raise NotImplemented

    def predict_on_batch(self, batch):
        """
        Predict given batch of input samples.

        @param batch - batch of input samples
        @return - output predictions
        """
        raise NotImplemented


class SampleModel(BaseModel):
    TYPE = 'sample'

    def __init__(self, model_config):
        self.model_config = model_config
        self.model = download_model(model_config['model_uri'])

    def fit(self, dataset, training_args):
        # TODO: clean up compilation/training config into task config
        batch_size = training_args.get('batch_size', 100)
        validation_size = training_args.get('validation_size', 500)
        epoch_size = training_args.get('epoch_size', 1000)
        epochs = training_args.get('epochs', 5)

        self.model.summary()

        history = self.model.fit_generator(
            dataset.training_generator(batch_size),
            validation_data=dataset.validation_generator(batch_size),
            samples_per_epoch=epoch_size,
            nb_val_samples=validation_size,
            nb_epoch=epochs,
            verbose=1,
            callbacks=[],
            pickle_safe=True,
            nb_worker=2)

    def evaluate(self, dataset):
        testing_size = dataset.get_testing_size()
        evaluation = self.model.evaluate_generator(
            dataset.testing_generator(testing_size),
            testing_size,
            nb_worker=2,
            pickle_safe=True)

        return evaluation


    def predict_on_batch(self, batch):
        return self.model.predict_on_batch(batch)

    @classmethod
    def create(cls,
               model_uri,
               loss='mean_squared_error',
               learning_rate=0.001,
               momentum=0.9,
               metrics=None):
        """
        Create and upload an untrained sample model.

        @param model_uri - path to upload tf model to in s3.
        @return - model_config dict compatible with SampleModel.
        """
        metrics = metrics or ['mse']
        sgd = SGD(lr=learning_rate,
                  momentum=momentum,
                  nesterov=False)

        # fix random seed for reproducibility
        seed = 7
        numpy.random.seed(seed)

        # Create the model
        model = Sequential()
        model.add(Convolution2D(
            nb_filter=1,
            nb_row=5,
            nb_col=5,
            input_shape=(80, 80, 3),
            init= "glorot_uniform",
            activation='relu',
            border_mode='same',
            W_regularizer=l2(0.01),
            bias=True,
            subsample=(1,1)))

        model.add(Dropout(0.5))
        model.add(MaxPooling2D(pool_size=(5, 5)))
        model.add(Flatten())
        model.add(Dense(
            output_dim=1024,
            init='glorot_uniform',
            activation='relu',
            bias=True,
            W_regularizer=l2(0.1)))
        model.add(Dropout(0.5))
        model.add(Dense(
            output_dim=1,
            init='glorot_uniform',
            W_regularizer=l2(0.01)))

        model.compile(
            loss=loss,
            optimizer=sgd,
            metrics=metrics)

        # Upload the model to designated path
        upload_model(model, model_uri)

        # Return model_config params compatible with constructor
        return {'model_uri': model_uri}


MODEL_CLASS_BY_TYPE = {
    SampleModel.TYPE: SampleModel
}


def load_from_config(model_type, model_config):
    """
    Loads a model from config by looking up it's model type and
    calling the right constructor.

    @param model_type - model type ID, should be in MODEL_CLASS_BY_TYPE.
    @param model_config - appropriate model_config dict for model type
    @return - model object
    """
    return MODEL_CLASS_BY_TYPE[model_type](model_config)


def upload_model(model, s3_uri, cache_dir='/tmp'):
    """
    Upload a keras model to s3.

    @param model - keras model
    @param s3_uri - formatted s3://bucket/key/path
    @param cache_dir - where to store cached models
    """
    _, key = parse_s3_uri(s3_uri)
    model_path = os.path.join(cache_dir, key)
    logger.info("Uploading model to %s", s3_uri)
    model.save(model_path)
    upload_file(model_path, s3_uri)


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
