"""
Creating tensorflow models.
"""
from collections import deque
import logging, os, tempfile

from keras.engine.topology import Merge
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.models import Sequential
from keras.models import load_model as keras_load_model
from keras.optimizers import SGD
from keras.regularizers import l2
import numpy as np

from util import download_file, parse_s3_uri, upload_file

logger = logging.getLogger(__name__)


class BaseModel(object):
    """
    Abstraction over model which allows fit/evaluate/predict
    given input data.
    """
    def save(self, task_id):
        """
        Save model and return the new model config.

        @param task_id - unique task id
        @return - model config
        """
        raise NotImplemented

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

    def as_encoder(self):
        """
        Transform this model into an encoder model.

        @return - tensorflow model
        """
        raise NotImplemented

    def output_dim(self):
        """
        @return - output dimension
        """
        raise NotImplemented


class SimpleModel(BaseModel):
    TYPE = 'simple'

    def __init__(self, model_config):
        self.model = load_model_from_uri(model_config['model_uri'])

    def fit(self, dataset, training_args):
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
        n_testing = dataset.get_testing_size()
        evaluation = self.model.evaluate_generator(
            dataset.testing_generator(16),
            (n_testing / 16) * 16,
            nb_worker=2,
            pickle_safe=True)

        return evaluation

    def predict_on_batch(self, batch):
        return self.model.predict_on_batch(batch)

    def as_encoder(self):
        # remove the last output layers to retain the feature maps
        deep_copy = deep_copy_model(self.model)
        for _ in xrange(4):
            deep_copy.pop()
        return deep_copy_model(deep_copy)

    def output_dim(self):
        return get_output_dim(self.model)

    def save(self, task_id):
        s3_uri = 's3://sdc-matt/simple/%s/model.h5' % task_id
        upload_model(self.model, s3_uri)

        return {
            'type': SimpleModel.TYPE,
            'model_uri': s3_uri,
        }

    @classmethod
    def create(cls,
               model_uri,
               loss='mean_squared_error',
               learning_rate=0.001,
               momentum=0.9,
               metrics=None):
        """
        Create and upload an untrained simple model.

        @param model_uri - path to upload tf model to in s3.
        @return - model_config dict compatible with SimpleModel.
        """
        metrics = metrics or ['mse']
        sgd = SGD(lr=learning_rate,
                  momentum=momentum,
                  nesterov=False)

        # Create the model
        model = Sequential()
        model.add(Convolution2D(20, 5, 5,
            input_shape=(80, 80, 3),
            init= "glorot_uniform",
            activation='relu',
            border_mode='same',
            W_regularizer=l2(0.01),
            bias=True))
        model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))
        model.add(Convolution2D(50, 5, 5,
            init= "glorot_uniform",
            activation='relu',
            border_mode='same',
            W_regularizer=l2(0.01),
            bias=True))
        model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(
            output_dim=128,
            init='glorot_uniform',
            activation='relu',
            bias=True,
            W_regularizer=l2(0.1)))
        model.add(Dropout(0.5))
        model.add(Dense(
            output_dim=1,
            init='glorot_uniform',
            W_regularizer=l2(0.01)))

        model.compile(loss=loss, optimizer=sgd, metrics=metrics)

        # Upload the model to designated path
        upload_model(model, model_uri)

        # Return model_config params compatible with constructor
        return {
            'type': SimpleModel.TYPE,
            'model_uri': model_uri
        }


class EnsembleModel(BaseModel):
    """
    """
    TYPE = 'ensemble'

    def __init__(self, model_config):
        self.input_model_config = model_config['input_model_config']
        self.input_model = load_from_config(
            self.input_model_config
        ).as_encoder()

        self.model = load_model_from_uri(
            model_config['model_uri'])

        self.timesteps = model_config['timesteps']
        self.timestep_noise = model_config['timestep_noise']
        self.timestep_dropout = model_config['timestep_dropout']

    def fit(self, dataset, training_args):
        batch_size = training_args.get('batch_size', 100)
        validation_size = training_args.get('validation_size', 500)
        epoch_size = training_args.get('epoch_size', 1000)
        epochs = training_args.get('epochs', 5)

        input_model = self.input_model
        timesteps = self.timesteps
        noise = self.timestep_noise
        dropout = self.timestep_dropout

        self.model.summary()

        batch, _ = (dataset
            .training_generator(batch_size)
            .next())

        # NOTE: for some reason if I don't call this then everything breaks
        # TODO: why?
        input_model.predict_on_batch(batch)

        training_generator = (dataset
            .training_generator(batch_size)
            .with_transform(input_model, timesteps, noise, dropout))

        validation_generator = (dataset
            .validation_generator(batch_size)
            .with_transform(input_model, timesteps, noise, dropout))

        # NOTE: can't use fit with parallel loading or it locks up
        # TODO: why?
        history = self.model.fit_generator(
            training_generator,
            validation_data=validation_generator,
            samples_per_epoch=epoch_size,
            nb_val_samples=validation_size,
            nb_epoch=epochs,
            verbose=1,
            callbacks=[])

    def evaluate(self, dataset):
        testing_size = min(dataset.get_testing_size() / 10, 100)
        testing_generator = (dataset
            .testing_generator(testing_size)
            .with_transform(self.input_model,
                            self.timesteps,
                            self.timestep_noise,
                            self.timestep_dropout))

        evaluation = self.model.evaluate_generator(
            testing_generator, testing_size)

        return evaluation

    def predict_on_batch(self, batch):
        default_prev = -0.0506
        ensemble_input = self.input_model.predict_on_batch(batch)
        if self.timesteps == 0:
            return self.model.predict_on_batch(ensemble_input)
        else:
            input_dim = ensemble_input.shape[1] + self.timesteps
            output_dim = get_output_dim(self.model)
            assert output_dim == 1, 'only support single output dim'

            output = np.empty((len(batch), output_dim))
            steps = deque([default_prev for _ in xrange(self.timesteps)])

            for i in xrange(len(batch)):
                sample = (np
                    .concatenate((ensemble_input[i], steps))
                    .reshape((1, input_dim)))
                prediction = self.model.predict([sample])[0, 0]
                output[i] = prediction
                steps.popleft()
                steps.append(prediction)

            return output

    def save(self, task_id):
        ensemble_s3_uri = 's3://sdc-matt/ensemble/%s/ensemble.h5' % task_id
        upload_model(self.model, ensemble_s3_uri)

        return {
            'type': EnsembleModel.TYPE,
            'timesteps': self.timesteps,
            'timestep_noise': self.timestep_noise,
            'timestep_dropout': self.timestep_dropout,
            'model_uri': ensemble_s3_uri,
            'input_model_config': self.input_model_config,
        }

    @classmethod
    def create(cls,
               model_uri,
               input_model_config,
               timesteps=0,
               timestep_noise=0,
               timestep_dropout=0,
               layers=None,
               loss='mean_squared_error',
               learning_rate=0.001,
               momentum=0.9,
               W_l2=0.1,
               metrics=None):

        input_model = load_from_config(input_model_config).as_encoder()
        input_dim = get_output_dim(input_model) + timesteps
        layers = [input_dim] + (layers or [128, ])
        metrics = metrics or ['mse']
        sgd = SGD(lr=learning_rate,
                  momentum=momentum,
                  nesterov=False)

        model = Sequential()
        for input_dim, output_dim in zip(layers, layers[1:]):
            logger.info('Adding layer (%d, %d)', input_dim, output_dim)
            model.add(Dense(
                input_dim=input_dim,
                output_dim=output_dim,
                activation='relu',
                init='glorot_uniform',
                bias=True,
                W_regularizer=l2(W_l2)))
            model.add(Dropout(0.5))

        model.add(Dense(
            output_dim=1,
            init='glorot_uniform',
            W_regularizer=l2(W_l2)))

        model.compile(loss=loss, optimizer=sgd, metrics=metrics)

        # Upload the model to designated path
        upload_model(model, model_uri)

        return {
            'type': EnsembleModel.TYPE,
            'timesteps': timesteps,
            'timestep_noise': timestep_noise,
            'timestep_dropout': timestep_dropout,
            'model_uri': model_uri,
            'input_model_config': input_model_config,
        }


MODEL_CLASS_BY_TYPE = {
    SimpleModel.TYPE: SimpleModel,
    EnsembleModel.TYPE: EnsembleModel,
}


def load_from_config(model_config):
    """
    Loads a model from config by looking up it's model type and
    calling the right constructor.

    NOTE: Requires model_config to have 'type' key.

    @param model_config - appropriate model_config dict for model type
    @return - model object
    """
    model_type = model_config['type']
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
    try: os.makedirs(os.path.dirname(model_path))
    except: pass
    logger.info("Uploading model to %s", s3_uri)
    model.save(model_path)
    upload_file(model_path, s3_uri)


def load_model_from_uri(s3_uri, skip_cache=False, cache_dir='/tmp'):
    """
    Download, deserialize, and load a keras model into memory.

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


def deep_copy_model(model):
    """
    Create a deep copy of a tensorflow model.

    @param model - tensorflow model
    @return - copy of model
    """
    _, tmp_path = tempfile.mkstemp()
    try:
        model.save(tmp_path)
        return keras_load_model(tmp_path)
    finally:
        os.remove(tmp_path)


def get_output_dim(model):
    """
    Infer output dimension from model by inspecting its layers.

    @param model - tensorflow model
    @return - output dimension
    """
    for layer in reversed(model.layers):
        if hasattr(layer, 'output_shape'):
            return layer.output_shape[-1]

    raise ValueError('Could not infer output dim')
