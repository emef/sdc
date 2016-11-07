"""
Creating tensorflow models.
"""
from collections import deque
import logging, os, tempfile

from keras import backend as K
from keras import metrics
from keras.engine.topology import Merge
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential
from keras.models import load_model as keras_load_model
from keras.optimizers import SGD
from keras.regularizers import l2
import numpy as np
from scipy.stats.mstats import mquantiles
import tensorflow as tf

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

    def fit(self, dataset, training_args, callbacks=None):
        """
        Fit the model with given dataset/args.

        @param dataset - a Dataset
        @param training_args - dict of training args
        @param callbacks - optional list of callbacks to use
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


class CategoricalModel(BaseModel):
    TYPE = 'categorical'

    def __init__(self, model_config):
        self.model = load_model_from_uri(model_config['model_uri'])
        self.thresholds = model_config.get('thresholds')

        if 'thresholds' not in model_config:
            logger.warning('Using old-style config, any use other than '
                           'as an encoder will fail')

    def fit(self, dataset, training_args, callbacks=None):
        batch_size = training_args.get('batch_size', 100)
        epoch_size = training_args.get(
            'epoch_size', dataset.get_training_size())
        validation_size = training_args.get(
            'validation_size', dataset.get_validation_size())
        epochs = training_args.get('epochs', 5)

        self.model.summary()

        training_generator = (dataset
            .training_generator(batch_size)
            .as_categorical(self.thresholds))

        validation_generator = (dataset
            .validation_generator(batch_size)
            .as_categorical(self.thresholds))

        if 'percentile_sampling' in training_args:
            sampling_type = training_args['percentile_sampling']
            training_generator = (training_generator
                .with_percentile_sampling(sampling_type))

        history = self.model.fit_generator(
            training_generator,
            validation_data=validation_generator,
            samples_per_epoch=epoch_size,
            nb_val_samples=validation_size,
            nb_epoch=epochs,
            verbose=1,
            callbacks=(callbacks or []))

    def evaluate(self, dataset):
        n_batches = 32 * (dataset.get_testing_size() / 32)
        testing_generator = (dataset
            .testing_generator(32)
            .as_categorical(self.thresholds))

        return self.model.evaluate_generator(
            testing_generator, n_batches)

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
        s3_uri = 's3://sdc-matt/categorical/%s/model.h5' % task_id
        upload_model(self.model, s3_uri)

        return {
            'type': CategoricalModel.TYPE,
            'model_uri': s3_uri,
            'thresholds': self.thresholds,
        }

    @classmethod
    def create(cls,
               model_uri,
               thresholds=[-0.03664, 0.03664],
               input_shape=(120, 320, 3),
               use_adadelta=True,
               learning_rate=0.01,
               W_l2=0.0001):
        """
        """
        model = Sequential()
        model.add(Convolution2D(16, 5, 5,
            input_shape=input_shape,
            init= "he_normal",
            activation='relu',
            border_mode='same'))
        model.add(MaxPooling2D(pool_size=(4, 5)))
        model.add(Convolution2D(32, 4, 4,
            init= "he_normal",
            activation='relu',
            border_mode='same'))
        model.add(MaxPooling2D(pool_size=(2, 3)))
        model.add(Convolution2D(64, 3, 3,
            init= "he_normal",
            activation='relu',
            border_mode='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(
            output_dim=64,
            init='he_normal',
            activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(
            output_dim=(1 + len(thresholds)),
            init='he_normal',
            W_regularizer=l2(W_l2),
            activation='sigmoid'))

        optimizer = ('adadelta' if use_adadelta
                     else SGD(lr=learning_rate, momentum=0.9))

	model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['categorical_accuracy', 'top_2'])

        # Upload the model to designated path
        upload_model(model, model_uri)

        # Return model_config params compatible with constructor
        return {
            'type': CategoricalModel.TYPE,
            'model_uri': model_uri,
            'thresholds': thresholds,
        }

class RegressionModel(BaseModel):
    TYPE = 'regression'

    def __init__(self, model_config):
        self.model = load_model_from_uri(model_config['model_uri'])

    def fit(self, dataset, training_args, callbacks=None):
        batch_size = training_args.get('batch_size', 100)
        epoch_size = training_args.get(
            'epoch_size', dataset.get_training_size())
        validation_size = training_args.get(
            'validation_size', dataset.get_validation_size())
        epochs = training_args.get('epochs', 5)

        self.model.summary()

        training_generator = dataset.training_generator(batch_size)
        if 'percentile_sampling' in training_args:
            sampling_type = training_args['percentile_sampling']
            training_generator = (training_generator
                .with_percentile_sampling(sampling_type))

        history = self.model.fit_generator(
            training_generator,
            validation_data=dataset.validation_generator(batch_size),
            samples_per_epoch=epoch_size,
            nb_val_samples=validation_size,
            nb_epoch=epochs,
            verbose=1,
            callbacks=(callbacks or []))

        return history

    def evaluate(self, dataset):
        n_testing = dataset.get_testing_size()
        return  self.model.evaluate_generator(
            dataset.testing_generator(256),
            (n_testing / 256) * 256)

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
        s3_uri = 's3://sdc-matt/regression/%s/model.h5' % task_id
        upload_model(self.model, s3_uri)

        return {
            'type': RegressionModel.TYPE,
            'model_uri': s3_uri,
        }

    @classmethod
    def create(cls,
               model_uri,
               input_shape=(120, 320, 3),
               use_adadelta=True,
               learning_rate=0.01,
               W_l2=0.0001):
        """
        """
        model = Sequential()
        model.add(Convolution2D(16, 5, 5,
            input_shape=input_shape,
            init= "he_normal",
            activation='relu',
            border_mode='same'))
        model.add(MaxPooling2D(pool_size=(4, 5)))
        model.add(Convolution2D(32, 4, 4,
            init= "he_normal",
            activation='relu',
            border_mode='same'))
        model.add(MaxPooling2D(pool_size=(2, 3)))
        model.add(Convolution2D(64, 3, 3,
            init= "he_normal",
            activation='relu',
            border_mode='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(
            output_dim=64,
            init='he_normal',
            activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(
            output_dim=1,
            init='he_normal',
            W_regularizer=l2(W_l2)))

        optimizer = ('adadelta' if use_adadelta
                     else SGD(lr=learning_rate, momentum=0.9))

	model.compile(
            loss='mean_squared_error',
            optimizer=optimizer,
            metrics=['rmse'])

        # Upload the model to designated path
        upload_model(model, model_uri)

        # Return model_config params compatible with constructor
        return {
            'type': RegressionModel.TYPE,
            'model_uri': model_uri,
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

    def fit(self, dataset, training_args, callbacks=None):
        validation_size = training_args.get(
            'validation_size', dataset.get_validation_size())
        epoch_size = training_args.get(
            'epoch_size', dataset.get_training_size())
        batch_size = training_args.get('batch_size', 100)
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
            callbacks=(callbacks or []))

    def evaluate(self, dataset):
        batch_size = 256
        testing_size = dataset.get_testing_size()
        testing_generator = dataset.testing_generator(batch_size)
        predictor = self.make_stateful_predictor()
        n_batches = testing_size / batch_size

        err_sum = 0.
        err_count = 0.
        for _ in xrange(n_batches):
            X_batch, y_batch = testing_generator.next()
            if self.timesteps == 0:
                y_pred = self.predict_on_batch(X_batch)
                err_sum += ((y_batch - y_pred) ** 2).sum()
                err_count += batch_size
            else:
                for i in xrange(batch_size):
                    transformed_batch = (self
                        .input_model
                        .predict_on_batch(X_batch))
                    y_pred = predictor(transformed_batch[i])
                    y_true = y_batch[i]
                    err_sum += (y_true - y_pred) ** 2
                    err_count += 1

        mse = err_sum / err_count
        return [mse, np.sqrt(mse)]

    def predict_on_batch(self, batch):
        transformed_batch = self.input_model.predict_on_batch(batch)
        if self.timesteps == 0:
            return self.model.predict_on_batch(transformed_batch)
        else:
            predictor = self.make_stateful_predictor()
            output_dim = get_output_dim(self.model)
            output = np.empty((len(batch), output_dim))

            for i in xrange(len(batch)):
                output[i] = predictor(transformed_batch[i])

            return output

    def make_stateful_predictor(self, apply_transform=False):
        default_prev = 0
        steps = deque([default_prev for _ in xrange(self.timesteps)])

        def predict_fn(input_features):
            if apply_transform:
                if len(input_features.shape) != 4:
                    input_features = input_features.reshape(
                        (1,) + input_features.shape)

                ensemble_features = (self.input_model
                    .predict([input_features])[0])
            else:
                ensemble_features = input_features.reshape(
                    (np.max(input_features.shape), ))

            input_dim = len(ensemble_features) + self.timesteps
            sample = (np
                .concatenate((ensemble_features, steps))
                .reshape((1, input_dim)))
            prediction = self.model.predict([sample])[0, 0]
            steps.popleft()
            steps.append(prediction)
            return prediction

        return predict_fn

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
               W_l2=0.001,
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
                init='he_normal',
                bias=True,
                W_regularizer=l2(W_l2)))
            model.add(Dropout(0.5))

        model.add(Dense(
            output_dim=1,
            init='he_normal',
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

class LstmModel(BaseModel):
    """
    """
    TYPE = 'lstm'

    def __init__(self, model_config):
        self.input_model_config = model_config['input_model_config']
        self.input_model = load_from_config(
            self.input_model_config
        ).as_encoder()

        self.model = load_model_from_uri(
            model_config['model_uri'])

        self.timesteps = model_config['timesteps']

    def fit(self, dataset, training_args, callbacks=None):
        validation_size = training_args.get(
            'validation_size', dataset.get_validation_size())
        epoch_size = training_args.get(
            'epoch_size', dataset.get_training_size())
        batch_size = training_args.get('batch_size', 100)
        epochs = training_args.get('epochs', 5)

        input_model = self.input_model
        timesteps = self.timesteps

        self.model.summary()

        batch, _ = (dataset.training_generator(batch_size).next())

        # NOTE: for some reason if I don't call this then everything breaks
        # TODO: why?
        input_model.predict_on_batch(batch)

        training_generator = (dataset
            .training_generator(batch_size)
            .with_timesteps('timestepped_images', timesteps=timesteps))

        validation_generator = (dataset
            .validation_generator(batch_size)
            .with_timesteps('timestepped_images', timesteps=timesteps))

        # NOTE: can't use fit with parallel loading or it locks up
        # TODO: why?
        history = self.model.fit_generator(
            training_generator,
            validation_data=validation_generator,
            samples_per_epoch=epoch_size,
            nb_val_samples=validation_size,
            nb_epoch=epochs,
            verbose=1,
            callbacks=(callbacks or []))

    def evaluate(self, dataset):
        batch_size = 64
        testing_size = dataset.get_testing_size()
        testing_generator = (dataset
            .testing_generator(batch_size)
            .with_timesteps('timestepped_images', timesteps=self.timesteps))
        n_batches = testing_size / batch_size

        err_sum = 0.
        err_count = 0.
        for _ in xrange(n_batches):
            X_batch, y_batch = testing_generator.next()
            y_pred = self.model.predict_on_batch(X_batch)
            err_sum = np.sum((y_batch - y_pred) ** 2)
            err_count += len(y_pred)
        mse = err_sum / err_count
        return [mse]

    def predict_on_batch(self, batch):
        inputs = np.empty([len(batch)] + [self.timesteps] + list(batch[0].shape))
        for index in xrange(len(batch)):
            for step in xrange(self.timesteps):
                if (index - step) >= 0:
                    inputs[index, self.timesteps - step - 1, :, :, :] = batch[index - step]
        return self.model.predict_on_batch(inputs)

    def save(self, task_id):
        ensemble_s3_uri = 's3://sdc-matt/lstm/%s/lstm.h5' % task_id
        upload_model(self.model, ensemble_s3_uri)

        return {
            'type': LstmModel.TYPE,
            'timesteps': self.timesteps,
            'model_uri': ensemble_s3_uri,
            'input_model_config': self.input_model_config,
        }

    @classmethod
    def create(cls,
      model_uri,
      input_model_config,
      input_shape,
      timesteps=0,
      loss='mean_squared_error',
      learning_rate=0.001,
      momentum=0.9,
      W_l2=0.001,
      metrics=None):
        """
        Creates an LstmModel using a model in the input_model_config

        @param model_uri - s3 uri to save the model
        @param input_model_config - model to use as the input time distributed layer
                                  in the lstm
        @param input_shape - timestepped shape (timesteps, feature dims)
        @param timesteps - timesteps inclusive of the current frame
                         (10 - current frame + 9 previous frames)
        @param loss - loss function on the model
        @param learning - learning rate parameter on the model
        @param momentum - learning momentum
        @param W_l2 - W_l2 regularization param
        @param metrics - metrics to track - (rmse, mse...)
        """

        input_model = load_from_config(input_model_config).as_encoder()
        metrics = metrics or ['mse']
        sgd = SGD(lr=learning_rate,
            momentum=momentum,
            nesterov=False)

        model = Sequential()
        model.add(TimeDistributed(input_model, input_shape=input_shape))
        model.add(LSTM(256))
        model.add(Dense(
            output_dim=1,
            init='he_normal',
            W_regularizer=l2(W_l2)))

        model.compile(loss=loss, optimizer=sgd, metrics=metrics)

        # Upload the model to designated path
        upload_model(model, model_uri)

        return {
            'type': LstmModel.TYPE,
            'timesteps': timesteps,
            'model_uri': model_uri,
            'input_model_config': input_model_config,
        }

MODEL_CLASS_BY_TYPE = {
    'simple': CategoricalModel,  # backwards compat
    CategoricalModel.TYPE: CategoricalModel,
    EnsembleModel.TYPE: EnsembleModel,
    RegressionModel.TYPE: RegressionModel,
    LstmModel.TYPE: LstmModel,
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

def rmse(y_true, y_pred):
    '''Calculates RMSE
    '''
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def top_2(y_true, y_pred):
    return K.mean(tf.nn.in_top_k(y_pred, K.argmax(y_true, axis=-1), 2))

metrics.rmse = rmse
metrics.top_2 = top_2
