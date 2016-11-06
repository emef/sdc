"""
Run a gridsearch on some model config.

* convolution layer parameters
  - nb filters
  - filter size
  - pooling
  - dropout

* dense layer parameters
  - neurons
  - dropout
  - regularization
"""
import argparse
import json
import numpy as np
import pprint
import time
import traceback

from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.models import Sequential
from keras.regularizers import l2

from callbacks import TimedEarlyStopping
from datasets import load_dataset
from models import upload_model, RegressionModel


class BaseParameter(object):
    """
    Parameter which has predefined values which it can take on.
    """

    def expand(self):
        """Return all possible values of this parameter"""
        raise NotImplemented()


class RangeParameter(BaseParameter):
    """
    Parameter which takes a value in a range with equal-valued steps.
    """
    def __init__(self,
                 min_value,
                 max_value,
                 delta=1):
        self.min_value = min_value
        self.max_value = max_value
        self.delta = delta

    def expand(self):
        return np.arange(
            self.min_value,
            self.max_value + self.delta,
            self.delta)


class ChoiceParameter(BaseParameter):
    """
    Parameter with fixed choice of parameters.
    """
    def __init__(self, choices):
        self.choices = choices

    def expand(self):
        return self.choices


class ParameterSpace(object):
    """
    Contains a set of parameters and can map a vector to/from
    the projected parameter space.

    Suppose we have a parameter which expands to possible values
    [0, 5, 6, 11]. You can map a value in [0, 1] to this parameter
    space by using equal size bins on the [0, 1] interval [0.25, 0.5, 0.75].

    Examples:
      in   -> out
      0.99 -> 11
      0.01 -> 0
      0.76 -> 11
      0.73 -> 6
    """

    def __init__(self, parameters):
        """
        @param parameters - (name, parameter) pairs
        """
        self.names = [
            name
            for name, _ in parameters]

        self.expanded = [
            p.expand()
            for _, p in parameters]

        self.bins = [
            np.linspace(0, 1, len(values) + 1)[1:]
            for values in self.expanded]

    @classmethod
    def empty(cls):
        return ParameterSpace([])

    def size(self):
        return len(self.names)

    def transform(self, projected):
        params = []
        for value, bins, expanded in zip(projected, self.bins, self.expanded):
            expanded_index = int(np.digitize([value], bins)[0])
            expanded_value = expanded[expanded_index]

            params.append(expanded_value)

        return params

class Layer(object):
    def get_parameter_space(self):
        raise NotImplemented()

    def add_to_model(self, model, projected_params, input_shape=None):
        raise NotImplemented()


class ConvLayer(Layer):
    def __init__(self,
                 min_nb_filters=1,
                 max_nb_filters=200,
                 delta_nb_filters=5,
                 min_filter_dim=1,
                 max_filter_dim=10,
                 delta_filter_dim=1,
                 min_pool_dim=1,
                 max_pool_dim=10,
                 delta_pool_dim=1,
                 min_dropout=0,
                 max_dropout=0.8,
                 delta_dropout=0.2):

        self.parameters = ParameterSpace((
            ('nb_filters', RangeParameter(
                min_nb_filters, max_nb_filters, delta_nb_filters)),
            ('filter_dim', RangeParameter(
                min_filter_dim, max_filter_dim, delta_filter_dim)),
            ('pool_dim', RangeParameter(
                min_pool_dim, max_pool_dim, delta_pool_dim)),
            ('dropout', RangeParameter(
                min_dropout, max_dropout, delta_dropout)),
        ))

    def get_parameter_space(self):
        return self.parameters

    def add_to_model(self, model, params, input_shape=None):
        nb_filters, filter_dim, pool_dim, dropout = params
        kwargs = {} if input_shape is None else {'input_shape': input_shape}

        model.add(Convolution2D(
            nb_filters,
            filter_dim, filter_dim,
            init="glorot_uniform",
            activation='relu',
            border_mode='same',
            **kwargs))
        model.add(MaxPooling2D(pool_size=(pool_dim, pool_dim)))
        model.add(Dropout(dropout))


class DenseLayer(Layer):
    def __init__(self,
                 min_output_dim=1,
                 max_output_dim=1000,
                 delta_output_dim=10,
                 min_dropout=0,
                 max_dropout=0.8,
                 delta_dropout=0.2):
        self.parameters = ParameterSpace((
            ('output_dim', RangeParameter(
                min_output_dim, max_output_dim, delta_output_dim)),
            ('dropout', RangeParameter(
                min_dropout, max_dropout, delta_dropout)),
            ('W_l2', ChoiceParameter((0, 0.1, 0.01, 0.001, 0.0001))),
        ))

    def get_parameter_space(self):
        return self.parameters

    def add_to_model(self, model, params, input_shape=None):
        output_dim, dropout, W_l2 = params
        kwargs = {} if input_shape is None else {'input_shape': input_shape}

        model.add(Dense(
            output_dim=output_dim,
            init='glorot_uniform',
            activation='relu',
            W_regularizer=l2(W_l2),
            **kwargs))
        model.add(Dropout(dropout))


class OutputLayer(Layer):
    def __init__(self, output_dim):
        self.output_dim = output_dim
        self.parameters = ParameterSpace((
            ('W_l2', ChoiceParameter((0, 0.1, 0.01, 0.001, 0.0001))),
        ))

    def get_parameter_space(self):
        return self.parameters

    def add_to_model(self, model, params, input_shape=None):
        W_l2, = params
        model.add(Dense(
            output_dim=self.output_dim,
            init='glorot_uniform',
            activation='relu',
            W_regularizer=l2(W_l2)))


class FlattenLayer(Layer):
    def get_parameter_space(self):
        return ParameterSpace.empty()

    def add_to_model(self, model, *args, **kwargs):
        model.add(Flatten())


class Topology(object):
    def __init__(self, name, layers, input_shape, output_dim):
        self.name = name
        self.input_shape = input_shape

        # append the output layer
        self.layers = tuple(layers) + (OutputLayer(output_dim), )

    def random_model(self):
        input_shape = self.input_shape
        model = Sequential()
        layer_params = {}
        for i, layer in enumerate(self.layers):
            space = layer.get_parameter_space()
            projected = np.random.rand(space.size())
            params = space.transform(projected)
            layer.add_to_model(model, params, input_shape)

            layer_name = '%s_%d' % (layer.__class__.__name__, i)
            layer_params[layer_name] = dict(zip(space.names, params))

        return model, layer_params


topologies = {
    '2conv-1dense': Topology(
        name='2conv-1dense',
        layers=[
            ConvLayer(min_nb_filters=5,
                      max_nb_filters=35,
                      delta_nb_filters=5,
                      min_filter_dim=2,
                      max_filter_dim=10,
                      delta_filter_dim=2,
                      min_pool_dim=2,
                      max_pool_dim=6,
                      delta_pool_dim=2),

        ConvLayer(min_nb_filters=10,
                  max_nb_filters=100,
                  delta_nb_filters=10,
                  min_filter_dim=2,
                  max_filter_dim=10,
                  delta_filter_dim=2,
                  min_pool_dim=2,
                  max_pool_dim=6,
                  delta_pool_dim=2),

        FlattenLayer(),

        DenseLayer(min_output_dim=40,
                   max_output_dim=400,
                   delta_output_dim=40)
        ],
        input_shape=(120, 320, 3),
        output_dim=1),
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run grid search')
    parser.add_argument('filename', type=str, metavar='F',
                        help='Where to read/write model output to')
    parser.add_argument('--duration', type=int, default=20,
                        help='Duration to train before early stopping')
    parser.add_argument('--max_params', type=int, default=1000000,
                        help='Max # params to consider training')
    parser.add_argument('--dataset_uri', type=str,
                        default='s3://sdc-matt/datasets/final_training',
                        help='Dataset uri to train on')
    parser.add_argument('--topology', type=str, default='2conv-1dense',
                        help='Which topology to search')

    args = parser.parse_args()

    dataset_uri = args.dataset_uri
    topology = topologies[args.topology]

    while True:
        try:
            rnd_model, gridsearch_params = topology.random_model()

            if rnd_model.count_params() > args.max_params:
                continue

            rnd_model.compile(
                loss='mean_squared_error',
                optimizer='adadelta',
                metrics=['rmse'])

            task_id = 'gridsearch.%d' % int(time.time())
            model_uri = 's3://sdc-matt/tmp/%s' % task_id
            upload_model(rnd_model, model_uri)

            model = RegressionModel({'model_uri': model_uri})
            dataset = load_dataset(dataset_uri)

            # TODO: grid search should be able to vary the training args
            training_args = {
                'epochs': 10000,  # let early stop callback end training
                'batch_size': 100,
                'percentile_sampling': 'uniform',
            }

            callbacks = [TimedEarlyStopping(args.duration)]
            history = model.fit(dataset, training_args, callbacks=callbacks)
            model_config = model.save(task_id)

            test_loss = model.evaluate(dataset)

            output = {
                'topology': topology.name,
                'dataset_uri': dataset_uri,
                'val_loss': history.history['val_loss'][-1],
                'test_loss': test_loss,
                'history': history.history,
                'model_config': model_config,
                'gridsearch_params': gridsearch_params,
                'model_free_parameters': rnd_model.count_params(),
            }

            with open(args.filename, 'a') as results_f:
                results_f.write('%s\n' % json.dumps(output))

        except Exception as e:
            traceback.print_exc()
