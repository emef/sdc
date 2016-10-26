"""
This is the worker process which reads from the task queue, trains the
model, validates it, and writes the results to the db.
"""
import logging, time

from keras.backend import binary_crossentropy
import numpy as np
import tensorflow as tf

from models import (
    load_from_config, upload_model, EnsembleModel, SimpleModel)
from datasets import load_dataset

logger = logging.getLogger(__name__)


def handle_task(task):
    """
    Runs a tensorflow task.
    """
    logger.info('loading model with config %s', task['model_config'])
    model = load_from_config(task['model_config'])
    dataset = load_dataset(task['dataset_uri'])
    print get_baseline_crossentropy(dataset)

    model.fit(dataset, task['training_args'])
    output_config = model.save(task['task_id'])

    # assume evaluation is mse
    evaluation = model.evaluate(dataset)
    training_mse = evaluation[0]
    baseline_mse = get_baseline_mse(dataset)
    improvement = -(training_mse - baseline_mse) / baseline_mse

    logger.info('Evaluation: %s', evaluation)
    logger.info('Baseline MSE %.5f, training MSE %.5f, improvement %.2f%%',
                baseline_mse, training_mse, improvement * 100)

    example_ranges = 10
    range_size = 20
    testing_size = dataset.get_testing_size()
    for _ in xrange(example_ranges):
        # print out some sample prediction/label pairs
        skip_to = int(np.random.random() * (testing_size - range_size))
        example_images, example_labels = (dataset
            .sequential_generator(range_size)
            .skip(skip_to)
            .next())

        predictions = model.predict_on_batch(example_images)
        for pred, label in zip(predictions, example_labels):
            logger.info('p=%.5f  l=%.5f', pred, label)

    print 'output config = %s' % output_config


def get_baseline_mse(dataset):
    """
    Get the baseline MSE of a dataset using a dummy predictor.

    @param - Dataset
    @return - mean squared error of dummy predictor on testing set
    """
    dummy_predictor = dataset.get_training_labels().mean()
    mse = ((dataset.get_testing_labels() - dummy_predictor) ** 2).mean()
    return mse


def get_baseline_crossentropy(dataset):
    """
    Get the baseline binary cross entry for a leftright dataset using
    dummy predictor.

    @param - Dataset
    @return - cross entropy of dummy predictor.
    """
    dataset = dataset.as_leftright()
    prior = dataset.get_training_labels().mean()
    y_true = dataset.get_testing_labels()
    y_pred = np.ones(y_true.shape) * prior
    return binary_crossentropy(
        tf.convert_to_tensor(y_pred),
        tf.convert_to_tensor(y_true))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    task_id = 'leftright-2.%s' % int(time.time())

    if False:
        sample_task = {
            'task_id': task_id,
            'dataset_uri': 's3://sdc-matt/datasets/elcamino_trip2',
            'output_uri': 's3://',
            'model_config': SimpleModel.create_leftright(
                's3://sdc-matt/tmp/' + task_id,
                learning_rate=0.01,
                input_shape=(160, 160, 3)),
            'training_args': {
                'batch_size': 32,
                'epochs': 100,
            },
        }

    if True:
        input_model_config = {
            'model_uri': 's3://sdc-matt/simple/leftright-2.1477421619/model.h5',
            'type': 'simple',
            'leftright': True,
        }

        ensemble_model_config = EnsembleModel.create(
            's3://sdc-matt/tmp/' + task_id,
            input_model_config,
            timesteps=3,
            timestep_noise=0.08,
            timestep_dropout=0.2)

        sample_task = {
            'task_id': task_id,
            'dataset_uri': 's3://sdc-matt/datasets/elcamino_trip2',
            'model_config': ensemble_model_config,
            'training_args': {
                'batch_size': 64,
                'epochs': 10,
            },
        }

    handle_task(sample_task)
