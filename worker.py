"""
This is the worker process which reads from the task queue, trains the
model, validates it, and writes the results to the db.
"""
import logging, time

from keras.backend import binary_crossentropy
import numpy as np
import tensorflow as tf

from callbacks import SnapshotCallback
from models import (
    load_from_config, upload_model,
    CategoricalModel, EnsembleModel, LstmModel, RegressionModel)
from datasets import load_dataset

logger = logging.getLogger(__name__)


def handle_task(task, datasets_dir):
    """
    Runs a tensorflow task.
    """
    logger.info('loading model with config %s', task['model_config'])
    model = load_from_config(task['model_config'])
    dataset = load_dataset(task['dataset_uri'], cache_dir=datasets_dir)
    baseline_mse = dataset.get_baseline_mse()

    snapshot = SnapshotCallback(
        model,
        task['task_id'],
        task.get('score_metric', 'mean_squared_error'))

    logger.info('Baseline mse = %.4f  rmse = %.4f' % (
        baseline_mse, np.sqrt(baseline_mse)))
    model.fit(dataset, task['training_args'], callbacks=[snapshot])
    output_config = model.save(task['task_id'])

    # assume evaluation is mse
    evaluation = model.evaluate(dataset)
    training_mse = evaluation[0]

    improvement = -(training_mse - baseline_mse) / baseline_mse
    logger.info('Evaluation: %s', evaluation)
    logger.info('Baseline MSE %.5f, training MSE %.5f, improvement %.2f%%',
                baseline_mse, training_mse, improvement * 100)
    logger.info('output config: %s' % output_config)

    if model.output_dim() == 1:
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


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    task_id = str(int(time.time()))
    datasets_dir = "/tmp"

    if True:
        task = {
            'task_id': task_id,
            'score_metric': 'val_rmse',
            'dataset_uri': 's3://sdc-matt/datasets/finale_full',
            'output_uri': 's3://',
            'model_config': RegressionModel.create(
                's3://sdc-matt/tmp/' + task_id,
                use_adadelta=True,
                learning_rate=0.001,
                input_shape=(120, 320, 3)),
            'training_args': {
                'batch_size': 32,
                'epochs': 20,
            },
        }

    if False:
        # sharp left vs center vs sharp right
        task = {
            'task_id': task_id,
            'dataset_uri': 's3://sdc-matt/datasets/finale_full',
            'output_uri': 's3://',
            'score_metric': 'val_categorical_accuracy',
            'model_config': CategoricalModel.create(
                's3://sdc-matt/tmp/' + task_id,
                use_adadelta=True,
                W_l2=0.001,
                thresholds=[-0.061, 0.061]
            ),
            'training_args': {
                'batch_size': 32,
                'epochs': 30,
                'pctl_sampling': 'uniform',
            },
        }


    if False:
        task = {
            'task_id': task_id,
            'dataset_uri': 's3://sdc-matt/datasets/finale_nonnegative',
            'output_uri': 's3://',
            'model_config': {
                'type': 'regression',
                'model_uri': 's3://sdc-matt/regression/from_cat_3/2/model.h5',
            },
            'training_args': {
                'pctl_sampling': 'uniform',
                'batch_size': 32,
                'epochs': 5,
            },
        }

    if False:
        task = {
            'task_id': task_id,
            'dataset_uri': 's3://sdc-matt/datasets/finale_timestepped_full',
            'output_uri': 's3://',
            'model_config': CategoricalModel.create(
                's3://sdc-matt/tmp/' + task_id,
                use_adadelta=True,
                learning_rate=0.001,
                thresholds=[-0.1, -0.03, 0.03, 0.1],
                input_shape=(120, 320, 3)),
            'training_args': {
                'pctl_sampling': 'uniform',
                'batch_size': 32,
                'epochs': 20,
            },
        }

    if False:
        input_model_config = {
            'model_uri': 's3://sdc-matt/simple/1477715388/model.h5',
            'type': 'simple',
            'cat_classes': 5
        }

        ensemble_model_config = EnsembleModel.create(
            's3://sdc-matt/tmp/' + task_id,
            input_model_config,
            timesteps=3,
            timestep_noise=0.1,
            timestep_dropout=0.5)

        task = {
            'task_id': task_id,
            'dataset_uri': 's3://sdc-matt/datasets/final_training',
            'model_config': ensemble_model_config,
            'training_args': {
                'batch_size': 64,
                'epochs': 3
            },
        }

    if False:
        input_model_config = {
            'model_uri': 's3://sdc-matt/simple/1477715388/model.h5',
            'type': 'simple',
            'cat_classes': 5
        }

        lstm_model_config = LstmModel.create(
            's3://sdc-matt/tmp/' + task_id,
            input_model_config,
            (10, 120, 320, 3),
            timesteps=10)

        task = {
            'task_id': task_id,
            'dataset_uri': 's3://sdc-matt/datasets/finale_lstm_seq',
            'model_config': lstm_model_config,
            'training_args': {
                'batch_size': 64,
                'epochs': 10
            },
        }

    handle_task(task, datasets_dir)
