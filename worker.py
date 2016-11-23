"""
This is the worker process which reads from the task queue, trains the
model, validates it, and writes the results to the db.
"""
import cProfile
import logging
import os
import signal
import StringIO
import pstats
import time

from keras.backend import binary_crossentropy
from keras.callbacks import TensorBoard
import numpy as np
import tensorflow as tf

from callbacks import SnapshotCallback
from models import (
    load_from_config,
    CategoricalModel, EnsembleModel, LstmModel, RegressionModel,
    TransferLstmModel)
from datasets import load_dataset

logger = logging.getLogger(__name__)

PROFILING = False


def profiling_sigint_handler(signal, frame):
    pr.disable()
    s = StringIO.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(.2)
    print s.getvalue()

    print '------------------------'
    if raw_input('continue? (y/n) ') != 'y':
        exit(0)

if PROFILING:
    pr = cProfile.Profile()
    pr.enable()
    signal.signal(signal.SIGINT, profiling_sigint_handler)


def handle_task(task,
                datasets_dir='/datasets',
                models_path='/models'):
    """
    Runs a tensorflow task.
    """
    model_config = task['model_config']
    model_type = model_config['type']
    logger.info('loading model with config %s', task['model_config'])
    model = load_from_config(task['model_config'])
    dataset_path = os.path.join(datasets_dir, task['dataset_path'])
    dataset = load_dataset(dataset_path)
    baseline_mse = dataset.get_baseline_mse()

    snapshot_dir = os.path.join(
        models_path, 'snapshots', model_type, task['task_id'])
    snapshot = SnapshotCallback(
        model,
        snapshot_dir=snapshot_dir,
        score_metric=task.get('score_metric', 'val_rmse'))

    tensorboard = TensorBoard(
        log_dir='/opt/tensorboard',
        histogram_freq=1,
        write_graph=True,
        write_images=True)

    callbacks = [snapshot, tensorboard]

    logger.info('Baseline mse = %.4f  rmse = %.4f' % (
        baseline_mse, np.sqrt(baseline_mse)))
    model.fit(
        dataset,
        task['training_args'],
        final=task.get('final', False),
        callbacks=callbacks)

    output_model_path = os.path.join(
        models_path, 'output', '%s.h5' % task['task_id'])
    output_config = model.save(output_model_path)
    logger.info('Maximum snapshot had score %s=%.6f, saved to %s',
                snapshot.score_metric,
                snapshot.max_score,
                snapshot.max_path)
    logger.info('Minimum snapshot had score %s=%.6f, saved to %s',
                snapshot.score_metric,
                snapshot.min_score,
                snapshot.min_path)
    logger.info('Wrote final model to %s', output_model_path)

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
    tmp_model_path = os.path.join('/tmp', '%s.h5' % task_id)

    if False:
        task = {
            'task_id': task_id,
            'score_metric': 'val_rmse',
            'dataset_path': 'showdown_full',
            'final': True,
            'model_config': TransferLstmModel.create(
                tmp_model_path,
                transform_model_config={
                    'model_uri': '/models/output/1479836278.h5',
                    'scale': 16,
                    'type': 'regression',
                },
                timesteps=10,
                W_l2=0.001,
                scale=16.,
                input_shape=(120, 320, 3)),
            'training_args': {
                'batch_size': 32,
                'epochs': 50,
            },
        }

    if False:
        task = {
            'task_id': task_id,
            'score_metric': 'val_rmse',
            'dataset_path': 'showdown_full',
            'final': True,
            'model_config': RegressionModel.create(
                tmp_model_path,
                use_adadelta=True,
                learning_rate=0.001,
                input_shape=(120, 320, 3)),
            'training_args': {
                'batch_size': 32,
                'epochs': 40,
            },
        }

    if False:
        # sharp left vs center vs sharp right
        task = {
            'task_id': task_id,
            'dataset_path': 'finale_full',
            'score_metric': 'val_categorical_accuracy',
            'model_config': CategoricalModel.create(
                tmp_model_path,
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
        # half degree model
        task = {
            'task_id': task_id,
            'dataset_path': 'finale_center',
            'model_config': CategoricalModel.create(
                tmp_model_path,
                use_adadelta=True,
                learning_rate=0.001,
                thresholds=np.linspace(-0.061, 0.061, 14)[1:-1],
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
            tmp_model_path,
            input_model_config,
            timesteps=3,
            timestep_noise=0.1,
            timestep_dropout=0.5)

        task = {
            'task_id': task_id,
            'dataset_path': 'final_training',
            'model_config': ensemble_model_config,
            'training_args': {
                'batch_size': 64,
                'epochs': 3
            },
        }

    if True:
        lstm_model_config = LstmModel.create(
            tmp_model_path,
            (10, 120, 320, 3),
            timesteps=10,
            W_l2=0.0001,
            scale=60.0)

        task = {
            'task_id': task_id,
            'dataset_path': 'showdown_full',
            'final': True,
            'model_config': lstm_model_config,
            'training_args': {
                'pctl_sampling': 'uniform',
                'batch_size': 32,
                'epochs': 10,
            },
        }

    handle_task(task)
