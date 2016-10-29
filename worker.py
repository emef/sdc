"""
This is the worker process which reads from the task queue, trains the
model, validates it, and writes the results to the db.
"""
import logging, time

from keras.backend import binary_crossentropy
from keras.callbacks import Callback
import numpy as np
import tensorflow as tf

from models import (
    load_from_config, upload_model, EnsembleModel, SimpleModel)
from datasets import load_dataset

logger = logging.getLogger(__name__)


class SnapshotCallback(Callback):
    """
    Callback which saves the model snapshot to s3 on each epoch
    """
    def __init__(self,
                 model_to_save,
                 task_id,
                 only_keep_best=True,
                 score_metric='val_categorical_accuracy'):
        self.model_to_save = model_to_save
        self.task_id = task_id
        self.only_keep_best = only_keep_best
        self.score_metric = score_metric
        self.best = None

    def on_epoch_end(self, epoch, logs):
        if self.only_keep_best:
            score = logs.get(self.score_metric)
            if self.best is None or score > self.best:
                self.best = score
            else:
                logger.info(
                    'Not snapshotting: %.2f less than previous %.2f',
                    score, self.best)
                return

        self.model_to_save.save(task_id)


def handle_task(task):
    """
    Runs a tensorflow task.
    """
    logger.info('loading model with config %s', task['model_config'])
    model = load_from_config(task['model_config'])
    dataset = load_dataset(task['dataset_uri'])

    snapshot = SnapshotCallback(model, task['task_id'])
    model.fit(dataset, task['training_args'], callbacks=[snapshot])
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


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    task_id = str(int(time.time()))

    if False:
        sample_task = {
            'task_id': task_id,
            'dataset_uri': 's3://sdc-matt/datasets/final_training',
            'output_uri': 's3://',
            'model_config': SimpleModel.create_categorical(
                's3://sdc-matt/tmp/' + task_id,
                cat_classes=5,
                learning_rate=0.01,
                input_shape=(120, 320, 3)),
            'training_args': {
                'batch_size': 32,
                'epochs': 50,
            },
        }

    if True:
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

        sample_task = {
            'task_id': task_id,
            'dataset_uri': 's3://sdc-matt/datasets/final_training',
            'model_config': ensemble_model_config,
            'training_args': {
                'batch_size': 64,
                'epochs': 10,
            },
        }

    handle_task(sample_task)
