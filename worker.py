"""
This is the worker process which reads from the task queue, trains the
model, validates it, and writes the results to the db.
"""
import logging

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

    # print out some sample prediction/label pairs
    example_images, example_labels = dataset.sequential_generator(100).next()
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

    if False:
        sample_task = {
            'task_id': 'simple-1',
            'dataset_uri': 's3://sdc-matt/datasets/sdc_processed_1',
            'output_uri': 's3://',
            'model_config': SimpleModel.create('s3://sdc-matt/tmp.h5'),
            'training_args': {
                'batch_size': 16,
                'validation_size': 512,
                'epoch_size': 1024,
                'epochs': 20,
            },
        }

    if True:
        input_model_config = {
            'type': SimpleModel.TYPE,
            'model_uri': 's3://sdc-matt/simple/simple-1/model.h5',
        }

        ensemble_model_config = EnsembleModel.create(
            's3://sdc-matt/ensemble-model.h5',
            input_model_config,
            timesteps=1,
            timestep_noise=0.2,
            timestep_dropout=0.5)

        sample_task = {
            'task_id': 'ensemble-1',
            'dataset_uri': 's3://sdc-matt/datasets/sdc_processed_1',
            'output_uri': 's3://',
            'model_config': ensemble_model_config,
            'training_args': {
                'batch_size': 16,
                'validation_size': 512,
                'epoch_size': 1024,
                'epochs': 20,
            },
        }

    handle_task(sample_task)
