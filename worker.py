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
    evaluation = model.evaluate(dataset)
    logger.info('Evaluation results %s', evaluation)

    # print out some sample prediction/label pairs
    example_images, example_labels = dataset.sequential_generator(50).next()
    predictions = model.predict_on_batch(example_images)
    for pred, label in zip(predictions, example_labels):
        logger.info('p=%.5f  l=%.5f', pred, label)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    if False:
        dataset = load_dataset('s3://sdc-matt/datasets/sdc_processed_1')
        input_model_config = SimpleModel.create('s3://sdc-matt/sample-model.h5')
        input_model = load_from_config(input_model_config)
        input_model.fit(dataset, {'epochs': 100})
        upload_model(
            input_model.as_encoder(),
            's3://sdc-matt/sample-trained-encoder.h5')

    input_model_config = {
        'type': SimpleModel.TYPE,
        'model_uri': 's3://sdc-matt/sample-trained-encoder.h5',
    }

    ensemble_model_config = EnsembleModel.create(
        's3://sdc-matt/ensemble-model.h5',
        input_model_config,
        timesteps=1)

    sample_task = {
        'task_id': '1',
        'dataset_uri': 's3://sdc-matt/datasets/sdc_processed_1',
        'output_uri': 's3://',
        'model_config': ensemble_model_config,
        'training_args': {
            'batch_size': 100,
            'validation_size': 500,
            'epoch_size': 1000,
            'epochs': 20,
        },
    }

    if False:
        sample_task = {
            'task_id': '1',
            'dataset_uri': 's3://sdc-matt/datasets/sdc_processed_1',
            'output_uri': 's3://',
            'model_config': SimpleModel.create('s3://sdc-matt/tmp.h5'),
            'training_args': {
                'batch_size': 100,
                'validation_size': 500,
                'epoch_size': 1000,
                'epochs': 20,
            },
        }


    handle_task(sample_task)
