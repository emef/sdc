"""
This is the worker process which reads from the task queue, trains the
model, validates it, and writes the results to the db.
"""
import logging

from keras.optimizers import SGD

from models import load_from_config, SampleModel
from datasets import load_dataset

logger = logging.getLogger(__name__)


def handle_task(task):
    """
    Runs a tensorflow task.
    """
    logger.info('loading model type %s with config %s',
                task['model_type'], task['model_config'])

    model = load_from_config(
        task['model_type'],
        task['model_config'])

    dataset = load_dataset(task['dataset_uri'])

    model.fit(dataset, task['training_args'])
    evaluation = model.evaluate(dataset)

    example_images, example_labels = dataset.sequential_generator(50).next()
    predictions = model.predict_on_batch(example_images)
    for pred, label in zip(predictions, example_labels):
        print 'p=%.5f  l=%.5f' % (pred, label)

    logger.info('Evaluation results %s', evaluation)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    model_config = SampleModel.create('s3://sdc-matt/sample-model.h5')
    model_config = {'model_uri': 's3://sdc-matt/sample-model.h5'}
    sample_task = {
        'task_id': '1',
        'dataset_uri': 's3://sdc-matt/datasets/sdc_processed_1',
        'output_uri': 's3://',
        'model_type': SampleModel.TYPE,
        'model_config': model_config,
        'training_args': {
            'batch_size': 100,
            'validation_size': 500,
            'epoch_size': 1000,
            'epochs': 5,
        },
    }

    handle_task(sample_task)
