"""
This is the worker process which reads from the task queue, trains the
model, validates it, and writes the results to the db.
"""
import logging

from models import load_from_config, SampleModel

logger = logging.getLogger(__name__)


def handle_task(task):
    """
    Runs a tensorflow task.
    """
    logger.info('handle task')
    model = load_from_config(
        task['model_type'],
        task['model_config']
    )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # model_config = SampleModel.create('s3://sdc-matt/sample-model.h5')
    model_config = {'model_path': 's3://sdc-matt/sample-model.h5'}
    sample_task = {
        'task_id': '1',
        'dataset_path': 's3://',
        'output_path': 's3://',
        'model_type': SampleModel.TYPE,
        'model_config': model_config,
    }

    handle_task(sample_task)
