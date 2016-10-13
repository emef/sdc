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

    # TODO: clean up compilation/training config into task config
    batch_size = 100
    validation_size = 500
    epoch_size = 1000
    epochs = 5
    lrate = 0.001
    decay = lrate/epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)

    model.tf_model.compile(
        loss='mean_squared_error',
        optimizer=sgd,
        metrics=['mse'])

    print model.tf_model.summary()

    history = model.tf_model.fit_generator(
        dataset.training_generator(batch_size),
        validation_data=dataset.validation_generator(batch_size),
        samples_per_epoch=epoch_size,
        nb_val_samples=validation_size,
        nb_epoch=epochs,
        verbose=1,
        callbacks=[],
        pickle_safe=True,
        nb_worker=2)

    _, mse = model.tf_model.evaluate_generator(
        dataset.testing_generator(batch_size),
        dataset.get_testing_size(),
        nb_worker=2,
        pickle_safe=True)

    example_images, example_labels = dataset.sequential_generator(50).next()
    predictions = model.tf_model.predict_on_batch(example_images)
    for pred, label in zip(predictions, example_labels):
        print 'p=%.5f  l=%.5f' % (pred, label)

    print "testing mse: %.6f" % mse


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
    }

    handle_task(sample_task)
