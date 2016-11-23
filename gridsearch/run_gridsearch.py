import argparse

from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.models import Sequential

from ..datasets import load_dataset
from ..models import RegressionModel, save_model

class TuneDenseTopology(object):
    def __init__(self):
        self.generator = self.parameter_iterator()

    def parameter_iterator(self):
        for dense_size in (128, 256, 512, 1024):
            yield dense_size

    def __iter__(self):
        return self

    def next(self):
        dense_size = self.generator.next()

        model = Sequential()
        model.add(Convolution2D(16, 5, 5,
            input_shape=(120, 320, 3),
            init= "he_normal",
            activation='relu',
            border_mode='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(20, 5, 5,
            init= "he_normal",
            activation='relu',
            border_mode='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(64, 3, 3,
            init= "he_normal",
            activation='relu',
            border_mode='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(64, 3, 3,
            init= "he_normal",
            activation='relu',
            border_mode='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(100, 3, 3,
            init= "he_normal",
            activation='relu',
            border_mode='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(
            output_dim=256,
            init='he_normal',
            activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(
            output_dim=1,
            init='he_normal',
            W_regularizer=l2(W_l2)))

        model.compile(
            loss='mean_squared_error',
            optimizer='adadelta',
            metrics=['rmse'])

        return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run grid search')
    parser.add_argument('filename', type=str, metavar='F',
                        help='Where to read/write model output to')
    parser.add_argument('--dataset_path', type=str,
                        default='s3://sdc-matt/datasets/final_training',
                        help='Dataset uri to train on')

    args = parser.parse_args()

    dataset_path = args.dataset_path
    dataset = load_dataset(dataset_path)
    model_iterator = TuneDenseTopology()
    for model in model_iterator:
        task_id = 'gridsearch.%d' % int(time.time())
        model_path = '/datasets/tmp/' + task_id
        save_model(rnd_model, model_uri)
        model = RegressionModel({'model_uri': model_uri})

        # TODO: grid search should be able to vary the training args
        training_args = {
            'epochs': 30,  # let early stop callback end training
            'batch_size': 32,
        }

        callbacks = []
        history = model.fit(dataset, training_args, callbacks=callbacks)
        model_config = model.save(model_path)
        test_loss = model.evaluate(dataset)

        output = {
            'topology': topology.name,
            'dataset_path': dataset_path,
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
