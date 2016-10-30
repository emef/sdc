"""
Auto encoder to try an reduce the dimensionality of feature maps.
"""
from keras.engine.topology import Merge
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.models import Sequential
from keras.models import load_model as keras_load_model
from keras.optimizers import SGD
from keras.regularizers import l2
import numpy as np

from datasets import load_dataset
from models import get_output_dim, load_from_config

model_config = {
    'model_uri': 's3://sdc-matt/simple/1477715388/model.h5',
    'type': 'simple',
    'cat_classes': 5,
}

input_model = load_from_config(model_config).as_encoder()
dataset = load_dataset('s3://sdc-matt/datasets/final_training')
training_size = dataset.get_training_size()
orig_dim = get_output_dim(input_model)

if False:
    batch_size = 256
    training_generator = (dataset
        .training_generator(batch_size)
        .with_transform(input_model))
    X = np.empty((training_size, orig_dim))

    n_batches = training_size / batch_size
    for i in xrange(n_batches):
        X_batch, _ = training_generator.next()
        start = batch_size * i
        end = batch_size * (i + 1)
        X[start:end] = X_batch

    np.save('/sdc/final_training_encoded.npy', X)
else:
    X = np.load('/sdc/final_training_encoded.npy')

def fit_autoencoder(latent_dim):
    model = Sequential()
    model.add(Dense(
        input_dim=orig_dim,
        output_dim=latent_dim,
        init='glorot_uniform',
        activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(
        output_dim=orig_dim,
        init='glorot_uniform',
        activation='sigmoid'))

    model.compile(
        loss='mean_squared_error',
        optimizer='adadelta')

    history = model.fit(
        X, X, batch_size=64, verbose=0)

    return history.history['loss'][-1]

for latent_dim in (128, 256, 384, 512):
    print latent_dim, fit_autoencoder(latent_dim)
