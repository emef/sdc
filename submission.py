"""
Evaluate the final test set.
"""
import os

import cv2
import numpy as np
from scipy.stats.mstats import mquantiles

from datasets import load_dataset
from models import load_from_config

#if __name__ == '__main__':
def main():
    images_path = '/media/drive/Challenge 2/Test/center'
    input_shape = (120, 320, 3)

    model_type = 'ensemble'

    if model_type == 'ensemble':
        final_model_config = {
            'type': 'ensemble',
            'input_model_config': {
                'model_uri': 's3://sdc-matt/simple/1477715388/model.h5',
                'type': 'simple',
                'cat_classes': 5,
            },
            'model_uri': 's3://sdc-matt/ensemble/1477725706/ensemble.h5',
            'timesteps': 3,
            'timestep_noise': 0.1,
            'timestep_dropout': 0.5,
        }
    elif model_type == 'lstm':
        final_model_config = {
            'type': 'lstm',
            'input_model_config': {
                'model_uri': 's3://sdc-matt/simple/1477715388/model.h5',
                'type': 'simple',
                'cat_classes': 5,
            },
            'model_uri': 's3://sdc-matt/lstm/1477805095/lstm.h5',
            'timesteps': 9
        }
    elif model_type == 'simple':
        final_model_config = {
            'type': 'simple',
            'model_uri': 's3://sdc-matt/simple/1477715388/model.h5',
            'cat_classes': 5,
        }
    model = load_from_config(final_model_config)

    if model_type == 'ensemble':
        predictor = model.make_stateful_predictor(True)
    if model_type == 'lstm':
        def predictor(x):
            return model.predict_on_batch(x)[0][0]
    elif model_type == 'simple':
        dataset = load_dataset(
            's3://sdc-matt/datasets/final_training')
        training_labels = dataset.get_training_labels()
        prob = np.arange(0, 1 + 1./5, 1. / 5)
        bins = mquantiles(training_labels, prob)
        cat_values = []
        for i in xrange(len(bins) - 1):
            cond = (
                (training_labels >= bins[i])
                & (training_labels < bins[i+1]))
            cat_values.append(training_labels[np.where(cond)].mean())

        def predictor(x):
            pred_cats = model.predict_on_batch(
                x.reshape((1,) + input_shape))[0]
            return cat_values[np.argmax(pred_cats)]
    model.model.summary()

    for filename in sorted(os.listdir(images_path)):
        src = os.path.join(images_path, filename)
        x = load_test_image(src).reshape((1, ) + input_shape)
        y_pred = predictor(x)
        print '%s,%s' % (filename, y_pred)

def load_test_image(src):
    cv_image = cv2.imread(src)
    cv_image = cv2.resize(cv_image, (320, 240))
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2YUV)
    cv_image = cv_image[120:240, :, :]
    cv_image = ((cv_image-(255.0/2))/255.0)
    return cv_image

main()
