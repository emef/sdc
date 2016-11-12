"""
Evaluate the final test set.
"""
import os, time

import cv2
import numpy as np
from scipy.stats.mstats import mquantiles

from datasets import load_dataset
from models import load_from_config

def main():
    images_path = '/media/drive/Challenge 2/Test/center'
    input_shape = (120, 320, 3)

    model_type = 'mixture'

    if model_type == 'ensemble':
        model = load_from_config({
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
        })

        model.model.summary()
        predictor = model.make_stateful_predictor(True)

    elif model_type == 'lstm':
        model = load_from_config({
            'type': 'lstm',
            'input_model_config': {
                'model_uri': 's3://sdc-matt/simple/1477715388/model.h5',
                'type': 'simple',
                'cat_classes': 5,
            },
            'model_uri': 's3://sdc-matt/lstm/1477805095/lstm.h5',
            'timesteps': 9
        })

        model.model.summary()
        predictor = lambda x: model.predict_on_batch(x)[0][0]

    elif model_type == 'regression':
        model = load_from_config({
            'type': 'regression',
            'model_uri': 's3://sdc-matt/regression/1477947323/model.h5',
        })

        model.model.summary()
        predictor = lambda x: model.predict_on_batch(x)[0][0]

    elif model_type == 'mixture':
        model = load_from_config({
            'type': 'mixture',
            'general_regression': {
                'model_uri': 's3://sdc-matt/regression/1478919380/model.h5',
                'type': 'regression'
            },
            'sharp_regression': {
                'model_uri': 's3://sdc-matt/regression/1478916261/model.h5',
                'type': 'regression'
            },
            'sharp_classifier': {
                'model_uri': 's3://sdc-matt/categorical/1478913870/model.h5',
                'thresholds': [-0.061, 0.061],
                'type': 'categorical'
            },
        })

        predictor = model.make_stateful_predictor(
            smoothing=True,
            interpolation_weight=0.5)

    with open('submission.%s.csv' % int(time.time()), 'w') as f:
        f.write('frame_id,steering_angle\n')
        for filename in sorted(os.listdir(images_path)):
            src = os.path.join(images_path, filename)
            x = load_test_image(src).reshape((1, ) + input_shape)
            y_pred = predictor(x)
            filename = filename.split('.')[0]
            f.write('%s,%s\n' % (filename, y_pred))

def load_test_image(src):
    cv_image = cv2.imread(src)
    cv_image = cv2.resize(cv_image, (320, 240))
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2YUV)
    cv_image = cv_image[120:240, :, :]
    cv_image[:,:,0] = cv2.equalizeHist(cv_image[:,:,0])
    cv_image = ((cv_image-(255.0/2))/255.0)
    return cv_image

main()
