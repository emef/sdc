"""
Evaluate models on the test set.
"""

from datasets import load_dataset
from models import load_from_config

if __name__ == '__main__':
    model_configs = [
        {
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
        },
        {
            'type': 'simple',
            'model_uri': 's3://sdc-matt/simple/1477715388/model.h5',
            'cat_classes': 5,
        },
    ]

    dataset = load_dataset('s3://sdc-matt/datasets/final_training')

    print '[baseline] %.4f' % dataset.get_baseline_mse()
    for i, model_config in enumerate(model_configs):
        model = load_from_config(model_config)
        metrics = model.evaluate(dataset)
        print '[%] ' + ', '.join('%.4f' % m for m in metrics)
