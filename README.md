### task config
```python
{
  'dataset_uri': 's3://sdc-matt/datasets/sdc_processed_1',
  'output_uri': 's3://',
  'model_config': {
    'type': 'ensemble',
    'timesteps': 3,
    'model_uri': 's3://sdc-matt/ensemble-model.h5',
    'input_model_config': {
      'type': 'simple',
      'model_uri': 's3://sdc-matt/sample-trained-encoder.h5',
    },
  },
  'training_args': {
    'batch_size': 100,
    'validation_size': 500,
    'epoch_size': 1000,
    'epochs': 20,
  }
```
