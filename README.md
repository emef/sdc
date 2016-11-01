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

### connecting to an network attached device for a dataset
```
sudo apt-get install -y open-iscsi
export INITIATOR_NAME=`hostname`
export TARGET_IP=172.30.0.108
sudo echo "InitiatorName=iqn.2016-10.local.nalapati:$INITIATOR_NAME" > /etc/iscsi/initiatorname.iscsi
sudo systemctl restart iscsid open-iscsi
sudo iscsiadm -m discovery -t sendtargets -p $TARGET_IP
sudo iscsiadm -m node --login
sudo cat /proc/partitions
sudo mkdir /media/drive
sudo chown ubuntu /media/drive
sudo chgrp ubuntu /media/drive
sudo mount /dev/sda1 /media/drive
```
NOTE: The device name could be different, if mounting /dev/sda1 fails, inspect the outout of cat /proc/partitions to mount the correct device.

### Encoding images in a directory into a video
cd directory_with_images/
mencoder "mf://*.png" -mf type=png:fps=20 -o /home/ubuntu/output-steering-reg.mpg -speed 1 -ofps 20 -ovc lavc -lavcopts vcodec=mpeg2video:vbitrate=2500 -oac copy -of mpeg
