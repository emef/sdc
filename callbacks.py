import time

from keras.callbacks import Callback

class SnapshotCallback(Callback):
    """
    Callback which saves the model snapshot to s3 on each epoch
    """
    def __init__(self,
                 model_to_save,
                 task_id,
                 only_keep_best=True,
                 score_metric='val_categorical_accuracy'):
        self.model_to_save = model_to_save
        self.task_id = task_id
        self.only_keep_best = only_keep_best
        self.score_metric = score_metric
        self.best = None

    def on_epoch_end(self, epoch, logs):
        if self.only_keep_best:
            score = logs.get(self.score_metric)
            if self.best is None or score > self.best:
                self.best = score
            else:
                logger.info(
                    'Not snapshotting: %.2f less than previous %.2f',
                    score, self.best)
                return

        self.model_to_save.save(task_id)


class TimedEarlyStopping(Callback):
    """
    Stop training after N minutes.
    """
    def __init__(self, duration_minutes):
        self.duration = duration_minutes
        self.started_at_ts = time.time()

    def on_batch_end(self, batch, logs):
        delta_ts = time.time() - self.started_at_ts
        delta_minutes = delta_ts / 60
        if delta_minutes >= self.duration:
            self.model.stop_training = True
