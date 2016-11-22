import logging, os, time

from keras.callbacks import Callback

logger = logging.getLogger(__name__)


class SnapshotCallback(Callback):
    """
    Callback which saves the model snapshot to s3 on each epoch
    """
    def __init__(self,
                 model_to_save,
                 snapshot_dir,
                 only_keep_best=True,
                 score_metric='val_categorical_accuracy'):
        self.model_to_save = model_to_save
        self.snapshot_dir = snapshot_dir
        self.only_keep_best = only_keep_best
        self.score_metric = score_metric
        self.best = None
        self.best_path = None
        self.nb = 0

        logger.info('Saving snapshots to %s', snapshot_dir)

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

        model_path = os.path.join(self.snapshot_dir, '%d.h5' % self.nb)

        self.best_path = model_path
        self.model_to_save.save(model_path)
        self.nb += 1

        logger.info('Snapshotted model to %s', model_path)


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
