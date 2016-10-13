"""
Loading/saving datasets.
"""
import logging, os, shutil, subprocess

import numpy as np
import requests

from util import download_dir, parse_s3_uri, upload_dir

logger = logging.getLogger(__name__)


class Dataset(object):
    """
    Used to create (image, label) tuple generators of an image dataset.
    """
    def __init__(self,
                 images_base_path,
                 labels,
                 training_indexes,
                 testing_indexes,
                 validation_indexes,
                 image_file_fmt='%d.png.npy'):
        """
        @param images_base_path - path to image files
        @param labels - 2d array of all label data
        @param training_indexes - 1d array of training indexes
        @param testing_indexes - 1d array of testing indexes
        @param validation_indexes - 1d array of validation indexes
        @param image_file_fmt - format string for image file names
        """
        self.images_base_path = images_base_path
        self.labels = labels
        self.training_indexes = training_indexes
        self.testing_indexes = testing_indexes
        self.validation_indexes = validation_indexes
        self.image_file_fmt = image_file_fmt

    def get_training_size(self):
        return len(self.training_indexes)

    def get_testing_size(self):
        return len(self.testing_indexes)

    def get_validation_size(self):
        return len(self.validation_indexes)

    def training_generator(self):
        return InfiniteImageLoadingGenerator(
            self.training_indexes,
            self.labels,
            self.images_base_path,
            self.image_file_fmt)

    def testing_generator(self):
        return InfiniteImageLoadingGenerator(
            self.testing_indexes,
            self.labels,
            self.images_base_path,
            self.image_file_fmt)

    def validation_generator(self):
        return InfiniteImageLoadingGenerator(
            self.validation_indexes,
            self.labels,
            self.images_base_path,
            self.image_file_fmt)


class InfiniteImageLoadingGenerator(object):
    """
    Iterable object which loads the next (image, label) tuple in
    the data set.
    """
    def __init__(self, indexes, labels, images_base_path, image_file_fmt):
        """
        @param indexes - array (N,) of image index IDs
        @param labels - array (N,) of corresponding labels
        @param images_base_path - local path to directory containing images
        @param image_file_fmt - format string (of image ID) for image filename
        """
        self.current_index = 0
        self.indexes = indexes
        self.labels = labels
        self.images_base_path = images_base_path
        self.image_file_fmt = image_file_fmt

    def __iter__(self):
        return self

    def next(self):
        next_index = self.indexes[self.current_index]
        image_path = os.path.join(
            self.images_base_path,
            self.image_file_fmt % next_index)

        image = np.load(image_path)
        label = self.labels[self.current_index]

        max_index = len(self.labels)
        self.current_index = (self.current_index + 1) % max_index

        return (image, label)

def load_dataset(s3_uri, cache_dir='/tmp'):
    """
    Downloads and loads an image dataset.

    A dataset directory should have this structure:
      /
          training_indexes.npy
          testing_indexes.npy
          validation_indexes.npy
          labels.npy
          images/
              1.png.npy
              2.png.npy
              ...
              N.png.npy

    @param s3_uri - formatted as s3://bucket/key/path.tar.gz
    @param cache_dir - local dir to cache datasets
    """
    _, s3_dir = parse_s3_uri(s3_uri)
    dataset_path = os.path.join(cache_dir, s3_dir)

    # Ensure we have the dataset downloaded locally
    if os.path.exists(dataset_path):
        logger.info('Dataset %s exists, using cache' % s3_dir)
    else:
        logger.info('Downloading dataset %s' % s3_dir)
        download_dir(s3_uri, dataset_path)

    # Load the dataset from the local directory
    labels = np.load(os.path.join(dataset_path, 'labels.npy'))
    training_indexes = np.load(
        os.path.join(dataset_path, 'training_indexes.npy'))
    testing_indexes = np.load(
        os.path.join(dataset_path, 'testing_indexes.npy'))
    validation_indexes = np.load(
        os.path.join(dataset_path, 'validation_indexes.npy'))

    images_base_path = os.path.join(dataset_path, 'images')

    return Dataset(
        images_base_path=images_base_path,
        labels=labels,
        training_indexes=training_indexes,
        testing_indexes=testing_indexes,
        validation_indexes=validation_indexes,
        image_file_fmt='%d.png.npy')


def prepare_dataset(
        archive_url,
        local_output_path,
        output_s3_uri,
        training_percent=0.7,
        testing_percent=0.2,
        validation_percent=0.1,
        cache_dir='/tmp'):
    """
    Prepare dataset from nishanth's preprocessed format.

    @param archive_url - url to download dataset archive from
    @param local_output_path - where to write prepared dataset locally
    @param output_s3_uri - where to write prepared dataset in s3
    @param training_percent - percent of samples used in training set
    @param testing_percent - percent of samples used in testing set
    @param validation_percent - percent of samples used in validation set
    @param cache_dir - where to store intermediate archives
    """
    filename = archive_url.split('/')[-1]
    archive_path = os.path.join(cache_dir, filename)
    local_raw_path = os.path.join(cache_dir, filename.split('.')[0])

    # download/decompress archive if necessary
    if not os.path.exists(local_raw_path):
        logger.info('Downloading dataset archive from %s', archive_url)

        # download
        r = requests.get(archive_url, stream=True)
        with open(archive_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=4096):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

        # decompress archive
        logger.info('Decompressing dataset archive %s to %s',
                     filename, local_raw_path)

        try: os.makedirs(local_raw_path)
        except: pass

        subprocess.call(['tar', 'xzf', archive_path, '-C', local_raw_path])

    dir_list = list(os.listdir(local_raw_path))
    assert len(dir_list) == 2
    assert 'labels' in dir_list

    images_dirname, = [f for f in dir_list if f != 'labels']
    base_images_path = os.path.join(local_raw_path, images_dirname)
    logger.info('Using %s as base images directory', base_images_path)

    with open(os.path.join(local_raw_path, 'labels')) as labels_f:
        labels = np.array([float(line.strip()) for line in labels_f])

    n_samples = len(labels)
    n_training = int(training_percent * n_samples)
    n_testing = int(testing_percent * n_samples)
    n_validation = n_samples - n_training - n_testing

    logger.info('%d total samples in the dataset', n_samples)
    logger.info('%d samples in training set', n_training)
    logger.info('%d samples in testing set', n_testing)
    logger.info('%d samples in validation set', n_validation)

    indexes = np.arange(0, n_samples)
    np.random.shuffle(indexes)

    training_indexes = indexes[:n_training]
    testing_indexes = indexes[n_training:(n_training + n_testing)]
    validation_indexes = indexes[-n_validation:]

    shutil.rmtree(local_output_path, ignore_errors=True)
    os.makedirs(local_output_path)

    # create the properly-formatted dataset directory
    np.save(os.path.join(local_output_path, 'labels.npy'), labels)
    np.save(
        os.path.join(local_output_path, 'training_indexes.npy'),
        training_indexes)
    np.save(
        os.path.join(local_output_path, 'testing_indexes.npy'),
        testing_indexes)
    np.save(
        os.path.join(local_output_path, 'validation_indexes.npy'),
        validation_indexes)
    shutil.copytree(
        base_images_path,
        os.path.join(local_output_path, 'images'))

    # upload dataset directory to s3
    upload_dir(local_output_path, output_s3_uri)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    if False:
        prepare_dataset(
            'https://s3-us-west-1.amazonaws.com/sdc-datasets/sdc_dataset_1.tar.gz',
            '/tmp/sdc_processed_1',
            's3://sdc-matt/datasets/sdc_processed_1')

    dataset = load_dataset('s3://sdc-matt/datasets/sdc_processed_1')
    training_generator = dataset.training_generator()
    img, label = training_generator.next()

    print img.max()
