""" Credits to https://github.com/tensorflow/tpu/blob/master/models/official/resnet/imagenet_input.py
MusDB18 input pipeline using tf.data.Dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
import functools

CHANNEL_NAMES = ['cello', 'clarinet', 'erhu', 'flute', 'trumpet', 'tuba', 'violin', 'xylophone',
                 'cf', 'clc', 'clf', 'clt', 'cltu', 'clv', 'ct', 'tf', 'tut', 'tuv', 'vc', 'vf', 'vt']

"""
Operations for source_map elements will be the following:
    a = source_map['clarinet'] 
    b = source_map['tut']
    # a | b represents vector of sources for both examples
    # format(a|b, "#010b") is a binary string with leading zeros, between '0b00000000' and '0b11111111'
    labels = [int(i) for i in format(a|b, "#010b")[2:]]
"""

source_map = {
    'cello':    0b00000001,
    'clarinet': 0b00000010,
    'erhu':     0b00000100,
    'flute':    0b00001000,
    'trumpet':  0b00010000,
    'tuba':     0b00100000,
    'violin':   0b01000000,
    'xylophone':0b10000000,
    'cf':       0b00001001,
    'clc':      0b00000011,
    'clf':      0b00001010,
    'clt':      0b00010010,
    'cltu':     0b00100010,
    'clv':      0b01000010,
    'ct':       0b00010001,
    'tf':       0b00011000,
    'tut':      0b00110000,
    'tuv':      0b01100000,
    'vc':       0b01000001,
    'vf':       0b01001000,
    'vt':       0b01010000,
    'xf':       0b10001000
}

BASENAMES = []

SAMPLE_RATE = 22050     # Set a fixed sample rate
NUM_SAMPLES = 16384     # get from parameters of the model
MIX_WITH_PADDING = 147443
CHANNELS = 1            # always work with mono!
NUM_SOURCES = 1         # 8 unique sources and 14 duets, but it's hardcoded here
CACHE_SIZE = 16         # load 16 audio files in memory, then shuffle examples and write a tf.record


class SOPInput(object):
    """Generates SOP input_fn for training or evaluation.
    The training data is assumed to be in TFRecord format with keys as specified
    in the dataset_parser below, sharded across 0024 files, named sequentially:
      train-00000-of-00024
      train-00001-of-00024
      ...
      train-00023-of-00024
    The validation data is in the same format but sharded in 12 files.
    The format of the data required is the following:
    example = tf.train.Example(features=tf.train.Features(feature={
        'audio/file_basename': _bytes_feature(os.path.basename(filename)),
        'audio/sample_rate': _int64_feature(sample_rate),
        'audio/sample_idx': _int64_feature(sample_idx),
        'audio/num_samples': _int64_feature(num_samples),
        'audio/channels': _int64_feature(channels),
        'audio/num_sources': _int64_feature(num_sources),
        'audio/encoded': _sources_floatlist_feature(data_buffer)}))
        data_buffer here is a vector of size num_samples*(num_sources+1), the first channel is always "mix"
    Args:
    is_training: `bool` for whether the input is for training
    data_dir: `str` for the directory of the training and validation data
    use_bfloat16: If True, use bfloat16 precision; else use float32.
    transpose_input: 'bool' for whether to use the double transpose trick # what is that??
    """

    def __init__(self, mode, data_dir, use_bfloat16=False, transpose_input=False):
        self.mode = mode
        self.use_bfloat16 = use_bfloat16
        self.data_dir = data_dir
        if self.data_dir == 'null' or self.data_dir == '':
            self.data_dir = None
        self.transpose_input = transpose_input

    def set_shapes(self, batch_size, features, sources):
        """Statically set the batch_size dimension."""

        features['mix'].set_shape(features['mix'].get_shape().merge_with(
            tf.TensorShape([batch_size, None, None])))
        sources.set_shape(sources.get_shape().merge_with(
            tf.TensorShape([batch_size, None, None, None])))
        features['labels'].set_shape(features['labels'].get_shape().merge_with(
            tf.TensorShape([batch_size, None])))
        features['filename'].set_shape(features['filename'].get_shape().merge_with(
            tf.TensorShape([batch_size])))
        features['sample_id'].set_shape(features['sample_id'].get_shape().merge_with(
            tf.TensorShape([batch_size])))

        return features, sources

    def combine_sources(self, sample1, sample2):
        bits = tf.constant((128, 64, 32, 16, 8, 4, 2, 1), dtype=tf.uint8)
        features1, sources1 = sample1[0], sample1[1]
        features2, sources2 = sample2[0], sample2[1]

        # here we have features: mix, labels(int->bfloat16), filename, sample_id + one source
        mix = tf.add(features1['mix'], features2['mix'])
        # TODO fix it
        filename = tf.constant(0, tf.float32)
        sample_id = features1['sample_id']
        labels = tf.cast(tf.bitwise.bitwise_or(features1['labels'], features2['labels']), tf.uint8)
        labels = tf.divide(tf.reshape(tf.bitwise.bitwise_and(labels, bits), [-1]), bits)

        sources = tf.stack([sources1, sources2])

        if self.use_bfloat16:
            labels = tf.cast(labels, tf.bfloat16)
            filename = tf.constant(0, dtype=tf.bfloat16)

        features = {'mix': mix, 'filename': filename,
                    'sample_id': sample_id, 'labels': labels}
        return features, sources


    def dataset_parser(self, value):
        """Parse an audio example record from a serialized string Tensor."""
        keys_to_features = {
            'audio/file_basename':
                tf.FixedLenFeature([], tf.int64, -1),
            'audio/encoded':
                tf.VarLenFeature(tf.float32),
            'audio/sample_rate':
                tf.FixedLenFeature([], tf.int64, SAMPLE_RATE),
            'audio/sample_idx':
                tf.FixedLenFeature([], tf.int64, -1),
            'audio/num_samples':
                tf.FixedLenFeature([], tf.int64, NUM_SAMPLES),
            'audio/channels':
                tf.FixedLenFeature([], tf.int64, CHANNELS),
            'audio/labels':
                tf.FixedLenFeature([], tf.int64, -1),
            'audio/num_sources':
                tf.FixedLenFeature([], tf.int64, NUM_SOURCES),
            'audio/source_names':
                tf.FixedLenFeature([], tf.string, ''),
        }

        parsed = tf.parse_single_example(value, keys_to_features)
        audio_data = tf.sparse_tensor_to_dense(parsed['audio/encoded'], default_value=0)
        audio_shape = tf.stack([MIX_WITH_PADDING + NUM_SAMPLES])
        audio_data = tf.reshape(audio_data, audio_shape)
        mix, sources = tf.reshape(audio_data[:MIX_WITH_PADDING], tf.stack([MIX_WITH_PADDING, CHANNELS])), \
                       tf.reshape(audio_data[MIX_WITH_PADDING:], tf.stack([NUM_SAMPLES, CHANNELS]))
        labels = tf.cast(parsed['audio/labels'], tf.int16)
        # labels = tf.sparse_tensor_to_dense(parsed['audio/labels'])
        # labels = tf.reshape(labels, tf.stack([NUM_SOURCES]))

        if self.use_bfloat16:
            mix = tf.cast(mix, tf.bfloat16)
            sources = tf.cast(sources, tf.bfloat16)
        features = {'mix': mix, 'filename': parsed['audio/file_basename'],
                    'sample_id': parsed['audio/sample_idx'], 'labels': labels}
        return features, sources

    def input_fn(self, params):
        """Input function which provides a single batch for train or eval.
            Args:
                params: `dict` of parameters passed from the `TPUEstimator`.
                `params['batch_size']` is always provided and should be used as the
                effective batch size.
            Returns:
                A `tf.data.Dataset` object.
        """

        def fetch_dataset(filename):
            buffer_size = 128 * 1024 * 1024     # 128 MiB cached data per file
            dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
            return dataset

        def _setup_dataset(mode, file_pattern, dataset_parser):
            _dataset = tf.data.Dataset.list_files(file_pattern, shuffle=(mode == 'train'))
            if mode == 'train':
                _dataset = _dataset.repeat()
            # Read the data from disk in parallel
            _dataset = _dataset.apply(
                tf.contrib.data.parallel_interleave(
                    fetch_dataset, cycle_length=6, sloppy=True))
            _dataset = _dataset.shuffle(1024, reshuffle_each_iteration=True)
            # Parse, preprocess, and batch the data in parallel
            _dataset = _dataset.map(dataset_parser)

            return _dataset

        # Retrieves the batch size for the current shard. The # of shards is
        # computed according to the input pipeline deployment. See
        # tf.contrib.tpu.RunConfig for details.
        batch_size = params['batch_size']

        # Shuffle the filenames to ensure better randomization.
        file_pattern = os.path.join(
            self.data_dir, 'train-*' if self.mode == 'train' else 'test-*')

        dataset1 = _setup_dataset(self.mode, file_pattern, self.dataset_parser)
        dataset2 = _setup_dataset(self.mode, file_pattern, self.dataset_parser)

        # TODO create second dataset, then zip them and map apply mapping to make a single element
        # Create a dataset which returns two random elements from tf.records
        dataset = tf.data.Dataset.zip((dataset1, dataset2))

        # Apply mapping to create a sum of sources
        dataset = dataset.apply(
                tf.contrib.data.map_and_batch(
                    self.combine_sources,
                    batch_size=batch_size,
                    num_parallel_batches=8,    # 8 == num_cores per host
                    drop_remainder=True))

        # Assign static batch size dimension
        dataset = dataset.map(functools.partial(self.set_shapes, batch_size))

        # Prefetch overlaps in-feed with training
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
        return dataset
