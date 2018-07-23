""" Credits to https://github.com/tensorflow/tpu/blob/master/models/official/resnet/imagenet_input.py
MusDB18 input pipeline using tf.data.Dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
import functools

#bn, cl, db, fl, hn, ob, sax, tba, tbn, tbt, va, vc, vn
CHANNEL_NAMES = ['.stem_mix.wav', '.stem_bn.wav', '.stem_cl.wav', '.stem_db.wav', '.stem_fl.wav', '.stem_hn.wav', '.stem_ob.wav',
                 '.stem_sax.wav', '.stem_tba.wav', '.stem_tbn.wav', '.stem_tbt.wav', '.stem_va.wav', '.stem_vc.wav', '.stem_vn.wav']
SAMPLE_RATE = 22050     # Set a fixed sample rate
NUM_SAMPLES = 16384     # get from parameters of the model
MIX_WITH_PADDING = 147443
CHANNELS = 1            # always work with mono!
NUM_SOURCES = 13         # fix 13 sources for urmp + mix
CACHE_SIZE = 16         # load 16 audio files in memory, then shuffle examples and write a tf.record


class URMPInput(object):
    """Generates URMP input_fn for training or evaluation.
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

    def __init__(self, is_training, data_dir, use_bfloat16=False, transpose_input=False):
        self.is_training = is_training
        self.use_bfloat16 = use_bfloat16
        self.data_dir = data_dir
        if self.data_dir == 'null' or self.data_dir == '':
            self.data_dir = None
        self.transpose_input = transpose_input

    def set_shapes(self, batch_size, mix, sources):
        """Statically set the batch_size dimension."""

        features['mix'].set_shape(features['mix'].get_shape().merge_with(
            tf.TensorShape([batch_size, None, None])))
        sources.set_shape(sources.get_shape().merge_with(
            tf.TensorShape([batch_size, None, None, None])))
        features['filename'].set_shape(features['filename'].get_shape().merge_with(
            tf.TensorShape([batch_size])))
        features['sample_id'].set_shape(features['sample_id'].get_shape().merge_with(
            tf.TensorShape([batch_size])))

        return features, sources

    def dataset_parser(self, value):
        """Parse an audio example record from a serialized string Tensor."""
        keys_to_features = {
            'audio/filename':
                tf.FixedLenFeature([], tf.string, ''),
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
            'audio/num_sources':
                tf.FixedLenFeature([], tf.int64, NUM_SOURCES)
        }

        parsed = tf.parse_single_example(value, keys_to_features)
        audio_data = tf.sparse_tensor_to_dense(parsed['audio/encoded'], default_value=0)
        #audio_shape = tf.stack([NUM_SOURCES+1, NUM_SAMPLES])
        audio_shape = tf.stack([MIX_WITH_PADDING + NUM_SOURCES*NUM_SAMPLES])
        audio_data = tf.reshape(audio_data, audio_shape)
        mix, sources = tf.reshape(audio_data[:MIX_WITH_PADDING], tf.stack([MIX_WITH_PADDING, CHANNELS])),
                       tf.reshape(audio_data[MIX_WITH_PADDING:], tf.stack([NUM_SAMPLES, CHANNELS, NUM_SOURCES]))
        features = {'mix': mix, 'filename': parsed['audio/filename'], 'sample_id': parsed['audio/sample_idx']}
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

        # Retrieves the batch size for the current shard. The # of shards is
        # computed according to the input pipeline deployment. See
        # tf.contrib.tpu.RunConfig for details.
        batch_size = params['batch_size']

        # Shuffle the filenames to ensure better randomization.
        file_pattern = os.path.join(
            self.data_dir, 'train-*' if self.is_training else 'test-*')
        dataset = tf.data.Dataset.list_files(file_pattern, shuffle=self.is_training)

        if self.is_training:
            dataset = dataset.repeat()

        def fetch_dataset(filename):
            buffer_size = 128 * 1024 * 1024     # 128 MiB cached data per file
            dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
            return dataset

        # Read the data from disk in parallel
        dataset = dataset.apply(
            tf.contrib.data.parallel_interleave(
                fetch_dataset, cycle_length=6, sloppy=True))
        dataset = dataset.shuffle(1024, reshuffle_each_iteration=True)

        # Parse, preprocess, and batch the data in parallel
        dataset = dataset.apply(
            tf.contrib.data.map_and_batch(
                self.dataset_parser, batch_size=batch_size,
                num_parallel_batches=8,    # 8 == num_cores per host
                drop_remainder=True))

        # Transpose for performance on TPU
        # TODO what does it do?
        # if self.transpose_input:
        #     dataset = dataset.map(
        #         lambda images, labels: (tf.transpose(images, [1, 2, 3, 0]), labels),
        #         num_parallel_calls=8)

        # Assign static batch size dimension
        dataset = dataset.map(functools.partial(self.set_shapes, batch_size))

        # Prefetch overlaps in-feed with training
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
        return dataset
