import math
import os
import random
import multiprocessing
from multiprocessing import Pool
from absl import flags
import tensorflow as tf
import numpy as np
# options for reading wav file
# from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
import soundfile as sf
import string

import librosa
from google.cloud import storage


flags.DEFINE_string(
    'project', os.environ["PROJECT_NAME"], 'Google cloud project id for uploading the dataset.')
flags.DEFINE_string(
    'gcs_output_path', 'gs://vimsstfrecords/sopjfm', 'GCS path for uploading the dataset.')
flags.DEFINE_string(
    'local_scratch_dir', os.path.expanduser('~/Downloads/sopjfm/tfrecords/'), 'Scratch directory path for temporary files.')
flags.DEFINE_string(
    'raw_data_dir', os.path.expanduser('~/Downloads/sopjfm/wav'), 'Directory path for raw SOP dataset. ')
flags.DEFINE_string(
    'valid_audio_segments', os.path.expanduser('~/Downloads/sopjfm/segment_times.npy'), 'File path for SOP valid segments.')


"""
This script converts wav files of Felipe SOP dataset to TF records.
We should split individual recordings into small chunks of size input_size.
Valid segments provided in os.path.expanduser('~/Downloads/sopjfm/segment_times.npy').
We have 472 recordings in total which leads to 111156 possible combinations of 2.
We can't store all those combinations because it will be about 1TB of data (audio only).
We can store the audio segments though and sample 2 random examples + make sum of them at loading time.
Notice, that the sum is just sum and clipping will happen sometimes.

"""

FLAGS = flags.FLAGS

TRAINING_DIRECTORY = 'train'
TEST_DIRECTORY = 'test'

TRAINING_SHARDS = 24
TEST_SHARDS = 12

CHANNEL_NAMES = ['cello', 'clarinet', 'erhu', 'flute', 'trumpet', 'tuba', 'violin', 'xylophone',
                 'cf', 'clc', 'clf', 'clt', 'cltu', 'clv', 'ct', 'tf', 'tut', 'tuv', 'vc', 'vf', 'vt']

MIX_WITH_PADDING = 147443
SAMPLE_RATE = 22050     # Set a fixed sample rate
NUM_SAMPLES = 16384     # get from parameters of the model
CHANNELS = 1            # always work with mono!
NUM_SOURCES = 8         # 8 unique sources + 14 duets
CACHE_SIZE = 16         # load 16 audio files in memory, then shuffle examples and write a tf.record

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


def make_shuffle_idx(n):
    order = list(range(n))
    random.shuffle(order)
    return order


def _check_or_create_dir(directory):
    """Check if directory exists otherwise create it."""
    if not tf.gfile.Exists(directory):
        tf.gfile.MakeDirs(directory)


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _floatlist_feature(value):
    """Wrapper for inserting float list features into Example proto."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _sources_floatlist_feature(value):
    """Wrapper for inserting list of sources into Example proto. """
    flatten = [item for sublist in value for item in sublist]
    return tf.train.Feature(float_list=tf.train.FloatList(value=flatten))


def _convert_to_example(filename, sample_idx, data_buffer, num_sources, labels,
                        sample_rate=SAMPLE_RATE, channels=CHANNELS, num_samples=NUM_SAMPLES):
    """Creating a training or testing example. These examples are aggregated later in a batch.
    Each data example should consist of [mix, bass, drums, other, vocals] data and corresponding metadata
    Each data example should have the same input_size (from base 16k to 244k samples), it needs to be fixed.

    data_buffer here is a vector of size num_samples*(num_sources+1), the first channel is always "mix"

    """
    example = tf.train.Example(features=tf.train.Features(feature={
        'audio/file_basename': _bytes_feature(filename),
        'audio/sample_rate': _int64_feature(sample_rate),
        'audio/sample_idx': _int64_feature(sample_idx),
        'audio/num_samples': _int64_feature(num_samples),
        'audio/channels': _int64_feature(channels),
        'audio/num_sources': _int64_feature(num_sources),
        'audio/labels': _int64_feature(labels),
        'audio/source_names': _bytes_feature(filename.rstrip(string.digits)),
        'audio/encoded': _sources_floatlist_feature(data_buffer)}))
    return example


def _get_segments_from_audio_cache(filename, data):
    """
    Args:
        filename: track file basename
        data: raw audio waveform to split onto segments
    Returns:
         segments: k segments of raw data. Each segment contains file_basename,
            sample_idx, 2 raw data audio frames, an expected input and output of the network
    """

    segments = list()
    offset = (MIX_WITH_PADDING - NUM_SAMPLES)//2
    start_idx = offset
    end_idx = len(data) - offset - 1
    for sample_idx in range((end_idx - start_idx)//NUM_SAMPLES):
        # sampling segments, ignore first and last incomplete segments
        # notice that we sample MIX_WITH_PADDING from the mix and central cropped NUM_SAMPLES from the sources
        segments_data = list()
        sample_offset_start = start_idx + sample_idx*NUM_SAMPLES
        sample_offset_end = start_idx + (sample_idx+1)*NUM_SAMPLES
        # adding big datasample for mix
        segments_data.append(data[sample_offset_start-offset:sample_offset_end+offset+1])
        assert len(segments_data[0]) == MIX_WITH_PADDING
        # adding "ground truth", which is a central crop of the big segment
        segments_data.append(data[sample_offset_start:sample_offset_end])
        segments.append([filename, sample_idx, segments_data])
    return segments


def _process_audio_files_batch(chunk_data):
    """Processes and saves list of audio files as TFRecords.
    Args:
        chunk_data: tuple of chunk_files and output_file
        chunk_files: list of strings; each string is a path to an wav file
        output_file: string, unique identifier specifying the data set
    """

    chunk_files, output_file = chunk_data[0], chunk_data[1]

    data_segments = np.load(FLAGS.valid_audio_segments)
    # create a dict (k, v) = (filename, (start, end))
    segment_dict = dict([(item['file'], (item['to'][0], item['tf'][-1])) for item in data_segments if
                         (len(item['to']) > 0 and len(item['tf']) > 0)])

    writer = tf.python_io.TFRecordWriter(output_file)

    chunk_data_cache = list()
    for track_filename in chunk_files:
        track_basename = os.path.basename(track_filename).split('.')[0]
        if track_basename not in segment_dict.keys():
            print("Segments not found for ", track_basename)
            continue

        # load a wave file into memory and create a buffer
        data, sr = sf.read(track_filename)

        data_to_process = data[sr*segment_dict[track_basename][0]: sr*segment_dict[track_basename][1]]

        for segment in _get_segments_from_audio_cache(track_basename, data_to_process):
            chunk_data_cache.append(segment)

    # shuffle all segments
    shuffle_idx = make_shuffle_idx(len(chunk_data_cache))
    chunk_data_cache = [chunk_data_cache[i] for i in shuffle_idx]

    for chunk in chunk_data_cache:

        labels = source_map[chunk[0].rstrip(string.digits)]
        example = _convert_to_example(filename=chunk[0], sample_idx=chunk[1],
                                      data_buffer=chunk[2], num_sources=1,  # only big original and small original
                                      labels=labels)
        writer.write(example.SerializeToString())

    writer.close()
    tf.logging.info('Finished writing file: %s' % output_file)


def _process_dataset(filenames,
                     output_directory,
                     prefix,
                     num_shards):
    """Processes and saves list of audio files as TFRecords.
    Args:
    filenames: list of strings; each string is a path to an audio file
    labels: map of string to integer; id for all channel name
    output_directory: path where output files should be created
    prefix: string; prefix for each file
    num_shards: number of chucks to split the filenames into
    Returns:
    files: list of tf-record filepaths created from processing the dataset.
    """
    _check_or_create_dir(output_directory)
    chunksize = int(math.ceil(len(filenames) / float(num_shards)))

    pool = Pool(multiprocessing.cpu_count()-1)

    def output_file(shard_idx):
        return os.path.join(output_directory, '%s-%.5d-of-%.5d' % (prefix, shard_idx, num_shards))

    # chunk data consists of chunk_filenames and output_file
    chunk_data = [(filenames[shard * chunksize: (shard + 1) * chunksize],
                  output_file(shard)) for shard in range(num_shards)]

    files = pool.map(_process_audio_files_batch, chunk_data)

    return files


def convert_to_tf_records(raw_data_dir):
    """Convert the dataset into TF-Record dumps."""

    # Glob all the training files
    training_files = tf.gfile.Glob(
        os.path.join(raw_data_dir, TRAINING_DIRECTORY, '*.wav'))
    # training_files = list(set([filename.split('.')[0] for filename in training_files]))

    # Glob all the validation files
    test_files = sorted(tf.gfile.Glob(
        os.path.join(raw_data_dir, TEST_DIRECTORY, '*.wav')))
    # test_files = list(set([filename.split('.')[0] for filename in test_files]))

    # Create training data
    tf.logging.info('Processing the training data.')
    training_records = _process_dataset(training_files,
                                        os.path.join(FLAGS.local_scratch_dir, TRAINING_DIRECTORY),
                                        TRAINING_DIRECTORY, TRAINING_SHARDS)

    # Create validation data
    tf.logging.info('Processing the validation data.')
    test_records = _process_dataset(test_files,
                                    os.path.join(FLAGS.local_scratch_dir, TEST_DIRECTORY),
                                    TEST_DIRECTORY, TEST_SHARDS)

    return training_records, test_records


def upload_to_gcs(training_records, test_records):
    """Upload TF-Record files to GCS, at provided path."""

    # Find the GCS bucket_name and key_prefix for dataset files
    path_parts = FLAGS.gcs_output_path[5:].split('/', 1)
    bucket_name = path_parts[0]
    if len(path_parts) == 1:
        key_prefix = ''
    elif path_parts[1].endswith('/'):
        key_prefix = path_parts[1]
    else:
        key_prefix = path_parts[1] + '/'

    client = storage.Client(project=FLAGS.project)
    bucket = client.get_bucket(bucket_name)

    def _upload_files(filenames):
        """Upload a list of files into a specifc subdirectory."""
        for i, filename in enumerate(filenames):
            blob = bucket.blob(key_prefix + os.path.basename(filename))
            blob.upload_from_filename(filename)
            if not i % 5:
                tf.logging.info('Finished uploading file: %s' % filename)

    # Upload training dataset
    tf.logging.info('Uploading the training data.')
    _upload_files(training_records)

    # Upload validation dataset
    tf.logging.info('Uploading the validation data.')
    _upload_files(test_records)


def main(argv):  # pylint: disable=unused-argument
    tf.logging.set_verbosity(tf.logging.INFO)

    if FLAGS.project is None:
        raise ValueError('GCS Project must be provided.')

    if FLAGS.gcs_output_path is None:
        raise ValueError('GCS output path must be provided.')
    elif not FLAGS.gcs_output_path.startswith('gs://'):
        raise ValueError('GCS output path must start with gs://')

    if FLAGS.local_scratch_dir is None:
        raise ValueError('Scratch directory path must be provided.')

    # Download the dataset if it is not present locally
    raw_data_dir = FLAGS.raw_data_dir

    # Convert the raw data into tf-records
    training_records, test_records = convert_to_tf_records(raw_data_dir)

    # Upload to GCS
    # upload_to_gcs(training_records, test_records)


if __name__ == '__main__':
    tf.app.run()
