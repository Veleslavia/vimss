import math
import os
import random

from absl import flags
import tensorflow as tf
# options for reading wav file
# from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
# import soundfile as sf

import librosa
from google.cloud import storage


flags.DEFINE_string(
    'project', 'plated-dryad-162216', 'Google cloud project id for uploading the dataset.')
flags.DEFINE_string(
    'gcs_output_path', 'gs://vimsstfrecords/musdb18', 'GCS path for uploading the dataset.')
flags.DEFINE_string(
    'local_scratch_dir', '/tmp', 'Scratch directory path for temporary files.')
flags.DEFINE_string(
    'raw_data_dir', '/mnt/disks/vimsstmp/musdb18', 'Directory path for raw MUSDB dataset. '
    'Should have train and test subdirectories inside it.')


"""
This script converts wav files of MusDB dataset to TF records.
An important difference between ImageNet and MusDB is that we should split individual recordings into small chunk of
size input_size. Perhaps, if we want to use bigger input_size later, if should be a multiple of input_size.

"""

FLAGS = flags.FLAGS

TRAINING_DIRECTORY = 'train'
TEST_DIRECTORY = 'test'

TRAINING_SHARDS = 24
TEST_SHARDS = 12

CHANNEL_NAMES = ['.stem_mix.wav', '.stem_vocals.wav', '.stem_bass.wav', '.stem_drums.wav', '.stem_other.wav']
SAMPLE_RATE = 22050     # Set a fixed sample rate
NUM_SAMPLES = 16384     # get from parameters of the model
CHANNELS = 1            # always work with mono!
NUM_SOURCES = 4         # fix 4 sources for musdb + mix
CACHE_SIZE = 16         # load 16 audio files in memory, then shuffle examples and write a tf.record


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


def _convert_to_example(filename, sample_idx, data_buffer,
                        sample_rate=SAMPLE_RATE, channels=CHANNELS,
                        num_sources=NUM_SOURCES, num_samples=NUM_SAMPLES):
    """Creating a training or testing example. These examples are aggregated later in a batch.
    Each data example should consist of [mix, bass, drums, other, vocals] data and corresponding metadata
    Each data example should have the same input_size (from base 16k to 244k samples), it needs to be fixed.

    data_buffer here is a vector of size num_samples*(num_sources+1), the first channel is always "mix"

    """
    example = tf.train.Example(features=tf.train.Features(feature={
        'audio/file_basename': _bytes_feature(os.path.basename(filename)),
        'audio/sample_rate': _int64_feature(sample_rate),
        'audio/sample_idx': _int64_feature(sample_idx),
        'audio/num_samples': _int64_feature(num_samples),
        'audio/channels': _int64_feature(channels),
        'audio/num_sources': _int64_feature(num_sources),
        'audio/encoded': _sources_floatlist_feature(data_buffer)}))
    return example


def _get_segments_from_audio_cache(file_data_cache):
    """
    Args:
        file_data_cache: list of raw audio files, mix and 4 sources data: [filename, len(data), data]
    Returns:
         segments: k segments of raw data
            each one contains file_basename, sample_idx, 5 raw data audio frames in a single list
    """
    segments = list()
    for sample_idx in range(file_data_cache[0][1]//SAMPLE_RATE): # sampling segments, ignore the last incomplete segment
        segments_data = list()
        for source in file_data_cache:
            segments_data.append(source[2][sample_idx*SAMPLE_RATE:(sample_idx+1)*SAMPLE_RATE])
        segments.append([file_data_cache[0][0], sample_idx, segments_data])
    return segments


def _process_audio_files_batch(chunk_files, output_file):
    """Processes and saves list of audio files as TFRecords.
    Args:
        chunk_files: list of strings; each string is a path to an image file
        coder: instance of AudioCoder to provide audio coding utils.
        output_file: string, unique identifier specifying the data set
    """

    # Get training files from the directory name

    writer = tf.python_io.TFRecordWriter(output_file)

    chunk_data_cache = list()
    for filename in chunk_files:
        # load all wave files into memory and create a buffer
        file_data_cache = list()
        for source in CHANNEL_NAMES:
            data, sr = librosa.core.load(filename+source, sr=SAMPLE_RATE, mono=True)
            file_data_cache.append([filename, len(data), data])

            # Option 1: use only tf to read and resample audio
            # audio_binary = tf.read_file(filename+source)
            # wav_decoder = contrib_audio.decode_wav(
            #     audio_binary,
            #     desired_channels=CHANNELS)
            # Option 2: use Soundfile and read binary files
            # SoundFile should be much more faster but it doesn't matter because we store everything in tf.records
            # with sf.SoundFile(filename+source, "r") as f:
            #     print(filename+source, f.samplerate, f.channels, len(f), f.read().tobytes())

        for segment in _get_segments_from_audio_cache(file_data_cache):
            chunk_data_cache.append(segment)

    # shuffle all segments
    shuffle_idx = make_shuffle_idx(len(chunk_data_cache))
    chunk_data_cache = [chunk_data_cache[i] for i in shuffle_idx]

    for chunk in chunk_data_cache:
        example = _convert_to_example(filename=chunk[0], sample_idx=chunk[1], data_buffer=chunk[2])
        writer.write(example.SerializeToString())

    writer.close()


def _process_dataset(filenames,
                     output_directory,
                     prefix,
                     num_shards):
    """Processes and saves list of audio files as TFRecords.
    Args:
    filenames: list of strings; each string is a path to an audio file
    channel_names: list of strings; each string is a channel name (vocals, bass, drums etc)
    labels: map of string to integer; id for all channel name
    output_directory: path where output files should be created
    prefix: string; prefix for each file
    num_shards: number of chucks to split the filenames into
    Returns:
    files: list of tf-record filepaths created from processing the dataset.
    """
    _check_or_create_dir(output_directory)
    chunksize = int(math.ceil(len(filenames) / num_shards))

    files = []

    for shard in range(num_shards):
        chunk_files = filenames[shard * chunksize: (shard + 1) * chunksize]
        output_file = os.path.join(
            output_directory, '%s-%.5d-of-%.5d' % (prefix, shard, num_shards))
        _process_audio_files_batch(chunk_files, output_file)

        tf.logging.info('Finished writing file: %s' % output_file)
        files.append(output_file)

    return files


def convert_to_tf_records(raw_data_dir):
    """Convert the MusDB dataset into TF-Record dumps."""

    # Glob all the training files
    training_files = tf.gfile.Glob(
        os.path.join(raw_data_dir, TRAINING_DIRECTORY, '*.wav'))
    training_files = list(set([filename.split('.')[0] for filename in training_files]))

    # Glob all the validation files
    test_files = sorted(tf.gfile.Glob(
        os.path.join(raw_data_dir, TEST_DIRECTORY, '*.wav')))
    test_files = list(set([filename.split('.')[0] for filename in test_files]))

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
        for i, filename in enumerate(sorted(filenames)):
            blob = bucket.blob(key_prefix + os.path.basename(filename))
            blob.upload_from_filename(filename)
            if not i % 20:
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
    upload_to_gcs(training_records, test_records)


if __name__ == '__main__':
    tf.app.run()