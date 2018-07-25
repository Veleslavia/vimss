import math
import os
import random
import multiprocessing
from multiprocessing import Pool
from absl import flags
import tensorflow as tf
# options for reading wav file
# from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
# import soundfile as sf

import librosa
from google.cloud import storage


flags.DEFINE_string(
    'project', os.environ["PROJECT_NAME"], 'Google cloud project id for uploading the dataset.')
flags.DEFINE_string(
    'gcs_output_path', 'gs://vimsstfrecords/urmpv2-labels', 'GCS path for uploading the dataset.')
flags.DEFINE_string(
    'local_scratch_dir', '/dev/tfrecords/urmpv2', 'Scratch directory path for temporary files.')
flags.DEFINE_string(
    'raw_data_dir', '/home/olga/urmpv2', 'Directory path for raw URMP dataset. '
    'Should have train and test subdirectories inside it.')


"""
This script converts wav files of URMP dataset to TF records.
An important difference between ImageNet and MusDB is that we should split individual recordings into small chunk of
size input_size. Perhaps, if we want to use bigger input_size later, if should be a multiple of input_size.

"""

FLAGS = flags.FLAGS

TRAINING_DIRECTORY = 'train'
TEST_DIRECTORY = 'test'

TRAINING_SHARDS = 6
TEST_SHARDS = 3

CHANNEL_NAMES = ['.stem_mix.wav', '.stem_bn.wav', '.stem_cl.wav', '.stem_db.wav', '.stem_fl.wav', '.stem_hn.wav', '.stem_ob.wav',
                 '.stem_sax.wav', '.stem_tba.wav', '.stem_tbn.wav', '.stem_tpt.wav', '.stem_va.wav', '.stem_vc.wav', '.stem_vn.wav']

MIX_WITH_PADDING = 147443
SAMPLE_RATE = 22050     # Set a fixed sample rate
NUM_SAMPLES = 16384     # get from parameters of the model
CHANNELS = 1            # always work with mono!
NUM_SOURCES = 13         # fix 4 sources for musdb + mix
CACHE_SIZE = 16         # load 16 audio files in memory, then shuffle examples and write a tf.record

source_map = {
    'mix': 0,
    'bn': 1,
    'cl': 2,
    'db': 3,
    'fl': 4,
    'hn': 5,
    'ob': 6,
    'sax': 7,
    'tba': 8,
    'tbn': 9,
    'tpt': 10,
    'va': 11,
    'vc': 12,
    'vn': 13,
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
        'audio/file_basename': _bytes_feature("_".join((os.path.basename(filename[0])).split("_")[:3])),
        'audio/sample_rate': _int64_feature(sample_rate),
        'audio/sample_idx': _int64_feature(sample_idx),
        'audio/num_samples': _int64_feature(num_samples),
        'audio/channels': _int64_feature(channels),
        'audio/num_sources': _int64_feature(num_sources),
        'audio/labels': _int64_feature(labels),
        'audio/source_names': _bytes_feature(",".join((os.path.basename(filename[0]).replace(".","_")).split("_")[3:-1])),
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
    offset = (MIX_WITH_PADDING - NUM_SAMPLES)//2
    start_idx = offset
    end_idx = file_data_cache[0][1] - offset - 1
    for sample_idx in range((end_idx - start_idx)//NUM_SAMPLES):
        # sampling segments, ignore first and last incomplete segments
        # notice that we sample MIX_WITH_PADDING from the mix and central cropped NUM_SAMPLES from the sources
        segments_data = list()
        sample_offset_start = start_idx + sample_idx*NUM_SAMPLES
        sample_offset_end = start_idx + (sample_idx+1)*NUM_SAMPLES
        # adding big datasample for mix
        segments_data.append(file_data_cache[0][2][sample_offset_start-offset:sample_offset_end+offset+1])
        # adding rest of the sources
        assert len(segments_data[0]) == MIX_WITH_PADDING
        for source in file_data_cache[1:]:
            segments_data.append(source[2][sample_offset_start:sample_offset_end])
        segments.append([file_data_cache[0][0], sample_idx, segments_data, len(file_data_cache)-1])
    return segments


def _process_audio_files_batch(chunk_data):
    """Processes and saves list of audio files as TFRecords.
    Args:
        chunk_data: tuple of chunk_files and output_file
        chunk_files: list of strings; each string is a path to an wav file
        output_file: string, unique identifier specifying the data set
    """

    chunk_files, output_file = chunk_data[0], chunk_data[1]
    # Get training files from the directory name

    writer = tf.python_io.TFRecordWriter(output_file)

    chunk_data_cache = list()
    for track in chunk_files:
        # load all wave files into memory and create a buffer
        file_data_cache = list()
        for source in track:
            data, sr = librosa.core.load(source, sr=SAMPLE_RATE, mono=True)
            file_data_cache.append([track, len(data), data])

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
        labels = get_labels_from_filename(chunk[0])
        example = _convert_to_example(filename=chunk[0], sample_idx=chunk[1],
                                      data_buffer=chunk[2], num_sources=chunk[3],
                                      labels=labels)
        writer.write(example.SerializeToString())

    writer.close()
    tf.logging.info('Finished writing file: %s' % output_file)


def get_labels_from_filename(filename):
    labels = [0]*(len(source_map)-1)
    label_names = (os.path.basename(filename[0]).replace(".", "_")).split("_")[3:-1]
    for label_name in label_names:
        labels[source_map[label_name]-1] = 1
    return labels


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
    chunksize = int(math.ceil(len(filenames) / float(num_shards)))

    pool = Pool(multiprocessing.cpu_count()-1)

    def output_file(shard_idx):
        return os.path.join(output_directory, '%s-%.5d-of-%.5d' % (prefix, shard_idx, num_shards))

    # chunk data consists of chunk_filenames and output_file
    chunk_data = [(filenames[shard * chunksize: (shard + 1) * chunksize],
                  output_file(shard)) for shard in range(num_shards)]

    files = pool.map(_process_audio_files_batch, chunk_data)

    return files


def get_wav(database_path):
    """ Iterate through .wav files from URMP dataset
        returns data_list: List[List[path_to_wavefiles]] """

    silence_path = '../silence.wav'
    track_list = []
    # for dir in os.listdir(database_path):
    #     source_list = []
    #     for filename in os.listdir(database_path+"/"+dir):
    #         source_list.append(os.path.join(database_path+"/"+dir, filename))
    #         source_list.sort()
    #     track_list.append(source_list)

    # tracks = [f for r,d,f in os.walk('test')]
    # waves = [os.path.join(database_path, wav) for sub_folders in tracks for wav in sub_folders]

    # Iterate through each tracks
    for folder in os.listdir(database_path):
        track_sources = [0 for i in range(14)]  # 1st index must be mix source + 13 individual sources

        # Create Sample object for each instrument source files present
        for filename in os.listdir(os.path.join(database_path, folder)):
            if filename.endswith(".wav"):
                if filename.startswith("AuMix"):
                    # Place mix source to the first index
                    mix_path = os.path.join(database_path, folder, filename)
                    # mix_audio, mix_rate = soundfile.read(mix_path, always_2d=True)
                    # mix_duration = mix_audio.shape[0] / mix_rate
                    # mix = Sample(mix_path, mix_rate, mix_audio.shape[1], mix_duration)
                    track_sources[0] = mix_path
                else:
                    # Place Sample object mapping to its instrument index
                    print(filename, filename.split('_'))
                    source_name = filename.split('_')[2]
                    source_idx = source_map[source_name]
                    source_path = os.path.join(database_path, folder, filename)
                    # source_audio, source_rate = soundfile.read(source_path, always_2d=True)
                    # source_duration = source_audio.shape[0] / source_rate
                    # source = Sample(source_path, source_rate, source_audio.shape[1], source_duration)
                    track_sources[source_idx] = source_path

        # Create and insert silence Sample object for instruments not present in the track
        # silence_audio, silence_rate = soundfile.read(silence_path, always_2d=True)
        # silence_duration = silence_audio.shape[0] / silence_rate
        # silence = Sample(silence_path, silence_rate, silence_audio.shape[1], silence_duration)
        for i, track in enumerate(track_sources):
            if track == 0:
                track_sources[i] = silence_path

        track_list.append(track_sources)


    return track_list

def convert_to_tf_records(raw_data_dir):
    """Convert the URMP dataset into TF-Record dumps."""

    training_files = get_wav(os.path.join(raw_data_dir, TRAINING_DIRECTORY))
    test_files= get_wav(os.path.join(raw_data_dir, TEST_DIRECTORY))

    # # Glob all the training files
    # training_files = tf.gfile.Glob(
    #     os.path.join(raw_data_dir, TRAINING_DIRECTORY, '*.wav'))
    # training_files = list(set([filename.split('.')[0] for filename in training_files]))
    #
    # # Glob all the validation files
    # test_files = sorted(tf.gfile.Glob(
    #     os.path.join(raw_data_dir, TEST_DIRECTORY, '*.wav')))
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
