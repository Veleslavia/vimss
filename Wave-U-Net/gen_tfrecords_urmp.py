""" Examples to demonstrate how to write an image file to a TFRecord,
and how to read a TFRecord file using TFRecordReader.
Author: Chip Huyen
Prepared for the class CS 20SI: "TensorFlow for Deep Learning Research"
cs20si.stanford.edu
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import sys
sys.path.append('..')

import tensorflow as tf
import soundfile as sf
from pydub import AudioSegment

MUSIC_PATH = '/home/leo/Documents/URMP'
NUM_FRAMES = 16384


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def get_URMP_data_lists(database_path):
    # iterate through .wav files from URMP dataset
    # returns data_list: List[List[path_to_wavefiles]]
    data_list = []
    for folder in os.listdir(database_path):


        if os.path.isdir(database_path+"/"+folder):
            music_list = []

            for filename in os.listdir(database_path+"/"+folder):
                if (filename.startswith(('AuMix'))):
                    # Append AuMix at the front of the list
                    music_list.insert(0, os.path.join(database_path+"/"+folder, filename))
                elif (filename.endswith(".wav") and not filename.startswith(".")):
                    music_list.append(os.path.join(database_path+"/"+folder, filename))
            data_list.append(music_list)

    return data_list


def get_music_data(music_list):
    # music_list: List[AuMix_path, AuSep_01_path, ...]

    # Extract info from path name
    file_basename = "_".join(music_list[0].split("_", 2)[:2])
    num_sources = len(music_list) - 1
    print('file_basename: ' + file_basename)
    print('num_sources: '+ str(num_sources))

    # Extract wav file detail using SoundFile lib
    with sf.SoundFile(music_list[0], "r") as f:
        sample_rate = f.samplerate
        duration = len(f) / f.samplerate
        tot_num_samples = len(f)
        channels = f.channels
        # print('sample_rate: ' + str(sample_rate))
        # print('duration: ' + str(duration))
        # print('tot_num_samples: ' + str(tot_num_samples))

    music_data = []

    # Split each audio source into multiple slices
    tot_iter = tot_num_samples / NUM_FRAMES

    for i in range(tot_iter):
        print('iter: ' + str(i) + ' / ' + str(tot_iter))

        concat_sound = AudioSegment.empty()

        for s in range(num_sources+1):
            print('source:' + music_list[s])
            tmpAudio = AudioSegment.from_wav(music_list[s])
            tmpAudioArr = tmpAudio.get_array_of_samples() # number of samples 4590186
            tmpAudioSlice = tmpAudio._spawn(tmpAudioArr[i*NUM_FRAMES:(i+1)*NUM_FRAMES])

            # tmpAudioSlice.export('/home/leo/Desktop/test/'+'source'+str(s)+'_'+str(i)+'.wav', format="wav")
            # print(len(tmpAudioSlice))

            # Concatenating AuMix & AuSeps
            concat_sound += tmpAudioSlice

        concat_sound_file_name = '/home/leo/Desktop/test/'+file_basename+'_'+str(i)+'.wav'
        # concat_sound.export(concat_sound_file_name, format="wav")

        music_data.append([file_basename, sample_rate, i*NUM_FRAMES, (i+1)*NUM_FRAMES, i, channels, num_sources, concat_sound.raw_data])

    return music_data

def main():
    data_list = get_URMP_data_lists(MUSIC_PATH)

    train_filename = 'train.tfrecords'  # address to save the TFRecords file

    # open the TFRecords file
    writer = tf.python_io.TFRecordWriter(train_filename)

    for i in range(len(data_list)):
        print('data_list_iter: ' + str(i))
        wav_list = get_music_data(data_list[i])

        for wav in wav_list:

            # Create a feature
            feature = {'audio/file_basename': _bytes_feature(wav[0]),
                       'audio/sample_rate': _int64_feature(wav[1]),
                       'audio/start_time':_int64_feature(wav[2]),
                       'audio/end_time':_int64_feature(wav[3]),
                       'audio/num_samples':_int64_feature(NUM_FRAMES),
                       'audio/channels': _int64_feature(wav[5]),
                       'audio/num_sources': _int64_feature(wav[6]),
                       'audio/encoded': _bytes_feature(wav[7])}

            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Serialize to string and write on the file
            writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()


if __name__ == '__main__':
    main()
