import tensorflow as tf
import numpy as np
import os

from Input import Input as Input
import Models.UnetAudioSeparator
import Evaluate
import Utils
import functools
from tensorflow.contrib.signal.python.ops import window_ops

import librosa

def test(model_config, audio_list, model_folder, load_model):
    # Determine input and output shapes
    disc_input_shape = [model_config["batch_size"], model_config["num_frames"], 0]  # Shape of discriminator input
    separator_class = Models.UnetAudioSeparator.UnetAudioSeparator(model_config["num_layers"], model_config["num_initial_filters"],
                                                                   output_type=model_config["output_type"],
                                                                   context=model_config["context"],
                                                                   mono=model_config["mono_downmix"],
                                                                   upsampling=model_config["upsampling"],
                                                                   num_sources=model_config["num_sources"],
                                                                   filter_size=model_config["filter_size"],
                                                                   merge_filter_size=model_config["merge_filter_size"])

    sep_input_shape, sep_output_shape = separator_class.get_padding(np.array(disc_input_shape))
    separator_func = separator_class.get_output

    # Creating the batch generators
    assert ((sep_input_shape[1] - sep_output_shape[1]) % 2 == 0)

    # Batch size of 1
    sep_input_shape[0] = 1
    sep_output_shape[0] = 1

    mix_context, sources = Input.get_multitrack_placeholders(sep_output_shape, model_config["num_sources"], sep_input_shape, "input")

    print("Testing...")

    # BUILD MODELS
    # Separator
    separator_sources = separator_func(mix_context, False, False, reuse=False)

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False, dtype=tf.int64)

    # Start session and queue input threads
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(model_config["log_dir"] + os.path.sep +  model_folder, graph=sess.graph)

    # CHECKPOINTING
    # Load pretrained model to test
    restorer = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2)
    print("Num of variables" + str(len(tf.global_variables())))
    restorer.restore(sess, load_model)
    print('Pre-trained model restored for testing')

    input_audio = tf.placeholder(tf.float32, shape=[None, 1])
    window = functools.partial(window_ops.hann_window, periodic=True)
    stft = tf.contrib.signal.stft(tf.squeeze(input_audio, 1), frame_length=1024, frame_step=768,
                                        fft_length=1024, window_fn=window)
    mag = tf.abs(stft)

    # Start training loop
    _global_step = sess.run(global_step)
    print("Starting!")

    total_loss = 0.0
    total_samples = 0
    for sample in audio_list: # Go through all tracks
        # Load mixture and fetch prediction for mixture
        mix_audio, mix_sr = Utils.load(sample[0].path, sr=None, mono=False)
        sources_pred = Evaluate.predict_track(model_config, sess, mix_audio, mix_sr, sep_input_shape, sep_output_shape, separator_sources, mix_context)

        # Load original sources
        sources_gt = list()
        for s in sample[1:]:
            s_audio, _ = Utils.load(s.path, sr=model_config["expected_sr"], mono=model_config["mono_downmix"], res_type="kaiser_fast")
            sources_gt.append(s_audio)

        # Determine mean squared error
        for (source_gt, source_pred) in zip(sources_gt, sources_pred):
            if model_config["network"] == "unet_spectrogram" and not model_config["raw_audio_loss"]:
                real_mag = sess.run(mag, feed_dict={input_audio : source_gt})
                pred_mag = sess.run(mag, feed_dict={input_audio: source_pred})
                total_loss += np.sum(np.abs(real_mag - pred_mag))
                total_samples += np.prod(real_mag.shape)  # Number of entries is product of number of sources and number of outputs per source
            else:
                total_loss += np.sum(np.square(source_gt - source_pred))
                total_samples += np.prod(source_gt.shape)  # Number of entries is product of number of sources and number of outputs per source

        print("MSE for track " + sample[0].path + ": " + str(total_loss / float(total_samples)))
    mean_mse_loss = total_loss / float(total_samples)

    summary = tf.Summary(value=[tf.Summary.Value(tag="test_loss", simple_value=mean_mse_loss)])
    writer.add_summary(summary, global_step=_global_step)

    writer.flush()
    writer.close()

    print("Finished testing - Mean MSE: " + str(mean_mse_loss))

    # Close session, clear computational graph
    sess.close()
    tf.reset_default_graph()

    return mean_mse_loss


def save_prediction(prediction, output_padding, estimates_path, maps=None, sample_rate=22050):
    if maps:
        basenames = maps[0]
        inv_source_map = {v: k for k, v in maps[1].iteritems()}
        estimates_dir = estimates_path + os.path.sep + basenames[prediction['filename']]
    else:
        inv_source_map = None
        estimates_dir = estimates_path + os.path.sep + prediction['filename']

    def _get_source_name(_source_id):
        if inv_source_map:
            source_name = inv_source_map[_source_id + 1]
        else:
            source_name = 'source_' + str(_source_id)
        return source_name

    if not os.path.exists(estimates_dir):
        os.makedirs(estimates_dir)
        for source_id in range(len(prediction['sources'])):
            os.makedirs(estimates_dir + os.path.sep + _get_source_name(source_id))
    for source_id in range(len(prediction['sources'])):
        source_name = _get_source_name(source_id)
        source_path = "{basedir}{sep}{sname}{sep}{sampleid}.wav".format(
            basedir=estimates_dir,
            sep=os.path.sep,
            sname=source_name,
            sampleid="%.4d" % prediction['sample_id']
        )
        librosa.output.write_wav(source_path,
                                 np.float32(prediction['sources'][source_id][output_padding[0]:-output_padding[1]]),
                                 sr=sample_rate)
