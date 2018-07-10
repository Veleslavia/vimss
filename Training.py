from sacred import Experiment
import tensorflow as tf
from tensorflow.contrib import tpu
import numpy as np
import os

from Input import musdb_input
import Utils
import Models.UnetSpectrogramSeparator
import Models.UnetAudioSeparator
import Test
import Evaluate

import functools
from tensorflow.contrib.cluster_resolver import TPUClusterResolver
from tensorflow.contrib import summary
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer
from tensorflow.contrib.training.python.training import evaluation
from tensorflow.python.estimator import estimator

ex = Experiment('Waveunet')

@ex.config
def cfg():
    # Base configuration
    model_config = {"musdb_path": "gs://vimsstfrecords/musdb18", # SET MUSDB PATH HERE
                    "estimates_path": "gs://vimsscheckpoints", # SET THIS PATH TO WHERE YOU WANT SOURCE ESTIMATES
                    # PRODUCED BY THE TRAINED MODEL TO BE SAVED. Folder itself must exist!
                    "model_base_dir": "checkpoints", # Base folder for model checkpoints
                    "log_dir": "logs", # Base folder for logs files
                    "batch_size": 16, # Batch size
                    "init_sup_sep_lr": 1e-4, # Supervised separator learning rate
                    "epoch_it" : 2000, # Number of supervised separator steps per epoch
                    "training_steps": 2000*100, # Number of training steps per training
                    "num_disc": 5,  # Number of discriminator iterations per separator update
                    'cache_size' : 16, # Number of audio excerpts that are cached to build batches from
                    'num_workers' : 6, # Number of processes reading audio and filling up the cache
                    "duration" : 2, # Duration in seconds of the audio excerpts in the cache. Has to be at least the output length of the network!
                    'min_replacement_rate' : 16,  # roughly: how many cache entries to replace at least per batch on average. Can be fractional
                    'num_layers' : 12, # How many U-Net layers
                    'filter_size' : 15, # For Wave-U-Net: Filter size of conv in downsampling block
                    'merge_filter_size' : 5, # For Wave-U-Net: Filter size of conv in upsampling block
                    'num_initial_filters' : 24, # Number of filters for convolution in first layer of network
                    "num_frames": 16384, # DESIRED number of time frames in the output waveform per samples (could be changed when using valid padding)
                    'expected_sr': 22050,  # Downsample all audio input to this sampling rate
                    'mono_downmix': True,  # Whether to downsample the audio input
                    'output_type' : 'direct', # Type of output layer, either "direct" or "difference". Direct output: Each source is result of tanh activation and independent. DIfference: Last source output is equal to mixture input - sum(all other sources)
                    'context' : False, # Type of padding for convolutions in separator. If False, feature maps double or half in dimensions after each convolution, and convolutions are padded with zeros ("same" padding). If True, convolution is only performed on the available mixture input, thus the output is smaller than the input
                    'network' : 'unet', # Type of network architecture, either unet (our model) or unet_spectrogram (Jansson et al 2017 model)
                    'upsampling' : 'linear', # Type of technique used for upsampling the feature maps in a unet architecture, either 'linear' interpolation or 'learned' filling in of extra samples
                    'task' : 'voice', # Type of separation task. 'voice' : Separate music into voice and accompaniment. 'multi_instrument': Separate music into guitar, bass, vocals, drums and other (Sisec)
                    'augmentation' : True, # Random attenuation of source signals to improve generalisation performance (data augmentation)
                    'raw_audio_loss' : True # Only active for unet_spectrogram network. True: L2 loss on audio. False: L1 loss on spectrogram magnitudes for training and validation and test loss
                    }
    seed=1337
    experiment_id = np.random.randint(0,1000000)

    model_config["num_sources"] = 4 if model_config["task"] == "multi_instrument" else 2
    model_config["num_channels"] = 1 if model_config["mono_downmix"] else 2

@ex.named_config
def baseline():
    print("Training baseline model")

@ex.named_config
def baseline_diff():
    print("Training baseline model with difference output")
    model_config = {
        "output_type" : "difference"
    }

@ex.named_config
def baseline_context():
    print("Training baseline model with difference output and input context (valid convolutions)")
    model_config = {
        "output_type" : "difference",
        "context" : True
    }

@ex.named_config
def baseline_stereo():
    print("Training baseline model with difference output and input context (valid convolutions)")
    model_config = {
        "output_type" : "difference",
        "context" : True,
        "mono_downmix" : False
    }

@ex.named_config
def full():
    print("Training full singing voice separation model, with difference output and input context (valid convolutions) and stereo input/output, and learned upsampling layer")
    model_config = {
        "output_type" : "difference",
        "context" : True,
        "upsampling": "learned",
        "mono_downmix" : False
    }

@ex.named_config
def baseline_context_smallfilter_deep():
    model_config = {
        "output_type": "difference",
        "context": True,
        "num_layers" : 14,
        "duration" : 7,
        "filter_size" : 5,
        "merge_filter_size" : 1
    }

@ex.named_config
def full_multi_instrument():
    print("Training multi-instrument separation with best model")
    model_config = {
        "output_type": "difference",
        "context": True,
        "upsampling": "linear",
        "mono_downmix": False,
        "task" : "multi_instrument"
    }

@ex.named_config
def baseline_comparison():
    model_config = {
        "batch_size": 4, # Less output since model is so big.
        # Doesn't matter since the model's output is not dependent on its output or input size (only convolutions)
        "cache_size": 4,
        "min_replacement_rate" : 4,

        "output_type": "difference",
        "context": True,
        "num_frames" : 768*127 + 1024,
        "duration" : 13,
        "expected_sr" : 8192,
        "num_initial_filters" : 34
    }

@ex.named_config
def unet_spectrogram():
    model_config = {
        "batch_size": 4, # Less output since model is so big.
        "cache_size": 4,
        "min_replacement_rate" : 4,

        "network" : "unet_spectrogram",
        "num_layers" : 6,
        "expected_sr" : 8192,
        "num_frames" : 768 * 127 + 1024, # hop_size * (time_frames_of_spectrogram_input - 1) + fft_length
        "duration" : 13,
        "num_initial_filters" : 16
    }

@ex.named_config
def unet_spectrogram_l1():
    model_config = {
        "batch_size": 4, # Less output since model is so big.
        "cache_size": 4,
        "min_replacement_rate" : 4,

        "network" : "unet_spectrogram",
        "num_layers" : 6,
        "expected_sr" : 8192,
        "num_frames" : 768 * 127 + 1024, # hop_size * (time_frames_of_spectrogram_input - 1) + fft_length
        "duration" : 13,
        "num_initial_filters" : 16,
        "loss" : "magnitudes"
    }


@ex.capture
def unet_separator(mix, sources, mode, model_config):
    disc_input_shape = [model_config["batch_size"], model_config["num_frames"], 0]
    if model_config["network"] == "unet":
        separator_class = Models.UnetAudioSeparator.UnetAudioSeparator(
            model_config["num_layers"], model_config["num_initial_filters"],
            output_type=model_config["output_type"],
            context=model_config["context"],
            mono=model_config["mono_downmix"],
            upsampling=model_config["upsampling"],
            num_sources=model_config["num_sources"],
            filter_size=model_config["filter_size"],
            merge_filter_size=model_config["merge_filter_size"])
    else:
        raise NotImplementedError

    sep_input_shape, sep_output_shape = separator_class.get_padding(np.array(disc_input_shape))
    separator_func = separator_class.get_output

    # Compute loss.
    separator_sources = separator_func(mix, True, not model_config["raw_audio_loss"], reuse=False)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'sources': separator_sources
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Supervised objective: MSE in log-normalized magnitude space
    separator_loss = 0
    for (real_source, sep_source) in zip(sources, separator_sources):
        separator_loss += tf.reduce_mean(tf.square(real_source - sep_source))
    separator_loss /= float(len(sources)) # Normalise by number of sources

    # SUMMARIES
    #tf.summary.scalar("sep_loss", separator_loss, collections=["sup"])
    #sup_summaries = tf.summary.merge_all(key='sup')

    # Creating evaluation estimator
    if mode == tf.estimator.ModeKeys.EVAL:
        return tpu_estimator.TPUEstimatorSpec(
            mode=mode,
            loss=separator_loss,
            eval_metrics=(tf.metrics.mean_squared_error, [sources, separator_sources]))


    # Create training op.
    if mode == tf.estimator.ModeKeys.TRAIN:
        sep_lr = tf.get_variable('unsup_sep_lr', [],
                                 initializer=tf.constant_initializer(model_config["init_sup_sep_lr"],
                                                                     dtype=tf.float32),
                                 trainable=False)
        separator_vars = Utils.getTrainableVariables("separator")
        print("Sep_Vars: " + str(Utils.getNumParams(separator_vars)))
        print("Num of variables" + str(len(tf.global_variables())))

        separator_solver = tpu_optimizer.CrossShardOptimizer(tf.train.AdamOptimizer(learning_rate=sep_lr))

        train_op = separator_solver.minimize(separator_loss,
                                             var_list=separator_vars,
                                             global_step=tf.train.get_global_step())
        return tpu_estimator.TPUEstimatorSpec(mode=mode,
                                              loss=separator_loss,
                                              train_op=train_op)


@ex.capture
def optimise(model_config, experiment_id, dataset):
    epoch = 0
    best_loss = 10000
    model_path = None
    best_model_path = None
    for i in range(2):
        worse_epochs = 0
        if i==1:
            print("Finished first round of training, now entering fine-tuning stage")
            model_config["batch_size"] *= 2
            model_config["cache_size"] *= 2
            model_config["min_replacement_rate"] *= 2
            model_config["init_sup_sep_lr"] = 1e-5
        while worse_epochs < 20:    # Early stopping on validation set after a few epochs
            print("EPOCH: " + str(epoch))
            musdb_train, musdb_eval = dataset[0], dataset[1]
            model_path = train(sup_dataset=musdb_train, load_model=model_path)
            curr_loss = Test.test(model_config, model_folder=str(experiment_id), audio_list=musdb_eval, load_model=model_path)
            epoch += 1
            if curr_loss < best_loss:
                worse_epochs = 0
                print("Performance on validation set improved from " + str(best_loss) + " to " + str(curr_loss))
                best_model_path = model_path
                best_loss = curr_loss
            else:
                worse_epochs += 1
                print("Performance on validation set worsened to " + str(curr_loss))
    print("TRAINING FINISHED - TESTING WITH BEST MODEL " + best_model_path)
    test_loss = Test.test(model_config, model_folder=str(experiment_id), audio_list=musdb_eval, load_model=best_model_path)
    return best_model_path, test_loss

@ex.automain
def dsd_100_experiment(model_config):
    print("SCRIPT START")
    # Create subfolders if they do not exist to save results
    for dir in [model_config["model_base_dir"], model_config["log_dir"]]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    print("TPU resolver started")
    config = tf.ConfigProto()
    tpu_grpc_url = TPUClusterResolver(tpu=[os.environ['TPU_NAME']]).get_master()

    config.gpu_options.allow_growth = True
    sess = tf.Session(target=tpu_grpc_url, config=config)
    sess.run(tpu.initialize_system())
    sess.run(tf.global_variables_initializer())


    print("Creating datasets")
    musdb_train, musdb_eval = [musdb_input.MusDBInput(
        is_training=is_training,
        data_dir=model_config['musdb_path'],
        transpose_input=False,
        use_bfloat16=False) for is_training in [True, False]]

    # Optimize in a +supervised fashion until validation loss worsens
    separator = tf.estimator.Estimator(
        model_fn=unet_separator)

    # Train the Model.
    separator.train(
        input_fn=musdb_train.input_fn,
        steps=model_config['training_steps']) # Should be an early stopping here, but it will come with tf 1.10

    print("Supervised training finished!")

    # Evaluate the model.
    eval_result = separator.evaluate(
        input_fn=musdb_eval.input_fn)

    #sup_model_path, sup_loss = optimise(dataset=[musdb_train, musdb_eval])
    #print("Supervised training finished! Saved model at " + sup_model_path + ". Performance: " + str(sup_loss))
    #Evaluate.produce_source_estimates(model_config, sup_model_path, model_config["musdb_path"], model_config[
    # "estimates_path"], "train")
