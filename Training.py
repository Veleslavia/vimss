from sacred import Experiment
import pickle
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

#tf.enable_eager_execution()

ex = Experiment('Waveunet')

@ex.config
def cfg():
    # Base configuration
    model_config = {"musdb_path": "gs://vimsstfrecords/musdb18", # SET MUSDB PATH HERE
                    "estimates_path": "gs://vimsscheckpoints", # SET THIS PATH TO WHERE YOU WANT SOURCE ESTIMATES
                    # PRODUCED BY THE TRAINED MODEL TO BE SAVED. Folder itself must exist!
                    "model_base_dir": "gs://vimsscheckpoints/baseline", # Base folder for model checkpoints
                    "log_dir": "logs", # Base folder for logs files
                    "batch_size": 64, # Batch size
                    "init_sup_sep_lr": 1e-4, # Supervised separator learning rate
                    "epoch_it" : 2000, # Number of supervised separator steps per epoch
                    "training_steps": 2000*50, # Number of training steps per training
                    "use_tpu": True,
                    "load_model": False,
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
        "mono_downmix": True,
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
def unet_separator(features, labels, mode, params):
    mix = features
    sources = labels
    model_config = params
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

    # Input context that the input audio has to be padded ON EACH SIDE
    # TODO move this to dataset function
    pad = (sep_input_shape[1] - sep_output_shape[1])
    pad_tensor = tf.constant([[0, 0], [pad//2+2, pad - pad//2+3], [0, 0]])
    mix = tf.pad(mix, pad_tensor, "CONSTANT")
    pad_tensor = tf.constant([[0, 0], [2, 3], [0, 0], [0, 0]])
    sources = tf.pad(sources, pad_tensor, "CONSTANT")

    separator_func = separator_class.get_output

    # Compute loss.
    separator_sources = separator_func(mix, True, not model_config["raw_audio_loss"], reuse=False)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'sources': separator_sources
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Supervised objective: MSE in log-normalized magnitude space
    separator_sources = tf.transpose(tf.stack(separator_sources), [1, 2, 3, 0])

    separator_loss = tf.reduce_mean(
        tf.reduce_mean(
            tf.squared_difference(sources, separator_sources), axis=[0, 1, 2]),
        axis=0)

    #separator_loss = tf.losses.mean_squared_error(sources, separator_sources)

    #tf.summary.scalar("sep_loss", separator_loss, collections=["sup"])
    #sup_summaries = tf.summary.merge_all(key='sup')

    # Creating evaluation estimator
    if mode == tf.estimator.ModeKeys.EVAL:
        def metric_fn(labels, predictions):
            #mean_mse_loss = 0.0
            #for i in range(sources.shape[0]):
            #    mean_mse_loss += tf.reduce_mean(tf.square(sources[i] - separator_sources[i]))
            #mean_mse_loss /= float(sources.shape[0].value) # Normalise by number of sources
            mean_mse_loss = tf.metrics.mean_squared_error(labels, predictions)
            return {'mse': mean_mse_loss}

        eval_params = {'labels': sources,
                       'predictions': separator_sources}

        return tpu_estimator.TPUEstimatorSpec(
            mode=mode,
            loss=separator_loss,
            eval_metrics=(metric_fn, eval_params))


    # Create training op.
    # TODO add learning rate schedule
    # TODO add early stopping
    if mode == tf.estimator.ModeKeys.TRAIN:
        sep_lr = tf.get_variable('unsup_sep_lr', [],
                                 initializer=tf.constant_initializer(model_config["init_sup_sep_lr"],
                                                                     dtype=tf.float32),
                                 trainable=False)
        separator_vars = Utils.getTrainableVariables("separator")
        print("Sep_Vars: " + str(Utils.getNumParams(separator_vars)))
        print("Num of variables: " + str(len(tf.global_variables())))

        separator_solver = tf.train.AdamOptimizer(learning_rate=sep_lr)
        if model_config["use_tpu"]:
            separator_solver = tpu_optimizer.CrossShardOptimizer(separator_solver)

        global_step = tf.train.get_global_step()
        batches_per_epoch = 10000 / model_config["batch_size"]
        current_epoch = (tf.cast(global_step, tf.float32) /
                         batches_per_epoch)
        learning_rate = sep_lr

        def host_call_fn(gs, loss, lr, ce):
            """Training host call. Creates scalar summaries for training metrics.
            This function is executed on the CPU and should not directly reference
            any Tensors in the rest of the `model_fn`. To pass Tensors from the
            model to the `metric_fn`, provide as part of the `host_call`. See
            https://www.tensorflow.org/api_docs/python/tf/contrib/tpu/TPUEstimatorSpec
            for more information.
            Arguments should match the list of `Tensor` objects passed as the second
            element in the tuple passed to `host_call`.
            Args:
              gs: `Tensor with shape `[batch]` for the global_step
              loss: `Tensor` with shape `[batch]` for the training loss.
              lr: `Tensor` with shape `[batch]` for the learning_rate.
              ce: `Tensor` with shape `[batch]` for the current_epoch.
            Returns:
              List of summary ops to run on the CPU host.
            """
            gs = gs[0]
            with summary.create_file_writer(model_config["model_base_dir"]).as_default():
                with summary.always_record_summaries():
                    summary.scalar('loss', loss[0], step=gs)
                    summary.scalar('learning_rate', lr[0], step=gs)
                    summary.scalar('current_epoch', ce[0], step=gs)

            return summary.all_summary_ops()

        # To log the loss, current learning rate, and epoch for Tensorboard, the
        # summary op needs to be run on the host CPU via host_call. host_call
        # expects [batch_size, ...] Tensors, thus reshape to introduce a batch
        # dimension. These Tensors are implicitly concatenated to
        # [params['batch_size']].
        gs_t = tf.reshape(global_step, [1])
        loss_t = tf.reshape(separator_loss, [1])
        lr_t = tf.reshape(learning_rate, [1])
        ce_t = tf.reshape(current_epoch, [1])

        host_call = (host_call_fn, [gs_t, loss_t, lr_t, ce_t])

        train_op = separator_solver.minimize(separator_loss,
                                             var_list=separator_vars,
                                             global_step=global_step)
        return tpu_estimator.TPUEstimatorSpec(mode=mode,
                                              loss=separator_loss,
                                              host_call=host_call,
                                              train_op=train_op)


@ex.automain
def dsd_100_experiment(model_config):
    print("SCRIPT START")
    # Create subfolders if they do not exist to save results
    for dir in [model_config["model_base_dir"], model_config["log_dir"]]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    print("TPU resolver started")

    tpu_cluster_resolver = TPUClusterResolver(
        tpu=[os.environ['TPU_NAME']],
        project='plated-dryad-162216',
        zone='us-central1-f')
    config = tpu_config.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=model_config['model_base_dir'],
        save_checkpoints_steps=500,
        tpu_config=tpu_config.TPUConfig(
            iterations_per_loop=500,
            num_shards=8,
            per_host_input_for_training=tpu_config.InputPipelineConfig.PER_HOST_V2))  # pylint: disable=line-too-long

    print("Creating datasets")
    musdb_train, musdb_eval = [musdb_input.MusDBInput(
        is_training=is_training,
        data_dir=model_config['musdb_path'],
        transpose_input=False,
        use_bfloat16=False) for is_training in [True, False]]

    # Optimize in a +supervised fashion until validation loss worsens
    separator = tpu_estimator.TPUEstimator(
        use_tpu=model_config["use_tpu"],
        model_fn=unet_separator,
        config=config,
        train_batch_size=model_config['batch_size'],
        eval_batch_size=model_config['batch_size'],
        predict_batch_size=model_config['batch_size'],
        params={i: model_config[i] for i in model_config if i != 'batch_size'})

    # Train the Model.
    if model_config['load_model']:
        current_step = estimator._load_global_step_from_checkpoint_dir(model_config['model_base_dir'])
    else:
        separator.train(
            input_fn=musdb_train.input_fn,
            steps=model_config['training_steps']) # Should be an early stopping here, but it will come with tf 1.10

    print("Supervised training finished!")

    # Evaluate the model.
    eval_result = separator.evaluate(
        input_fn=musdb_eval.input_fn,
        steps=1000)

    print("Test results and save predicted sources:")
    predictions = separator.predict(
        input_fn=musdb_eval.input_fn)

    for i, predicted in enumerate(predictions):
        pickle.dump(predicted, open('predicted_' + str(i)+'.pkl', 'w'))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
