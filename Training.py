from sacred import Experiment
import tensorflow as tf
import numpy as np
import os

from Input import musdb_input
import Utils
import Models.UnetSpectrogramSeparator
import Models.UnetAudioSeparator

from tensorflow.contrib.cluster_resolver import TPUClusterResolver
from tensorflow.contrib import summary
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer
from tensorflow.python.estimator import estimator

import librosa

ex = Experiment('Waveunet')

@ex.config
def cfg():
    # Base configuration
    model_config = {"musdb_path": "gs://vimsstfrecords/musdb18context", # SET MUSDB PATH HERE
                    "estimates_path": "estimates", # SET THIS PATH TO WHERE YOU WANT SOURCE ESTIMATES
                    # PRODUCED BY THE TRAINED MODEL TO BE SAVED. Folder itself must exist!
                    "model_base_dir": "gs://vimsscheckpoints", # Base folder for model checkpoints
                    "log_dir": "logs", # Base folder for logs files
                    "batch_size": 64, # Batch size
                    "init_sup_sep_lr": 1e-5, # Supervised separator learning rate
                    "epoch_it" : 2000, # Number of supervised separator steps per epoch
                    "training_steps": 2000*100, # Number of training steps per training
                    "evaluation_steps": 1000,
                    "use_tpu": True,
                    "load_model": False,
                    "predict_only": False,
                    "write_audio_summaries": False,
                    "audio_summaries_every_n_steps": 10000,
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
                    'raw_audio_loss' : True, # Only active for unet_spectrogram network. True: L2 loss on audio. False: L1 loss on spectrogram magnitudes for training and validation and test loss
                    'experiment_id': np.random.randint(0, 1000000)
                    }

    model_config["num_sources"] = 4 if model_config["task"] == "multi_instrument" else 2
    model_config["num_channels"] = 1 if model_config["mono_downmix"] else 2

@ex.named_config
def baseline():
    print("Training baseline model")


@ex.named_config
def baseline_stereo():
    print("Training baseline model with difference output and input context (valid convolutions)")
    model_config = {
        "output_type" : "difference",
        "context" : True,
        "mono_downmix" : False
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

@ex.capture
def unet_separator(features, labels, mode, params):

    # Define host call function
    def host_call_fn(gs, loss, lr, 
            mix=tf.placeholder(tf.float32), 
            gt_sources=tf.placeholder(tf.float32), 
            est_sources=tf.placeholder(tf.float32)):
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
              input: `Tensor` with shape `[batch, mix_samples, 1]`
              gt_sources: `Tensor` with shape `[batch, sources_n, output_samples, 1]`
              est_sources: `Tensor` with shape `[batch, sources_n, output_samples, 1]`
            Returns:
              List of summary ops to run on the CPU host.
            """
            gs = gs[0]
            with summary.create_file_writer(model_config["model_base_dir"]+os.path.sep+str(model_config["experiment_id"])).as_default():
                with summary.always_record_summaries():
                    summary.scalar('loss', loss[0], step=gs)
                    summary.scalar('learning_rate', lr[0], step=gs)
                if model_config["write_audio_summaries"]:
                    with summary.record_summaries_every_n_global_steps(model_config["audio_summaries_every_n_steps"]):
                        summary.audio('mix', mix, model_config['expected_sr'], max_outputs=4)
                        for source_id in range(gt_sources.shape[1].value):
                            summary.audio('gt_sources_{source_id}'.format(source_id=source_id), gt_sources[:, source_id, :, :],
                                          model_config['expected_sr'], max_outputs=4)
                            summary.audio('est_sources_{source_id}'.format(source_id=source_id), est_sources[:, source_id, :, :],
                                          model_config['expected_sr'], max_outputs=4)
            return summary.all_summary_ops()

    mix = features['mix']
    sources = labels
    model_config = params
    disc_input_shape = [model_config["batch_size"], model_config["num_frames"], 0]

    with tf.contrib.tpu.bfloat16_scope():
        separator_class = Models.UnetAudioSeparator.UnetAudioSeparator(
            model_config["num_layers"], model_config["num_initial_filters"],
            output_type=model_config["output_type"],
            context=model_config["context"],
            mono=model_config["mono_downmix"],
            upsampling=model_config["upsampling"],
            num_sources=model_config["num_sources"],
            filter_size=model_config["filter_size"],
            merge_filter_size=model_config["merge_filter_size"])

    sep_input_shape, sep_output_shape = separator_class.get_padding(np.array(disc_input_shape))

    # Input context that the input audio has to be padded ON EACH SIDE
    # TODO move this to dataset function
    assert mix.shape[1].value == sep_input_shape[1]
    if mode != tf.estimator.ModeKeys.PREDICT:
        pad_tensor = tf.constant([[0, 0], [0, 0], [2, 3], [0, 0]])
        sources = tf.pad(sources, pad_tensor, "CONSTANT")

    separator_func = separator_class.get_output

    # Compute loss.
    separator_sources = tf.stack(separator_func(mix, True, not model_config["raw_audio_loss"], reuse=False), axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'mix': mix,
            'sources': separator_sources,
            'filename': features['filename'],
            'sample_id': features['sample_id']
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    separator_loss = tf.losses.mean_squared_error(sources, separator_sources)

    if mode != tf.estimator.ModeKeys.PREDICT:
        global_step = tf.train.get_global_step()
        sep_lr = tf.get_variable('unsup_sep_lr', [],
                                 initializer=tf.constant_initializer(model_config["init_sup_sep_lr"],
                                                                     dtype=tf.float32),
                                 trainable=False)

        gs_t = tf.reshape(global_step, [1])
        loss_t = tf.reshape(separator_loss, [1])
        lr_t = tf.reshape(sep_lr, [1])

        if model_config["write_audio_summaries"]:
            host_call = (host_call_fn, [gs_t, loss_t, lr_t, mix, sources, separator_sources])
        else:
            host_call = (host_call_fn, [gs_t, loss_t, lr_t])

    # Creating evaluation estimator
    if mode == tf.estimator.ModeKeys.EVAL:
        def metric_fn(labels, predictions):
            mean_mse_loss = tf.metrics.mean_squared_error(labels, predictions)
            return {'mse': mean_mse_loss}

        eval_params = {'labels': sources,
                       'predictions': separator_sources}

        return tpu_estimator.TPUEstimatorSpec(
            mode=mode,
            loss=separator_loss,
            host_call=host_call,
            eval_metrics=(metric_fn, eval_params))


    # Create training op.
    # TODO add learning rate schedule
    # TODO add early stopping
    if mode == tf.estimator.ModeKeys.TRAIN:
        separator_vars = Utils.getTrainableVariables("separator")
        print("Sep_Vars: " + str(Utils.getNumParams(separator_vars)))
        print("Num of variables: " + str(len(tf.global_variables())))

        separator_solver = tf.train.AdamOptimizer(learning_rate=sep_lr)
        if model_config["use_tpu"]:
            separator_solver = tpu_optimizer.CrossShardOptimizer(separator_solver)

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
    for dir in [model_config["model_base_dir"], model_config["log_dir"], model_config["estimates_path"]]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    print("TPU resolver started")

    tpu_cluster_resolver = TPUClusterResolver(
        tpu=[os.environ['TPU_NAME']],
        project='plated-dryad-162216',
        zone='us-central1-f')
    config = tpu_config.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=model_config['model_base_dir'] + os.path.sep + str(model_config["experiment_id"]),
        save_checkpoints_steps=500,
        save_summary_steps=250,
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
        params={i: model_config[i] for i in model_config if i != 'batch_size'}
    )

    # Train the Model.
    if model_config['load_model']:
        current_step = estimator._load_global_step_from_checkpoint_dir(
            model_config['model_base_dir'] + os.path.sep + str(model_config["experiment_id"]))
    else:
        separator.train(
            input_fn=musdb_train.input_fn,
            steps=model_config['training_steps']) # Should be an early stopping here, but it will come with tf 1.10

    print("Supervised training finished!")

    if not model_config["predict_only"]:
        print("Evaluate model")
        # Evaluate the model.
        eval_result = separator.evaluate(
            input_fn=musdb_eval.input_fn,
            steps=model_config['evaluation_steps'])
    else:
        print("Test results and save predicted sources:")
        predictions = separator.predict(
            input_fn=musdb_eval.input_fn)

        for prediction in predictions:
            estimates_dir = model_config["estimates_path"] + os.path.sep + prediction['filename']
            if not os.path.exists(estimates_dir):
                os.makedirs(estimates_dir)
                os.makedirs(estimates_dir + os.path.sep + 'mix')
                for source_name in range(len(prediction['sources'])):
                    os.makedirs(estimates_dir + os.path.sep + "source_" + str(source_name))
            mix_audio_path = "{basedir}{sep}mix{sep}{sampleid}.wav".format(
                basedir=estimates_dir,
                sep=os.path.sep,
                sampleid="%.4d" % prediction['sample_id']
            )
            librosa.output.write_wav(mix_audio_path,
                                     prediction['mix'],
                                     sr=model_config["expected_sr"])
            for source_name in range(len(prediction['sources'])):
                source_path = "{basedir}{sep}source_{sname}{sep}{sampleid}.wav".format(
                    basedir=estimates_dir,
                    sep=os.path.sep,
                    sname=source_name,
                    sampleid="%.4d" % prediction['sample_id']
                )
                librosa.output.write_wav(source_path,
                                         prediction['sources'][source_name],
                                         sr=model_config["expected_sr"])
        Utils.concat_and_upload(model_config["estimates_path"],
                                model_config['model_base_dir'] + os.path.sep + str(model_config["experiment_id"]))

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
