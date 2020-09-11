# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Main function to train various object detection models."""

from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging
import functools
import os
import pprint
import tensorflow as tf

from tensorflow.python.keras.optimizers import Adam, SGD
from official.modeling.hyperparams import params_dict
from official.modeling.training import distributed_executor as executor
from official.utils import hyperparams_flags
from official.vision.image_classification import optimizer_factory
from configs import factory as config_factory
from dataloader import input_reader
from dataloader import mode_keys as ModeKeys
from executor.detection_executor import DetectionDistributedExecutor
from modeling import factory as model_factory
from official.utils.misc import distribution_utils
from official.utils.misc import keras_utils

hyperparams_flags.initialize_common_flags()

flags.DEFINE_bool(
    'enable_xla',
    default=False,
    help='Enable XLA for GPU')

flags.DEFINE_string(
    'mode',
    default='train',
    help='Mode to run: `train`, `eval` or `train_and_eval`.')

flags.DEFINE_string(
    'model', default='retinanet',
    help='Model to run: `retinanet` or `shapemask`.')

flags.DEFINE_string('training_file_pattern', None,
                    'Location of the train data.')

flags.DEFINE_string('eval_file_pattern', None, 'Location of ther eval data')

flags.DEFINE_string(
    'checkpoint_path', None,
    'The checkpoint path to eval. Only used in eval_once mode.')

FLAGS = flags.FLAGS


def build_model_fn(features, labels, mode, params):
    features = features
    model_builder = model_factory.model_generator(params)
    model = model_builder.build_model(params, mode=mode)
    # model.summary()
    loss_fn = model_builder.build_loss_fn()
    global_step = tf.train.get_or_create_global_step()
    outputs = model(features, training=True)
    prediction_loss = loss_fn(labels, outputs)
    total_loss = tf.reduce_mean(prediction_loss['total_loss'])
    # total_loss = tf.reduce_mean(outputs["cls_outputs"][3])
    optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(total_loss, global_step)
    return tf.estimator.EstimatorSpec(
                              mode=mode,
                              loss=total_loss,
                              train_op=train_op)

def input_pipeline():
    def input_fn():
        image = tf.ones(shape=[4, 640, 640, 3], dtype=tf.float32)
        return image, {'shape': image}
    return input_fn


def run_executor(params,
                 train_input_fn=None,
                 eval_input_fn=None,
                 callbacks=None,
                 strategy=None):
  """Runs Retinanet model on distribution strategy defined by the user."""

  if params.architecture.use_bfloat16:
    policy = tf.compat.v2.keras.mixed_precision.experimental.Policy(
        'mixed_bfloat16')
    tf.compat.v2.keras.mixed_precision.experimental.set_policy(policy)

  model_builder = model_factory.model_generator(params)

  if strategy is None:
    strategy_config = params.strategy_config
    distribution_utils.configure_cluster(strategy_config.worker_hosts,
                                         strategy_config.task_index)
    strategy = distribution_utils.get_distribution_strategy(
        distribution_strategy=params.strategy_type,
        num_gpus=strategy_config.num_gpus,
        all_reduce_alg=strategy_config.all_reduce_alg,
        num_packs=strategy_config.num_packs,
        tpu_address=strategy_config.tpu)
    strategy_scope = distribution_utils.get_strategy_scope(strategy)

  num_workers = int(strategy.num_replicas_in_sync + 7) // 8
  is_multi_host = (int(num_workers) >= 2)

  if FLAGS.mode == 'train':


    train_dataset = train_input_fn()
    logging.info(
        'Train num_replicas_in_sync %d num_workers %d is_multi_host %s',
        strategy.num_replicas_in_sync, num_workers, is_multi_host)

    with strategy_scope:
      model = model_builder.build_model(params.as_dict(), mode=ModeKeys.TRAIN)
      #learning_rate = optimizer_factory.build_learning_rate(
      #                params=params.model.learning_rate,
      #                batch_size=64,
      #                train_steps=1000)
      #optimizer = optimizer_factory.build_optimizer(
      #        optimizer_name="momentum",
      #        base_learning_rate=0.01)
      #        #params=params.model.optimizer.as_dict())
      model.compile(optimizer=Adam(lr=1e-3), 
                    loss=model_builder.build_loss_fn(),)
      model.fit(train_dataset,
                epochs=10,
                steps_per_epoch=1000,
                initial_epoch=0)
  else:
    raise ValueError('Mode not found: %s.' % FLAGS.mode)

def estimator_run(params, train_input_fn):
    ssd_detector = tf.estimator.Estimator(
        model_fn=build_model_fn,
        model_dir="/algo_proj/changming/model-zoo/tmp-dt",
        params=params
    )
    ssd_detector.train(input_fn=train_input_fn, max_steps=10000)


def run(callbacks=None):
  keras_utils.set_session_config(enable_xla=FLAGS.enable_xla)

  params = config_factory.config_generator(FLAGS.model)

  params = params_dict.override_params_dict(
      params, FLAGS.config_file, is_strict=True)

  params = params_dict.override_params_dict(
      params, FLAGS.params_override, is_strict=True)
  params.override(
      {
          'strategy_type': FLAGS.strategy_type,
          'model_dir': FLAGS.model_dir,
          'strategy_config': executor.strategy_flags_dict(),
      },
      is_strict=False)
  params.validate()
  params.lock()
  pp = pprint.PrettyPrinter()
  params_str = pp.pformat(params.as_dict())
  logging.info('Model Parameters: {}'.format(params_str))

  train_input_fn = None
  eval_input_fn = None
  training_file_pattern = FLAGS.training_file_pattern or params.train.train_file_pattern
  eval_file_pattern = FLAGS.eval_file_pattern or params.eval.eval_file_pattern
  if not training_file_pattern and not eval_file_pattern:
    raise ValueError('Must provide at least one of training_file_pattern and '
                     'eval_file_pattern.')

  if training_file_pattern:
    # Use global batch size for single host.
    train_input_fn = input_reader.InputFn(
        file_pattern=training_file_pattern,
        params=params,
        mode=input_reader.ModeKeys.TRAIN,
        batch_size=params.train.batch_size)

  if eval_file_pattern:
    eval_input_fn = input_reader.InputFn(
        file_pattern=eval_file_pattern,
        params=params,
        mode=input_reader.ModeKeys.PREDICT_WITH_GT,
        batch_size=params.eval.batch_size,
        num_examples=params.eval.eval_samples)
  estimator_run(params, train_input_fn)
  # return run_executor(
  #     params,
  #     train_input_fn=train_input_fn,
  #     eval_input_fn=eval_input_fn,
  #     callbacks=callbacks)


def main(argv):
  del argv  # Unused.

  run()


if __name__ == '__main__':
  #assert tf.version.VERSION.startswith('2.')
  #tf.config.set_soft_device_placement(True)
  app.run(main)
