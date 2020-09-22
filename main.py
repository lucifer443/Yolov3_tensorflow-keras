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
from modeling.hyperparams import params_dict
from utils import hyperparams_flags
from configs import factory as config_factory
from dataloader import input_reader
from dataloader import mode_keys as ModeKeys
from modeling import factory as model_factory

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


def estimator_run(params, train_input_fn):
    ssd_detector = tf.estimator.Estimator(
        model_fn=build_model_fn,
        model_dir=FLAGS.model_dir,
        params=params
    )
    ssd_detector.train(input_fn=train_input_fn, max_steps=10000)

def run_executor(params, mode, train_input_fn, callbacks):
    model_builder = model_factory.model_generator(params)
    model = model_builder.build_model(params, mode=mode)
    loss_fn = [model_builder.build_cls_loss_fn(), model_builder.build_box_loss_fn()]
    model.compile(optimizer=model.optimizer,
                  loss=loss_fn)
    model.fit(train_input_fn(), epochs=10, steps_per_epoch=100, callbacks=callbacks)


def run(callbacks=None):
  # keras_utils.set_session_config(enable_xla=FLAGS.enable_xla)

  params = config_factory.config_generator(FLAGS.model)

  params = params_dict.override_params_dict(
      params, FLAGS.config_file, is_strict=True)

  params = params_dict.override_params_dict(
      params, FLAGS.params_override, is_strict=True)
  params.override(
      {
          'strategy_type': FLAGS.strategy_type,
          'model_dir': FLAGS.model_dir,
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
  # estimator_run(params, train_input_fn)
  return run_executor(
      params,
      mode=ModeKeys.TRAIN,
      train_input_fn=train_input_fn,
      callbacks=callbacks)


def main(argv):
  del argv  # Unused.

  run()


if __name__ == '__main__':
  #assert tf.version.VERSION.startswith('2.')
  #tf.config.set_soft_device_placement(True)
  app.run(main)
