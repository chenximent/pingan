# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""Run BERT on SQuAD 1.1 and SQuAD 2.0."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import math
import os
import random
import modeling
import optimization as optimization
import re
import tokenization
import six
import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

# flags.DEFINE_string(
#     "export_dir", None,
#     "The output directory where the model pb will be written.")
## Other parameters
flags.DEFINE_string("train_file", None,
                    "SQuAD json for training. E.g., train-v1.1.json")

flags.DEFINE_string(
    "predict_file", None,
    "SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")

flags.DEFINE_string(
    "eval_file", None,
    "SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 100,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_predict", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_export", False, "Whether to save model with pb file.")
flags.DEFINE_bool("do_eval", False, "Whether to run eval.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("predict_batch_size", 1,
                     "Total batch size for predictions.")

flags.DEFINE_integer("eval_batch_size", 1, "Total batch size for eval.")


flags.DEFINE_float("learning_rate", 1e-3, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 10.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


label_list=['alignment','no_alignment']

class GenExample(object):
    """A single training/test example for simple sequence classification.

       For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 text_a,
                 text_b,
                 label=None):
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "src_text: %s" % (
            tokenization.printable_text(self.text_a))
        s += ", tar_text: %s" % (tokenization.printable_text(self.text_b))
        s += ", label: %s" % (self.label)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 segment_ids,
                 input_mask,
                 label=None,
                 weight=None
                 ):
        self.input_ids = input_ids
        self.segment_ids = segment_ids
        self.input_mask = input_mask
        self.label = label
        self.weight=weight


def read_gen_examples(input_file):
    examples = []
    with tf.gfile.Open(input_file, "r") as reader:
        for line in reader:
            line = line.strip().lower().split('\t')
            text = line[1]
            for fac in line[2:]:
                fac=fac.strip().split('#')
                if fac[1]=='alignment':
                    example = GenExample(text, fac[0],'alignment')
                if fac[1]=='no_alignment':
                    example = GenExample(text, fac[0],'no_alignment')
                    
                examples.append(example)


    return examples



def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 output_fn):
    """Loads a data file into a list of `InputBatch`s."""
    label_map = {label: i for i, label in enumerate(label_list)}
    memory={}
    for (example_index, example) in enumerate(examples):
        if example.text_a not in memory:
            tokens_a = tokenizer.tokenize(example.text_a)
            memory[example.text_a]=tokens_a
        else:
            tokens_a = memory[example.text_a]
        if example.text_b not in memory:
            tokens_b = tokenizer.tokenize(example.text_b)
            memory[example.text_b] = tokens_b
        else:
            tokens_b = memory[example.text_b]
        tokens_a_tmp = tokens_a[:min(len(tokens_a),max_seq_length-3-len(tokens_b))]
        tokens=["[CLS]"]+tokens_a_tmp+["[SEP]"]+tokens_b+["[SEP]"]
        segment_ids = [0]*(len(tokens_a_tmp)+2)+[1]*(len(tokens_b)+1)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        label = label_map[example.label]
        weight= 1 if example.label=='alignment' else 1

        if example_index < 20:
            tf.logging.info("*** Example ***")
            tf.logging.info("tokens: %s" % (' '.join(tokens)))
            tf.logging.info("input_ids: %s" % (' '.join([str(x) for x in input_ids])))
            tf.logging.info("segment_ids: %s" % (' '.join([str(x) for x in segment_ids])))
            tf.logging.info("input_mask: %s" % (' '.join([str(x) for x in input_mask])))
            tf.logging.info("label: %s, label_id: %d" % (example.label,label))
            tf.logging.info("weight: %d" % (weight))

        feature = InputFeatures(input_ids=input_ids,
                                input_mask=input_mask,
                                segment_ids=segment_ids,
                                label=label,
                                weight=weight)

        # Run callback
        output_fn(feature)



def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,label,weight,
                 use_one_hot_embeddings):
    """Creates a classification model."""
    with tf.variable_scope('SIM_1') as scope:
        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings,
            scope=scope)
        final_hidden = model.get_pooled_output()
        logits=tf.layers.dense(final_hidden,2,kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               name='output')
        prob=tf.reduce_max(tf.nn.softmax(logits=logits),axis=-1)
        pred=tf.argmax(logits,axis=-1)
        #weight=tf.cast(weight,tf.float32)
        #loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label,logits=logits)*weight)

        def focal_loss(logits, labels, alpha=0.25, gamma=2):
            epsilon = 1.e-7
            targets = tf.one_hot(labels, depth=2, dtype=tf.float32)
            p = tf.clip_by_value(tf.nn.softmax(logits=logits), epsilon, 1. - epsilon)
            p_t = p*targets+(1.-targets)*(1.-p)
            alpha_t = alpha*targets+(1.-alpha)*(1.-targets)

            f_l = - alpha_t * tf.pow(1 - p_t, gamma) *tf.log(p_t) # * tf.expand_dims(weights, axis=2)

            return tf.reduce_sum(f_l)

        loss = focal_loss(logits, label)
    return prob,pred,loss

def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
  """Compute the union of the current variables and checkpoint variables."""
  assignment_map = {}
  initialized_variable_names = {}

  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
    name_to_variable[name] = var

  init_vars = tf.train.list_variables(init_checkpoint)

  assignment_map = collections.OrderedDict()
  for x in init_vars:
    (name, var) = (x[0], x[1])
    name_tmp='SIM_1'+name[4:]
    if name_tmp not in name_to_variable:
      continue
    assignment_map[name] = name_tmp
    initialized_variable_names[name_tmp] = 1
    initialized_variable_names[name_tmp + ":0"] = 1

  return (assignment_map, initialized_variable_names)

def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label = features["label"]
        weight=features["weight"]

        prob, pred, loss = create_model(
            bert_config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label=label,
            weight=weight,
            use_one_hot_embeddings=use_one_hot_embeddings)

        tvars = tf.trainable_variables()

        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            # print(assignment_map)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op,lr = optimization.create_optimizer(
                loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            logging_hook = tf.train.LoggingTensorHook({"loss": loss,"learning_rate":lr, "steps":tf.train.get_or_create_global_step()}, every_n_iter=1000)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                training_hooks=[logging_hook],
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(label_ids, preds):
                accuracy = tf.metrics.accuracy(
                    labels=label_ids, predictions=preds)
                return {
                    "eval_accuracy": accuracy
                }

            eval_metrics = (metric_fn,
                            [label, pred])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:

            predictions = {
                "input_ids": input_ids,
                "label":label,
                "prob": prob,
                "pred": pred
            }
            if FLAGS.do_export:
                predictions = {
                    "prob": prob,
                    "pred": pred
                }
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)

        return output_spec

    return model_fn


def input_fn_builder(input_file, seq_length, is_training, drop_remainder,batch_size):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label": tf.FixedLenFeature([], tf.int64),
        "weight":tf.FixedLenFeature([], tf.int64)
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def write_predictions(tokenizer, all_results, output_prediction_file):
    """Write final predictions to the json file and log-odds of null if needed."""
    tf.logging.info("Writing predictions to: %s" % (output_prediction_file))
    with tf.gfile.GFile(output_prediction_file, "w") as writer:
        for result in all_results:
            tokens = tokenizer.convert_ids_to_tokens(result["input_ids"])
            if "[PAD]" in tokens:
                tokens_pad = tokens.index("[PAD]")
                text = ''.join(tokens[1:tokens_pad])
            else:
                text = ''.join(tokens[1:])
            text=re.sub("\[SEP\]",'\t',text)

            writer.write(text+str(label_list[result['label']])+'\t'+str(label_list[result['pred']])+'\n')


class FeatureWriter(object):
    """Writes InputFeature to TF example file."""

    def __init__(self, filename):
        self.filename = filename
        self.num_features = 0
        self._writer = tf.python_io.TFRecordWriter(filename)

    def process_feature(self, feature):
        """Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
        self.num_features += 1

        def create_int_feature(values):
            feature = tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(values)))
            return feature

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label"] = create_int_feature([feature.label])
        features["weight"] = create_int_feature([feature.weight])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        self._writer.write(tf_example.SerializeToString())

    def close(self):
        self._writer.close()


def validate_flags_or_throw(bert_config):
    """Validate the input FLAGS or throw an exception."""
    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)

    if not FLAGS.do_train and not FLAGS.do_predict and not FLAGS.do_eval:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if FLAGS.do_train:
        if not FLAGS.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")

    if FLAGS.do_eval:
        if not FLAGS.eval_file:
            raise ValueError(
                "If `do_eval` is True, then `eval_file` must be specified.")

    if FLAGS.do_predict:
        if not FLAGS.predict_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_file` must be specified.")

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    validate_flags_or_throw(bert_config)

    tf.gfile.MakeDirs(FLAGS.output_dir)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        train_examples = read_gen_examples(
            input_file=FLAGS.train_file)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        print(len(train_examples),num_train_steps)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
        rng = random.Random(12345)
        rng.shuffle(train_examples)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        predict_batch_size=FLAGS.predict_batch_size,
        eval_batch_size = FLAGS.eval_batch_size)

    if FLAGS.do_train:
        # We write to a temporary file to avoid storing very large constant tensors
        # in memory.
        train_writer = FeatureWriter(
            filename=os.path.join(FLAGS.output_dir, "train.tf_record"))
        convert_examples_to_features(
            examples=train_examples,
            tokenizer=tokenizer,
            max_seq_length=FLAGS.max_seq_length,
            output_fn=train_writer.process_feature)
        train_writer.close()

        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num orig examples = %d", len(train_examples))
        tf.logging.info("  Num split examples = %d", train_writer.num_features)
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        del train_examples

        train_input_fn = input_fn_builder(
            input_file=train_writer.filename,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True,
            batch_size=FLAGS.train_batch_size)
        # train_input_fn = input_fn_builder(
        #     input_file=os.path.join(FLAGS.output_dir, "train.tf_record"),
        #     seq_length=FLAGS.max_seq_length,
        #     is_training=True,
        #     drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval:
        eval_examples = read_gen_examples(
            input_file=FLAGS.eval_file)

        eval_writer = FeatureWriter(
            filename=os.path.join(FLAGS.output_dir, "eval.tf_record"))
        eval_features = []

        def append_feature(feature):
            eval_features.append(feature)
            eval_writer.process_feature(feature)

        convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=FLAGS.max_seq_length,
            output_fn=append_feature)
        eval_writer.close()

        tf.logging.info("***** Running eval *****")
        tf.logging.info("  Num orig examples = %d", len(eval_examples))
        tf.logging.info("  Num split examples = %d", len(eval_features))
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        eval_input_fn = input_fn_builder(
            input_file=eval_writer.filename,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False,
            batch_size=FLAGS.eval_batch_size)

        result = estimator.evaluate(input_fn=eval_input_fn)

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if FLAGS.do_predict:
        eval_examples = read_gen_examples(
            input_file=FLAGS.predict_file)

        eval_writer = FeatureWriter(
            filename=os.path.join(FLAGS.output_dir, "pred.tf_record"))
        eval_features = []

        def append_feature(feature):
            eval_features.append(feature)
            eval_writer.process_feature(feature)

        convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=FLAGS.max_seq_length,
            output_fn=append_feature)
        eval_writer.close()

        tf.logging.info("***** Running predictions *****")
        tf.logging.info("  Num orig examples = %d", len(eval_examples))
        tf.logging.info("  Num split examples = %d", len(eval_features))
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)


        predict_input_fn = input_fn_builder(
            input_file=eval_writer.filename,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False,
            batch_size=FLAGS.predict_batch_size)

        # If running eval on the TPU, you will need to specify the number of
        # steps.
        all_results = []
        for result in estimator.predict(
                predict_input_fn, yield_single_examples=True):
            if len(all_results) % 1000 == 0:
                tf.logging.info("Processing example: %d" % (len(all_results)))
            all_results.append(
                result)

        output_prediction_file = os.path.join(FLAGS.output_dir, "predictions.res")

        write_predictions(tokenizer, all_results, output_prediction_file)
    if FLAGS.do_export:
        estimator._export_to_tpu = False
        estimator.export_savedmodel(FLAGS.output_dir, serving_input_receiver_fn=serving_input_fn_receiver)
def serving_input_fn_receiver():
    input_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='input_ids')
    input_mask = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='input_mask')
    segment_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='segment_ids')
    label = tf.placeholder(tf.int32, [None], name='label')
    weight = tf.placeholder(tf.int32, [None], name='weight')
    input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
        {
            'input_ids': input_ids,
            'input_mask': input_mask,
            'segment_ids': segment_ids,
            'label': label,
            'weight':weight
        }
    )()

    return input_fn


if __name__ == "__main__":
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
