from __future__ import print_function
import tornado.web
import tornado.web
import grpc
from tensorflow_serving.apis import get_model_metadata_pb2
from tensorflow_serving.apis import model_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow_serving.apis import predict_pb2
import tensorflow as tf
import os
from . import tokenization
import time

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
dir = root_path + "/controller"

flags = tf.flags
FLAGS = flags.FLAGS
##flags.DEFINE_string("vocab_file_muti", root_path + '/vocab.txt',"The vocabulary file that the BERT model was trained on.")
flags.DEFINE_integer(
    "max_seq_length", 40,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")
flags.DEFINE_string("port", 'port', "port")

tokenizer = tokenization.FullTokenizer(vocab_file=dir + '/vocab.txt', do_lower_case=True)

def process_query(query,com):
    tokens_a = tokenizer.tokenize(query)
    tokens_com = tokenizer.tokenize(com)
    tokens_a = tokens_a[:min(len(tokens_a),(FLAGS.max_seq_length - 3-len(tokens_com)))]
    tokens = ["[CLS]"]+tokens_a+["[SEP]"]+tokens_com+["[SEP]"]
    segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_com) + 1)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < FLAGS.max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
    label = 1
    assert len(input_ids) == FLAGS.max_seq_length
    assert len(input_mask) == FLAGS.max_seq_length
    assert len(segment_ids) == FLAGS.max_seq_length
    return input_ids,input_mask,segment_ids,label,1


def tf_serving_grpc_muticlass(model_name, grpc_target, query, elements):
        """
        tf_serving grpc调用方式
        Parameters:
          model_name(str):模型名称
          query(str) - 用户输入的句子
          elements(list) - 召回的要素集合
          grpc_target（str）:grpc ip:port
        Returns:
        	sim_score,prob:要素是否命中 要素得分值
        """
        start = int(round(time.time() * 1000))

        with grpc.insecure_channel(str(grpc_target)) as channel:
            stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
            '''
            meta_request = get_model_metadata_pb2.GetModelMetadataRequest(
                model_spec=model_pb2.ModelSpec(name=model_name),
                metadata_field=['signature_def']
            )
            meta_response = stub.GetModelMetadata(meta_request)
            '''

            all_input_ids, all_input_mask, all_segment_ids, all_labels, all_weight = [], [], [], [], []
            for element in elements:
                input_ids, input_mask, segment_ids, label, weight = process_query(query, element)
                all_input_ids.append(input_ids)
                all_input_mask.append(input_mask)
                all_segment_ids.append(segment_ids)
                all_labels.append(label)
                all_weight.append(weight)
            print(all_input_ids)
            print(type(all_input_ids[0]))
            predict_request = predict_pb2.PredictRequest(model_spec=model_pb2.ModelSpec(name=model_name),
                                                         inputs={'input_ids': tf.make_tensor_proto(all_input_ids,
                                                                                                   dtype=tf.int32),
                                                                 'input_mask': tf.make_tensor_proto(all_input_mask),
                                                                 'segment_ids': tf.make_tensor_proto(all_segment_ids),
                                                                 'label': tf.make_tensor_proto(all_labels),
                                                                 'weight': tf.make_tensor_proto(all_weight)})
            predict_response = stub.Predict(predict_request)
            sim_score = tf.make_ndarray(predict_response.outputs["pred"])
            prob = tf.make_ndarray(predict_response.outputs["prob"])
            end = int(round(time.time() * 1000))
            print("tf_serving_aligment_grpc waste time=%s ms"% str(end - start))

            return sim_score, prob



