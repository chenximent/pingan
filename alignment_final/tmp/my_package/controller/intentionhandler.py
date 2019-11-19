from __future__ import print_function
import tornado.web
import tornado.web
from util import tokenization
import time
from util.response import ResponseMixin
import numpy as np
from constant.constant import logger
import traceback
from config_new import settings
from tornado.concurrent import run_on_executor
from concurrent.futures import ThreadPoolExecutor
from tensorflow_serving.apis import get_model_metadata_pb2
from tensorflow_serving.apis import model_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow_serving.apis import predict_pb2
import tensorflow as tf
import grpc
import json
from myelasticsearch.myelasticsearch import esclient
from util.tool import process_query
from .MutiClass import tf_serving_grpc_muticlass
class Intention(tornado.web.RequestHandler,ResponseMixin):
    executor = ThreadPoolExecutor(max_workers=10)
    model_name = settings.SIM["model_name"]
    grpc_target = settings.SIM["grpc_target"]
    indexname = settings.SIM["indexname"]
    indextype = settings.SIM["indextype"]


    @tornado.gen.coroutine
    def post(self, *args, **kwargs):
        try:
            ###之前全量意图接口的入参是form-data格

            start = int(round(time.time() * 1000))
            data =  self.request.body.decode('utf-8')
            req_arg = json.loads(data,encoding="utf-8")
            userInput=req_arg["userInput"]
            intentions=req_arg["intentions"]

            if userInput is None or len(str(userInput))==0:
                logger.info ("userInput can not be empty")
                raise Exception("userInput can not be empty")
            else:
                    sim_score, prob = tf_serving_grpc_muticlass(self.model_name,
                                                                self.grpc_target,
                                                                userInput,
                                                                intentions)
                    s = ""
                    for i in range(len(sim_score)):
                        score = 0
                        ###0命中元素  1不命中元素
                        if sim_score[i] == 1:prob[i]=float(1-prob[i])
                        else:prob[i]=float(prob[i])
                    data={}
                    max={}
                    mscore=0.0
                    data["topn_intention"]=[]
                    for index, item in enumerate(prob):
                            d={}
                            d["intention"]=intentions[index]
                            d["score"]=str(prob[index])
                            data["topn_intention"].append(d)

                            if prob[index] >= mscore:
                                mscore=prob[index]
                                max=d
                    data["top1_intention"]=max

                    self.success = True
                    self.status = '00000'
                    print(data)
                    self.write(self.get_json_response(data))
                    end = int(round(time.time() * 1000))

                    logger.info("%s Intention waste time=%s ms"%
                          (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),str(end - start)))
        except Exception as e:
            ##logger.error(e)
            traceback.format_exc()
            logger.info(str(traceback.format_exc()))
            self.success = False
            self.status = '10000'
            self.msg=traceback.format_exc()
            self.write(self.get_json_response({}))
