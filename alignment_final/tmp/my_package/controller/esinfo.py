import tornado
import re
from myelasticsearch.myelasticsearch import esclient
from config_new import settings

from constant.constant import logger
import traceback
class ESInfo(tornado.web.RequestHandler):
    def get(self):
        try:
            indexname = settings.SIM["indexname"]
            indextype = settings.SIM["indextype"]
            count = esclient.index_count(indexname, indextype)
            self.write(indexname + " Index Sucess. " + str(count) + " docs.")

        except Exception as e:
            logger.error(traceback.format_exc())
            self.write(str(e))
