import argparse
import random
import json
import requests
import re
import numpy as np
def add_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument("--query", default='./tmp/data/data', type=str, required=False,
                        help="query file")

    parser.add_argument("--train_file", default='./tmp/data/train', type=str, required=False,
                        help="train file")

    parser.add_argument("--test_file", default='./tmp/data/test', type=str, required=False,
                        help="test file")

    parser.add_argument("--class_file",default='tmp/data/class',type=str, required=False,
                        help="class file")

    parser.add_argument("--max_len", default=20, type=int, required=False,
                        help="the max len of factor")

    args = parser.parse_args()
    return args


def get_candidate(query):
    url = "http://10.25.80.121:8888/v2/recall?query="+query
    headers = {
        "Content-Type": "application/json; charset=UTF-8",
        "Referer": "http://jinbao.pinduoduo.com/index?page=5",
        "User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.87 Safari/537.36",
    }
    req_arg = {}
    repose = requests.post(url, data=json.dumps(req_arg), headers=headers)
    data=repose.json()['data']['elements']
    return data

def get_all_elements():
    all_elements=set()
    with open(args.query, encoding='utf-8', errors='ignore') as fp:
        for line in fp:
            line=line.strip().lower()
            if len(line)==0:
                continue
            line=line.split('\t')
            text = line[-1]
            elements=set()
            for element in line[:-1]:
                element = re.sub(r'#|\?|,|!|？|！|√|，', '', element)
                if len(element)>args.max_len:
                    continue
                elements.add(element)
            all_elements.update(elements)
    with open(args.class_file,'w',encoding='utf-8', errors='ignore') as fp:
        for element in all_elements:
            fp.write(element+'\n')
    return list(all_elements)

def get_neg_label(query,label):
    # print(query)
    all_neg=set()
    recall=set()
    for element in get_candidate(query):
        element = re.sub(r'#|\?|,|!|？|！|√|，', '', element)
        if len(element) > args.max_len:
            continue
        recall.add(element)
    # print(recall)
    recall.update(['boolean', 'where', 'how', 'what', 'why', 'number', 'period', 'who', 'condition'])
    recall.update(list(np.random.choice(all_elements, size=2*len(recall), replace=False)))
    for y in label:
        if y in recall:
            recall.remove(y)
    for y in recall:
        if y in all_elements:
            all_neg.add(y)

    return list(all_neg)



if __name__=='__main__':
    # get_candidate("海外债券四期3号c最新净值是多少")
    args=add_argument()
    all_elements=get_all_elements()

    with open(args.query,encoding='utf-8',errors='ignore') as fp,\
        open(args.train_file,'w',encoding='utf-8',errors='ignore') as fwp_train,\
        open(args.test_file,'w',encoding='utf-8',errors='ignore') as fwp_test:
        for line in fp:
            line=line.strip().lower()
            if len(line)==0:
                continue
            line=line.split('\t')
            text = line[-1]
            elements=set()
            for element in line[:-1]:
                element = re.sub(r'#|\?|,|!|？|！|√|，', '', element)
                if len(element)>args.max_len:
                    continue
                elements.add(element)
            label=list(elements)
            neg_label = get_neg_label(text, label)
            if random.random()<0.01:
                fwp_test.write('text\t' + text + '\t' + '#alignment\t'.join(label) \
                      + '#alignment\t' + '#no_alignment\t'.join(neg_label) + '#no_alignment\n')
            else:
                fwp_train.write('text\t' + text + '\t' + '#alignment\t'.join(label) \
                      + '#alignment\t' + '#no_alignment\t'.join(neg_label) + '#no_alignment\n')
