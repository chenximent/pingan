# encoding:utf-8

import requests
import json
import urllib.request
from urllib.parse import urlencode
def read_test_file(testfile):
    l=[]
    file = open(testfile)
    for line in file:
        if len(str(line).strip())>0:
            line=str(line).strip()
            arr=line.split("###")
            l.append({"i":arr[0],"s":arr[1]})

    file.close()
    return l
l=read_test_file("test2")

num=0
total=0
for d in l:
    i=d["i"]
    s=d["s"]
    url = "http://localhost:7771/sim_intention"
    data = {"userInput":"哦，对，我现在盘点，没空","intentions":["I-08敏感拒绝","I-04不好"]}


    headers = {
        'User-Agent': 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'
    }

    ##data = urllib.parse.urlencode(data).encode('utf-8')
    ##data = urlencode(data, encoding='utf-8')
    data = json.dumps(data)

    ret = requests.post(url,data=data,headers=headers)
    ##print(ret.text)

    d=json.loads(ret.text,encoding='utf-8')
    if d["data"]["top1_intention"]["intention"]==i:num=num+1
    total=total+1
    print(float(d["data"]["top1_intention"]["score"]))
    print(d["data"]["top1_intention"])
    print(i)
    print("right:"+str(float(num)/total))

