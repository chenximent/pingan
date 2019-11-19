# coding=utf-8
import pandas as pd
import random
def generate_train_file(rows,train_file):
    try:
        inverted_index={}
        negative_elements_list=[]
        all_elements=[]
        for i in range(0, len(rows)):
            elements=rows[i]["elements"]
            for e in elements:
                if e not in all_elements:all_elements.append(e)

        ######3+,-样本的生成
        for i in range(0, len(rows)):
            elements=rows[i]["elements"]
            negative_elements=[]
            for e in all_elements:
                if e not in elements:
                    negative_elements.append(e)
            negative_elements_list.append(negative_elements)

        ######4.写文件
        with open(train_file, 'w', encoding='utf-8', errors='ignore') as fp:
         for i in range(0, len(rows)):
              text=rows[i]["extends"]
              elements=rows[i]["elements"]
              negative_elements=negative_elements_list[i]
              end="\n"
              if i == len(rows)-1:end=""
              fp.write('text\t' + text + '\t' + '#alignment\t'.join(elements) \
                       + '#alignment\t' + '#no_alignment\t'.join(negative_elements) + '#no_alignment'+end)
    except Exception as e:
        print(str(e))
        return False
    return True

#rows=[{"extends":"aaaa","elements":["A","B"]},{"extends":"aaaa2","elements":["A2","B2"]},{"extends":"aaaa3","elements":["A3","B3"]}]

def readExcel(file,columns,intention,script):
    df = pd.read_excel(file)
    rows=[]
    intentions = df[intention].tolist()
    scripts = df[script].tolist()

    for i in range(0, len(intentions)):
        rows.append({"extends":scripts[i],"elements":[intentions[i]]})
    return rows

def split(full_list,shuffle=False,ratio=0.2):
      n_total = len(full_list)
      offset = int(n_total * ratio)
      if n_total==0 or offset<1:
          return [],full_list
      if shuffle:
          random.shuffle(full_list)
      sublist_1 = full_list[:offset]
      sublist_2 = full_list[offset:]
      return sublist_1,sublist_2

def split_dat(rows,ratio):
    map={}
    train_rows=[]
    test_rows=[]
    for i in range(0, len(rows)):
        element = rows[i]["elements"][0]
        if element in map:
            l=map[element]
            l.append(rows[i]["extends"])
            map[element] = l
        else:
            map[element] = [rows[i]["extends"]]

    for key in map:
        l=map[key]
        train,test=split(l, shuffle=True, ratio=ratio)
        for t in train:
            train_rows.append({"extends":t,"elements":[key]})

        for t in test:
            test_rows.append({"extends":t,"elements":[key]})

        print("train "+str(len(train)))
        print(train)
        print("test "+str(len(test)))
        print(test)
    return train_rows,test_rows

def generate_test_file(rows,testfile):
    with open(testfile, 'w', encoding='utf-8', errors='ignore') as fp:

        for i in range(0, len(rows)):
            ##print(rows[i])
            intention = rows[i]["elements"][0]
            script=rows[i]["extends"]
            fp.write(intention+"###"+script+"\n")

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


excel="es_data.xlsx"
train_file="liutao2"
test_file="test2"
ratio=0.9
rows=readExcel("es_data.xlsx",["intention","script"],"intention","script")
print (rows)

train,test=split_dat(rows,ratio)
print("===============")
##print(train)
generate_train_file(train,train_file)
print("===============>>>>>>>")
##print(test)
generate_test_file(test,test_file)

'''
a=[1,2,3,4,5,6,7,8,9,0]
x,y=split(a, shuffle=True, ratio=0.1)
print(x)
print(y)
'''

l=read_test_file(test_file)
print(l)