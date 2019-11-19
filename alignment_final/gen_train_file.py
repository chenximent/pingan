# coding=utf-8
import pandas as pd
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

rows=readExcel("es_data.xlsx",["intention","script"],"intention","script")
print (rows)
generate_train_file(rows,"liutao")