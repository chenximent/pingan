import argparse
import re
import random

def add_argument():
    parser = argparse.ArgumentParser()


    parser.add_argument("--query", default='./tmp/data/data', type=str, required=False,
                        help="query file")

    parser.add_argument("--train_file", default='./tmp/data/train', type=str, required=False,
                        help="train file")

    parser.add_argument("--test_file", default='./tmp/data/test', type=str, required=False,
                        help="test file")

    parser.add_argument("--class_file", default='./tmp/data/class', type=str, required=False,
                        help="class file")

    parser.add_argument("--max_len", default=20, type=int, required=False,
                        help="the max len of factor")

    args = parser.parse_args()
    return args



if __name__=='__main__':
    args=add_argument()
    all_elements=set()
    with open(args.query,encoding='utf-8',errors='ignore') as fp,\
        open(args.train_file,'w',encoding='utf-8',errors='ignore') as fwp_train,\
        open(args.test_file,'w',encoding='utf-8',errors='ignore') as fwp_test,\
        open(args.class_file,'w',encoding='utf-8',errors='ignore') as fwp_class:
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
            all_elements.update(elements)
            if random.random()<0.01:
                fwp_test.write(text+'\t'+'\t'.join(label)+'\n')
            else:
                fwp_train.write(text+'\t'+'\t'.join(label)+'\n')

        for element in all_elements:
            fwp_class.write(element+'\n')

