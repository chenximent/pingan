if __name__=='__main__':
    m=21
    an = 1
    cor=0
    num=0
    r_cor=0
    r_num=0
    p_cor=0
    p_num=0
    with open('./tmp/predictions.res',encoding='utf-8',errors='ignore') as fp:
        for cur,line in enumerate(fp):
            line=line.strip().split('\t')
            answer,pred=line[2],line[3]
            c = cur%m
            an*=1 if answer==pred else 0
            if c==m-1:
                cor+=an
                num+=1
                an=1
            if answer=='alignment':
                r_num+=1
                if pred=='alignment':
                    r_cor +=1
            if answer=='no_alignment':
                p_num+=1
                if pred=='no_alignment':
                    p_cor +=1

    print(cor/num)
    print(r_cor/r_num)
    print(p_cor/p_num)
