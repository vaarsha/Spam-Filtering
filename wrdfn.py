import numpy as np 

def extractor(lst):
    wrds = []
    # Spliting the words in a line
    for i,ln in enumerate(lst):
        if i == 2:
            wrds += ln.split()

    return wrds

def pop_words(d, pop_val):
    #div = sorted(d.values())[::-1]
    # Most Popular Words
    l = sorted(d.items(),key=lambda x:x[-1],reverse=True)[:pop_val]

    return l

def words_freq(wrds):
    fn_wrds = []
    wrds_dict = {}

    #print(wrds)
    # Unique words list
    for j in wrds:
        if (j not in fn_wrds) and (j.isalpha()) and (len(j) != 1):
            fn_wrds.append(j)

    #print("Unique word list ",len(fn_wrds))

    for f_w in fn_wrds:
        cnt = 0
        for w in wrds:
            if w == f_w:
                cnt += 1
        wrds_dict[f_w] = cnt
    return wrds_dict

# Word count vector contains the frequency of most popular words in the training file
def extract_features(wrd_ls, key_ls, flen):
    f_mtx = np.zeros((flen,len(key_ls)))
    #print("Dimensions of feature vector",np.shape(f_mtx))

    for i,k in enumerate(wrd_ls):
        for j,w in enumerate(key_ls):
            f_mtx[i,j] = k.count(w[0])

    return f_mtx
