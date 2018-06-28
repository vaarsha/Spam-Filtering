import gzip
import random, os, sys
import numpy as np

import time
import multiprocessing
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score, f1_score, recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from math import floor

from wrdfn import * 

res_f1  = []
res_ac = []
res_pr = []
res_t = []
nm = []

def set_popval(fl_list):

    cmpl_words = []
    fn_wrds = []
    
    #print("Setting pop word cnt")

    for f1 in fl_list :
        with gzip.open(os.path.join(dt, f1), 'rt') as f:
            words = f.read().splitlines()
            ls1 = extractor(words)
            cmpl_words.extend(ls1)

    for j in cmpl_words:
        if (j not in fn_wrds) and (j.isalpha()) and (len(j) != 1):
            fn_wrds.append(j)

    ret = floor(len(fn_wrds)/1.5)
    #print(ret)

    return ret

# Tag spam mails
def dataset_labels(ls, flen):
    l_mtx = np.zeros(flen)

    for i, k in enumerate(ls):
        l_mtx[i] = 0
        if 'spmsg' in k:
            l_mtx[i] = 1

    return l_mtx

'''
def matrix_lb(fpath):
    fl_list = os.listdir(fpath)
    fl_len = len(fl_list)
    ls = list()
    cmpl_words = list()
    cmpl_dict = dict()
    
    for f1 in fl_list :
        with gzip.open(os.path.join(fpath, f1), 'rt') as f:
            words = f.read().splitlines()
            ls1 = extractor(words)
            cmpl_words.extend(ls1)
            ls.append(ls1)
    
    cmpl_dict = words_freq(cmpl_words)
    pop_ls = pop_words(cmpl_dict)
    matrix = extract_features(ls, pop_ls, fl_len)
    label = dataset_labels(fl_list, fl_len)

    return matrix, label
'''

def matrix_lb(fl_list):
    fl_len = len(fl_list)
    ls = list()
    cmpl_words = list()
    cmpl_dict = dict()
    
    for f1 in fl_list :
        with gzip.open(os.path.join(dt, f1), 'rt') as f:
            words = f.read().splitlines()
            ls1 = extractor(words)
            cmpl_words.extend(ls1)
            ls.append(ls1)
    
    cmpl_dict = words_freq(cmpl_words)
    pop_ls = pop_words(cmpl_dict, pop_val)
    matrix = extract_features(ls, pop_ls, fl_len)
    label = dataset_labels(fl_list, fl_len)

    return matrix, label

def model_pr(fname, name, model):
    t1 = time.time()
    model.fit(tr_matrix, tr_lb)
    pred_lb = model.predict(ts_matrix)
    res1 = f1_score(ts_lb, pred_lb, average = 'micro')
    res2 = precision_score(ts_lb, pred_lb, average = 'micro')
    res3 = recall_score(ts_lb, pred_lb, average = 'micro')

    res_f1.append(res1*100)
    res_ac.append(res2*100)
    res_pr.append(res3*100)

    nm.append(name)
    print(("Precision is %.5f")%res2)
    print(("Recall Score is %.5f")%res3)
    print(fname, "%.5f"%res1)
    t2 = time.time()
    res4 = t2-t1
    res_t.append(res4)
    print("Run time is %.4f \n"% (res4))

def plot_graph(results, lb):
    # Plotting graph
    y_axis = np.arange(len(nm))
    
    plt.bar(y_axis, results, align='center')
    plt.xticks(y_axis, nm)
    plt.ylabel(lb)
    plt.title("Model Comparisons")
    
    plt.show()

def stacking_func():
    model2 = GaussianNB()
    model1 = SVC()

    t1 = time.time()
    model1.fit(tr_matrix, tr_lb)
    model2.fit(tr_matrix, tr_lb)
    
    pred1 = model1.predict(vl_matrix)
    pred2 = model2.predict(vl_matrix)

    ts_pred1 = model1.predict(ts_matrix)
    ts_pred2 = model2.predict(ts_matrix)

    stacked_pred = np.column_stack((pred1,pred2))
    test_stacked_pred = np.column_stack((ts_pred1,ts_pred2))
    
    model1.fit(stacked_pred, vl_lb)

    final_pred = model1.predict(test_stacked_pred)

    res1 = f1_score(ts_lb, final_pred, average = 'micro')
    res2 = precision_score(ts_lb, final_pred, average = 'micro')
    res3 = recall_score(ts_lb, final_pred, average = 'micro')
    print("SVM-NB")
    print(("Precision is %.5f")%res2)
    print(("Recall Score is %.5f")%res3)
    print("Final predictions %.5f"%res1)
    
    res_f1.append(res1*100)
    res_ac.append(res2*100)
    res_pr.append(res3*100)

    nm.append("SVM-NB")

    t2 = time.time()
    res4 = t2-t1
    res_t.append(res4)
    print("Run time is %.4f \n"% (res4))

def main():
    # Reading from file
    '''
    tr_dt = '/home/varsha/fl-proj/lingspam_public/lemm_stop/train'
    ts_dt = '/home/varsha/fl-proj/lingspam_public/lemm_stop/test'
    vl_dt = '/home/varsha/fl-proj/lingspam_public/lemm_stop/val'
    '''

    global dt

    dt = '/home/varsha/fl-proj/lingspam_public/lemm_stop/train'

    file_list = os.listdir(dt)
    random.shuffle(file_list)
 
    split1 = 0.5
    split2 = 0.7
    split_index1 = floor(len(file_list) * split1)
    split_index2 = floor(len(file_list) * split2)

    training = file_list[:split_index1]
    validation = file_list[split_index1: split_index2] 
    testing = file_list[split_index2:]

    global tr_matrix
    global tr_lb
    global ts_matrix
    global ts_lb
    global vl_matrix
    global vl_lb

    global pop_val

    pop_val = set_popval(validation)
    
    tr_matrix, tr_lb = matrix_lb(training)
    ts_matrix, ts_lb = matrix_lb(testing)
    vl_matrix, vl_lb = matrix_lb(validation)
 
    print("Start computation")
    model_pr("Support Vector Machines", "SVM", SVC())
    model_pr("Naive Bayes", "NB", GaussianNB())
    stacking_func()
    plot_graph(res_f1,"F1 Score")
    plot_graph(res_ac,"Recall Rate")
    plot_graph(res_pr,"Precision")
    plot_graph(res_t,"Computation time")

if __name__== "__main__":
    main()
