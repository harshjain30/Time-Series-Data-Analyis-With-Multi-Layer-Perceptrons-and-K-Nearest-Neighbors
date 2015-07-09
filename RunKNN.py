import numpy as np
from KNN import KNN_get_label,KNN_reg,getKDTree

def KNN_label_leave_one_out(x,t,k=1,norm=1):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    N = len(x)
    labels = [None]*N
    for i in range(N):
        traind = np.array(list(x[:i]) + list(x[i+1:]))
        traint = np.array(t[:i] + t[i+1:])
        testd = np.array(x[i])
        testt = t[i]
        
        kd_tree = getKDTree(x=traind, t=traint)
        testlabel = KNN_get_label(testd=testd, kd_tree=kd_tree, k=k, norm=norm)
        labels[i] = testlabel
        if (testlabel == 1):
            if (testt[0] == 1):
                tp += 1
            else:
                fp += 1
        elif (testlabel == 0):
            if (testt[0] == 0):
                tn += 1
            else:
                fn += 1
     
    try:
        acc = (tp+tn)/N
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        sensitivity = recall
        specificity = tn/(tn+fp)
        f1_measure = (2*(precision*recall))/(precision+recall)
        conf_matrix = np.matrix([[tn,fp],[fn,tp]])
        return labels,acc,precision,recall,sensitivity,specificity,f1_measure,conf_matrix
    except ZeroDivisionError:
        print("One of the evaluation metrics is 0")
        
def KNN_reg_leave_one_out(x,t,k=1,norm=1):
    N = len(x)
    vals = [None]*N
    for i in range(N):
        traind = np.array(list(x[:i]) + list(x[i+1:]))
        testd = np.array(x[i])
        traint = np.array(t[:i] + t[i+1:])
        
        kd_tree = getKDTree(x=traind, t=traint)
        val = KNN_reg(testd=testd, kd_tree=kd_tree, k=k, norm=norm)
        vals[i] = val
    return vals
