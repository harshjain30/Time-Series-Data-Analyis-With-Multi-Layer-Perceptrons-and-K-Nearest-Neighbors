import numpy as np
from MLP import MLP_getLabel,MLP_reg,trainMLP

def MLP_label_leave_one_out(x,t,T=50000,n1=3,n2=1,ita=0.3,th=0.5):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    N = len(x)
    labels = [None]*N
    for i in range(N):
        print("Training MLP model #:",i+1)
        traind = np.array(list(x[:i]) + list(x[i+1:]))
        testd = np.array(x[i])
        traint = np.array(t[:i] + t[i+1:])
        testt = t[i]
        
        wih,who = trainMLP(x=traind, t=traint, T=T, n1=n1, n2=n2, ita=ita)
        testlabel = MLP_getLabel(testd, wih, who, th)
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

def MLP_reg_leave_one_out(x,t,T=50000,n1=3,n2=1,ita=0.3):
    N = len(x)
    vals = [None]*N
    for i in range(N):
        traind = np.array(list(x[:i]) + list(x[i+1:]))
        testd = np.array(x[i])
        traint = np.array(t[:i] + t[i+1:])
        
        wih,who = trainMLP(x=traind, t=traint, T=T, n1=n1, n2=n2, ita=ita)
        val = MLP_reg(testd, wih, who)
        vals[i] = val
    return vals
