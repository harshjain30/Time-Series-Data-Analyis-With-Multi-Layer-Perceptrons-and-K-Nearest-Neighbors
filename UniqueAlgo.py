import numpy as np
from MLP import trainMLP,MLP_reg,MLP_getLabel
from KNN import getKDTree,KNN_reg
from PrepareData import import_data

def UA_leave_one_out(path,features1,features2,T=50000,k=1,ita=0.3,th=0.5):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    xMLP,t = import_data(path,features1)
    xKNN,t = import_data(path,features2)
    
    N = len(xMLP)
    m = len(xMLP[0])
    labels = [None]*N
    
    for i in range(N):
        print("Training MLP model #:",i+1)
        traind = np.array(list(xMLP[:i]) + list(xMLP[i+1:]))
        traint = np.array(t[:i] + t[i+1:])
        testt = t[i]
        
        #train on training data
        wih,who = trainMLP(x=traind, t=traint, T=T, n1=3, n2=1, ita=ita)
        
        #get values for entire data
        valsMLP=[None]*N
        for i in range(N):
            valsMLP[i] = MLP_reg(testd=np.array(xMLP[i]), wih=wih, who=who, n1=3, n2=1)
        
        #insert absolute errors as feature
        err = abs(np.array(valsMLP)-t[:][0])
        
        xKNN = np.insert(arr=xKNN, obj=m-1, values=err, axis=1)
        traind = np.array(list(xKNN[:i]) + list(xKNN[i+1:]))
        traint = np.array(t[:i] + t[i+1:])
        
        #make tree with training data
        kd_tree = getKDTree(x=traind, t=traint)
        valsKNN=[None]*N
        for i in range(N):
            valsKNN[i] = KNN_reg(testd=np.array(xKNN[i]), kd_tree=kd_tree, k=k)
            
        #inputs for MLP2
        x2 = np.ndarray([N,2]) 
        x2[:,0] = valsMLP
        x2[:,1] = valsKNN
        
        traind = np.array(list(x2[:i]) + list(x2[i+1:]))
        traint = np.array(t[:i] + t[i+1:])
        testd=np.array(x2[i])
        
        #train on training data
        wih,who = trainMLP(x=traind, t=traint, T=T, n1=1, n2=1, ita=ita)
        
        #get value for test point
        label = MLP_getLabel(testd, wih=wih, who=who, th=th, n1=1, n2=1)
        labels[i] = label
        
        if(label==1):
            if(testt[0] == 1):
                tp+=1
            else:
                fp+=1
        elif(label==0):
            if(testt[0] == 0):
                tn+=1
            else:
                fn+=1
    
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
