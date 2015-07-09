import csv
import os
import numpy as np

def f(feature,x1):
    return {
            "mean" : list(np.mean(x1,axis=0,dtype=float)),
            "var" : list(np.var(x1,axis=0,dtype=float)),
            "max" : list(np.max(x1, axis=0)),
            "min" : list(np.min(x1, axis=0)),
            "rms" : list(np.sqrt(np.mean(np.square(x1),axis=0,dtype=float))),
            "cov" :  [(np.cov(np.array(x1),rowvar=0))[0,1]],
            }.get(feature,list(np.mean(x1,axis=0,dtype=float)))

def import_data(path,features=["mean"]):
    x = []
    t = []
    for file in os.listdir(path):
        file_path = os.path.join(path,file)
        if (os.path.isfile(file_path)) and (file_path.lower().endswith(".csv")):
            #get target class
            t.append([int(file.split("_",1)[1].split(".")[0])])
            #open file
            csv_file = csv.reader(open(file_path, mode='r'))
            #read file data skipping the labels row
            x1 = []
            skip_row = True
            for row in csv_file:
                if not skip_row:
                    x1.append([int(row[1]),int(row[2])])
                skip_row = False
            
            l = []
            for feature in features:
                l.extend(f(feature,x1))
            x.append(l)
            
    x = np.array(x)
    for i in range(len(x[0])):
        x[:,i] = x[:,i]/np.linalg.norm(x[:,i])
    
    return x,t
