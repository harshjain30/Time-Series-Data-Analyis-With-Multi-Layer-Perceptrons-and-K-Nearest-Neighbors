import numpy as np
import timeit

def get_hidden_activations(x,w,beta=0.5):
    h = np.dot(x,w)
    act = 1/(1+np.exp(-beta*h))
    return act

def get_output_activations(a,w,beta=0.5):
    o = np.dot(a,w)
    yact = 1/(1+np.exp(-beta*o))
    return yact

def trainMLP(x,t,T=50000,n1=3,n2=1,ita=0.3):
    starttime = timeit.default_timer()
    a = np.ndarray([n1])
    y = np.ndarray([n2])
    eh = np.ndarray([n1])
    eo = np.ndarray([n2])
    N = len(x)
    m = len(x[0])+1 #+1 for the bias node in the input layer
    ####add bias to input and hidden layers, bias weights already taken care of
    b = 1
    x = np.insert(arr=x, obj=m-1, values=b, axis=1)
    a = np.append(arr=a, values=[b])
    t = np.matrix(t)
    #initialize weights
    wih = np.random.rand(m,n1)-0.5
    who = np.random.rand(n2,n1+1)-0.5 #+1 for the bias node in the hidden layer, #dimensions flipped for less computations
    
    for i in range(T):
        if ((i+1)%1000 == 0):
            print("iteration #",i+1)
        
        #randomize input order
        iporder = np.random.permutation(N)
        
        for r in iporder:
            #get activations
                #hidden layer
            for j in range(n1):
                a[j] = get_hidden_activations(x[r,:],wih[:,j])
                #output layer
            for k in range(n2):
                y[k] = get_output_activations(a,who[k,:])
            
            #calculate errors
            for k in range(n2):
                eo[k] = (y[k]-t[r][k])*y[k]*(1.0-y[k])
            for j in range(n1): #make sure to exclude the weights for the bias node, because there is no error for bias node activation
                eh_part1 = a[j]*(1.0-a[j])
                eh_part2 = 0
                for k in range(n2):
                    eh_part2 += np.dot(eo[k],who[k,j])
                eh[j] = np.dot(eh_part1,eh_part2)
            
            #update weights
            for j in range(n1):
                for k in range(n2):
                    who[k,j] -= ita*a[j]*eo[k]
            
            for l in range(m):
                for j in range(n1):
                    wih[l,j] -= ita*x[r,l]*eh[j]
    
    endtime = timeit.default_timer()
    print("Training complete. Iterations:",T,"Time taken:",endtime-starttime,"secs")
    return wih,who

def MLP_getLabel(testd,wih,who,th=0.5,n1=3,n2=1):
    a = np.ndarray([n1])
    y = np.ndarray([n2])
    m = len(testd)+1 #+1 for the bias node in the input layer
    ####add bias to input and hidden layers, bias weights already taken care of
    b = 1
    testd = np.insert(arr=np.matrix(testd), obj=m-1, values=b, axis=1)
    a = np.append(arr=a, values=[b])
    
    #hidden layer
    for j in range(n1):
        a[j] = get_hidden_activations(testd,wih[:,j])
    #output layer
    for k in range(n2):
        y[k] = get_output_activations(a,who[k,:])
    if (y[k] >= th):
        return 1
    else:
        return 0

def MLP_reg(testd,wih,who,n1=3,n2=1):
    a = np.ndarray([n1])
    y = np.ndarray([n2])
    m = len(testd)+1 #+1 for the bias node in the input layer
    ####add bias to input and hidden layers, bias weights already taken care of
    b = 1
    testd = np.insert(arr=np.matrix(testd), obj=m-1, values=b, axis=1)
    a = np.append(arr=a, values=[b])
    
    #hidden layer
    for j in range(n1):
        a[j] = get_hidden_activations(testd,wih[:,j])
    #output layer
    for k in range(n2):
        y[k] = get_output_activations(a,who[k,:])
    return y[0]