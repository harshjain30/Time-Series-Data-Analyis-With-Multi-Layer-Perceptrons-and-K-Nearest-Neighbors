import numpy as np
import operator
  
class TreeNode:
    val = None
    label = None
    left = None
    right = None
    
    def _init_(self,v,l,r):
        self.val = v
        self.left = l
        self.right = r

def maketree(feature,m,a,l):
    node = TreeNode()
    n = len(a)
    
    if(n==0):
        return None
    if(n==1):
        node.val = a[0]
        node.label = l[0][0]
        node.left = None
        node.right = None
        return node
    
    if (n%2 == 0):
        med_index = n/2
    else:
        med_index = np.ceil(n/2)-1
    
    sorted_indices = a[:,feature].argsort()
    a = a[sorted_indices]
    l = l[sorted_indices]
    
    med = a[med_index,feature]
    
    i=1
    while((a[med_index-i,feature] == med) and (i<=med_index)):
        i += 1
    med_index_f = med_index-i+1
    
    node.val = a[med_index_f]
    node.label = l[med_index_f][0]
    node.left = maketree((feature+1)%m, m, a[:med_index_f], l[:med_index_f])
    node.right = maketree((feature+1)%m, m, a[med_index_f+1:], l[med_index_f+1:])
    return node

def calc_dist(node1,node2,norm=1):
    dist = 0
    for i in range(len(node1)):
        if(norm==1):
            dist += abs(node1[i]-node2[i])
        elif(norm==2):
            dist += (node1[i]-node2[i])**2
        else:
            print("Incorrect Norm Value")
    return np.sqrt(dist)

def sort_lists(knns,dists):
    sorted_indices = sorted(range(len(dists)),key=lambda l: dists[l])
    temp1 = [None]*(len(dists)) #deep copy
    temp2 = [None]*(len(knns))
    for ii,i in enumerate(sorted_indices):
        temp1[ii] = dists[i]
        temp2[ii] = knns[i]
    dists = temp1
    knns = temp2
    return knns,dists

def traverse(testd,feature,m,node,k,norm,knns,dists):
    if node == None:
        return knns, dists
    dist = calc_dist(node.val,testd,norm)
    if (testd[feature] < node.val[feature]):
        knns, dists = traverse(testd,(feature+1)%m, m, node.left,k,norm,knns,dists)
        
        if (len(knns)<k):
            knns.append(node)
            dists.append(dist)
            knns, dists = sort_lists(knns,dists)
            knns, dists = traverse(testd,(feature+1)%m, m, node.right,k,norm,knns,dists)
            
        else:
            if (dist<dists[-1]):
                knns[-1] = node
                dists[-1] = dist
                knns, dists = sort_lists(knns,dists)
        
            if (np.abs(node.val[feature]-testd[feature])<dists[-1]):
                knns, dists = traverse(testd,(feature+1)%m, m, node.right,k,norm,knns,dists)        
        
    elif (testd[feature] >= node.val[feature]):
        knns, dists = traverse(testd,(feature+1)%m, m, node.right,k,norm,knns,dists)
        
        if (len(knns)<k):
            knns.append(node)
            dists.append(dist)
            knns, dists = sort_lists(knns,dists)
            knns, dists = traverse(testd,(feature+1)%m, m, node.left,k,norm,knns,dists)
            
        else:
            if (dist<dists[-1]):
                knns[-1] = node
                dists[-1] = dist
                knns, dists = sort_lists(knns,dists)
        
            if (np.abs(node.val[feature]-testd[feature])<dists[-1]):
                knns, dists = traverse(testd,(feature+1)%m, m, node.left,k,norm,knns,dists)
    
    return knns,dists

def KNN_get_label(testd,kd_tree,k=1, norm=1):
    knns, dists = traverse(testd,feature=0,m=len(testd),node=kd_tree,k=k,norm=norm,knns=[],dists=[])
    label_freq = {}
    for i in range(k):
        label = knns[i].label
        if label in label_freq:
            label_freq[label] += 1
        else:
            label_freq[label] = 1
    testlabel,testlabel_freq = max(label_freq.items(), key=operator.itemgetter(1))
    return testlabel

def KNN_reg(testd,kd_tree,k=1,norm=1):
    knns, dists = traverse(testd,feature=0,m=len(testd),node=kd_tree,k=k,norm=norm,knns=[],dists=[])
    reg_value = 0
    lda = dists[-1]+(dists[-1]/10)
    for i in range(k):
        d = dists[i]
        if(d<lda):
            reg_value += ((1-abs((d/lda)**3))**3)*knns[i].label
    return reg_value

def getKDTree(x,t):
    return maketree(feature=0, m=len(x[0]), a=x, l=t)