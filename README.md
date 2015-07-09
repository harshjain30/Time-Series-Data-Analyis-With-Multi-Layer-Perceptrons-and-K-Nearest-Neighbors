# CS260-Project
Multi-Layer Perceptron and KNN with KD-Tree

Please make sure you import all 6 files into the workspace

********Common Variables********
x: input data  ~~~~~~~~format~ [[feature1, feature2, ..., featuren], [feature1, feature2, ..., featuren], ..., [feature1, feature2, ..., featuren]]
t: targets ~~~~~~~~~~~~format~ [[target1], [target2], ..., [targetn]]
testd: test data point
*************************

*******MLP Variables*****
T: number of iterations, defalut 50000
n1: number of neurons in the hidden layer, default 3
n2: number of output neurons, default 1
ita: learning rate, default 0.3
wih: input layer-hidden layer weights
who: hidden layer-output layer weights
th: activation threshold, default 0.5
*********************

********KNN Variables*******
k: number of nearest neighbors, default 1
norm: the norm to be used for distance calculations, 1 or 2, default 1
************************

To import data:
	x,t = import_data(path,features)
path: string specifying the path to the folder that contains all the data files
features: list of features that you want to extract from the dataset, in python list format

*************************MLP**************************

*******************MLP Inner functions:

To train a MLP:
	wih,who = trainMLP(x,t,T,n1,n2,ita)

To get a label for a test point from a trained MLP model:
	label = MLP_getLabel(testd,wih,who,th,n1,n2)

To get a value for a test point from a trained MLP(regression):
	value = MLP_reg(testd,wih,who,n1,n2)

*******************

To run Leave-One-Out on MLP:
	labels,acc,precision,recall,sensitivity,specificity,f1_measure,conf_matrix = MLP_label_leave_one_out(x,t,T,n1,n2,ita,th)
		
		or for regression
	
	vals = MLP_reg_leave_one_out(x,t,T)

***********************************************************************

****************************KNN************************

*****************Inner functions:

To make a KD-tree:
	kd_tree = getKDTree(x,t)

To get a label for a test point for a KD-tree:
	label = KNN_get_label(testd,kd_tree,k,norm)

TO get a value for the test data poin usin the KD-tree:
	value = KNN_reg(testd,kd_tree,k,norm)

******************

To run Leave-One-Out on KNN:
	labels,acc,precision,recall,sensitivity,specificity,f1_measure,conf_matrix = KNN_label_leave_one_out(x,t,k,norm)

		or for regression

	vals = KNN_reg_leave_one_out(x,t,k,norm)

*************************************************************************

***************************MLP1-KNN-MLP2***************************

To run Leave-One_Out on the unique algorithm:
	labels,acc,precision,recall,sensitivity,specificity,f1_measure,conf_matrix = UA_leave_one_out(path,features1,features2,T,k,ita,th):
path: string specifying the path to the folder that contains all the data files
features1: python list of features for MLP1
features2: python list of features for KNN
