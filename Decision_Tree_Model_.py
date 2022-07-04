#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import the required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import randrange
import pandas as pd


# In[3]:


# Recursice Binary Splitting (Classification Tree) and K-Fold Cross Validation for assesing the model

# Lets first create a function that is partitioning the dataset into left and right branch based on some value conditions on a specific variable
# Index: Index of the variable, value: the value used for partitioning the Feature space
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# Now we want to construct a function thas is going to serve as the criterion for selecting the right index and value for the splitting.
# The criterion that we are going to use is the GINI Index.

def gini_index(regions, classes):
    regions_size = float(sum([len(region) for region in regions]))
    gini_node = 0
    for region in regions:
        size = len(region)
        score_region = 0
        if size == 0:
            continue
        for class_val in classes:
            p = [row[-1] for row in region].count(class_val)/size
            score_region += p**2
        gini_node +=  score_region
    gini_node = 1 - gini_node
    return gini_node


# Now, we combine the previous two functions, in order to examine which (index, value) selection yields the minimum Gini Index
def decide_split(dataset):
    classes = set([row[-1] for row in dataset])# To take only the distinct classes
    f_gini_thes, f_value, f_index = 99,99,99
    for index in range(len(dataset[0])-1):
        for row in dataset:
            regions = test_split(index, row[index], dataset)
            gini_node = gini_index(regions,classes)
            if gini_node < f_gini_thes :
                f_gini_thes, f_value, f_index, f_regions = gini_node, row[index], index, regions
    return {'node_gini' : f_gini_thes , 'node_value' : f_value, 'node_index' : f_index, "node_regions" : f_regions}


# Before continuing in how to create a sequence of nodes, we create that is returning the terminal node response value
def terminal_response_value(region):
    responses = [row[-1] for row in region]
    return max(set(responses), key = responses.count)# Returns the class existing the most times.


# At this point, we start considering what are the conditions for stopping the tree growing process.
# A reasonable choice, would be to predetermine the maximum amaount of terminal nodes that we want(max_nodes).
# Also, we can select the minimum size that a region must have in order to continue the process(min_region_size)
# Finally, we use the parameter depth as an indication of how many nodes have been already constructed.
def split(node, max_nodes, min_region_size, depth):
    left, right = node['node_regions']
    if len(left) == 0 or len(right) == 0:
        node['left'] =  node["right"] = terminal_response_value(left + right)
        return
    if depth >= max_nodes:
        node['left'], node["right"] = terminal_response_value(left), terminal_response_value(right)
        return
    if len(left) < min_region_size:
        node["left"] = terminal_response_value(left)
    else:
        node['left'] = decide_split(left)
        split(node['left'], max_nodes, min_region_size, depth + 1)
    if len(right) < min_region_size:
        node["right"] = terminal_response_value(right)
    else:
        node['right'] = decide_split(right)
        split(node['right'], max_nodes, min_region_size, depth + 1)
        
# The function below, classifies one observation based on the branch that it belongs and based on whether 
# this branch has reached its terminal node or not.
def predict_class(node,test_row):
    if test_row[node['node_index']] < node['node_value']:
        if isinstance(node['left'],dict):
            return predict_class(node['left'], test_row)
        else:
            return node['left']
    else:
        if isinstance(node['right'],dict):
            return predict_class(node['right'], test_row)
        else:
            return node['right']
    
        
        
# Now, we are going to use K-fold Validation for increasing the efficiency of the model.
# So, firstly we need to build a tree based on the training part of the data, as follows.
def build_tree(train, max_nodes, min_region_size):
    root_node = decide_split(train)
    split(root_node, max_nodes, min_region_size, 1)
    return root_node
        
#Then, we split the data into k folds randomly.        
def k_fold_splitting(dataset, k_folds):
    folds = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset)/k_folds)
    for i in range(k_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))# Randomeness in Fold Selection
            fold.append(dataset_copy.pop(index))
        folds.append(fold)
    return folds

# K-Fold Cross Validation of the Recursive Binary Splitting. Namely, we split the dataset into K Folds of approximately
# the same size, and then we train the model to K-1 folds aand we test on the left out fold (hence we build K trees).
# Then we take the average of the Classification Error Rate(or, Sucess as in this occasion)
def K_fold_validation(dataset, model,  k_folds, max_nodes, min_region_size):
    folds = k_fold_splitting(dataset, k_folds)
    CSR = 0
    for i in range(len(folds)):
        train = list(folds)
        train.pop(i)
        train = sum(train,[])
        test = folds[i]
        node = model(train, max_nodes, min_region_size)
        correct = 0
        for row in test:
            if predict_class(node, row) == row[-1]:
                    correct += 1
        CSR += correct/len(folds[i])#Classification Sucess Rate
    return CSR/k_folds
            


# Lets now perform a random splitting of the data when the proportion of data for testing is given.   
def train_test_split(dataset, test_proportion):
    train_index = list()
    test_index = list()
    test_size = round(len(dataset)*test_proportion)
    train_size = len(dataset) - test_size
    while len(test_index) < test_size:
        index = randrange(len(dataset))
        if index not in test_index:
            test_index.append(index)
    for i in range(len(dataset)):
        if i not in test_index:
            train_index.append(i)
    test_set = [dataset[i] for i in test_index]
    train_set = [dataset[i] for i in train_index]
    return train_set, test_set



# Now we create a function which builds a decision tree on the train data, provides the predictions on the test data 
# and then we assess its accuracy based one how many correct predictions were performed.
def forecast(dataset, model, test_proportion, max_nodes, min_region_size):
    train_set, test_set = train_test_split(dataset, test_proportion)
    node = model(train_set, max_nodes, min_region_size)
    predictions = list()
    correct = 0
    for row in test_set:
            prediction = predict_class(node,row)
            predictions.append(prediction)
            if prediction == row[-1]:
                correct +=1
    score = correct / len(test_set)
    return predictions, node, score


        
        
        
        
    

    

    


# In[17]:






forecast(dataset_copy_1, build_tree, 0.05, 10, 2)
#k_folds, max_nodes, min_region_size = 10, 5, 5
#K_fold_validation(dataset_copy_1, build_tree, k_folds, max_nodes, min_region_size)


# In[3]:


# Random Forest Classification and K-Fold Cross Validation for assesing the model

# Lets first create a function that is partitioning the dataset into left and right branch based on some value conditions on a specific variable
# Index: Index of the variable, value: the value used for partitioning the Feature space
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# Now we want to construct a function thas is going to serve as the criterion for selecting the right index and value for the splitting.
# The criterion that we are going to use is the GINI Index.

def gini_index(regions, classes):
    regions_size = float(sum([len(region) for region in regions]))
    gini_node = 0
    for region in regions:
        size = len(region)
        score_region = 0
        if size == 0:
            continue
        for class_val in classes:
            p = [row[-1] for row in region].count(class_val)/size
            score_region += p*(1-p)
        gini_node += score_region
    return gini_node


# Now, we combine the previous two functions, in order to examine which (index, value) selection yields the minimum Gini Index.
# In addition to Recursive Binary splitting Algorithm, here we set one extra variable for determining the number of variables
# used on each split decision. This is the variable expresses the distinction between bagging and Random forest Classification. 
def decide_split(dataset, n_variables):
    classes = set([row[-1] for row in dataset])# To take only the distinct classes
    f_gini_thes = 1000
    used_variables = list()
    while (len(used_variables) < n_variables) :
        index = randrange(len(dataset[0])-1)# To restrict only to predictors indices
        if index not in used_variables:
            used_variables.append(index)
    for index in used_variables:
        for row in dataset:
                regions = test_split(index, row[index], dataset)
                gini_node = gini_index(regions,classes)
                if gini_node < f_gini_thes :
                    f_gini_thes, f_value, f_index, f_regions = gini_node, row[index], index, regions
    return {'node_gini' : f_gini_thes , 'node_value' : f_value, 'node_index' : f_index, "node_regions" : f_regions}


# Before continuing in how to create a sequence of nodes, we create that is returning the terminal node response value
def terminal_response_value(region):
    responses = [row[-1] for row in region]
    return max(set(responses), key = responses.count)# Returns the class existing the most times.


# At this point, we start considering what are the conditions for stopping the tree growing process.
# A reasonable choice, would be to predetermine the maximum amaount of terminal nodes that we want(max_nodes).
# Also, we can select the minimum size that a region must have in order to continue the process(min_region_size)
# Finally, we use the parameter depth as an indication of how many nodes have been already constructed.
def split(node, max_nodes, min_region_size, depth, n_variables):
    left, right = node['node_regions']
    if len(left) == 0 or len(right) == 0:
        node['left'] =  node["right"] = terminal_response_value(left + right)
        return
    if depth >= max_nodes:
        node['left'], node["right"] = terminal_response_value(left), terminal_response_value(right)
        return
    if len(left) < min_region_size:
        node["left"] = terminal_response_value(left)
    else:
        node['left'] = decide_split(left, n_variables)
        split(node['left'], max_nodes, min_region_size, depth + 1, n_variables)
    if len(right) < min_region_size:
        node["right"] = terminal_response_value(right)
    else:
        node['right'] = decide_split(right, n_variables)
        split(node['right'], max_nodes, min_region_size, depth + 1, n_variables)
        
# The function below, classifies one observation based on the branch that it belongs and based on whether 
# this branch has reached its terminal node or not.
def predict_class(node,test_row):
    if test_row[node['node_index']] < node['node_value']:
        if isinstance(node['left'],dict):
            return predict_class(node['left'], test_row)
        else:
            return node['left']
    else:
        if isinstance(node['right'],dict):
            return predict_class(node['right'], test_row)
        else:
            return node['right']
    
        
        
# Now, we are going to use K-fold Validation for increasing the efficiency of the model.
# So, firstly we need to build a tree based on the training part of the data, as follows.
def build_tree(train, max_nodes, min_region_size, n_variables):
    root_node = decide_split(train,n_variables)
    split(root_node, max_nodes, min_region_size, 1, n_variables)
    return root_node
        
#Then, we split the data into k folds randomly.        
def k_fold_splitting(dataset, k_folds):
    folds = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset)/k_folds)
    for i in range(k_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))# Randomeness in Fold Selection
            fold.append(dataset_copy.pop(index))
        folds.append(fold)
    return folds


#Now, we create a function that is taking samples of the training data(Bootstapping)
def boot_sample(dataset, sample_size):
    sample = list()
    while len(sample) < sample_size:
        row_index = randrange(len(dataset)) 
        sample.append(dataset[row_index])
    return sample

# Moreover, we can now generate trees for any bootstrapped sample, where the number of samples used is indicated 
# by the n_trees variable. Then, we obtain the final prediction by considering the class found most for a specific input
# among all trees. Firsty, we perform the prediction for just one row input, and then we create the random forest function
# where all predictions for a given test data are collected.

def bagging_prediction(trees, row):
    predictions = list()
    for tree in trees:
        predictions.append(predict_class(tree,row))
    return(max(set(predictions), key = predictions.count()))


# Lets now perform a random splitting of the data when the proportion of data for testing the model is given.   
def train_test_split(dataset, test_proportion):
    train_index = list()
    test_index = list()
    test_size = round(len(dataset)*test_proportion)
    train_size = len(dataset) - test_size
    while len(test_index) < test_size:
        index = randrange(len(dataset))
        if index not in test_index:
            test_index.append(index)
    for i in range(len(dataset)):
        if i not in test_index:
            train_index.append(i)
    test_set = [dataset[i] for i in test_index]
    train_set = [dataset[i] for i in train_index]
    return train_set, test_set


def random_forest_pred(dataset, n_trees, sample_size, test_proportion, *args):
    train, test = train_test_split(dataset, test_proportion)
    trees = list()
    predictions = list()
    for i in range(n_trees):
        sample_train = boot_sample(train,sample_size)
        tree = build_tree(sample_train, *args)
        trees.append(tree)
    predictions = [bagging_prediction(trees,row) for row in test]
    return predictions

# K-Fold Cross Validation of the Random Forest
def evaluation(dataset, model,  k_folds, n_trees, sample_size, *args):
    folds = k_fold_splitting(dataset, k_folds)
    CER = 0
    for i in range(len(folds)):
        train = list(folds)
        train.pop(i)
        train = sum(train,[])
        correct = 0
        test = folds[i]
        for row in folds[i]:
            if model(train, test, n_trees, sample_size, *args) == row[-1]:
                    correct += 1
        CER += correct/len(folds[i])#Classification Sucess Rate
    return CER/k_folds
            





        
        
        
        
    

    

    


# In[ ]:


dataset = pd.read_csv("winequality-white.csv", delimiter = ';')
dataset.isnull().values.any()
codes, uniques = pd.factorize(dataset.iloc[:,-1])

dataset['quality_1'] = codes

dataset.pop('quality')
dataset_copy = dataset.iloc[range(0,len(dataset),10),:]
dataset_copy_1 = list(dataset_copy.values)

k_folds , max_nodes,min_region_size, n_variables, n_trees, sample_size, test_proportion = 5, 5, 5, 3, 2, 50, 0.05
random_forest_pred(dataset_copy_1, n_trees, sample_size, max_nodes, min_region_size, n_variables, test_proportion)

