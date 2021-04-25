# -*- coding: utf-8 -*-

#This code is copied from the sample solution:
#It converts the table csv files into dataframes

import pandas as pd
import numpy as np
from os.path import join

# 1. read data

ltable = pd.read_csv(join('data', "ltable.csv"))
rtable = pd.read_csv(join('data', "rtable.csv"))
train = pd.read_csv(join('data', "train.csv"))

###End copied code


#Blocking - Minhashing Method

import datasketch as ds
import re

def Block(qtable, stable):

    #First, initialize an LSH index for the right table
    LSH = ds.MinHashLSH(threshold=0.25)
    #Create a MinHash based on the product name of each record in the right table
    for row in stable.iterrows():
        prod_id = row[0]
        name = str(row[1][1])
        name = re.split(' |-', name) #remove spaces and dashes
        name = [i for i in name if i != ''] #remove empty strings from split operation
        name = set(name) #Convert to set for use in MinHash
        currentmin = ds.MinHash()
        for d in name:
            currentmin.update(d.encode('utf-8'))
       
        LSH.insert(prod_id, currentmin)

    #Then, create a MinHash for each record in the left table and query the index
    candset = []
    for record in qtable.iterrows():
        #Create a query minhash according to the same process as above
        prod_id = record[0]
        title = str(record[1][1])
        title = re.split(' |-', title)
        title = [i for i in title if i != '']
        title = set(title)
        currentmin = ds.MinHash()
        for d in title:
            currentmin.update(d.encode('utf-8'))

        result = LSH.query(currentmin) #Query the index
        for pair in result:
            candset.append([prod_id, pair]) #Append resulting pairs to the candidate set

    return candset

'''
This next function is also copied from the sample solution
It takes a list of candidate pairs and retrieves the tuples associated with the pair's ids
It returns a table where each row contains a pair of matched tuples, one from the left table and one from the right
'''

def pairs2LR(ltable, rtable, candset):
    ltable.index = ltable.id
    rtable.index = rtable.id
    pairs = np.array(candset)
    tpls_l = ltable.loc[pairs[:, 0], :]
    tpls_r = rtable.loc[pairs[:, 1], :]
    tpls_l.columns = [col + "_l" for col in tpls_l.columns]
    tpls_r.columns = [col + "_r" for col in tpls_r.columns]
    tpls_l.reset_index(inplace=True, drop=True)
    tpls_r.reset_index(inplace=True, drop=True)
    LR = pd.concat([tpls_l, tpls_r], axis=1)
    return LR

print('Blocking...')

cand = Block(ltable, rtable)
candset_df = pairs2LR(ltable, rtable, cand)

print(str(len(cand)) + " pairs")

###End copied code


#Feature Engineering

'''
For feature engineering, I will be using Levenshtein distance and a modified Jaccard Similarity
for a couple of the attributes
I copied the respective functions below
'''
import Levenshtein as lev

def jaccard_similarity(row, attr):
    #String splitting changed to match what I did above
    x = row[attr + "_l"].lower()
    x = re.split(' |-', x)
    x = [i for i in x if i != '']
    x = set(x)
    
    y = row[attr + "_r"].lower()
    y = re.split(' |-', y)
    y = [j for j in y if j != '']
    y = set(y)
    return len(x.intersection(y)) / min(len(x), len(y)) #changed to min

def levenshtein_distance(row, attr):
    x = row[attr + "_l"].lower()
    y = row[attr + "_r"].lower()
    return lev.distance(x, y)

###End copied code

import scipy

#Load GloVe word vectors
#Pretrained vectors downloaded from https://nlp.stanford.edu/projects/glove/
glove = open(join('data', 'glove.6B.50d.txt'), encoding='utf-8')
words = dict()

for word in glove:
    word = word.split()
    vector = np.asarray(word[1:], dtype='float32')
    words[word[0]] = vector

glove.close()
    
def semantic(row, attr):
    #Get embedding for each word in the input string
    l = row[attr + '_l'] #Get strings from attributes
    r = row[attr + '_r']
    
    if l == r:
        return 1
    
    l = re.split(' |-', l)
    l = [i for i in l if i != ''] #Split into words
    lwords = list() #Get GloVe vector for each word, skip if oov
    for word in l:
        try:
            lwords.append(words[word])
        except KeyError:
            continue
    larray = np.asarray(lwords)
    laverage = larray.mean(axis=0) #Get average vector across all words
    
    r = re.split(' |-', r)
    r = [j for j in r if j != '']
    rwords = list()
    for word in r:
        try:
            rwords.append(words[word])
        except KeyError:
            continue
    rarray = np.asarray(rwords)
    raverage = rarray.mean(axis=0)
    
    #Calculate cosine distance between average vectors
    dist = scipy.spatial.distance.cosine(laverage, raverage)
    
    if pd.isna(dist) == True:
        return 0
    
    return dist

def pricecomp(row, attr):
    lprice = float(row[attr + '_l'])
    rprice = float(row[attr + '_r'])
    
    if pd.isna(lprice) == True or pd.isna(rprice) == True:
        return 0.5

    diff = abs(lprice - rprice) / max(lprice, rprice)
    
    return diff

def FE(df):
    df = df.astype(str)
    features = []
    
    #Product title:
    titlejsim = df.apply(jaccard_similarity, attr='title', axis=1)
    features.append(titlejsim)
    
    titlecosdist = df.apply(semantic, attr='title', axis=1)
    features.append(titlecosdist)
    
    #Category
    catcosdist = df.apply(semantic, attr='category', axis=1)
    features.append(catcosdist)
    
    #Brand
    brandjsim = df.apply(jaccard_similarity, attr='brand', axis=1)
    features.append(brandjsim)
    
    #Modelno
    modellev = df.apply(levenshtein_distance, attr='modelno', axis=1)
    features.append(modellev)
    
    #Price
    pricediff = df.apply(pricecomp, attr='price', axis=1)
    features.append(pricediff)
    
    features = np.array(features).T
    
    return features

print('Feature Engineering...')

candset_features = FE(candset_df)

print('Training and Prediction...')

#Next 4 lines copied from sample solution
training_pairs = list(map(tuple, train[["ltable_id", "rtable_id"]].values))
training_df = pairs2LR(ltable, rtable, training_pairs)
training_features = FE(training_df)
training_label = train.label.values

#For debugging
#tf = pd.DataFrame(training_features)
#tf.to_csv("4400log.csv", index=False)

'''
The last two steps, model training and output, are also copied from the sample solution
I've added comments, both for my own reference, and to show that I know what each part does
I may try to change some of this later, in which case I'll mark my changes
'''


# 4. Model training and prediction
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(class_weight="balanced", random_state=0, n_estimators=300) #number of estimators increased
rf.fit(training_features, training_label) #Fit the model to the training data's engineered features
y_pred = rf.predict(candset_features) #Use model to predict matching pairs in whole dataset

# 5. output

matching_pairs = candset_df.loc[y_pred == 1, ["id_l", "id_r"]] #Get locations of predicted matches
matching_pairs = list(map(tuple, matching_pairs.values)) #Convert locations to a list

matching_pairs_in_training = training_df.loc[training_label == 1, ["id_l", "id_r"]] #Do the same for the training data
matching_pairs_in_training = set(list(map(tuple, matching_pairs_in_training.values)))

pred_pairs = [pair for pair in matching_pairs if
              pair not in matching_pairs_in_training]  # remove the matching pairs already in training
pred_pairs = np.array(pred_pairs)   #Convert matching pairs to an array
pred_df = pd.DataFrame(pred_pairs, columns=["ltable_id", "rtable_id"]) #Then to a dataframe
pred_df.to_csv("output.csv", index=False) #Use pandas's built in function to convert to a csv file

print('Done')






