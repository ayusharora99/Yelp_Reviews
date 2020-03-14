#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv, json
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import pandas as pd 

inFile = open('data_train.json') # open the json file
outFile = open("fileOutput.csv", 'w') #load a writable csv file

content = json.load(inFile) #load json content
inFile.close() # close the input file

output = csv.writer(outFile) # create a csv writer
output.writerow(content[0].keys())  # label header row with first row of data

for row in content:
    output.writerow(row.values()) # fill csv with values row by row


with open('fileOutput.csv', 'r', encoding = "ISO-8859-1") as reviews:
    rev =  csv.reader(reviews, delimiter=',')
    texts = []
    stars = []
    for lines in rev:
        texts.append(lines[4]) # append all text reviews to texts array
        stars.append(lines[0]) # apend all star ratings to stars array
    texts.pop(0) # delete intializer
    stars.pop(0) # delete intializer


# In[2]:


# Balancing out input data to avoid bias

def balance_data(txt, st):
    frequency = Counter(st)
 
    maximum_allowed = frequency.most_common()[-1][1]

    freq = {var: 0 for var in frequency.keys()}

    new_stars = []
    new_txt = []
    for i,y in enumerate(st):
        if freq[y] < maximum_allowed:
            new_stars.append(y)
            new_txt.append(txt[i])
            freq[y] += 1
            
    return new_txt, new_stars


balanced_texts, balanced_stars = balance_data(texts,stars)

print(Counter(balanced_stars))


# In[3]:


vect = TfidfVectorizer(ngram_range=(1,2))

vectors = vect.fit_transform(balanced_texts)


# In[4]:


SVM = LinearSVC()

SVM.fit(vectors, balanced_stars)

inputFile_wo_label = open('data_test_wo_label.json')  # open the json file
outputFile_wo_label = open("fileOutput_wo_label.csv", 'w') # load a writable csv file

data_wo_label = json.load(inputFile_wo_label) # load json content
inputFile_wo_label.close() # close the input file

output_wo_label = csv.writer(outputFile_wo_label) # create a csv writer
output_wo_label.writerow(content[0].keys())  # label header row with first row of data

for row in data_wo_label:
    output_wo_label.writerow(row.values()) #fill csv with values row by row

with open('fileOutput_wo_label.csv', 'r', encoding = "ISO-8859-1") as reviews_wo_label:
    revs_wo_label =  csv.reader(reviews_wo_label, delimiter=',')
    texts_wo_label = []
    for lines in revs_wo_label:
        texts_wo_label.append(lines[3]) # append all text reviews to texts array
    texts_wo_label.pop(0) # delete intializer


vectors_test = vect.transform(texts_wo_label) # transform vectorizer to fit new vocabulary 


# In[5]:


preds_wo_label = SVM.predict(vectors_test) # prediction


# In[6]:


predictions = preds_wo_label


# In[7]:


predictions = predictions.astype('float') 
predictions = predictions.astype('int')


# In[8]:


pd.DataFrame(predictions).to_csv("AroraKrishnanNampoothiri_predictions.csv", header = None, index = None) # write to CSV
with open('AroraKrishnanNampoothiri_predictions.csv',newline='') as f:
    r = csv.reader(f)
    data = [line for line in r]
with open('AroraKrishnanNampoothiri_predictions.csv','w',newline='') as f:
    w = csv.writer(f)
    w.writerow(['Predictions'])
    w.writerows(data)

