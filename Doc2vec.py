import pandas as pd
import gensim
from gensim.models import doc2vec
from collections import namedtuple
from Word2VecUtility import Word2VecUtility
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from gensim.models.doc2vec import TaggedDocument
import pickle
from sklearn import linear_model
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
from sklearn import svm

if __name__ == '__main__':

    Data = pd.read_csv('C:\waitingforprocess\InputDataclean.csv', names=['ID', 'TEXT', 'CUSTOMER', 'ORIGINAL','LABEL'],
                       encoding='latin1')
    train=Data.copy()

# Initialize an empty list to hold the clean reviews
    clean_train_text = []

    # Loop over each review; create an index i that goes from 0 to the length
    # of the movie review list

    print ("Cleaning and parsing the training set raw text...\n")
    for i in range( 0, len(train["TEXT"])):
        clean_train_text.append(" ".join(Word2VecUtility.text_to_wordlist(train["TEXT"][i], True)))

#Transform data into doc format
documents=[]
index=[]
with open('c:\dutch.pickle', 'rb') as resource:
    tokenizer = pickle.load(resource)
for i,text in enumerate(clean_train_text):
    tags=[i]
    doc=tokenizer.tokenize(text)
    index.append(tags)
    documents.append(TaggedDocument(doc,tags))

#Train models
# model=doc2vec.Doc2Vec(documents,window=30,min_count=1,pretrained_emb='300features_40minwords_10context_CBOW_vector')
model=doc2vec.Doc2Vec(window=30,min_count=1,dm=1,size=300)
model.build_vocab(documents)
model.train(documents)

train_data_features=[]
# print(model.docvecs[index[0]])
for item in index:
    train_data_features.append(model.docvecs[item][0])
# print(train_data_features[0])
# print(len(train_data_features))

np.save('Model/DMPVparagraphvectorfeature',train_data_features)
# np.save('Model/DBOWparagraphvectorfeature',train_data_features)

# Initialize a Random Forest classifier with 100 trees
# forest = RandomForestClassifier(n_estimators=300)
# forest = AdaBoostClassifier(n_estimators=300)
# forest = GradientBoostingRegressor(n_estimators=100)

# Fit the forest to the training set, using the bag of words as
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
# Evaluate via accuracy
# final = pd.DataFrame(train_data_features, train["LABEL"])
# final.to_csv('C:\waitingforprocess\output\modelbagofwords.csv')
# print(len(train_data_features))
# print(len(train["LABEL"]))
# forest= linear_model.LinearRegression()
# classmodel = forest.fit(train_data_features, train["LABEL"])
# forest = svm.SVC(gamma=0.00001,C=150)
# print(len(train_data_features))
# print(train_data_features[0])
# print(len(train["LABEL"]))
# print(train["LABEL"][0])
# scores = cross_val_score(classmodel,train_data_features, train["LABEL"], cv=5,scoring='precision')
# print(classmodel.score(train_data_features, train["LABEL"]))
# scores_accuracy = cross_val_score(forest, train_data_features, train["LABEL"], cv=5)
# scores_precision = cross_val_score(forest, train_data_features, train["LABEL"], cv=5, scoring='precision')
# scores_recall = cross_val_score(forest, train_data_features, train["LABEL"], cv=5, scoring='recall')
# print('accuracy:',scores_accuracy.mean())
# print('precison:' ,scores_precision.mean())
# print('recall:' ,scores_recall.mean())