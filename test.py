import nltk
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import tfidfmodel
Data = pd.read_csv('C:\waitingforprocess\InputDataclean.csv', header=0, encoding='latin1',names=['ID', 'TEXT', 'CUSTOMER', 'ORIGINAL', 'LABEL'])
#
# # text1 = 'hello he heloo hello hi '
# # text1 = text1.split(' ')
# # fdist1 = nltk.FreqDist(text1)
# # print (fdist1.most_common(50))
# print(list(Data))
# print(Data['TEXT'])
# model_name = 'Model/paragraphvectorfeature.npy'
# # model_name = 'Model/tfidffeature.npy'
# trainDataVecs=np.load(model_name)
# print(trainDataVecs)
# print(pd.DataFrame(trainDataVecs))

np.save('C:\waitingforprocess\InputDataclean.npy',Data['TEXT'])

# text=[]
# text.append("just need test")
# text.append("like english")
# text.append("You like nothing")
# print(text)

# tf-idf vector
# vectorizer = TfidfVectorizer(analyzer="word", \
#                              tokenizer=None, \
#                              preprocessor=None, \
#                              stop_words=None, \
#                              max_features=5000)
# tfidf = vectorizer.fit_transform(Data['TEXT'])
# dict={}
# tfidf.shape
# features=vectorizer.get_feature_names()
# scores=zip(vectorizer.get_feature_names(),np.asarray(tfidf.sum(axis=0)).ravel())
# sorted_scores=sorted(scores,key=lambda x:x[1] ,reverse=True)
# id=round(len(sorted_scores)*0.8)
# print(list(sorted_scores)[id])

# for i in range(0,len(vectorizer.get_feature_names())):
#     if tfidf[counter,h]==0 and counter<len(text):
#         h=0
#         counter=counter+1
#         try:
#             dict[vectorizer.get_feature_names()[i]]=tfidf[counter,h]
#         except:
#             print(i,counter,h)
#     else:
#         h=h+1
#         dict[vectorizer.get_feature_names()[i]] = tfidf[counter, h]
# print(vectorizer.idf_)
# print(len(vectorizer.idf_)==len(vectorizer.get_feature_names()))
# print(tfidf)
# print(vectorizer.get_feature_names())
# print(dict)
# print(dict['nothing'])

# tfidf= tfidfmodel.TfidfModel(text)
# print(tfidf)



