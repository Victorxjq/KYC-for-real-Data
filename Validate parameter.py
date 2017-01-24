from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
from scipy.stats import randint as sp_randint
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
import pandas as pd
from time import time
Data = pd.read_csv('C:\waitingforprocess\InputDataclean.csv', names=['ID', 'TEXT', 'CUSTOMER', 'ORIGINAL', 'LABEL'],
                   encoding='latin1')
train=Data.copy()
msk=np.load('Model/index.npy')

# model_name = 'Model/bagofwords.npy'
# model_name = 'Model/CBOW.npy'
# model_name = 'Model/CBOWtfidf.npy'
model_name = 'Model/DMPVparagraphvectorfeature.npy'
# model_name = 'Model/paragraphvectorfeature.npy'
# model_name = 'Model/skip.npy'
# model_name = 'Model/skiptfidf.npy'
trainDataVecs=np.load(model_name)

# msk=np.random.rand(len(trainDataVecs))<0.8
trainset=trainDataVecs[msk]
testset=trainDataVecs[~msk]
traintarget=train["LABEL"][msk]
# traintarget=traintarget.to_arrary()
testtarget=train["LABEL"][~msk]
# testtarget=testtarget.to_arrary()
# print(testtarget.value_counts())
# print(len(trainset))
# np.save('Model/index',msk)

# finalmodel=RandomForestClassifier(n_estimators=100,min_samples_leaf=2,criterion='gini',max_features=1,min_samples_split=10,max_depth=8,bootstrap=True)
# finalmodel=svm.SVC(C=100,gamma=0.0001)
finalmodel=AdaBoostClassifier(n_estimators=100,learning_rate=1)
finalmodel = finalmodel.fit(trainset, traintarget)
predict=finalmodel.predict(testset)
# print(predict)
# print(testtarget)
from sklearn.metrics import classification_report
# print(finalmodel.score(testset,testtarget))
from sklearn import metrics
print('accuracy:',metrics.accuracy_score(testtarget,predict))
print('precision:',metrics.precision_score(testtarget,predict))
print('recall:',metrics.recall_score(testtarget,predict))
print('AUC:',metrics.roc_auc_score(testtarget,predict))
# # scores=cross_val_score(forest,trainDataVecs,train["LABEL"],cv=5,scoring='precision')
# scores_accuracy = cross_val_score(finalmodel, trainDataVecs, train["LABEL"], cv=5)
# scores_precision = cross_val_score(finalmodel, trainDataVecs, train["LABEL"], cv=5, scoring='precision')
# scores_recall = cross_val_score(finalmodel, trainDataVecs, train["LABEL"], cv=5, scoring='recall')
# print('accuracy:', scores_accuracy.mean())
# print('precison:', scores_precision.mean())
# print('recall:', scores_recall.mean())