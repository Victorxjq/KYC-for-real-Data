#!/usr/bin/env python

# *************************************** #


# Load a pre-trained model
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import time
from sklearn.cross_validation import cross_val_score
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import numpy as np
import os
from Word2VecUtility import Word2VecUtility


# Define a function to create bags of centroids
#
def create_bag_of_centroids( wordlist, word_centroid_map ):
    #
    # The number of clusters is equal to the highest cluster index
    # in the word / centroid map
    num_centroids = max( word_centroid_map.values() ) + 1
    #
    # Pre-allocate the bag of centroids vector (for speed)
    bag_of_centroids = np.zeros( num_centroids, dtype="float32" )
    #
    # Loop over the words in the review. If the word is in the vocabulary,
    # find which cluster it belongs to, and increment that cluster count
    # by one
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    #
    # Return the "bag of centroids"
    return bag_of_centroids


if __name__ == '__main__':
    #load model from AverageVectors
    import gensim
    modelname='300features_40minwords_10context_CBOW_vector'
    #modelname='300features_40minwords_10context_skip_vector'
    model=gensim.models.Word2Vec.load_word2vec_format(modelname)


    # ****** Run k-means on the word vectors and print a few clusters
    #

    start = time.time() # Start time

    # Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
    # average of 5 words per cluster
    word_vectors = model.syn0
    num_clusters = round(word_vectors.shape[0] / 5)

    # Initalize a k-means object and use it to extract centroids
    print ("Running K means")
    kmeans_clustering = KMeans( n_clusters = num_clusters )
    idx = kmeans_clustering.fit_predict( word_vectors )

    # Get the end time and print how long the process took
    end = time.time()
    elapsed = end - start
    print ("Time taken for K Means clustering: ", elapsed, "seconds.")


    # Create a Word / Index dictionary, mapping each vocabulary word to
    # a cluster number
    word_centroid_map = dict(zip( model.wv.index2word, idx ))

    # Print the first ten clusters
    for cluster in range(0,10):
        #
        # Print the cluster number
        print ("\nCluster %d" % cluster)
        #
        # Find all of the words for that cluster number, and print them out
        words = []
        for i in range(0,len(word_centroid_map.values())):
            v=list(word_centroid_map.values())
            if( v[i] == cluster):
                words.append(v[i])
        print (words)




    # Create clean_train_reviews and clean_test_reviews as we did before
    #

    # Read data from files
    Data = pd.read_csv('C:\waitingforprocess\InputDataclean.csv', names=['ID', 'TEXT', 'CUSTOMER', 'ORIGINAL', 'LABEL'],
                       encoding='latin1')
    train = Data.copy()


    print ("Cleaning training datas")
    clean_train_text = []
    for text in train["TEXT"]:
        clean_train_text.append(Word2VecUtility.text_to_wordlist(text, \
                                                                 remove_stopwords=True))



    # ****** Create bags of centroids
    #
    # Pre-allocate an array for the training set bags of centroids (for speed)
    train_centroids = np.zeros( (train["TEXT"].size, num_clusters), \
        dtype="float32" )

    # Transform the training set reviews into bags of centroids
    counter = 0
    for text in clean_train_text:
        train_centroids[counter] = create_bag_of_centroids( text, \
            word_centroid_map )
        counter += 1


    # ****** Fit a random forest and extract predictions
    #
    forest = RandomForestClassifier(n_estimators = 300)

    # Fitting the forest may take a few minutes
    print ("Fitting a random forest to labeled training data...")
    forest = forest.fit(train_centroids, train["LABEL"])
    # scores=cross_val_score(forest,trainDataVecs,train["LABEL"],cv=5,scoring='precision')
    scores_accuracy = cross_val_score(forest, train_centroids, train["LABEL"], cv=5)
    scores_precision = cross_val_score(forest, train_centroids, train["LABEL"], cv=5, scoring='precision')
    scores_recall = cross_val_score(forest, train_centroids, train["LABEL"], cv=5, scoring='recall')
    print('accuracy:', scores_accuracy.mean())
    print('precison:', scores_precision.mean())
    print('recall:', scores_recall.mean())
