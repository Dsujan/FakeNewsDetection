import pickle
import scipy
import os
import sys
import numpy as np
import data_refractor as dr
import warnings

#ignoring module warnings
warnings.filterwarnings("ignore")

#ignores print outputs on console
def blockLog():
    sys.stdout = open(os.devnull, 'w')

#brings back console output
def enableLog():
    sys.stdout = sys.__stdout__


#loading saved models
print('\nLoading Model...\n')
filename = '../pickled_model/classifier_model.pkl'
pickled_model = open(filename,'rb')
all_models = pickle.load(pickled_model)
pickled_model.close()

#fetching all model objects
print('Preparing Model Objects..\n')
uni_tfidf_vector = all_models['uni_tfidf_vector']
char_tfidf_vector = all_models['char_tfidf_vector']
ngram_tfidf_vector = all_models['ngram_tfidf_vector']
count_vector = all_models['count_vector']
doc2vec_model = all_models['doc2vec_model']
lda_topic_model = all_models['lda_topic_model']
log_reg_model = all_models['log_reg_model']

print('*********Model Ready***********')


def  prepare_features():

    news = []
    news.append(input('\n\nEnter News article for detection\n\n\n'))
    print('\n')    
    #blocking print logs on console.
    blockLog()
    #preprocessing news article
    dummy,preprocessed_news = dr.perform_pre_processing(news)
    detokenized_news = dr.detokenize_data(news)
    



    #creating feature vectors of the article.
    uni_tfidf_feature_vector = uni_tfidf_vector.transform(preprocessed_news)
    char_tfidf_feature_vector = char_tfidf_vector.transform(detokenized_news)
    ngram_tfidf_feature_vector = ngram_tfidf_vector.transform(preprocessed_news)
    count_feature_vector =  count_vector.transform(preprocessed_news)
    doc2vec_feature_vector = np.vstack(doc2vec_model.infer_vector(singletoken) for singletoken in preprocessed_news)
    lda_topic_feature_vector = lda_topic_model.transform(count_feature_vector)


    #stacking features for prediction
    feature_vectors = scipy.sparse.hstack([uni_tfidf_feature_vector,ngram_tfidf_feature_vector
    ,char_tfidf_feature_vector,count_feature_vector,lda_topic_feature_vector,doc2vec_feature_vector])
    #enabaling print logs on console
    enableLog()
    return feature_vectors

while True:
    feature_vectors = prepare_features()
    #predicting news by logistic regrssion
    if 1 in log_reg_model.predict(feature_vectors):
        print('************Fake News*******')
    else:
        print("*********Real News**********")
    confirm = input("\nPredict Another News(Y/N)\n")
    if confirm.lower() == 'y':
        pass
    else:
        break






