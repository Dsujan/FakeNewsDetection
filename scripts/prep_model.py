#importing necessary modeules
import pickle
import numpy as np
import pandas as pd
import pickle
import scipy.sparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from feature_extractor import FeatureXtractor
import matplotlib.pyplot as plt


#fake-news dataset location
filename = '../datasets/train.csv'

#trainning data in provided classifer
def ml_pipeline(classifier,train_label,train_features):

    classifier.fit(train_features,train_label)
    
    return classifier


#splitting test and train data and extracting feature vectors
def extract_features_from_data(filename,rows=5000):

    print("\nReading CSV data..")
    df = pd.read_csv("../datasets/train.csv")
    main_df = df.loc[:,['id','text','label']]


    main_df = main_df.dropna()

    #using all dataset
    if rows != 'all':
         main_df = main_df.head(rows)

    print(len(main_df))

        

    df_x= main_df['text']
    df_y = main_df['label']


    print('\nSplitting dataset')
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)


    train_label=y_train.astype('int')
    test_label = y_test.astype('int')


    print('\nExtracting Features')
    fe = FeatureXtractor(x_train,x_test)


    return fe,train_label,test_label


def plot_confusion_matrix(naive_bayes_cm,logistic_cm,svm_cm):
    titles = ['Naive Bayes CM','Logistic Regression CM','SVM CM']

    label_names = ['Fake News','Real News']
    ticks_mrk = np.arange(len(label_names))

    confusion_matrix_labels = [
                                ['TN','FP'],
                                ['FN', 'TP']
                            ]
    
    all_cms =[naive_bayes_cm,logistic_cm,svm_cm]

    for index,matrix in enumerate(all_cms):

        #calculating different metrics for classifier
        accuracy = round((matrix[0][0] + matrix[1][1])/np.sum(matrix),2)
        recall = round(matrix[1][1]/(matrix[1][0]+matrix[1][1]),2)
        precession = round(matrix[1][1]/(matrix[0][1]+matrix[1][1]),2)
        f1_score = round((2*precession*recall) / (precession + recall),2)


        plt.subplot(2,2,index+1)
        plt.title(titles[index],fontsize=10)
        plt.colorbar(plt.imshow(matrix, interpolation='nearest',cmap='Blues'))
        plt.ylabel('Actual')
        plt.xlabel('Predicted\nAccuracy = '+str(accuracy)+'\n'+'F1 Score = '+str(f1_score),fontsize=8)  
        plt.xticks(ticks_mrk, label_names,fontsize=8)
        plt.yticks(ticks_mrk,label_names,fontsize=8)

        for i in range(2):
            for j in range(2):
                plt.text(j,i, str(confusion_matrix_labels[i][j])+" = "+str(matrix[i][j]),horizontalalignment="center",color='g')
    plt.tight_layout()
    plt.show()






#trainning multiple classifiers and  testing for 
def train_model(fe,train_label,test_label):

    print('\nStacking  training features')
    #stacking train features without doc2vec features for naive bayes classifier
    train_features_nb = scipy.sparse.hstack([fe.uni_train_tf_feature,fe.ngram_train_tf_feature,fe.char_train_tf_feature,fe.count_vector_train_feature,fe.lda_train_vec_feature])
    #train features including doc2vec features for other classifiers.
    train_features_otr = scipy.sparse.hstack([train_features_nb,fe.train_doc_vec])

    print('\nStacking  testing features')
    #stacking test features without doc2vec features for naive bayes classifier
    test_features_nb = scipy.sparse.hstack([fe.uni_test_tf_feature,fe.ngram_test_tf_feature,fe.char_test_tf_feature,fe.count_vector_test_feature,fe.lda_test_vec_feature])
    #test features with doc2vec features for other classifiers
    test_features_otr = scipy.sparse.hstack([test_features_nb,fe.test_doc_vec])


    naive_bayes_classifier = ml_pipeline(MultinomialNB(),train_label,train_features_nb)

    logistic_reg_classifier = ml_pipeline(LogisticRegression(),train_label,train_features_otr)

    svm_model_classifier = ml_pipeline(SVC(kernel='linear'),train_label,train_features_otr)


    naive_bayes_cm = confusion_matrix(test_label,naive_bayes_classifier.predict(test_features_nb))
    logistic_cm = confusion_matrix(test_label,logistic_reg_classifier.predict(test_features_otr))
    svm_cm = confusion_matrix(test_label,svm_model_classifier.predict(test_features_otr))




    # print('Accuracy for Naive Bayes: '+str(naive_bayes_classifier.score(test_features_nb,test_label)))
    # print('Accuracy for Logistic Regression: '+str(logistic_reg_classifier.score(test_features_otr,test_label)))
    # print('Accuracy for Linear SVM: '+str(svm_model_classifier.score(test_features_otr,test_label)))

    # plot_confusion_matrix(naive_bayes_cm,logistic_cm,svm_cm)


    

    return naive_bayes_classifier,logistic_reg_classifier,svm_model_classifier


def tune_hyperparameters(fe,train_label):

    print('\nStacking  training features')
    train_features= scipy.sparse.hstack([fe.uni_train_tf_feature,fe.ngram_train_tf_feature,fe.char_train_tf_feature,fe.count_vector_train_feature,fe.lda_train_vec_feature,fe.train_doc_vec])

    logistic_reg =LogisticRegression()

    # Setup the hyperparameter grid
    c_space = np.logspace(-5, 5, 15)
    c_penalty = ['l1', 'l2']
    c_solver = ['liblinear','saga']
    param_grid = {'C': c_space,'penalty':c_penalty,'solver':c_solver}

    cv_tuner = GridSearchCV(logistic_reg, param_grid, cv=3,verbose=True,n_jobs=-1)

    #fitting gridcv to data
    cv_tuner.fit(train_features,train_label)

    print("Tuned Logistic Regression Parameters: {}".format(cv_tuner.best_params_))
    print("Best score is {}".format(cv_tuner.best_score_))



def save_model(classifer,feature_vector):
    filename = '../pickled_model/classifier_model.pkl'
    model_grid = {'uni_tfidf_vector':feature_vector.uni_tfidf_vector,'char_tfidf_vector':feature_vector.char_tfidf_vector,'ngram_tfidf_vector':feature_vector.ngram_tfidf_vector,'count_vector':feature_vector.count_vector,'doc2vec_model':feature_vector.doc2vec_model,'lda_topic_model':feature_vector.lda_topic_model,'log_reg_model':classifer}

    pickle_file = open(filename,'wb')
    print('\nSaving model...')
    pickle.dump(model_grid,pickle_file)
    print('\nModel Saved!!')
    pickle_file.close()


# fe,train_label,test_label = extract_features_from_data(filename,rows=500)
# naive_bayes_classifier,logistic_reg_classifier,svm_model_classifier = train_model(fe,train_label,test_label)
# save_model(logistic_reg_classifier,fe)







    

