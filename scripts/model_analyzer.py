import prep_model as model 
import pandas as pd
import pickle
import os
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import scipy.sparse



pickles_location = '../extras/fe'
filename = '../datasets/train.csv'


def save_features_with_labels():
    #loading dataset for determing total length
    df = pd.read_csv(filename)
    main_df = df.loc[:,['id','text','label']]
    main_df = main_df.dropna()

    total_articles = len(main_df)
    increase_article_by = 1597

    for dataset in range(increase_article_by,total_articles+increase_article_by,increase_article_by):

        fe,train_label,test_label = model.extract_features_from_data(filename,dataset)
        features_with_labels = {'fe':fe,'train_label':train_label,'test_label':test_label}
        analyzer_file = open(pickles_location+str(dataset),'wb')
        pickle.dump(features_with_labels,analyzer_file)
        analyzer_file.close()
        print('\nFile Saved!!')

def load_saved_object_files():
    print('Reading available files')
    all_files = []
    file_path = '../extras/'

    for r, d, f in os.walk(file_path):
        for file in f:
            all_files.append(os.path.join(r, file))

    return all_files


def perform_composite_feature_test_analysis(files):

    for file in files:
        print('Loading '+file)
        pickled_files = open(file, 'rb')
        pickled_obj = pickle.load(pickled_files)
        train_label = pickled_obj['train_label']
        test_label  = pickled_obj['test_label']
        fe =  pickled_obj['fe']


        print('\nStacking  testing features')
        #stacking test features without doc2vec features for naive bayes classifier
        test_features_nb = scipy.sparse.hstack([fe.uni_test_tf_feature,fe.ngram_test_tf_feature,fe.char_test_tf_feature,fe.count_vector_test_feature,fe.lda_test_vec_feature])
        #test features with doc2vec features for other classifiers
        test_features_otr = scipy.sparse.hstack([test_features_nb,fe.test_doc_vec])
        
        naive_bayes_classifier,logistic_reg_classifier,svm_model_classifier = model.train_model(fe,train_label,test_label);
        acc_nb =  naive_bayes_classifier.score(test_features_nb,test_label)
        acc_lr = logistic_reg_classifier.score(test_features_otr,test_label)
        acc_svm = svm_model_classifier.score(test_features_otr,test_label)

        accuracies = [acc_nb,acc_lr,acc_svm]

        print('\nWritting result')
        with open('composite_results.txt','a+') as f:
            f.write(str(accuracies)+'\n')

def perform_individual_feature_test_analysis(files):

    for file in files:
        print('Loading '+file)
        pickled_files = open(file, 'rb')
        pickled_obj = pickle.load(pickled_files)
        train_label = pickled_obj['train_label']
        test_label  = pickled_obj['test_label']
        fe =  pickled_obj['fe']

        print('Fetching individual trainning features')
        uni_train_tf_feature= fe.uni_train_tf_feature
        ngram_train_tf_feature= fe.ngram_train_tf_feature
        char_train_tf_feature= fe.char_train_tf_feature
        count_vector_train_feature= fe.count_vector_train_feature
        lda_train_vec_feature= fe.lda_train_vec_feature

        print('Fetching individual testing features')
        uni_test_tf_feature = fe.uni_test_tf_feature
        ngram_test_tf_feature = fe.ngram_test_tf_feature
        char_test_tf_feature = fe.char_test_tf_feature
        count_vector_test_feature = fe.count_vector_test_feature
        lda_test_vec_feature = fe.lda_test_vec_feature


        acc_nb_uni = model.ml_pipeline(MultinomialNB(),train_label,uni_train_tf_feature).score(uni_test_tf_feature,test_label)
        acc_lr_uni = model.ml_pipeline(LogisticRegression(),train_label,uni_train_tf_feature).score(uni_test_tf_feature,test_label) 
        acc_sv_uni = model.ml_pipeline(SVC(kernel='linear'),train_label,uni_train_tf_feature).score(uni_test_tf_feature,test_label)

        uni_accuracies = [acc_nb_uni,acc_lr_uni,acc_sv_uni]

        acc_nb_char = model.ml_pipeline(MultinomialNB(),train_label,char_train_tf_feature).score(char_test_tf_feature,test_label)
        acc_lr_char = model.ml_pipeline(LogisticRegression(),train_label,char_train_tf_feature).score(char_test_tf_feature,test_label) 
        acc_sv_char = model.ml_pipeline(SVC(kernel='linear'),train_label,char_train_tf_feature).score(char_test_tf_feature,test_label)

        char_accuracies = [acc_nb_char,acc_lr_char,acc_sv_char]

        acc_nb_ngram = model.ml_pipeline(MultinomialNB(),train_label,ngram_train_tf_feature).score(ngram_test_tf_feature,test_label)
        acc_lr_ngram = model.ml_pipeline(LogisticRegression(),train_label,ngram_train_tf_feature).score(ngram_test_tf_feature,test_label) 
        acc_sv_ngram = model.ml_pipeline(SVC(kernel='linear'),train_label,ngram_train_tf_feature).score(ngram_test_tf_feature,test_label)

        ngram_accuracies = [acc_nb_ngram,acc_lr_ngram,acc_sv_ngram]        





                

                
                

                


 




















perform_composite_feature_test_analysis(load_saved_object_files())




    

    






